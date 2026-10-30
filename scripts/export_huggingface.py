# Copyright 2026 Co-developed by KDA (Kernel Design Agents)
"""Export a validated CuTe kernel directory to a Hugging Face kernel package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import keyword
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

_HEADER = "# Copyright 2026 Co-developed by KDA (Kernel Design Agents)\n"


@dataclass(frozen=True)
class ExportConfig:
    repo_id: str
    version: int
    license: str
    license_file: Path
    cuda_capabilities: tuple[str, ...]
    python_depends: tuple[str, ...] = ("nvidia-cutlass-dsl",)
    readme_file: Path | None = None

    @property
    def name(self) -> str:
        return self.repo_id.split("/")[1]

    @classmethod
    def read(cls, path: Path) -> ExportConfig:
        path = Path(path).resolve()
        data = tomllib.loads(path.read_text())
        for field in ("license_file", "readme_file"):
            if field in data:
                data[field] = (path.parent / data[field]).resolve()
        for field in ("cuda_capabilities", "python_depends"):
            if field in data:
                values = data[field]
                if not isinstance(values, list) or not all(
                    isinstance(value, str) and value for value in values
                ):
                    raise ValueError(f"{field} must be a list of nonempty strings")
                data[field] = tuple(values)
        config = cls(**data)
        if not re.fullmatch(r"[\w-]+/[A-Za-z_][\w-]*", config.repo_id):
            raise ValueError("repo_id must be an owner/name with a Python module name")
        if keyword.iskeyword(config.name.replace("-", "_")):
            raise ValueError("the kernel name must not be a Python keyword")
        if type(config.version) is not int or config.version < 1:
            raise ValueError("version must be a positive integer")
        if not config.cuda_capabilities or not all(
            re.fullmatch(r"\d+\.\d+[af]?", capability) for capability in config.cuda_capabilities
        ):
            raise ValueError("cuda_capabilities must explicitly list supported GPUs")
        if not isinstance(config.license, str) or not config.license.strip():
            raise ValueError("license must be explicitly specified")
        if not config.license_file.read_text().strip():
            raise ValueError("license_file must contain the license text")
        return config


def _snapshot_solution(solution: Path, destination: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(solution.rglob("*")):
        relative = path.relative_to(solution)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"source symlinks are not supported: {relative}")
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".json"}:
            raise ValueError(f"only Python and JSON source files can export: {relative}")
        data = path.read_bytes()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def _relative_prefix(source_dir: Path, import_root: Path) -> str:
    common = 0
    for source_part, root_part in zip(source_dir.parts, import_root.parts):
        if source_part != root_part:
            break
        common += 1
    parents = "." * (len(source_dir.parts) - common + 1)
    destination = ".".join(import_root.parts[common:])
    return parents + (destination + "." if destination else "")


def _rewrite_imports(
    source: str,
    relative_path: Path,
    import_root: Path,
    local_modules: set[str],
    solution_modules: set[str] | None = None,
) -> str:
    tree = ast.parse(source, filename=str(relative_path))
    solution_modules = solution_modules or set()

    def local_prefix(module: str) -> str | None:
        name = module.split(".")[0]
        if name in local_modules:
            root = import_root
        elif name in solution_modules:
            root = Path()
        else:
            return None
        return _relative_prefix(relative_path.parent, root)

    lines = source.encode().splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    replacements = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            dynamic = (isinstance(node.func, ast.Name) and node.func.id == "__import__") or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"import_module", "spec_from_file_location"}
            )
            if dynamic:
                raise ValueError(
                    f"{relative_path}:{node.lineno}: dynamic imports require manual packaging"
                )
        replacement = None
        if isinstance(node, ast.Import):
            statements = []
            changed = False
            for alias in node.names:
                prefix = local_prefix(alias.name)
                if prefix is None:
                    statements.append(f"import {ast.unparse(alias)}")
                    continue
                changed = True
                if "." in alias.name:
                    if alias.asname is None:
                        raise ValueError(
                            f"{relative_path}:{node.lineno}: dotted local imports need an alias"
                        )
                    parent, name = alias.name.rsplit(".", 1)
                    statements.append(f"from {prefix}{parent} import {name} as {alias.asname}")
                else:
                    statements.append(
                        f"from {prefix.rstrip('.') or prefix} import {ast.unparse(alias)}"
                    )
            if changed:
                replacement = "; ".join(statements)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and (prefix := local_prefix(node.module)) is not None
        ):
            names = ", ".join(ast.unparse(alias) for alias in node.names)
            replacement = f"from {prefix}{node.module} import {names}"
        if replacement is not None:
            start = starts[node.lineno - 1] + node.col_offset
            end = starts[node.end_lineno - 1] + node.end_col_offset
            replacements.append((start, end, replacement.encode()))
    encoded = source.encode()
    for start, end, replacement in sorted(replacements, reverse=True):
        encoded = encoded[:start] + replacement + encoded[end:]
    rewritten = encoded.decode()
    ast.parse(rewritten, filename=str(relative_path))
    return rewritten


def _package_solution(solution: Path, entrypoint: Path, package: Path) -> None:
    import_root = entrypoint.parent
    local_modules = {
        path.stem if path.is_file() else path.name
        for path in (solution / import_root).iterdir()
        if path.suffix == ".py" or path.is_dir()
    }
    solution_modules = {
        path.stem if path.is_file() else path.name
        for path in solution.iterdir()
        if path.suffix == ".py" or path.is_dir()
    }
    adapted = package / "_solution"
    shutil.copytree(solution, adapted)
    for path in sorted(adapted.rglob("*.py")):
        relative = path.relative_to(adapted)
        if any(
            not part.isidentifier() or keyword.iskeyword(part)
            for part in relative.with_suffix("").parts
        ):
            raise ValueError(f"not a Python module path: {relative}")
        source = path.read_text()
        rewritten = _rewrite_imports(
            source, relative, import_root, local_modules, solution_modules
        )
        if rewritten != source:
            path.write_text(rewritten)
        for parent in path.parents:
            if not parent.is_relative_to(adapted):
                break
            initializer = parent / "__init__.py"
            if not initializer.exists():
                initializer.write_text(_HEADER)
    module_parts = ["_solution", *entrypoint.with_suffix("").parts]
    if module_parts[-1] == "__init__":
        module_parts.pop()
    parent = "." + ".".join(module_parts[:-1])
    (package / "__init__.py").write_text(
        _HEADER
        + f"from {parent} import {module_parts[-1]} as _entrypoint\n"
        + '\nrun = _entrypoint.run\n__all__ = ["run"]\n'
        + 'if hasattr(_entrypoint, "prepare"):\n'
        + '    prepare = _entrypoint.prepare\n    __all__.append("prepare")\n'
    )


def _build(source_dir: Path, kernel_builder: str, unique_id: str) -> None:
    subprocess.run(
        [kernel_builder, "create-pyproject", "--unique-id", unique_id, str(source_dir)],
        check=True,
    )
    metadata_path = source_dir / "metadata-cuda.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.get("provenance", {}).pop("kernel", None)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    subprocess.run(
        [sys.executable, "-B", "setup.py", "build_kernel", "--backends=cuda"],
        cwd=source_dir,
        check=True,
    )
    subprocess.run([kernel_builder, "hash", str(source_dir)], check=True)


def export_huggingface(
    solution_dir: Path,
    entrypoint: Path,
    config_path: Path,
    output_dir: Path,
    *,
    kernel_builder: str = "kernel-builder",
) -> Path:
    """Package a selected candidate; task validation and upload are separate."""
    solution_dir = Path(solution_dir).resolve()
    entrypoint = Path(entrypoint)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    if not solution_dir.is_dir():
        raise ValueError(f"solution directory does not exist: {solution_dir}")
    if output_dir.is_relative_to(solution_dir):
        raise ValueError("output directory must be outside the solution directory")
    if entrypoint.is_absolute() or ".." in entrypoint.parts or entrypoint.suffix != ".py":
        raise ValueError("entrypoint must be a relative Python file within the solution")
    config = ExportConfig.read(config_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kda-export-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        snapshot = staging / "solution"
        digest = _snapshot_solution(solution_dir, snapshot)
        if not (snapshot / entrypoint).is_file():
            raise ValueError("entrypoint must be a Python file in the solution directory")
        export_root = staging / "export"
        source_dir = export_root / "source"
        package = source_dir / "torch-ext" / config.name.replace("-", "_")
        package.mkdir(parents=True)
        _package_solution(snapshot, entrypoint, package)
        shutil.copyfile(config.license_file, package / "LICENSE.txt")
        build_config = (
            "[general]\n"
            f"name = {json.dumps(config.name)}\n"
            f"version = {config.version}\nedition = 5\n"
            f"license = {json.dumps(config.license)}\n"
            'backends = ["cuda"]\n'
            f"python-depends = {json.dumps([name for name in config.python_depends if name != 'nvidia-cutlass-dsl'])}\n\n"
            "[general.cuda]\n"
            f"python-depends = {json.dumps([name for name in config.python_depends if name == 'nvidia-cutlass-dsl'])}\n\n"
            "[general.hub]\n"
            f"repo-id = {json.dumps(config.repo_id)}\n\n"
            "[torch-noarch]\n"
            f"cuda_capabilities = {json.dumps(config.cuda_capabilities)}\n"
        )
        (source_dir / "build.toml").write_text(build_config)
        identity = hashlib.sha256(("kda-export-1\n" + digest + build_config).encode()).hexdigest()[
            :16
        ]
        _build(source_dir, kernel_builder, identity)
        hub_dir = export_root / "hub"
        shutil.copytree(source_dir / "build", hub_dir / "build")
        readme = (
            config.readme_file.read_text()
            if config.readme_file
            else (
                "---\nlibrary_name: kernels\n"
                f"license: {json.dumps(config.license)}\n"
                "tags: [cuda, cute-dsl]\n---\n\n"
                f"# {config.name}\n\n"
                "CuTe-DSL kernel.\n\n"
                "Exposes the candidate's `run` and optional `prepare` functions.\n"
            )
        )
        (hub_dir / "README.md").write_text(readme)
        (hub_dir / "build/CARD.md").write_text(readme)
        export_root.rename(output_dir)
    return output_dir / "hub"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-dir", type=Path, required=True)
    parser.add_argument("--entrypoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-builder", default="kernel-builder")
    args = parser.parse_args(argv)
    try:
        hub_dir = export_huggingface(
            args.solution_dir,
            args.entrypoint,
            args.config,
            args.output,
            kernel_builder=args.kernel_builder,
        )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        SyntaxError,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(f"Packaged kernel: {hub_dir}\nValidate the exported package before publishing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
