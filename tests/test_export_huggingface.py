# Copyright 2026 Co-developed by KDA (Kernel Design Agents)

import base64
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from scripts import export_huggingface as export


def _candidate(root, files=None, entrypoint="kernel.py"):
    solution = root / "solution"
    solution.mkdir(parents=True)
    for name, content in (files or {"kernel.py": "def run(): return 7\n"}).items():
        path = solution / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (root / "LICENSE.txt").write_text("Example license text\n")
    config = root / "release.toml"
    config.write_text(
        'repo_id = "example/demo-kernel"\nversion = 1\n'
        'license = "MIT"\nlicense_file = "LICENSE.txt"\n'
        'cuda_capabilities = ["10.0"]\n'
    )
    return solution, Path(entrypoint), config, root / "export"


@pytest.fixture
def fake_builder(monkeypatch):
    def build(source_dir, kernel_builder, unique_id):
        package = next((source_dir / "torch-ext").iterdir())
        variant = source_dir / "build/torch-cuda"
        shutil.copytree(package, variant)
        (variant / "metadata.json").write_text(
            json.dumps({"id": f"_demo_{unique_id}", "digest": {"files": {}}})
        )

    monkeypatch.setattr(export, "_build", build)


def _load(hub_dir):
    variant = hub_dir / "build/torch-cuda"
    metadata = json.loads((variant / "metadata.json").read_text())
    spec = importlib.util.spec_from_file_location(metadata["id"], variant / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[metadata["id"]] = module
    spec.loader.exec_module(module)
    return module


def test_export_preserves_inputs_and_separates_same_named_modules(tmp_path, fake_builder):
    modules = []
    search_path = sys.path.copy()
    for value in (1, 2):
        root = tmp_path / str(value)
        arguments = _candidate(
            root,
            {
                "kernel.py": (
                    "import helper\n\ndef run(): return helper.VALUE\ndef prepare(): return run\n"
                ),
                "helper.py": f"VALUE = {value}\n",
            },
        )
        original = {
            path.relative_to(arguments[0]): path.read_bytes()
            for path in arguments[0].rglob("*")
            if path.is_file()
        }
        hub_dir = export.export_huggingface(*arguments)
        modules.append(_load(hub_dir))
        assert {
            path.relative_to(arguments[0]): path.read_bytes()
            for path in arguments[0].rglob("*")
            if path.is_file()
        } == original
    assert [module.run() for module in modules] == [1, 2]
    assert [module.prepare()() for module in modules] == [1, 2]
    assert sys.path == search_path


@pytest.mark.parametrize(
    "entrypoint",
    [
        "kernel.py",
        "package/kernel.py",
        "package/__init__.py",
    ],
)
def test_entrypoints_and_local_import_roots(tmp_path, fake_builder, entrypoint):
    relative = Path(entrypoint)
    arguments = _candidate(
        tmp_path,
        {
            relative.as_posix(): "from helper import VALUE\n\ndef run(): return VALUE\n",
            (relative.parent / "helper.py").as_posix(): "VALUE = 9\n",
        },
        entrypoint=entrypoint,
    )
    assert _load(export.export_huggingface(*arguments)).run() == 9


def test_nested_entrypoint_rewrites_absolute_package_import(tmp_path, fake_builder):
    arguments = _candidate(
        tmp_path,
        {
            "package/kernel.py": (
                "from package.helper import VALUE\n\ndef run(): return VALUE\n"
            ),
            "package/helper.py": "VALUE = 42\n",
        },
        entrypoint="package/kernel.py",
    )
    assert _load(export.export_huggingface(*arguments)).run() == 42


@pytest.mark.parametrize(
    ("source", "path", "expected"),
    [
        (
            "import os, helper as local\n",
            "kernel.py",
            "import os; from . import helper as local\n",
        ),
        (
            "from helper import value as result\n",
            "nested/module.py",
            "from ..helper import value as result\n",
        ),
        (
            "import helper.child as local\n",
            "kernel.py",
            "from .helper import child as local\n",
        ),
        (
            "if True: import helper; result = helper.VALUE\n",
            "kernel.py",
            "if True: from . import helper; result = helper.VALUE\n",
        ),
        (
            'text = "汉字"; import helper\n',
            "kernel.py",
            'text = "汉字"; from . import helper\n',
        ),
    ],
)
def test_rewrite_imports_preserves_binding_and_surrounding_code(source, path, expected):
    assert export._rewrite_imports(source, Path(path), Path("."), {"helper"}) == expected


@pytest.mark.parametrize(
    "source",
    [
        "import helper.child\n",
        "__import__('helper')\n",
        "importlib.import_module(name)\n",
    ],
)
def test_unsupported_imports_are_reported(source):
    with pytest.raises(ValueError, match="kernel.py:1"):
        export._rewrite_imports(source, Path("kernel.py"), Path("."), {"helper"})


@pytest.mark.parametrize("entrypoint", ["missing.py", "../outside.py", "/absolute.py", "data.json"])
def test_invalid_entrypoint_is_rejected(tmp_path, entrypoint):
    arguments = _candidate(tmp_path, entrypoint=entrypoint)
    with pytest.raises(ValueError, match="entrypoint"):
        export.export_huggingface(*arguments)
    assert not arguments[3].exists()


def test_output_cannot_be_inside_solution(tmp_path):
    solution, entrypoint, config, _ = _candidate(tmp_path)
    with pytest.raises(ValueError, match="outside the solution"):
        export.export_huggingface(solution, entrypoint, config, solution / "export")


@pytest.mark.parametrize("directory", [False, True])
def test_source_symlinks_are_rejected(tmp_path, directory):
    arguments = _candidate(tmp_path)
    target = tmp_path / "helper"
    if directory:
        target.mkdir()
    else:
        target.write_text("VALUE = 1\n")
    (arguments[0] / "helper.py").symlink_to(target, target_is_directory=directory)
    with pytest.raises(ValueError, match="symlinks"):
        export.export_huggingface(*arguments)
    assert not arguments[3].exists()


def test_json_data_is_preserved_and_bytecode_is_ignored(tmp_path, fake_builder):
    arguments = _candidate(
        tmp_path, {"kernel.py": "def run(): return 7\n", "data.json": '{"v": 7}\n'}
    )
    cache = arguments[0] / "__pycache__"
    cache.mkdir()
    (cache / "kernel.pyc").write_bytes(b"bytecode")
    hub = export.export_huggingface(*arguments)
    source = hub / "build/torch-cuda/_solution"
    assert (source / "data.json").read_bytes() == (arguments[0] / "data.json").read_bytes()
    assert not (source / "__pycache__").exists()


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("version = 1", "version = true"),
        ("version = 1", "version = 0"),
        ('cuda_capabilities = ["10.0"]', "cuda_capabilities = []"),
        ('repo_id = "example/demo-kernel"', 'repo_id = "example/../outside"'),
        ('repo_id = "example/demo-kernel"', 'repo_id = "example/class"'),
    ],
)
def test_invalid_configuration(tmp_path, original, replacement):
    arguments = _candidate(tmp_path)
    arguments[2].write_text(arguments[2].read_text().replace(original, replacement))
    with pytest.raises(ValueError):
        export.export_huggingface(*arguments)


def test_native_sources_require_a_different_build(tmp_path):
    arguments = _candidate(tmp_path, {"kernel.py": "def run(): pass\n", "kernel.cu": "source"})
    with pytest.raises(ValueError, match="only Python and JSON"):
        export.export_huggingface(*arguments)


def test_build_failure_leaves_no_partial_output(tmp_path, monkeypatch):
    arguments = _candidate(tmp_path)

    def fail(*args):
        raise subprocess.CalledProcessError(1, "kernel-builder")

    monkeypatch.setattr(export, "_build", fail)
    with pytest.raises(subprocess.CalledProcessError):
        export.export_huggingface(*arguments)
    assert not arguments[3].exists()
    assert not list(tmp_path.glob(".kda-export-*"))


def test_existing_output_is_not_overwritten(tmp_path):
    arguments = _candidate(tmp_path)
    arguments[3].mkdir()
    with pytest.raises(ValueError, match="already exists"):
        export.export_huggingface(*arguments)


@pytest.mark.parametrize("custom_card", [False, True])
def test_export_includes_card_for_official_upload(tmp_path, fake_builder, custom_card):
    arguments = _candidate(tmp_path)
    if custom_card:
        (tmp_path / "CARD.md").write_text("# Custom kernel card\n")
        with arguments[2].open("a") as config:
            config.write('readme_file = "CARD.md"\n')
    hub_dir = export.export_huggingface(*arguments)
    readme = (hub_dir / "README.md").read_text()
    assert (hub_dir / "build/CARD.md").read_text() == readme
    if custom_card:
        assert readme == "# Custom kernel card\n"
    else:
        assert "# demo-kernel" in readme


def test_builder_separates_cuda_and_general_dependencies(tmp_path, fake_builder):
    arguments = _candidate(tmp_path)
    with arguments[2].open("a") as config:
        config.write('python_depends = ["einops", "nvidia-cutlass-dsl"]\n')
    export.export_huggingface(*arguments)
    build = tomllib.loads((arguments[3] / "source/build.toml").read_text())
    assert build["general"]["python-depends"] == ["einops"]
    assert build["general"]["cuda"]["python-depends"] == ["nvidia-cutlass-dsl"]
    assert build["torch-noarch"]["cuda_capabilities"] == ["10.0"]


def test_builder_does_not_claim_parent_repository_provenance(tmp_path, monkeypatch):
    commands = []
    metadata_path = tmp_path / "metadata-cuda.json"

    def run(command, **kwargs):
        commands.append(command)
        if "create-pyproject" in command:
            metadata_path.write_text(
                json.dumps(
                    {
                        "provenance": {
                            "kernel": {"commit": "unrelated"},
                            "kernel-builder": {"version": "test"},
                        }
                    }
                )
            )

    monkeypatch.setattr(subprocess, "run", run)
    export._build(tmp_path, "kernel-builder", "example")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["provenance"] == {"kernel-builder": {"version": "test"}}
    assert len(commands) == 3


def test_real_kernel_builder_round_trip(tmp_path):
    builder = shutil.which("kernel-builder")
    if builder is None:
        pytest.skip("kernel-builder is not installed")
    arguments = _candidate(
        tmp_path,
        {
            "kernel.py": "from helper import VALUE\n\ndef run(): return VALUE\n",
            "helper.py": "VALUE = 37\n",
        },
    )
    hub_dir = export.export_huggingface(*arguments, kernel_builder=builder)
    assert _load(hub_dir).run() == 37
    assert (hub_dir / "build/CARD.md").read_text() == (hub_dir / "README.md").read_text()
    variant = hub_dir / "build/torch-cuda"
    metadata = json.loads((variant / "metadata.json").read_text())
    assert metadata["backend"] == {"type": "cuda", "archs": ["10.0"]}
    assert metadata["python-depends"] == ["nvidia-cutlass-dsl"]
    assert "kernel" not in metadata["provenance"]
    for relative_path, digest in metadata["digest"]["files"].items():
        observed = hashlib.sha256((variant / relative_path).read_bytes()).digest()
        assert base64.b64encode(observed).decode() == digest
