export function buildWishlistPrUrl(fork: string, branch: string): string {
  const repository = fork.trim().replace(/^https:\/\/github\.com\//i, '').replace(/\/$/, '');
  const match = /^([a-z\d](?:[a-z\d-]{0,37}[a-z\d])?)\/([a-z\d._-]+)$/i.exec(repository);
  if (!match || match[2] === '.' || match[2] === '..') {
    throw new Error('Enter your fork as username/repository or its GitHub repository URL.');
  }

  const [, owner, repo] = match;
  const ref = branch.trim();
  const hasControlCharacter = Array.from(ref).some((char) => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);
  if (!ref || hasControlCharacter || ref.startsWith('-') || ref === '@' || /[\s~^:?*\[\\]/.test(ref)
    || ref.includes('..') || ref.includes('@{')
    || ref.split('/').some((part) => !part || part.startsWith('.') || part.endsWith('.') || part.endsWith('.lock'))) {
    throw new Error('Enter the branch name that contains your committed request files.');
  }
  if (owner.toLowerCase() === 'nvlabs' && repo.toLowerCase() === 'kda' && ref === 'wishlist') {
    throw new Error('Enter your fork and request branch so GitHub can compare your new files.');
  }

  const url = new URL(`https://github.com/NVlabs/kda/compare/wishlist...${owner}:${repo}:${encodeURIComponent(ref)}`);
  url.search = new URLSearchParams({ quick_pull: '1', template: 'wishlist.md', title: '[wishlist] ' }).toString();
  return url.toString();
}
