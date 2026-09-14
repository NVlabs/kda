'use client';

import { useState, type FormEvent } from 'react';
import { buildWishlistPrUrl } from './wishlist-pr-url';

export default function RequestForm() {
  const [error, setError] = useState('');

  function openPullRequest(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    try {
      const url = buildWishlistPrUrl(String(data.get('fork') ?? ''), String(data.get('branch') ?? ''));
      window.location.assign(url);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Check your fork and branch, then try again.');
    }
  }

  return (
    <section className="section shell request-section" id="submit" aria-labelledby="request-heading">
      <div className="section-heading">
        <p className="section-label">READY TO CONTRIBUTE</p>
        <h2 id="request-heading">Submit your<br />request</h2>
        <p className="request-intro">Push your request files to a branch based on <code>wishlist</code>, then enter your fork and branch.</p>
        <a className="text-link" href="https://github.com/NVlabs/kda/tree/wishlist#submit">Read the submission guide <span aria-hidden="true">↗</span></a>
      </div>
      <form className="request-form" onSubmit={openPullRequest} onInput={() => setError('')}>
        <div className="request-field">
          <label htmlFor="request-fork">Your fork</label>
          <input id="request-fork" name="fork" placeholder="your-name/kda" required autoComplete="off" spellCheck={false} aria-describedby="request-fork-help" />
          <p id="request-fork-help">Enter username/repository or the GitHub repository URL.</p>
        </div>
        <div className="request-field">
          <label htmlFor="request-branch">Your branch</label>
          <input id="request-branch" name="branch" placeholder="my-kernel-request" required autoComplete="off" spellCheck={false} aria-describedby="request-branch-help" />
          <p id="request-branch-help">The branch in your fork that contains the request files.</p>
        </div>
        <p className="request-target">Target: <strong>NVlabs/kda → wishlist</strong></p>
        {error && <p className="request-error" role="alert">{error}</p>}
        <button className="button button-primary" type="submit">Continue to GitHub <span aria-hidden="true">↗</span></button>
        <p className="request-note">Review your files and complete the wishlist template on GitHub before submitting.</p>
        <noscript><p>Enable JavaScript to build the link, or open a pull request from your fork targeting NVlabs/kda&apos;s wishlist branch.</p></noscript>
      </form>
    </section>
  );
}
