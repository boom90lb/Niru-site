# Niru ML site

The public site for Niru ML: plain HTML and one stylesheet, with no build step
and no JavaScript of its own. Everything served lives in `docs/`.

- `docs/index.html`, `docs/about/index.html` and `docs/404.html` are the pages.
  Each carries the same header and footer, so a change to either goes in all
  three.
- `docs/style.css` holds every rule. Dark mode follows the reader's system
  setting.
- `docs/fonts/` holds Source Serif 4 (latin subset, upright and italic) under
  the SIL Open Font License.
- `docs/_headers` and `docs/_redirects` are read by Cloudflare Pages, which
  serves `docs/` from `main` as the project `niru-site`.
- Cloudflare adds two scripts as it serves a page: the decoder for its email
  address obfuscation and the Web Analytics beacon. The Content-Security-Policy
  in `docs/_headers` allows those two and nothing else.

To preview locally:

```sh
python3 -m http.server -d docs 8000
```

What the site may say is set by the Niru ML brief. While Niru ML is in stealth
it says that, gives a contact address and carries the About page, with no
results, figures or method detail.
