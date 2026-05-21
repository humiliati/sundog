/**
 * sd-strand-nav.mjs — scroll-spy for the in-page strand index.
 *
 * Looks up `<nav class="sd-strand-nav">` and each `<a class="sd-strand-link"
 * href="#section-id">`. Observes the matching `<section id="section-id">`
 * elements with IntersectionObserver, sets `aria-current="true"` on the link
 * whose section is currently in the upper-viewport active band.
 *
 * The active band is shaped by a rootMargin that shrinks the viewport's top
 * to 15% and bottom to 70%, leaving a ~15%-tall strip near the top as the
 * "current" zone. Multiple short sections can be in that strip at once; we
 * pick the topmost one (document order) so the highlight progresses linearly
 * as the reader scrolls.
 *
 * If the browser doesn't support IntersectionObserver, the nav still renders
 * as static anchors. Hash-jump clicks still work because section ids carry
 * scroll-margin-top in the theme CSS.
 */

(function () {
  const nav = document.querySelector('.sd-strand-nav');
  if (!nav) return;

  const links = Array.from(nav.querySelectorAll('.sd-strand-link'));
  if (!links.length) return;

  // Map href -> link element, and collect target sections in document order.
  const linkByHref = new Map();
  const targets = [];
  links.forEach((link) => {
    const href = link.getAttribute('href') || '';
    if (!href.startsWith('#')) return;
    const id = href.slice(1);
    const target = document.getElementById(id);
    if (!target) return;
    linkByHref.set(href, link);
    targets.push(target);
  });

  if (!targets.length || !('IntersectionObserver' in window)) return;

  // Sort targets by document position so "topmost visible" is well-defined.
  targets.sort((a, b) => {
    const rel = a.compareDocumentPosition(b);
    if (rel & Node.DOCUMENT_POSITION_FOLLOWING) return -1;
    if (rel & Node.DOCUMENT_POSITION_PRECEDING) return 1;
    return 0;
  });

  const visible = new Set();

  function setCurrent(id) {
    links.forEach((link) => {
      if (link.getAttribute('href') === '#' + id) {
        link.setAttribute('aria-current', 'true');
      } else {
        link.removeAttribute('aria-current');
      }
    });
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) visible.add(entry.target.id);
        else visible.delete(entry.target.id);
      });

      // Pick the topmost (first in document order) visible section.
      const current = targets.find((t) => visible.has(t.id));
      if (current) setCurrent(current.id);
      // If nothing is in the active band (e.g. between sections), keep the
      // last current marker. Removing it on every miss makes the highlight
      // flicker on slow scrolls.
    },
    {
      // ~15%-tall strip near the top of the viewport is the "current" zone.
      rootMargin: '-15% 0px -70% 0px',
      threshold: 0,
    },
  );

  targets.forEach((t) => observer.observe(t));
})();
