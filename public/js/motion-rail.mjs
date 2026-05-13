/* Highlights motion rail — Slice 1A: layout + seam.
 *
 * Responsibilities in this slice (see docs/HIGHLIGHTS_RAIL_ROADMAP.md
 * Build Phase 1):
 *
 *   1. Track which card is the centred one and set [data-rail-active]
 *      on it. The CSS in index.html dims and scales the rest into peek
 *      state.
 *   2. Wire the always-overlaid prev/next skip arrows. Clicking either
 *      arrow immediately lands the current card's verdict stamp before
 *      scrolling away — verdicts are never silently skipped.
 *   3. Inject a no-op `.sd-verdict-stamp` span into every card carrying
 *      data-stamp / data-stamp-meaning. The base styles live in
 *      public/css/sundog-theme.css; the [data-stamp-armed] toggle that
 *      makes them visible lands in Slice 1B.
 *
 * Out of scope for this slice (lands in Slice 1B):
 *
 *   - Auto-cycle controller (timer + per-verdict dwells + IO-driven
 *     advance on clip-end).
 *   - Pause-on-hover / pause-on-focus dwell tracking.
 *   - Replay-sequence affordance at end of sequence.
 *   - The rubber-stamp ink-bleed visual refinement.
 */

const RAIL_SELECTOR = "[data-motion-rail]";
const CARD_SELECTOR = ".motion-card";
const ACTIVE_ATTR = "data-rail-active";
const ARMED_ATTR = "data-stamp-armed";
const STAMP_CLASS = "sd-verdict-stamp";
const PREV_SELECTOR = "[data-rail-prev]";
const NEXT_SELECTOR = "[data-rail-next]";

const reducedMotion = window.matchMedia(
  "(prefers-reduced-motion: reduce)"
).matches;

initMotionRail();

function initMotionRail() {
  const rail = document.querySelector(RAIL_SELECTOR);
  if (!rail) {
    return;
  }
  const track = rail.querySelector(".motion-rail-track");
  const cards = Array.from(rail.querySelectorAll(CARD_SELECTOR));
  if (!track || cards.length === 0) {
    return;
  }

  applyVerdictStamps(cards);

  const state = {
    rail,
    track,
    cards,
    activeIndex: 0,
  };

  setupActiveTracking(state);
  wireSkipArrows(state, rail);

  // Make sure the first card is centred on initial load. Without this the
  // browser sometimes restores a scroll position from a previous session.
  centreCard(state, 0, { instant: true });
}

/**
 * Walks every .motion-card and ensures it carries a .sd-verdict-stamp
 * span. The span is positioned by the CSS in sundog-theme.css; this
 * function only ensures it exists and carries the right data-stamp
 * attribute. The span is invisible until [data-stamp-armed] is set on
 * the parent card (Slice 1B work).
 */
function applyVerdictStamps(cards) {
  for (const card of cards) {
    const stamp = card.dataset.stamp;
    if (!stamp) {
      continue;
    }

    let stampEl = card.querySelector(`.${STAMP_CLASS}`);
    if (!stampEl) {
      stampEl = document.createElement("span");
      stampEl.className = STAMP_CLASS;
      stampEl.setAttribute("aria-hidden", "true");
      // The card body is wrapped in an anchor (.motion-card-link); the
      // stamp must sit OUTSIDE that anchor so peeked-card clicks still
      // route to data-href without the stamp swallowing pointer events.
      card.appendChild(stampEl);
    }

    stampEl.setAttribute("data-stamp", stamp);
    stampEl.textContent = stamp;

    // Screen-reader exposure: the visual stamp is aria-hidden, but the
    // verdict and its gloss are read out via a visually-hidden span
    // inside the card copy.
    const meaning = card.dataset.stampMeaning;
    if (meaning) {
      ensureSrVerdict(card, stamp, meaning);
    }
  }
}

function ensureSrVerdict(card, stamp, meaning) {
  let sr = card.querySelector(".sd-sr-verdict");
  if (!sr) {
    sr = document.createElement("span");
    sr.className = "sd-sr-verdict";
    // Inline screen-reader-only style; the shared SR utility class may
    // not exist yet so we self-host the minimum here.
    sr.style.cssText = [
      "position:absolute",
      "width:1px",
      "height:1px",
      "padding:0",
      "margin:-1px",
      "overflow:hidden",
      "clip:rect(0,0,0,0)",
      "white-space:nowrap",
      "border:0",
    ].join(";");
    card.appendChild(sr);
  }
  sr.textContent = `Verdict: ${humanize(stamp)}. ${meaning}`;
}

function humanize(stamp) {
  return stamp
    .toLowerCase()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/* -------------------------------------------------------------------------- *
 * Active-card tracking
 * -------------------------------------------------------------------------- */

function setupActiveTracking(state) {
  if (!("IntersectionObserver" in window)) {
    // Fallback for environments without IO: just mark the first card
    // active and let manual nav handle the rest.
    setActive(state, 0);
    return;
  }

  // Observer fires when a card crosses the track's centre line. The
  // threshold list lets us pick the card with the largest intersection
  // ratio on each event.
  const observer = new IntersectionObserver(
    (entries) => {
      let bestIndex = state.activeIndex;
      let bestRatio = 0;
      for (const entry of entries) {
        if (!entry.isIntersecting) {
          continue;
        }
        const idx = state.cards.indexOf(entry.target);
        if (idx >= 0 && entry.intersectionRatio > bestRatio) {
          bestRatio = entry.intersectionRatio;
          bestIndex = idx;
        }
      }
      if (bestIndex !== state.activeIndex && bestRatio > 0.55) {
        setActive(state, bestIndex);
      }
    },
    {
      root: state.track,
      // The centred card occupies roughly the middle 64% of the track;
      // setting rootMargin to negative left/right values means only the
      // truly-centred card breaches the threshold.
      rootMargin: "0px -20% 0px -20%",
      threshold: [0.4, 0.6, 0.8, 0.95],
    }
  );

  for (const card of state.cards) {
    observer.observe(card);
  }
}

function setActive(state, index) {
  if (index < 0 || index >= state.cards.length) {
    return;
  }
  for (let i = 0; i < state.cards.length; i++) {
    const card = state.cards[i];
    if (i === index) {
      card.setAttribute(ACTIVE_ATTR, "");
    } else {
      card.removeAttribute(ACTIVE_ATTR);
    }
  }
  state.activeIndex = index;
}

/* -------------------------------------------------------------------------- *
 * Skip arrows — always-overlaid prev/next
 * -------------------------------------------------------------------------- */

function wireSkipArrows(state, rail) {
  const prev = rail.querySelector(PREV_SELECTOR);
  const next = rail.querySelector(NEXT_SELECTOR);

  if (prev) {
    prev.addEventListener("click", () => skipTo(state, state.activeIndex - 1));
  }
  if (next) {
    next.addEventListener("click", () => skipTo(state, state.activeIndex + 1));
  }
}

function skipTo(state, targetIndex) {
  // Clamp; sequence does not wrap in Slice 1A (Slice 1B handles
  // end-of-sequence settle + replay).
  const clamped = Math.max(
    0,
    Math.min(state.cards.length - 1, targetIndex)
  );
  if (clamped === state.activeIndex) {
    return;
  }

  // Land the current card's stamp before scrolling away. This is the
  // load-bearing rule: skipping a card drops its remaining clip beat,
  // but its verdict is never silently dropped.
  armStampIfPresent(state.cards[state.activeIndex]);

  centreCard(state, clamped, { instant: reducedMotion });
}

function armStampIfPresent(card) {
  if (!card || !card.dataset.stamp) {
    return;
  }
  if (!card.hasAttribute(ARMED_ATTR)) {
    card.setAttribute(ARMED_ATTR, "");
  }
}

function centreCard(state, index, { instant = false } = {}) {
  const card = state.cards[index];
  if (!card) {
    return;
  }
  // Programmatic scroll. The IntersectionObserver will pick up the new
  // centre and call setActive() — but we also set the active state
  // synchronously here so the visual response is instant.
  setActive(state, index);
  card.scrollIntoView({
    behavior: instant ? "auto" : "smooth",
    block: "nearest",
    inline: "center",
  });
}
