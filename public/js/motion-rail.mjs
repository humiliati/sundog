/* Highlights motion rail — Slice 1B: stamp-cued auto-cycle.
 *
 * Per-card lifecycle (see docs/HIGHLIGHTS_RAIL_ROADMAP.md):
 *
 *   centre → clip plays → arm stamp → dwell → advance
 *
 * Inputs on each .motion-card:
 *   data-stamp           (required) one of:
 *                         CONFIRMED | OPERATING ENVELOPE | PLAUSIBLE |
 *                         BOUNDARY FOUND | STALLED | UNTESTED
 *   data-stamp-meaning   (required) one-line gloss for SR layer
 *   data-stamp-source    (required) repo path to the owning roadmap
 *   data-clip-ms         (optional) override for the implied clip beat
 *                         (default 1600ms for static-poster cards;
 *                          for cards with data-media this is ignored —
 *                          media duration is used directly)
 *   data-dwell-ms        (optional) post-stamp hold before advancing.
 *                         Default = VERDICT_DWELL_DEFAULTS[stamp].
 *
 * Pause rules:
 *   - Hover on track:      pauses dwell, resumes on leave
 *   - Focus inside track:  pauses dwell, resumes on focusout
 *   - Manual nav:          DISABLES auto-cycle for the rest of the page
 *                          session. No resume. data-rail-state="user".
 *   - prefers-reduced-motion: no auto-cycle at all. All stamps armed
 *                          on init.
 *
 * End of sequence:
 *   The controller wraps last -> first and first -> last so the rail can
 *   keep moving in either direction. The replay arrow remains a reset affordance
 *   after manual takeover.
 */

const RAIL_SELECTOR = "[data-motion-rail]";
const CARD_SELECTOR = ".motion-card";
const ACTIVE_ATTR = "data-rail-active";
const ARMED_ATTR = "data-stamp-armed";
const ARMED_JUST_FIRED_ATTR = "data-stamp-armed-just-fired";
const RAIL_STATE_ATTR = "data-rail-state";
const LOOP_CLONE_ATTR = "data-rail-clone";
const STAMP_CLASS = "sd-verdict-stamp";
const PREV_SELECTOR = "[data-rail-prev]";
const NEXT_SELECTOR = "[data-rail-next]";
const REPLAY_SELECTOR = "[data-rail-replay]";

const DEFAULT_CLIP_MS = 1600;
const SHAKE_MS = 80;

// Per-verdict dwell defaults (ms). The roadmap commits to these.
const VERDICT_DWELL_DEFAULTS = {
  "CONFIRMED": 2200,
  "OPERATING ENVELOPE": 1800,
  "PLAUSIBLE": 1800,
  "BOUNDARY FOUND": 3600,
  "STALLED": 3000,
  "UNTESTED": 1500,
};

const reducedMotionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");

initMotionRail();

function initMotionRail() {
  const rail = document.querySelector(RAIL_SELECTOR);
  if (!rail) {
    return;
  }
  const section = rail.matches(".motion-rail-section")
    ? rail
    : rail.closest(".motion-rail-section");
  const track = rail.querySelector(".motion-rail-track");
  const cards = Array.from(rail.querySelectorAll(CARD_SELECTOR));
  if (!track || cards.length === 0) {
    return;
  }

  applyVerdictStamps(cards);

  const state = {
    rail,
    section,
    track,
    cards,
    activeIndex: 0,
    /** Pending timer for the current phase (clip OR dwell). */
    timer: null,
    /** Phase the controller is currently in. */
    phase: "idle", // "idle" | "clip" | "dwell" | "settled" | "user"
    /** If the dwell is paused, this holds the ms remaining when paused. */
    pausedRemainingMs: null,
    /** When the current phase started, ms since epoch. Used to compute
     *  how much of a phase has elapsed when pausing. */
    phaseStartedAt: 0,
    /** Total ms the current phase was scheduled for. */
    phaseDurationMs: 0,
    /** Once user clicks prev/next manually, the rail goes user-driven
     *  for the rest of the session. */
    userDriven: false,
    /** Track whether we're currently in a pause condition (hover/focus). */
    paused: false,
    /** Circular-scroll clone buffers for the visual infinite rail. */
    loopEnabled: false,
    loopSpan: 0,
    loopJumping: false,
    beforeClones: [],
    afterClones: [],
    /** Direction of the current programmed move: -1, 0, or 1. */
    moveDirection: 0,
  };

  setupCircularScroll(state);
  setupActiveTracking(state);
  wireSkipArrows(state);
  wireKeyboardNavigation(state);
  wireReplayButton(state);
  wireManualTakeover(state);
  setupPauseListeners(state);
  setupReducedMotion(state);

  // Centre the first card immediately. The browser may have restored a
  // scroll position from a previous session — undo that.
  setActive(state, 0);
  centreCard(state, 0, { instant: true });

  if (reducedMotionQuery.matches) {
    // No auto-cycle; arm every stamp now.
    armAllStamps(state);
    setRailState(state, "settled");
  } else {
    // Begin the auto-cycle on the first card. A small delay gives the
    // initial scroll a chance to settle before the first clip starts.
    state.timer = window.setTimeout(() => startClipPhase(state), 320);
    state.phase = "clip";
    setRailState(state, "cycling");
  }
}

/* -------------------------------------------------------------------------- *
 * Stamp DOM injection
 * -------------------------------------------------------------------------- */

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
      card.appendChild(stampEl);
    }

    stampEl.setAttribute("data-stamp", stamp);
    stampEl.textContent = stamp;

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
    setActive(state, 0);
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      let bestIndex = state.activeIndex;
      let bestRatio = 0;
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const idx = state.cards.indexOf(entry.target);
        if (idx >= 0 && entry.intersectionRatio > bestRatio) {
          bestRatio = entry.intersectionRatio;
          bestIndex = idx;
        }
      }
      // Only trust the observer when not in a programmatic scroll. The
      // controller drives the canonical activeIndex; the observer is a
      // secondary signal for the cases where the user manually swiped
      // or scrolled. We deliberately *don't* re-enter cycle phases on
      // observer-only changes — that would race with the timer.
      if (bestIndex !== state.activeIndex && bestRatio > 0.6) {
        // Update the active flag silently — the controller's phase
        // machine still owns scheduling.
        markActiveOnly(state, bestIndex);
      }
    },
    {
      root: state.track,
      rootMargin: "0px -20% 0px -20%",
      threshold: [0.4, 0.6, 0.8, 0.95],
    }
  );

  for (const card of state.cards) {
    observer.observe(card);
  }
}

function markActiveOnly(state, index) {
  state.activeIndex = index;
  applyActiveFlags(state);
}

function setActive(state, index) {
  state.activeIndex = index;
  applyActiveFlags(state);
}

function applyActiveFlags(state) {
  for (let i = 0; i < state.cards.length; i++) {
    const card = state.cards[i];
    if (i === state.activeIndex) {
      card.setAttribute(ACTIVE_ATTR, "");
    } else {
      card.removeAttribute(ACTIVE_ATTR);
    }
  }
}

function setupCircularScroll(state) {
  if (state.cards.length < 2) return;

  const before = document.createDocumentFragment();
  const after = document.createDocumentFragment();
  state.beforeClones = state.cards.map((card, index) => cloneLoopCard(card, "before", index));
  state.afterClones = state.cards.map((card, index) => cloneLoopCard(card, "after", index));

  for (const clone of state.beforeClones) before.appendChild(clone);
  for (const clone of state.afterClones) after.appendChild(clone);
  state.track.insertBefore(before, state.track.firstChild);
  state.track.appendChild(after);
  state.loopEnabled = true;

  window.requestAnimationFrame(() => {
    state.loopSpan = state.track.scrollWidth / 3;
  });

  state.track.addEventListener("scroll", () => maintainCircularScroll(state), { passive: true });
}

function cloneLoopCard(card, position, index) {
  const clone = card.cloneNode(true);
  clone.setAttribute(LOOP_CLONE_ATTR, position);
  clone.dataset.railCloneIndex = String(index);
  clone.setAttribute("aria-hidden", "true");
  clone.removeAttribute(ACTIVE_ATTR);
  clone.removeAttribute(ARMED_ATTR);
  clone.removeAttribute(ARMED_JUST_FIRED_ATTR);

  for (const focusable of clone.querySelectorAll("a, button, input, select, textarea, [tabindex]")) {
    focusable.setAttribute("tabindex", "-1");
  }

  return clone;
}

function maintainCircularScroll(state) {
  if (!state.loopEnabled || state.loopJumping) return;
  const span = state.loopSpan || state.track.scrollWidth / 3;
  if (!Number.isFinite(span) || span <= 0) return;

  const low = span * 0.5;
  const high = span * 1.5;
  let delta = 0;
  if (state.track.scrollLeft < low) {
    delta = span;
  } else if (state.track.scrollLeft > high) {
    delta = -span;
  }
  if (delta === 0) return;

  state.loopJumping = true;
  state.track.scrollLeft += delta;
  window.requestAnimationFrame(() => {
    state.loopJumping = false;
  });
}

function centreCard(state, index, { instant = false } = {}) {
  let card = state.cards[index];
  if (!card) return;
  if (state.loopEnabled) {
    if (state.moveDirection > 0 && index === 0 && state.afterClones[0]) {
      card = state.afterClones[0];
    } else if (
      state.moveDirection < 0 &&
      index === state.cards.length - 1 &&
      state.beforeClones[state.cards.length - 1]
    ) {
      card = state.beforeClones[state.cards.length - 1];
    }
  }
  state.moveDirection = 0;
  const track = state.track;
  // Scroll ONLY the rail's own horizontal track. Element.scrollIntoView()
  // walks every scrollable ancestor up to the document, so each auto-cycle
  // advance yanked the whole page back to the rail while the reader was
  // elsewhere. Translating the card-vs-track centre delta into a track-local
  // horizontal scroll leaves the window scroll position untouched.
  const cardRect = card.getBoundingClientRect();
  const trackRect = track.getBoundingClientRect();
  const delta =
    cardRect.left + cardRect.width / 2 - (trackRect.left + trackRect.width / 2);
  if (Math.abs(delta) < 1) return;
  track.scrollBy({
    left: delta,
    behavior: instant ? "auto" : "smooth",
  });
}

/* -------------------------------------------------------------------------- *
 * Auto-cycle phase machine
 * -------------------------------------------------------------------------- */

function startClipPhase(state) {
  if (state.userDriven) return;
  const card = state.cards[state.activeIndex];
  if (!card) return;

  centreCard(state, state.activeIndex);
  setActive(state, state.activeIndex);

  state.phase = "clip";
  const clipMs = readClipMs(card);
  state.phaseDurationMs = clipMs;
  state.phaseStartedAt = performance.now();

  if (state.paused) {
    state.pausedRemainingMs = clipMs;
    return;
  }

  state.timer = window.setTimeout(() => onClipEnd(state), clipMs);
}

function onClipEnd(state) {
  if (state.userDriven) return;
  const card = state.cards[state.activeIndex];
  if (!card) return;

  armStamp(card);
  state.phase = "dwell";
  const dwellMs = readDwellMs(card);
  state.phaseDurationMs = dwellMs;
  state.phaseStartedAt = performance.now();

  if (state.paused) {
    state.pausedRemainingMs = dwellMs;
    return;
  }

  state.timer = window.setTimeout(() => onDwellEnd(state), dwellMs);
}

function onDwellEnd(state) {
  if (state.userDriven) return;
  state.activeIndex = wrapIndex(state, state.activeIndex + 1);
  state.moveDirection = 1;
  startClipPhase(state);
}

function armStamp(card) {
  if (!card || !card.dataset.stamp) return;
  if (card.hasAttribute(ARMED_ATTR)) return;
  card.setAttribute(ARMED_ATTR, "");
  card.setAttribute(ARMED_JUST_FIRED_ATTR, "");
  window.setTimeout(() => {
    card.removeAttribute(ARMED_JUST_FIRED_ATTR);
  }, SHAKE_MS + 40);
}

function armAllStamps(state) {
  for (const card of state.cards) {
    if (card.dataset.stamp) {
      card.setAttribute(ARMED_ATTR, "");
    }
  }
}

function readClipMs(card) {
  const override = Number.parseInt(card.dataset.clipMs || "", 10);
  if (Number.isFinite(override) && override > 0) {
    return override;
  }
  return DEFAULT_CLIP_MS;
}

function readDwellMs(card) {
  const override = Number.parseInt(card.dataset.dwellMs || "", 10);
  if (Number.isFinite(override) && override > 0) {
    return override;
  }
  const stamp = card.dataset.stamp;
  return VERDICT_DWELL_DEFAULTS[stamp] || VERDICT_DWELL_DEFAULTS["PLAUSIBLE"];
}

/* -------------------------------------------------------------------------- *
 * Pause / resume on hover and focus
 * -------------------------------------------------------------------------- */

function setupPauseListeners(state) {
  state.section.addEventListener("pointerenter", () => onPauseEnter(state));
  state.section.addEventListener("pointerleave", () => onPauseLeave(state));
  state.section.addEventListener("focusin", () => onPauseEnter(state));
  state.section.addEventListener("focusout", (event) => {
    if (!state.section.contains(event.relatedTarget)) {
      onPauseLeave(state);
    }
  });
}

function onPauseEnter(state) {
  if (state.paused || state.userDriven) return;
  if (state.phase !== "clip" && state.phase !== "dwell") return;
  state.paused = true;
  if (state.timer != null) {
    window.clearTimeout(state.timer);
    state.timer = null;
  }
  const elapsed = performance.now() - state.phaseStartedAt;
  state.pausedRemainingMs = Math.max(0, state.phaseDurationMs - elapsed);
}

function onPauseLeave(state) {
  if (!state.paused) return;
  state.paused = false;
  if (state.userDriven) return;

  const remaining = state.pausedRemainingMs;
  state.pausedRemainingMs = null;
  if (remaining == null) return;

  state.phaseStartedAt = performance.now();
  state.phaseDurationMs = remaining;

  if (state.phase === "clip") {
    state.timer = window.setTimeout(() => onClipEnd(state), remaining);
  } else if (state.phase === "dwell") {
    state.timer = window.setTimeout(() => onDwellEnd(state), remaining);
  }
}

/* -------------------------------------------------------------------------- *
 * Manual navigation (prev / next) — disables auto-cycle for session
 * -------------------------------------------------------------------------- */

function wireSkipArrows(state) {
  const prev = state.rail.querySelector(PREV_SELECTOR);
  const next = state.rail.querySelector(NEXT_SELECTOR);

  if (prev) {
    prev.addEventListener("click", () => skipTo(state, state.activeIndex - 1));
  }
  if (next) {
    next.addEventListener("click", () => skipTo(state, state.activeIndex + 1));
  }
}

function wireKeyboardNavigation(state) {
  state.track.addEventListener("keydown", (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) {
      return;
    }

    let targetIndex = null;
    if (event.key === "ArrowLeft") {
      targetIndex = state.activeIndex - 1;
    } else if (event.key === "ArrowRight") {
      targetIndex = state.activeIndex + 1;
    } else if (event.key === "Home") {
      targetIndex = 0;
    } else if (event.key === "End") {
      targetIndex = state.cards.length - 1;
    }

    if (targetIndex == null) {
      return;
    }

    event.preventDefault();
    const nextIndex = event.key === "Home" || event.key === "End"
      ? Math.max(0, Math.min(state.cards.length - 1, targetIndex))
      : targetIndex;
    const resolvedIndex = event.key === "Home" || event.key === "End"
      ? nextIndex
      : wrapIndex(state, nextIndex);
    if (resolvedIndex === state.activeIndex) {
      armStamp(state.cards[state.activeIndex]);
      goUserDriven(state);
      return;
    }

    skipTo(state, nextIndex);
  });
}

function wireManualTakeover(state) {
  const takeOver = () => {
    if (state.userDriven) return;
    armStamp(state.cards[state.activeIndex]);
    goUserDriven(state);
  };

  state.track.addEventListener("pointerdown", takeOver, { passive: true });
  state.track.addEventListener("wheel", takeOver, { passive: true });
}

function skipTo(state, targetIndex) {
  const wrapped = wrapIndex(state, targetIndex);
  if (wrapped === state.activeIndex) return;

  armStamp(state.cards[state.activeIndex]);
  goUserDriven(state);

  state.moveDirection = movementDirection(state, targetIndex, wrapped);
  state.activeIndex = wrapped;
  setActive(state, wrapped);
  centreCard(state, wrapped, { instant: reducedMotionQuery.matches });
}

function movementDirection(state, targetIndex, wrappedIndex) {
  const count = state.cards.length;
  if (targetIndex >= count) return 1;
  if (targetIndex < 0) return -1;
  return Math.sign(wrappedIndex - state.activeIndex);
}

function wrapIndex(state, index) {
  const count = state.cards.length;
  if (count <= 0) return 0;
  return ((index % count) + count) % count;
}

function goUserDriven(state) {
  if (state.userDriven) return;
  state.userDriven = true;
  if (state.timer != null) {
    window.clearTimeout(state.timer);
    state.timer = null;
  }
  state.phase = "user";
  setRailState(state, "user");
}

/* -------------------------------------------------------------------------- *
 * Replay button — restarts from card 1, clears everything
 * -------------------------------------------------------------------------- */

function wireReplayButton(state) {
  const replay = state.rail.querySelector(REPLAY_SELECTOR);
  if (!replay) return;
  replay.addEventListener("click", () => restartSequence(state));
}

function restartSequence(state) {
  if (state.timer != null) {
    window.clearTimeout(state.timer);
    state.timer = null;
  }
  for (const card of state.cards) {
    card.removeAttribute(ARMED_ATTR);
    card.removeAttribute(ARMED_JUST_FIRED_ATTR);
  }
  state.activeIndex = 0;
  state.userDriven = false;
  state.paused = false;
  state.pausedRemainingMs = null;

  setActive(state, 0);
  centreCard(state, 0, { instant: reducedMotionQuery.matches });

  if (reducedMotionQuery.matches) {
    armAllStamps(state);
    setRailState(state, "settled");
    return;
  }

  setRailState(state, "cycling");
  state.phase = "clip";
  state.timer = window.setTimeout(() => startClipPhase(state), 320);
}

/* -------------------------------------------------------------------------- *
 * Reduced-motion handling — also reacts to runtime changes
 * -------------------------------------------------------------------------- */

function setupReducedMotion(state) {
  const handler = (event) => {
    if (event.matches) {
      if (state.timer != null) {
        window.clearTimeout(state.timer);
        state.timer = null;
      }
      armAllStamps(state);
      setRailState(state, "settled");
      state.phase = "settled";
    }
  };
  if (typeof reducedMotionQuery.addEventListener === "function") {
    reducedMotionQuery.addEventListener("change", handler);
  } else if (typeof reducedMotionQuery.addListener === "function") {
    reducedMotionQuery.addListener(handler);
  }
}

/* -------------------------------------------------------------------------- *
 * Rail-state attribute (read by CSS for the replay-button affordance)
 * -------------------------------------------------------------------------- */

function setRailState(state, value) {
  if (!state.section) return;
  state.section.setAttribute(RAIL_STATE_ATTR, value);
}
