# Third-Party Reuse Ledger

This file records third-party design, code, or asset references used by Sundog
so borrowed material remains traceable during promotion work.

## Gontary101 / lounas-portfolio

Source:

- GitHub: `https://github.com/Gontary101/lounas-portfolio`
- Public README summary: Vite, React 18, Three.js, and interactive robotics
  simulations.

Permission record:

- Status: project-owner reported reuse permission granted through a LinkedIn
  agreement.
- Recorded in Sundog on: 2026-05-08.
- Scope as currently understood: Sundog may borrow heavily from the Lounas
  portfolio for the Sundog Balance visual background and for a BoxForge/CSS arm
  template derived from the portfolio's robotics visual language.

Expected Sundog usage:

- `docs/SUNDOG_V_BALANCE.md`: roadmap and implementation plan.
- `balance.html` and `public/js/balance-browser.mjs`: workbench background and
  robotics balance motif.
- Future BoxForge/CSS arm template: reusable motion/arm visual component for
  Balance and the eventual index-page motion clip.

Attribution expectation:

- Credit Lounas / `Gontary101/lounas-portfolio` in implementation comments or
  nearby docs when code, layout, motion language, or visual structure is derived
  from that project.
- Preserve a link to the source repo in any detailed implementation writeup.
- If direct code or assets are copied rather than reimplemented, identify the
  copied files and commit the attribution in the same change.

Guardrails:

- Do not silently mix borrowed pieces into generic Sundog components.
- Keep the cart-pole equations and Sundog controller logic original unless a
  future entry explicitly records otherwise.
- If the permission scope changes, update this ledger before expanding reuse.
