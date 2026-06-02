# NSE C1 Review Targets

Date: 2026-06-02

Purpose: internal dispatch note for the `internal/memo/NSE_Priority.md`
priority shift. This is not a public artifact and should not be copied into
`docs/**` without a separate privacy / publication pass.

## Read First

Technical review and IP/legal intake are separate jobs.

- A PDE reviewer can answer whether the C1 framing is mathematically honest,
  standard/renamable, or overclaimed.
- A technical reviewer cannot settle patentability, novelty-bar timing,
  inventorship, privilege, assignment, or disclosure strategy.
- Do not send mechanism-level patent-claim-specificity detail to technical
  reviewers before counsel. Send only the bounded C1 review bundle and its
  existing finite-Galerkin/numerical scope language.
- Use one-to-one outreach. No bulk mail, no BCC list, no "spam for peer
  review" framing. Send one ask, wait several business days, then send one
  follow-up or move to the next target.

The technical packet is already staged:

- `docs/proof/PDE_C1_EXTERNAL_REVIEW_EMAIL_DRAFT.md`
- `internal/outreach/PDE_C1_REVIEW_BUNDLE.md`
- `docs/REVIEW_TRACKER.md` (dispatch status)

## Recommended First Wave

These are the best first targets for the actual C1 question: finite-Galerkin
2D NSE, determining modes, data assimilation, observer/control language, and
whether the decision-observable/state-unobservable distinction is a real
bounded distinction or just ordinary observer/LES language.

| Order | Target | Public contact | Why this fit | Source |
| --- | --- | --- | --- | --- |
| 1 | Eric Olson, University of Nevada, Reno | `ejolson@unr.edu` | Direct hit for determining modes + continuous data assimilation in 2D turbulence; likely to judge whether this is standard observer/data-assimilation language. | https://www.unr.edu/math/eric-olson |
| 2 | Vincent R. Martinez, Hunter College / CUNY | `vrmartinez@hunter.cuny.edu` | PDE/fluid dynamics/data assimilation; good fit for "state-insufficient on sampled support" and kNN/disintegration language sanity check. | https://math.hunter.cuny.edu/vmartine/ |
| 3 | Adam Larios, University of Nebraska-Lincoln | `alarios@unl.edu` | PDE, fluid dynamics, numerical analysis, and recent CDA/LES/NSE work; good for numerical-methodology and objective-legitimacy checks. | https://adamlarios.github.io/ |
| 4 | Edriss Titi, Texas A&M | `titi@tamu.edu` | Very senior exact-fit authority for data assimilation, control theory, determining modes, dissipative dynamical systems. Use a shorter, deferential note. | https://artsci.tamu.edu/mathematics/contact/profiles/edriss-titi.html |

Recommended dispatch: start with Olson or Martinez, not the most senior name.
The ask is bounded and negative-friendly; a practical "rename this / cite this"
reply is more valuable than prestige.

## Turbulence / LES / MZ Mechanism Wave

These targets are useful if the first-wave question turns specifically toward
the MZ energy-budget mechanism, LES closure comparison, or the "this is just
coarse closure" objection.

| Target | Public contact | Why this fit | Source |
| --- | --- | --- | --- |
| Gregory Eyink, Johns Hopkins | `eyink@jhu.edu` | Turbulence, PDE, dynamical systems, coarse-grained turbulence, data assimilation; best fit for the MZ/coarse-graining mechanism language. | https://engineering.jhu.edu/faculty/gregory-eyink/ |
| Charles Meneveau, Johns Hopkins | `meneveau@jhu.edu` | LES, turbulence modeling, multiscale turbulence, subgrid-scale modeling, JHTDB; best fit for "is this just LES/closure?" | https://engineering.jhu.edu/faculty/charles-meneveau/ |
| Jonathan Mattingly, Duke | `jonm@duke.edu` | Stochastic dynamics, SPDEs, 2D fluids, invariant measures; good for sampled-SRB/support language and dynamical-systems caution. | https://sites.math.duke.edu/~jonm/ |

## Stretch / Senior PDE Authority

| Target | Public contact | Why this fit | Source |
| --- | --- | --- | --- |
| Peter Constantin, Princeton | `const@math.princeton.edu` | Senior Navier-Stokes / determining-modes authority; high signal if he replies, lower probability. | https://web.math.princeton.edu/~const/ |

Use only after a first-wave send unless there is a warm introduction.

## Secondary Lanes Already Well Into Experiments

Do not let these displace C1. They are separate communities and separate asks.

### Yang-Mills finite-lattice informative null

Status: already sent to Dr. Biagio Lucini via LinkedIn DM on 2026-06-01.
If no response, the next email-capable lattice gauge theory fallbacks are:

| Target | Public contact | Why this fit | Source |
| --- | --- | --- | --- |
| Biagio Lucini, Swansea | `b.lucini@swan.ac.uk` | Already chosen; SU(N), finite-T / Polyakov-loop fit. | https://www.swansea.ac.uk/physics/research-and-impact/particle-physics-cosmology/ |
| David Schaich, University of Liverpool | `David.Schaich@liverpool.ac.uk` | Lattice field theory and advanced computing; good practical reviewer for finite-lattice methodology. | https://www.liverpool.ac.uk/people/david-schaich |
| Simon Catterall, Syracuse | `smcatter@syr.edu` | Lattice gauge theory, supersymmetric lattice gauge work, center-symmetry-adjacent expertise. | https://artsandsciences.syracuse.edu/people/faculty/catterall-simon/ |
| David Kaplan, University of Washington | `dbkaplan@uw.edu` | QFT/lattice field theory; higher-level sanity check, but less directly Polyakov-targeted than Lucini/Schaich. | https://phys.washington.edu/people/david-kaplan |

### Chat / trace-conditioned safety eval

Status: experimentally strong, but not an IP-settling lane. It is useful for
AI-eval methodology peer review after the C1 dispatch because it has a clean
5,670-trial zero-unsafe-accept surface.

Good outreach targets are organization/lab channels rather than individual
professor cold calls unless there is a warm intro:

- Stanford CRFM / HELM contact: `contact-crfm@stanford.edu`
  (https://crfm.stanford.edu/support.html)
- METR: use public site / GitHub / LinkedIn channels first; no public general
  email was confirmed on the official site during this pass
  (https://metr.org/about)
- Jacob Steinhardt, UC Berkeley: `jsteinhardt@berkeley.edu`
  (https://statistics.berkeley.edu/people/jacob-steinhardt)

### Lattice deduction

Status: paper-design-only. Do not ask for result review until an access/build
gate passes. The existing `docs/lattice/LITPASS_MEMO.md` shortlist should stay
as a future Phase-7 list, not current outreach.

## IP / Counsel Lane

Use counsel for IP questions before broad technical disclosure.

- USPTO Patent Pro Bono Program: `probono@uspto.gov`
  Source: https://www.uspto.gov/patents/basics/using-legal-services/pro-bono/patent-pro-bono-program
- California Inventors Assistance Program / California Lawyers for the Arts:
  start from the USPTO map or CLA LRIS intake, not a random scraped email.
  Source: https://www.uspto.gov/about-us/uspto-office-locations/california
  and https://www.calawyersforthearts.org/lris
- Internal intake draft already exists:
  `docs/501c3/SUNDOG_USPTO_PROBONO_INTAKE_v0.1_DRAFT.md`

## PDE C1 Subject Lines

Best:

```text
Technical framing check request: finite-Galerkin NSE signature/control result
```

Shorter / specialist:

```text
Bounded sanity check: determining modes vs control-sufficiency framing
```

Avoid:

```text
Novel IP / patentable method in Navier-Stokes
```

The latter confuses the reviewer job and invites legal/process worries.

## Send Discipline

1. Regenerate/check the bundle if any C1 source file has changed:
   `node scripts/build-c1-review-bundle.mjs`.
2. Pick exactly one first-wave reviewer.
3. Fill `[Name]`, specialty phrase, and `[link]` in
   `docs/proof/PDE_C1_EXTERNAL_REVIEW_EMAIL_DRAFT.md`.
4. Attach/link only the bounded packet. Do not attach IP strategy or apparatus
   graph documents.
5. Add a row to `docs/REVIEW_TRACKER.md` send log after sending.
6. If no reply after 7-10 days, send one concise follow-up, then move to the
   next first-wave target.

