# Standalone App Roadmap

Working title: **Sundog Observer**

Goal: someone should be able to enter the repo, double-click one file, and
observe the Sundog experiment and theorem on their own device with little or no
dependency setup.

The ideal first-run experience:

1. Download or clone the repo.
2. Open `dist/SundogObserver/`.
3. Double-click `run.exe` on Windows, `run.command` on macOS, or `index.html`
   for the web build.
4. Watch the theorem explained, run the mirror-alignment demo, compare
   baselines, inspect stress tests, and leave with a clear sense of what Sundog
   claims and what it does not claim.

## Product Promise

Sundog Observer is not a paper PDF and not a dev portal. It is a guided,
inspectable demonstration of the research program:

- explain the theorem in plain language;
- replay the photometric mirror-alignment experiment;
- show the photometric agent, oracle baseline, noisy oracle, and random policy;
- visualize the scan/seek/track loop;
- surface stress-test boundaries honestly;
- link the core experiment to EyesOnly, Dungeon Gleaner, and Money Bags;
- let technical readers inspect exported data and reproducibility notes.

## Packaging Targets

| Target | Priority | User experience | Dependency posture |
| --- | --- | --- | --- |
| Static `.html` | P0 | Open `index.html` directly. | No install; replay-only at first. |
| Windows `.exe` | P1 | Double-click `run.exe`; app opens in browser or native window. | Bundled runtime. |
| Portable folder | P1 | Zip with app, assets, data, and launcher scripts. | No global install. |
| macOS/Linux launchers | P2 | Double-click script or app bundle. | Bundled or system Python fallback. |
| Android `.apk` | P3 | Install demo app; view guided replay on phone/tablet. | Bundled WebView/native shell. |

Recommendation: ship static HTML first. It gives the strongest "no
dependencies" story and can be reused inside `.exe` and `.apk` wrappers.

## Core Product Modes

### 1. Guided Story Mode

Purpose: make the theorem legible in five minutes.

Flow:

1. The problem: full target state is often unavailable or unrealistic.
2. The setup: mirror, laser, detector ring, agent observations.
3. The trick: scan, seek, track.
4. The result: terminal accuracy comparable to oracle, slower acquisition.
5. The boundary: tight joint limits break the claim.
6. The future: applications in games, simulation, and softbody telemetry.

Acceptance criteria:

- a non-researcher can explain the experiment back after one pass;
- the app never implies that the theorem is fully proven;
- every bold claim has an "inspect the data" escape hatch.

### 2. Experiment Replay Mode

Purpose: observe the experiment without installing MuJoCo or Python.

Approach:

- export representative `.npz` run data to compact JSON or binary blobs;
- render detector intensities, joint angles, phase state, and beam/floor
  geometry in the browser;
- allow seed selection from recorded runs;
- animate photometric, DOA-direct, DOA-noisy, and random side by side.

Initial data sources:

- `results/photometric/seed_*.npz`
- `results/doa_direct/seed_*.npz`
- `results/doa_noisy/seed_*.npz`
- `results/random/seed_*.npz`
- `results/analysis/analysis_summary.json`
- `results/stress_tests/stress_summary.csv`

Acceptance criteria:

- no Python required;
- playback is deterministic;
- the displayed headline numbers match `analysis_summary.json`;
- the app can show at least one full seed animation and one summary chart.

### 3. Live Local Run Mode

Purpose: let technical users rerun the experiment on their machine.

Approach:

- use a bundled Python runtime for `run.exe`;
- launch a local app server on `127.0.0.1`;
- run a small smoke experiment by default;
- offer "full 30-seed run" as an advanced option;
- write outputs to `runs/<timestamp>/`.

Acceptance criteria:

- Windows user double-clicks `run.exe` and gets the app without installing
  Python globally;
- live mode clearly labels itself as slower and hardware-dependent;
- failure to load MuJoCo falls back to replay mode with a friendly message;
- outputs include a manifest with app version, commit hash, settings, and
  platform info.

### 4. Research Inspector Mode

Purpose: make the demo credible to skeptical readers.

Features:

- show metrics definitions;
- show bootstrap CI and Mann-Whitney test summaries;
- expose run config and seed logic;
- show stress-test plots;
- show negative result for joint limits;
- link to relevant source files.

Acceptance criteria:

- one screen answers "what exactly was measured?";
- one screen answers "where does the claim fail?";
- one screen answers "how do I reproduce this?";
- all file links are local repo paths or embedded source excerpts.

### 5. Application Futures Mode

Purpose: connect the theorem to product systems without overclaiming.

Sections:

- EyesOnly: procedural agent under compressed/occluded state.
- Dungeon Gleaner: cheap physical-feeling glass and pressure-washing behavior.
- Money Bags: graph-readable softbody telemetry.

Acceptance criteria:

- each application has one highlight, one inspectable surface, and one future
  experiment;
- product evidence is clearly separated from controlled research evidence.

## Architecture Options

### Option A: Static Web App

Stack:

- `app/observer/index.html`
- plain JavaScript or TypeScript build output;
- Canvas or SVG for visualization;
- exported JSON data under `app/observer/data/`;
- no server required.

Pros:

- best no-dependency story;
- easiest to host on GitHub Pages;
- can be wrapped by `.exe` or `.apk` later;
- safest for reviewers.

Cons:

- replay only unless paired with a local runner;
- large data exports need pruning or compression.

Best first milestone.

### Option B: Local Web App With Launcher

Stack:

- reuse ideas from `portal/`;
- Python stdlib server or small packaged backend;
- browser opens automatically;
- optional live experiment execution.

Pros:

- close to current repo;
- can run real Python code;
- `run.exe` can be PyInstaller-packed.

Cons:

- native packaging and MuJoCo bundling need careful testing;
- backend errors can make the first-run experience feel fragile.

Best second milestone.

### Option C: Native Desktop Shell

Stack candidates:

- Tauri or Electron wrapping the static web app;
- bundled data and assets;
- optional sidecar Python runner.

Pros:

- polished desktop feel;
- easy double-click distribution;
- app menu, file export, and local storage are straightforward.

Cons:

- more build tooling;
- larger artifacts if Electron;
- live simulation still needs a backend/sidecar.

Best after the static app has stabilized.

### Option D: APK Wrapper

Stack candidates:

- Godot UI;
- Capacitor WebView wrapping static web app;
- native Android WebView shell.

Pros:

- mobile/tablet demos;
- good for public-facing theorem walkthrough.

Cons:

- replay-only at first;
- live MuJoCo experiment is not realistic on Android without a separate
  architecture;
- additional signing/release steps.

Best as a polished communication artifact, not as the first research tool.

## Recommended Build Path

### Phase 0: Define The Artifact

Deliverables:

- name: `Sundog Observer`;
- app tagline: "Alignment without direct sight.";
- choose static web app as first target;
- decide which runs and stress tests ship as embedded data;
- define the first five-minute guided story.

Exit criteria:

- one-page product brief;
- list of included datasets;
- rough wireframe for the main screens.

### Phase 1: Export Replay Data

Deliverables:

- `tools/export_observer_data.py`;
- `observer_data/manifest.json`;
- compact per-seed playback files;
- summary stats copied from `results/analysis/analysis_summary.json`;
- stress-test summary export.

Data schema sketch:

```json
{
  "version": 1,
  "commit": "<git-sha>",
  "conditions": ["photometric", "doa_direct", "doa_noisy", "random"],
  "seeds": [0, 1, 2],
  "runs": {
    "photometric/seed_000": {
      "target_intensity": [],
      "joint_angles": [],
      "actions": [],
      "laser_xy": [0.0, 0.0],
      "phase": []
    }
  }
}
```

Exit criteria:

- exported data can be loaded without NumPy;
- exported numbers match existing result summaries;
- exported dataset is small enough to commit or attach as release asset.

### Phase 2: Build Static Observer

Deliverables:

- `app/observer/index.html`;
- `app/observer/app.js`;
- `app/observer/styles.css`;
- `app/observer/data/`;
- local file-open smoke test.

Screens:

- Home / theorem hook;
- Experiment setup;
- Replay viewer;
- Baseline comparison;
- Stress-test boundary;
- Applications and future;
- Research caveat / reproduce.

Visualization requirements:

- detector ring with live intensity glow;
- mirror/pole state as a simple 2D/3D schematic;
- target intensity timeline;
- phase labels: SCAN, SEEK, TRACK;
- side-by-side condition comparison;
- stress-test chart or table.

Exit criteria:

- works by double-clicking `index.html`;
- works offline;
- no network calls;
- no build step required for a user;
- includes a visible "what is proven vs what is future work" panel.

### Phase 3: Create Portable Windows Launcher

Deliverables:

- `packaging/windows/run.ps1`;
- `packaging/windows/run.bat`;
- `dist/SundogObserver/run.exe`;
- embedded static app;
- optional local server for better browser behavior.

Launcher behavior:

1. find a free local port;
2. serve the static app from bundled files;
3. open the default browser;
4. write logs to `SundogObserver/logs/`;
5. if anything fails, open `index.html` directly.

Implementation candidates:

- small Go/Rust launcher that serves static files;
- PyInstaller-packed Python launcher using `http.server`;
- Tauri/Electron shell after the static app proves itself.

Exit criteria:

- clean Windows machine can run it without installing Python;
- double-click opens the app within five seconds;
- antivirus false-positive risk is evaluated;
- release zip includes `README_FIRST.txt`.

### Phase 4: Add Live Run Backend

Deliverables:

- "Replay" and "Live Run" tabs;
- bundled smoke-run profile;
- full-run advanced profile;
- output manifest and run folder export;
- clear fallback when simulation dependencies fail.

Implementation notes:

- current `portal/` can provide useful server/job patterns;
- public UI should not reuse the developer portal unchanged;
- live mode should use a constrained runner first, not the full EyesOnly bridge;
- MuJoCo native libraries must be bundled or detected.

Exit criteria:

- smoke run completes from `run.exe`;
- run output can be inspected in the app;
- recorded and live runs use the same metric definitions;
- failure paths are friendly and do not strand the user in a terminal window.

### Phase 5: Polish The Narrative

Deliverables:

- guided tour script;
- short glossary;
- theorem animation;
- application futures section;
- copy pass using `docs/PROMO_HIGHLIGHTS.md`;
- research boundary copy using `docs/SCIENTIFIC_CRITERIA.md`.

Exit criteria:

- first-time user can understand the point without reading the paper;
- technical user can reach source/data within two clicks;
- no screen overclaims beyond the evidence tier it belongs to.

### Phase 6: Build APK

Deliverables:

- Android wrapper around the static observer;
- touch-friendly layout;
- embedded replay data;
- signed debug APK first, release APK later.

Candidate paths:

- Capacitor wrapping the web app;
- Godot implementation using exported data;
- minimal native Android WebView.

Exit criteria:

- installs on a modern Android device;
- works offline;
- replay mode performs smoothly;
- app clearly labels itself as replay/educational, not live MuJoCo execution.

## Repository Layout Proposal

```text
app/
  observer/
    index.html
    app.js
    styles.css
    data/
      manifest.json
      summary.json
      runs/
      stress/
tools/
  export_observer_data.py
packaging/
  windows/
    run.ps1
    run.bat
    launcher/
  android/
docs/
  STANDALONE_APP_ROADMAP.md
dist/
  SundogObserver/        # generated, not necessarily committed
```

## Minimum Viable Demo

The smallest version worth shipping:

- static `index.html`;
- one photometric seed replay;
- one oracle replay;
- headline stats table;
- convergence curve;
- joint-limit failure table;
- short explanation of `H(x)`;
- links to paper draft, researcher guide, applications, and scientific
  criteria.

This version can be done before solving native packaging.

## Polished Demo

The version that feels real:

- multi-seed replay selector;
- side-by-side baseline animation;
- stress-test explorer;
- guided voiceover-ready script;
- exportable run card image or summary;
- `run.exe` launcher;
- optional live smoke run;
- release zip with no global dependency requirement.

## Risks

| Risk | Mitigation |
| --- | --- |
| MuJoCo bundling is brittle. | Make replay mode the default; live mode optional. |
| Data exports become too large. | Ship selected seeds first; compress or lazy-load later. |
| Demo overclaims the theorem. | Keep evidence-tier labels visible. |
| `run.exe` triggers security warnings. | Consider signed builds later; provide static HTML fallback. |
| UI becomes a paper in disguise. | Use guided visuals first, details on demand. |
| Android live simulation is impractical. | APK is replay/education only. |

## Acceptance Checklist

Before calling the app polished:

- [ ] A nontechnical user can launch it without installing dependencies.
- [ ] The app works offline.
- [ ] The first screen explains what Sundog is in under 30 seconds.
- [ ] The experiment replay makes SCAN, SEEK, and TRACK visible.
- [ ] Baselines are shown, not merely described.
- [ ] The joint-limit failure boundary is included.
- [ ] The application futures are exciting but labeled as application evidence.
- [ ] The research claim is stated narrowly and accurately.
- [ ] The app links to source files and exported data.
- [ ] A release zip can be tested on a clean Windows machine.

## Immediate Next Tasks

1. Choose the MVP seed set, probably seeds `000`, `001`, and one visually
   interesting stress-test case.
2. Write `tools/export_observer_data.py`.
3. Create `app/observer/index.html` with static mock data.
4. Render the detector ring and convergence chart.
5. Add the theorem/story panels.
6. Add `run.bat` or `run.ps1` that opens the static app.
7. Package a first `dist/SundogObserver.zip`.

## Decision Record

Current recommendation:

> Build replay-first static HTML, then wrap it in a Windows launcher, then add
> optional live local runs, then ship APK as a replay-focused educational app.

This avoids letting native simulation packaging block the core promise:
anyone can double-click and observe the theorem.
