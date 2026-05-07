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
- provide knobs and widgets for changing experiment variables without code;
- rerun or replay the experiment against different surface and geometry presets;
- print graphs and run documents from adjusted experiments;
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
- Dungeon Gleaner: verb-field NPC behavior driven by unmet-need gradients over
  satisfier nodes.
- Money Bags: graph-readable softbody telemetry.

Acceptance criteria:

- each application has one highlight, one inspectable surface, and one future
  experiment;
- product evidence is clearly separated from controlled research evidence.

### 6. Experiment Workbench Mode

Purpose: let a user adjust the experiment without opening code.

This is the bridge from "watch the theorem" to "try to break the theorem."
Users should be able to change the operating conditions with ordinary UI
controls, rerun a small batch, and immediately see how the controller behaves.

Knob families:

| Family | Example controls | UI widget |
| --- | --- | --- |
| Geometry | laser height, laser xy range, detector ring radius, target detector | sliders, number inputs, segmented target picker |
| Optics | beam sigma, detector noise, intensity clamp, floor absorption | sliders and toggles |
| Controller | scan duration, scan amplitude, probe amplitude, gradient gain, reacquire threshold | sliders with safe ranges |
| Mechanics | joint limit, servo stiffness, damping, initial joint perturbation | sliders and presets |
| Surface presets | mirror, frosted glass, occluder block, noisy detector field, narrow beam, wide beam | visual preset cards |
| Run plan | seed count, seed start, steps per episode, policies to include | steppers, checkboxes |

Default posture:

- safe preset first: baseline paper settings;
- "stress me" preset second: detector noise, narrow beam, tight joint limits;
- "explore" mode hides dangerous ranges until the user expands advanced
  controls;
- every modified run gets a manifest so screenshots and reports are traceable.

Acceptance criteria:

- a user can change at least five meaningful variables without editing files;
- the app distinguishes replay-only knobs from live-run knobs;
- invalid combinations are blocked or explained;
- the workbench always preserves the baseline settings as a one-click reset;
- custom runs produce results that can be compared against the recorded
  baseline.

### 7. Results Printer Mode

Purpose: turn app runs into shareable research artifacts.

The user should be able to click "Print Results" after a replay comparison or
live experiment and receive a compact document package.

Outputs:

- `summary.html`: human-readable report;
- `summary.md`: markdown report for GitHub or notes;
- `summary.json`: machine-readable run manifest and metrics;
- `convergence.png`: target intensity over time;
- `terminal_intensity.png`: condition comparison chart;
- `time_to_threshold.png`: convergence-time chart;
- `stress_curve.png`: if a sweep was run;
- `runs.csv`: per-seed metrics table;
- `manifest.json`: app version, commit hash, settings, environment, and data
  provenance.

Report sections:

1. Title and timestamp.
2. Claim being tested.
3. Settings changed from baseline.
4. Conditions/policies run.
5. Headline metrics.
6. Graphs.
7. Failure or boundary notes.
8. Reproduction block.
9. Caveat language.

Acceptance criteria:

- report generation works offline;
- a report from baseline settings matches the known summary numbers;
- changed knobs are printed prominently;
- every generated graph includes axis labels and units;
- the report can be opened without the app.

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

### Phase 6: Add No-Code Knobs

Deliverables:

- Workbench panel with baseline, stress, and custom tabs;
- typed experiment config schema;
- UI controls for geometry, optics, controller, mechanics, surface presets,
  and run plan;
- config validation layer;
- "reset to paper baseline" button;
- "explain this knob" help text for every control;
- config diff view showing what changed from the baseline.

Initial safe controls:

- detector noise;
- beam sigma;
- scan duration;
- laser height;
- joint limit;
- seed count;
- condition/policy selection.

Second-wave controls:

- scan amplitude;
- probe amplitude;
- gradient gain;
- detector count or target detector;
- occluder enabled/disabled;
- occluder position and attenuation;
- surface preset selection.

Surface preset model:

| Preset | Meaning | First implementation |
| --- | --- | --- |
| clean mirror | baseline paper setup | current optics |
| noisy detector field | tests observation noise | additive intensity noise |
| narrow beam | sharper alignment landscape | beam sigma sweep |
| tight joints | reachable-workspace stress | joint-limit sweep |
| high laser | easier reflection geometry | laser-height sweep |
| occluded path | Phase-2 robustness | block attenuation model |
| frosted glass | rougher reflection approximation | broadened/scattered intensity model |

Config schema sketch:

```json
{
  "name": "custom_joint_limit_scan",
  "base": "paper_baseline",
  "seed_start": 0,
  "seed_count": 10,
  "steps": 500,
  "geometry": {
    "laser_height": 2.5,
    "detector_radius": 1.2,
    "target_detector": 0
  },
  "optics": {
    "beam_sigma": 0.15,
    "detector_noise": 0.0,
    "surface_preset": "clean_mirror"
  },
  "controller": {
    "scan_duration_s": 4.0,
    "scan_amplitude_rad": 1.4,
    "gradient_gain": 8.0
  },
  "mechanics": {
    "joint_limit_rad": 1.5
  },
  "conditions": ["photometric", "doa_direct", "doa_noisy", "random"]
}
```

Exit criteria:

- a custom config can be saved, reloaded, and rerun;
- invalid values produce helpful UI messages;
- the app can compare "baseline" vs "custom" without manual file handling;
- every result has a config manifest attached.

### Phase 7: Add Preset Sweeps

Deliverables:

- sweep builder UI;
- one-variable sweep mode;
- preset sweeps for detector noise, beam sigma, scan duration, laser height,
  joint limit, and occlusion attenuation;
- progress indicator for multi-run sweeps;
- pause/cancel support;
- automatic stress-curve chart generation.

Sweep builder controls:

- variable to sweep;
- min/max/step or explicit values;
- seeds per level;
- conditions to include;
- quick labels such as "fast smoke", "paper-scale", and "deep run".

Example sweep presets:

| Sweep | Values | Purpose |
| --- | --- | --- |
| detector noise | 0.00, 0.02, 0.05, 0.10, 0.20 | observation robustness |
| beam sigma | 0.05, 0.10, 0.15, 0.25, 0.40 | landscape sharpness |
| scan duration | 1, 2, 4, 8, 16 seconds | acquisition/refinement trade |
| laser height | 1.5, 2.0, 2.5, 3.0, 3.5 meters | geometry sensitivity |
| joint limit | 0.8, 1.0, 1.2, 1.5 radians | reachability boundary |
| occlusion attenuation | 0.0, 0.25, 0.5, 0.75, 0.9 | Phase-2 block robustness |

Exit criteria:

- a user can create a one-variable sweep without code;
- completed sweeps produce a chart and metrics table;
- sweep outputs are stored under a named run folder;
- cancellation leaves a partial report rather than corrupt output.

### Phase 8: Build Results Printer

Deliverables:

- report generator service/module;
- graph generator for convergence, terminal intensity, time-to-threshold, and
  stress curves;
- HTML and Markdown report templates;
- CSV and JSON exports;
- "Print Results" button;
- "Open Report Folder" button;
- optional browser print stylesheet for PDF export.

Report folder layout:

```text
runs/
  2026-05-01_1530_custom_joint_limit_scan/
    manifest.json
    summary.html
    summary.md
    summary.json
    runs.csv
    graphs/
      convergence.png
      terminal_intensity.png
      time_to_threshold.png
      stress_curve.png
    data/
      seed_000.json
      seed_001.json
```

Graph requirements:

- title includes run name and compared conditions;
- axes include units;
- baseline settings are annotated;
- stress curves highlight the best and worst levels;
- joint-limit reports call out the known failure boundary automatically;
- charts have color palettes that remain readable in print.

Document requirements:

- one-page executive summary at the top;
- expanded technical section below;
- exact knob changes from baseline;
- condition table;
- per-seed metrics appendix;
- reproduction command or app action list;
- caveat block copied from `docs/SCIENTIFIC_CRITERIA.md` language.

Exit criteria:

- report generation works for replay, live run, and sweep outputs;
- baseline report reproduces existing headline numbers;
- markdown report can be pasted into GitHub without manual cleanup;
- HTML report opens outside the app;
- generated artifacts are deterministic for the same input data.

### Phase 9: Add Comparison Library

Deliverables:

- local run library;
- saved configs;
- report browser;
- compare two or more prior runs;
- tag runs as baseline, stress, exploratory, or publication candidate;
- delete/archive controls.

Useful comparisons:

- baseline vs custom knob run;
- photometric only vs all baselines;
- replay data vs fresh live run;
- pre-occlusion vs occlusion preset;
- short smoke run vs full 30-seed run.

Exit criteria:

- a user can return later and understand what they ran;
- reports and configs remain linked;
- no-code comparison produces charts and a summary table.

### Phase 10: Build APK

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
- app clearly labels itself as replay/educational, not live MuJoCo execution;
- reports can be exported or shared from the device when generated from replay
  data.

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
  print_observer_results.py
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
runs/
  <timestamped-run>/     # generated local outputs
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
- no-code knobs for common stress variables;
- generated HTML/Markdown reports;
- graph export for convergence and stress curves;
- `run.exe` launcher;
- optional live smoke run;
- release zip with no global dependency requirement.

## Workbench Polish

The version that feels like a usable research toy:

- knob panel with safe/advanced groups;
- surface preset gallery with small visual previews;
- baseline-vs-custom diff;
- "rerun with these settings" button;
- queue for sweeps and multi-seed batches;
- live progress with estimated time remaining;
- warnings when a setting leaves the paper-supported envelope;
- result cards for each run;
- one-click report generation;
- local library of past runs.

This is the level where the app stops being a demo and becomes a small lab.

## Risks

| Risk | Mitigation |
| --- | --- |
| MuJoCo bundling is brittle. | Make replay mode the default; live mode optional. |
| Data exports become too large. | Ship selected seeds first; compress or lazy-load later. |
| Demo overclaims the theorem. | Keep evidence-tier labels visible. |
| Knobs create invalid or misleading experiments. | Validate configs and show envelope warnings. |
| Users mistake exploratory output for peer-reviewed results. | Stamp reports as exploratory unless they match a locked protocol. |
| Graph/report generation drifts from analysis scripts. | Share metric definitions and add golden-output tests. |
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
- [ ] Common variables can be changed without editing code.
- [ ] Custom runs preserve the exact config used.
- [ ] Generated reports include graphs, tables, settings, and caveats.
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
8. Draft the experiment config schema before adding live knobs.
9. Prototype a results-printer report from existing `results/analysis` data.

## Decision Record

Current recommendation:

> Build replay-first static HTML, then wrap it in a Windows launcher, then add
> optional live local runs, then add no-code knobs and results printing, then
> ship APK as a replay-focused educational app.

This avoids letting native simulation packaging block the core promise:
anyone can double-click and observe the theorem.
