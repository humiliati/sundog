import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

// Probe 05 - Nonlinear zero-statistics S2 gap-pair reversibility test.
// Spec: docs/riemann/PROBE_05_NONLINEAR_ZERO_STATISTICS_SPEC.md
// Bridge: docs/riemann/NONLINEAR_PAIR_CORRELATION_BRIDGE_NOTES.md (S2 hook only).
// This is a reversibility test for the unfolded gap sequence. It is NOT a
// structural-zero probe, not a D3/S3 descent, and not evidence for or against RH.

const REGISTERED_SOURCE_SHA =
  "3436c916a7878261ac183fd7b9448c9a4736b8bbccf1356874a6ce1788541632";
const DEFAULT_SOURCE_FILE =
  "results/riemann/probe01-isotropy-zero-pairs/source/zeros1.txt";
const DEFAULT_SOURCE_URL =
  "https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros1";
const DEFAULT_OUT = "results/riemann/probe05-nonlinear-zero-statistics";
const TWO_PI = 2 * Math.PI;

// Registered domain / floor parameters (frozen by the spec; do not tune).
const N_ZERO = 5000;
const MAX_HEIGHT_CEILING = 10000;
const EXPECTED_MAX_HEIGHT = 5447.861998301;
const TIE_TOL = 1e-8;
const BLOCK_LENGTH = 64;
const B = 10000;
const SEED = 20260528;
const BOOT_QUANTILE = 0.9975;

function parseArgs(argv) {
  const args = {
    sourceFile: DEFAULT_SOURCE_FILE,
    sourceUrl: DEFAULT_SOURCE_URL,
    out: DEFAULT_OUT,
    forceDownload: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--source-file") args.sourceFile = argv[++i];
    else if (arg === "--source-url") args.sourceUrl = argv[++i];
    else if (arg === "--out") args.out = argv[++i];
    else if (arg === "--force-download") args.forceDownload = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

function sha256Buffer(buffer) {
  return crypto.createHash("sha256").update(buffer).digest("hex");
}

function sha256File(filePath) {
  return sha256Buffer(fs.readFileSync(filePath));
}

// Deterministic, seedable PRNG (mulberry32) so the registered SEED reproduces
// the bootstrap floor exactly. Math.random() is not seedable and must not be
// used here.
function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function acquireSource(args, sourcePathOut) {
  // Prefer the registered local cache (byte-identical to the Probe 01 source).
  if (!args.forceDownload && fs.existsSync(args.sourceFile)) {
    const buf = fs.readFileSync(args.sourceFile);
    fs.writeFileSync(sourcePathOut, buf);
    return buf;
  }
  const response = await fetch(args.sourceUrl);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch ${args.sourceUrl}: ${response.status} ${response.statusText}`,
    );
  }
  const buffer = Buffer.from(await response.arrayBuffer());
  fs.writeFileSync(sourcePathOut, buffer);
  return buffer;
}

function parseZeros(buffer, n) {
  const text = buffer.toString("utf8");
  const zeros = [];
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const value = Number.parseFloat(trimmed);
    if (Number.isFinite(value)) zeros.push(value);
    if (zeros.length === n) break;
  }
  if (zeros.length < n) {
    throw new Error(`Source supplied ${zeros.length} zeros; need ${n}`);
  }
  return zeros;
}

function validateZeros(zeros, maxHeightCeiling) {
  const rows = [];
  let finiteOk = true;
  let positiveOk = true;
  let orderOk = true;
  for (let i = 0; i < zeros.length; i += 1) {
    if (!Number.isFinite(zeros[i])) finiteOk = false;
    if (!(zeros[i] > 0)) positiveOk = false;
    if (i > 0 && !(zeros[i] > zeros[i - 1])) orderOk = false;
    rows.push({
      zero_index: i + 1,
      gamma: zeros[i].toFixed(12),
      source: "odlyzko_zeros1",
      validation_status:
        Number.isFinite(zeros[i]) &&
        zeros[i] > 0 &&
        (i === 0 || zeros[i] > zeros[i - 1])
          ? "ok"
          : "invalid",
    });
  }
  const maxHeight = zeros.at(-1);
  return {
    rows,
    maxHeight,
    checks: {
      finiteOk,
      positiveOk,
      orderOk,
      maxHeightOk: maxHeight < maxHeightCeiling,
      maxHeightCeiling,
    },
  };
}

function localZeroDensity(t) {
  return Math.log(t / TWO_PI) / TWO_PI;
}

function quantileSorted(sortedValues, q) {
  if (sortedValues.length === 0) return null;
  const idx = (sortedValues.length - 1) * q;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sortedValues[lo];
  const w = idx - lo;
  return sortedValues[lo] * (1 - w) + sortedValues[hi] * w;
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (/[",\r\n]/.test(text)) return `"${text.replaceAll('"', '""')}"`;
  return text;
}

function writeCsv(filePath, headers, rows) {
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map((h) => csvEscape(row[h])).join(","));
  }
  fs.writeFileSync(filePath, `${lines.join("\n")}\n`);
}

function gitValue(gitArgs) {
  try {
    return execFileSync("git", gitArgs, { encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(args.out);
  const sourceDir = path.join(outDir, "source");
  fs.mkdirSync(sourceDir, { recursive: true });
  const sourcePathOut = path.join(sourceDir, "zeros1.txt");

  // acquireSource is async only when it must download; resolve synchronously
  // via a small wrapper so the rest of main stays linear.
  return acquireSource(args, sourcePathOut).then((sourceBuffer) => {
    const sourceHash = sha256Buffer(sourceBuffer);
    if (sourceHash !== REGISTERED_SOURCE_SHA) {
      throw new Error(
        `Source SHA mismatch: got ${sourceHash}, registered ${REGISTERED_SOURCE_SHA}. ` +
          `Refusing to run on unregistered data.`,
      );
    }

    const zeros = parseZeros(sourceBuffer, N_ZERO);
    const validation = validateZeros(zeros, MAX_HEIGHT_CEILING);
    if (
      !validation.checks.finiteOk ||
      !validation.checks.positiveOk ||
      !validation.checks.orderOk ||
      !validation.checks.maxHeightOk
    ) {
      throw new Error(
        `Zero validation failed: ${JSON.stringify(validation.checks)}`,
      );
    }
    const heightLockOk =
      Math.abs(validation.maxHeight - EXPECTED_MAX_HEIGHT) <= 1e-6;

    // Unfolded nearest-neighbor gaps (Probe 01 local-density convention).
    const gapRows = [];
    const s = [];
    for (let i = 0; i < zeros.length - 1; i += 1) {
      const gap = zeros[i + 1] - zeros[i];
      const center = (zeros[i] + zeros[i + 1]) / 2;
      const rho = localZeroDensity(center);
      const si = gap * rho;
      s.push(si);
      gapRows.push({
        gap_index: i + 1,
        gamma_left: zeros[i].toFixed(12),
        gamma_right: zeros[i + 1].toFixed(12),
        gap: gap.toFixed(12),
        center: center.toFixed(12),
        rho: rho.toFixed(15),
        s: si.toFixed(15),
      });
    }

    // Consecutive gap pairs and the S2 sign statistic.
    const m = s.length - 1; // = N_ZERO - 2
    const signs = new Array(m);
    const pairRows = [];
    let nPlus = 0;
    let nMinus = 0;
    let nTie = 0;
    for (let i = 0; i < m; i += 1) {
      const delta = s[i] - s[i + 1];
      let sign;
      if (delta > TIE_TOL) {
        sign = 1;
        nPlus += 1;
      } else if (delta < -TIE_TOL) {
        sign = -1;
        nMinus += 1;
      } else {
        sign = 0;
        nTie += 1;
      }
      signs[i] = sign;
      pairRows.push({
        pair_index: i + 1,
        s_i: s[i].toFixed(15),
        s_ip1: s[i + 1].toFixed(15),
        delta: delta.toExponential(12),
        sign,
      });
    }

    const D = signs.reduce((acc, v) => acc + v, 0) / m;
    const tauInd = 3 / Math.sqrt(m);

    // Circular moving-block bootstrap on the centered sign sequence.
    const centered = signs.map((v) => v - D);
    const rng = mulberry32(SEED);
    const bootMeans = new Array(B);
    for (let b = 0; b < B; b += 1) {
      let sum = 0;
      let count = 0;
      while (count < m) {
        const start = Math.floor(rng() * m);
        const take = Math.min(BLOCK_LENGTH, m - count);
        for (let k = 0; k < take; k += 1) {
          sum += centered[(start + k) % m];
        }
        count += take;
      }
      bootMeans[b] = sum / m;
    }
    const absMeansSorted = bootMeans.map(Math.abs).sort((a, b2) => a - b2);
    const tauBoot = quantileSorted(absMeansSorted, BOOT_QUANTILE);
    const tauD = Math.max(tauInd, tauBoot);

    const tieFraction = nTie / m;
    const tieQuarantine = tieFraction > 0.001;

    let disposition;
    if (tieQuarantine) {
      disposition = {
        verdict: "source / precision quarantine",
        branch: "tie fraction exceeds 0.1% of m",
        falsifier: "source/precision quarantine",
        summary: `Tie fraction ${tieFraction} exceeds the registered 0.1% guard; interpret no further until the source precision is re-checked.`,
      };
    } else if (Math.abs(D) <= tauD) {
      disposition = {
        verdict: "bounded reversibility-test null (expected)",
        branch: "abs(D) <= tau_D",
        falsifier: "R-NL-NEG-A (GUE dominance)",
        summary:
          "Consecutive-gap orientation is balanced within the registered finite-window floor. This is the expected GUE / sine-kernel reversibility null; it is NOT a Sundog structural-zero and NOT evidence for or against RH.",
      };
    } else {
      disposition = {
        verdict: "GUE-reversibility anomaly flag",
        branch: "abs(D) > tau_D",
        falsifier:
          "not R-NL-NEG-A; requires independent replication on a separately registered window with a magnitude-aware statistic",
        summary:
          "abs(D) exceeded the registered floor. Per the spec this is a finite-window reversibility flag ONLY, not a Sundog structural-zero; it must be replicated before any further discussion.",
      };
    }

    const summary = {
      pairCount: m,
      nPlus,
      nMinus,
      nTie,
      tieFraction,
      D,
      tauInd,
      tauBoot,
      tauD,
      bootMeanOfMeans:
        bootMeans.reduce((acc, v) => acc + v, 0) / bootMeans.length,
      bootMaxAbsMean: absMeansSorted.at(-1),
      maxHeight: validation.maxHeight,
      heightLockOk,
    };

    const manifest = {
      probe: "riemann_probe05_nonlinear_zero_statistics",
      runId: "probe05-s2-reversibility-odlyzko-zeros1-n5000",
      createdAtUtc: new Date().toISOString(),
      command: process.argv.join(" "),
      git: {
        commit: gitValue(["rev-parse", "HEAD"]),
        statusShort: gitValue(["status", "--short"]),
      },
      code: {
        scriptPath: path
          .relative(process.cwd(), fileURLToPath(import.meta.url))
          .replaceAll("\\", "/"),
        scriptSha256: sha256File(fileURLToPath(import.meta.url)),
      },
      bridge: {
        admittedHook: "S2 gap-pair swap (s_i, s_{i+1}) -> (s_{i+1}, s_i)",
        quarantinedHooks: ["C3 triple", "residual-bin sectors", "S3/D3 upgrade"],
        note: "Reversibility test only; not a structural-zero probe.",
      },
      source: {
        name: "Odlyzko zeta zero table zeros1",
        localPath: path.relative(process.cwd(), sourcePathOut).replaceAll("\\", "/"),
        url: args.sourceUrl,
        sha256: sourceHash,
        registeredSha256: REGISTERED_SOURCE_SHA,
        shaMatch: sourceHash === REGISTERED_SOURCE_SHA,
        declaredAccuracyAbs: "3e-9",
      },
      domain: {
        nZero: N_ZERO,
        maxHeightCeiling: MAX_HEIGHT_CEILING,
        expectedMaxHeight: EXPECTED_MAX_HEIGHT,
        observedMaxHeight: validation.maxHeight,
        heightLockOk,
        statistic: "consecutive_unfolded_gap_pair_sign_statistic_D",
        unfoldingFormula:
          "s_i = (gamma_{i+1} - gamma_i) * log(((gamma_i + gamma_{i+1})/2) / (2*pi)) / (2*pi)",
      },
      thresholds: {
        tieTol: TIE_TOL,
        tieQuarantineFraction: 0.001,
        tauInd: tauInd,
        blockLength: BLOCK_LENGTH,
        bootstrapReplicates: B,
        seed: SEED,
        bootstrapQuantile: BOOT_QUANTILE,
      },
      validation: validation.checks,
      summary,
      disposition,
    };

    // Write artifacts.
    writeCsv(
      path.join(outDir, "zeros.csv"),
      ["zero_index", "gamma", "source", "validation_status"],
      validation.rows,
    );
    writeCsv(
      path.join(outDir, "unfolded_gaps.csv"),
      ["gap_index", "gamma_left", "gamma_right", "gap", "center", "rho", "s"],
      gapRows,
    );
    writeCsv(
      path.join(outDir, "gap_pairs.csv"),
      ["pair_index", "s_i", "s_ip1", "delta", "sign"],
      pairRows,
    );
    fs.writeFileSync(
      path.join(outDir, "reversibility_summary.json"),
      `${JSON.stringify({ summary, thresholds: manifest.thresholds, disposition }, null, 2)}\n`,
    );
    // Bootstrap floor audit: histogram of |bootstrap means| so tau_boot is
    // reproducible/checkable without dumping all B raw means.
    const histBins = 50;
    const histMax = absMeansSorted.at(-1);
    const hist = new Array(histBins).fill(0);
    for (const v of absMeansSorted) {
      let bin = histMax > 0 ? Math.floor((v / histMax) * histBins) : 0;
      if (bin >= histBins) bin = histBins - 1;
      hist[bin] += 1;
    }
    const histRows = hist.map((cnt, i) => ({
      bin_index: i,
      abs_mean_low: ((i / histBins) * histMax).toExponential(9),
      abs_mean_high: (((i + 1) / histBins) * histMax).toExponential(9),
      count: cnt,
    }));
    histRows.push({
      bin_index: "quantile",
      abs_mean_low: `q=${BOOT_QUANTILE}`,
      abs_mean_high: tauBoot.toExponential(12),
      count: B,
    });
    writeCsv(
      path.join(outDir, "bootstrap_floor.csv"),
      ["bin_index", "abs_mean_low", "abs_mean_high", "count"],
      histRows,
    );
    writeCsv(
      path.join(outDir, "quarantine.csv"),
      ["quarantine_id", "reason", "disposition"],
      tieQuarantine
        ? [
            {
              quarantine_id: "tie_fraction",
              reason: `tie fraction ${tieFraction} > 0.001`,
              disposition: "source/precision quarantine",
            },
          ]
        : [],
    );

    const readme = `# Probe 05 - Nonlinear Zero-Statistics Reversibility Test (run)

Status: ${disposition.verdict} (${disposition.falsifier})

S2 gap-pair reversibility test on the unfolded consecutive-gap sequence. This is
NOT a structural-zero probe and NOT evidence for or against RH. Expected result
per spec: R-NL-NEG-A (GUE dominance).

## Frozen domain

- Source: Odlyzko zeros1, SHA256 ${sourceHash} (registered match: ${sourceHash === REGISTERED_SOURCE_SHA})
- N_zero: ${N_ZERO}
- Observed max height: ${validation.maxHeight} (expected ${EXPECTED_MAX_HEIGHT}; lock ${heightLockOk ? "ok" : "FAILED"})
- Unfolding: s_i = gap_i * log(center_i/(2*pi))/(2*pi)
- Statistic: D = sum_i sign(s_i - s_{i+1}) / m, tie_tol ${TIE_TOL}

## Result

- pairs m: ${m}
- ascents (sign -1, s_i<s_{i+1}): ${nMinus}
- descents (sign +1, s_i>s_{i+1}): ${nPlus}
- ties: ${nTie} (fraction ${tieFraction})
- D: ${D}
- tau_ind (3/sqrt(m)): ${tauInd}
- tau_boot (block=${BLOCK_LENGTH}, B=${B}, seed=${SEED}, q=${BOOT_QUANTILE}): ${tauBoot}
- tau_D = max(tau_ind, tau_boot): ${tauD}

## Disposition

Branch: ${disposition.branch}
Falsifier: ${disposition.falsifier}

${disposition.summary}
`;
    fs.writeFileSync(path.join(outDir, "README.md"), readme);

    const artifactFiles = [
      "zeros.csv",
      "unfolded_gaps.csv",
      "gap_pairs.csv",
      "reversibility_summary.json",
      "bootstrap_floor.csv",
      "quarantine.csv",
      "README.md",
      "source/zeros1.txt",
    ];
    manifest.artifacts = artifactFiles.map((fileName) => {
      const artifactPath = path.join(outDir, fileName);
      return {
        path: path.relative(process.cwd(), artifactPath).replaceAll("\\", "/"),
        sha256: sha256File(artifactPath),
      };
    });
    fs.writeFileSync(
      path.join(outDir, "manifest.json"),
      `${JSON.stringify(manifest, null, 2)}\n`,
    );

    console.log(
      JSON.stringify(
        {
          out: path.relative(process.cwd(), outDir).replaceAll("\\", "/"),
          verdict: disposition.verdict,
          falsifier: disposition.falsifier,
          D,
          tauD,
          nPlus,
          nMinus,
          nTie,
          maxHeight: validation.maxHeight,
          heightLockOk,
          sourceShaMatch: sourceHash === REGISTERED_SOURCE_SHA,
        },
        null,
        2,
      ),
    );
  });
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
