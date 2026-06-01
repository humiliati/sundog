// scripts/lib/pvnp-phase3-attacker-core.mjs
//
// Pure-Node deterministic attacker models for the Phase 3 capacity battery.
// Frozen classes (docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V0_SLATE.md):
//   small : logistic regression (L2 1e-3, lr 0.05, 2000 epochs)
//   medium: 2-layer MLP widths [32,32] ReLU (L2 1e-4, lr 0.01, 2000 epochs,
//           seeds 0/1/2)
//
// Scoring protocol (frozen): deterministic leave-one-policy-group-out. Inside
// each fold, fit on all other policy groups and score the held-out group. Any
// binarization threshold is selected on the fold's TRAINING groups only. Final
// AUROC and balanced accuracy are computed from concatenated held-out scores.
//
// Standardization (mean/std) is fit on the training fold only, then applied to
// the held-out group, so no held-out statistics leak into the attacker.

import { makeRng, gaussian } from "./pvnp-phase1-rng.mjs";

// ---------- metrics ----------

// Mann-Whitney rank AUROC with tie handling. scores/labels concatenated.
export function auroc(scores, labels) {
  const n = scores.length;
  const pos = labels.reduce((s, y) => s + (y === 1 ? 1 : 0), 0);
  const neg = n - pos;
  if (pos === 0 || neg === 0) return null;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[a] - scores[b]);
  const ranks = new Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && scores[idx[j + 1]] === scores[idx[i]]) j += 1;
    const avgRank = (i + j) / 2 + 1; // 1-based average rank for ties
    for (let k = i; k <= j; k += 1) ranks[idx[k]] = avgRank;
    i = j + 1;
  }
  let sumPosRanks = 0;
  for (let k = 0; k < n; k += 1) if (labels[k] === 1) sumPosRanks += ranks[k];
  const u = sumPosRanks - (pos * (pos + 1)) / 2;
  return u / (pos * neg);
}

// Balanced accuracy at a given threshold (score >= threshold => predict 1).
export function balancedAccuracy(scores, labels, threshold) {
  let tp = 0; let fn = 0; let tn = 0; let fp = 0;
  for (let i = 0; i < scores.length; i += 1) {
    const pred = scores[i] >= threshold ? 1 : 0;
    if (labels[i] === 1) { if (pred === 1) tp += 1; else fn += 1; }
    else { if (pred === 0) tn += 1; else fp += 1; }
  }
  const tpr = tp + fn > 0 ? tp / (tp + fn) : 0;
  const tnr = tn + fp > 0 ? tn / (tn + fp) : 0;
  return (tpr + tnr) / 2;
}

// Pick the threshold on the train fold that maximizes balanced accuracy.
// Candidate thresholds are midpoints between sorted unique train scores plus
// open ends, so the choice is deterministic and uses train data only.
export function selectThresholdTrainOnly(trainScores, trainLabels) {
  const uniq = Array.from(new Set(trainScores)).sort((a, b) => a - b);
  const candidates = [uniq[0] - 1e-6];
  for (let i = 0; i + 1 < uniq.length; i += 1) candidates.push((uniq[i] + uniq[i + 1]) / 2);
  candidates.push(uniq[uniq.length - 1] + 1e-6);
  let best = candidates[0];
  let bestBa = -1;
  for (const t of candidates) {
    const ba = balancedAccuracy(trainScores, trainLabels, t);
    if (ba > bestBa) { bestBa = ba; best = t; }
  }
  return best;
}

export function mae(preds, targets) {
  if (preds.length === 0) return null;
  let s = 0;
  for (let i = 0; i < preds.length; i += 1) s += Math.abs(preds[i] - targets[i]);
  return s / preds.length;
}

// ---------- standardization ----------

function fitStandardizer(rows) {
  const dim = rows[0].length;
  const mean = new Array(dim).fill(0);
  const std = new Array(dim).fill(0);
  for (const r of rows) for (let d = 0; d < dim; d += 1) mean[d] += r[d];
  for (let d = 0; d < dim; d += 1) mean[d] /= rows.length;
  for (const r of rows) for (let d = 0; d < dim; d += 1) std[d] += (r[d] - mean[d]) ** 2;
  for (let d = 0; d < dim; d += 1) std[d] = Math.sqrt(std[d] / rows.length) || 1;
  return { mean, std };
}

function applyStandardizer(row, st) {
  return row.map((v, d) => (v - st.mean[d]) / st.std[d]);
}

// ---------- logistic regression (small tier) ----------

function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

export function trainLogistic(X, y, { l2, learning_rate, epochs }) {
  const dim = X[0].length;
  const w = new Array(dim).fill(0);
  let b = 0;
  const n = X.length;
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const gradW = new Array(dim).fill(0);
    let gradB = 0;
    for (let i = 0; i < n; i += 1) {
      let z = b;
      for (let d = 0; d < dim; d += 1) z += w[d] * X[i][d];
      const p = sigmoid(z);
      const err = p - y[i];
      for (let d = 0; d < dim; d += 1) gradW[d] += err * X[i][d];
      gradB += err;
    }
    for (let d = 0; d < dim; d += 1) w[d] -= learning_rate * (gradW[d] / n + l2 * w[d]);
    b -= learning_rate * (gradB / n);
  }
  return { w, b, paramCount: dim + 1 };
}

export function predictLogistic(model, X) {
  return X.map((row) => {
    let z = model.b;
    for (let d = 0; d < row.length; d += 1) z += model.w[d] * row[d];
    return sigmoid(z);
  });
}

// ---------- 2-layer MLP (medium tier) ----------

function reluV(v) { return v.map((x) => (x > 0 ? x : 0)); }

function initMlp(inDim, hidden, seed) {
  const rng = makeRng(`pvnp-phase3-mlp-seed-${seed}-in-${inDim}`);
  const layer = (rows, cols, scale) => {
    const W = [];
    for (let r = 0; r < rows; r += 1) {
      const row = [];
      for (let c = 0; c < cols; c += 1) row.push(gaussian(rng) * scale);
      W.push(row);
    }
    return { W, b: new Array(rows).fill(0) };
  };
  // He-ish init scaled by 1/sqrt(fan_in).
  const l1 = layer(hidden[0], inDim, Math.sqrt(2 / inDim));
  const l2 = layer(hidden[1], hidden[0], Math.sqrt(2 / hidden[0]));
  const l3 = layer(1, hidden[1], Math.sqrt(2 / hidden[1]));
  return { l1, l2, l3 };
}

function paramCountMlp(inDim, hidden) {
  return inDim * hidden[0] + hidden[0]
    + hidden[0] * hidden[1] + hidden[1]
    + hidden[1] * 1 + 1;
}

function forwardLayer(layer, x) {
  const out = new Array(layer.W.length);
  for (let o = 0; o < layer.W.length; o += 1) {
    let s = layer.b[o];
    const row = layer.W[o];
    for (let i = 0; i < row.length; i += 1) s += row[i] * x[i];
    out[o] = s;
  }
  return out;
}

// Forward returning intermediate activations for backprop.
function forwardMlp(net, x) {
  const z1 = forwardLayer(net.l1, x);
  const a1 = reluV(z1);
  const z2 = forwardLayer(net.l2, a1);
  const a2 = reluV(z2);
  const z3 = forwardLayer(net.l3, a2)[0];
  const out = sigmoid(z3);
  return { z1, a1, z2, a2, out };
}

export function trainMlp(X, y, { hidden, l2: l2pen, learning_rate, epochs, seed }) {
  const inDim = X[0].length;
  const net = initMlp(inDim, hidden, seed);
  const n = X.length;
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    // accumulate full-batch gradients
    const gW1 = net.l1.W.map((r) => r.map(() => 0));
    const gb1 = net.l1.b.map(() => 0);
    const gW2 = net.l2.W.map((r) => r.map(() => 0));
    const gb2 = net.l2.b.map(() => 0);
    const gW3 = net.l3.W.map((r) => r.map(() => 0));
    const gb3 = [0];
    for (let i = 0; i < n; i += 1) {
      const x = X[i];
      const f = forwardMlp(net, x);
      const dOut = f.out - y[i]; // dL/dz3 for BCE+sigmoid
      // layer3
      for (let j = 0; j < hidden[1]; j += 1) gW3[0][j] += dOut * f.a2[j];
      gb3[0] += dOut;
      // backprop into a2
      const dA2 = new Array(hidden[1]).fill(0);
      for (let j = 0; j < hidden[1]; j += 1) dA2[j] = dOut * net.l3.W[0][j];
      const dZ2 = dA2.map((v, j) => (f.z2[j] > 0 ? v : 0));
      // layer2
      for (let o = 0; o < hidden[1]; o += 1) {
        for (let k = 0; k < hidden[0]; k += 1) gW2[o][k] += dZ2[o] * f.a1[k];
        gb2[o] += dZ2[o];
      }
      // backprop into a1
      const dA1 = new Array(hidden[0]).fill(0);
      for (let k = 0; k < hidden[0]; k += 1) {
        let s = 0;
        for (let o = 0; o < hidden[1]; o += 1) s += dZ2[o] * net.l2.W[o][k];
        dA1[k] = s;
      }
      const dZ1 = dA1.map((v, k) => (f.z1[k] > 0 ? v : 0));
      // layer1
      for (let o = 0; o < hidden[0]; o += 1) {
        for (let d = 0; d < inDim; d += 1) gW1[o][d] += dZ1[o] * x[d];
        gb1[o] += dZ1[o];
      }
    }
    const step = (Wm, gW, bm, gb) => {
      for (let o = 0; o < Wm.length; o += 1) {
        for (let d = 0; d < Wm[o].length; d += 1) Wm[o][d] -= learning_rate * (gW[o][d] / n + l2pen * Wm[o][d]);
        bm[o] -= learning_rate * (gb[o] / n);
      }
    };
    step(net.l1.W, gW1, net.l1.b, gb1);
    step(net.l2.W, gW2, net.l2.b, gb2);
    step(net.l3.W, gW3, net.l3.b, gb3);
  }
  return { net, paramCount: paramCountMlp(inDim, hidden) };
}

export function predictMlp(model, X) {
  return X.map((row) => forwardMlp(model.net, row).out);
}

// ---------- leave-one-policy-group-out binary scoring ----------

// items: [{ groupId, features:[...], label:0|1 }]. Returns concatenated
// held-out scores/labels and the per-fold train-selected thresholds. tier
// selects logistic (small) vs MLP-seed-ensemble (medium).
export function leaveOnePolicyGroupOutBinary(items, tier, classes) {
  const groups = Array.from(new Set(items.map((it) => it.groupId)));
  const heldScores = [];
  const heldLabels = [];
  const heldPreds = [];
  const foldRows = [];
  let paramCount = 0;
  for (const heldGroup of groups) {
    const train = items.filter((it) => it.groupId !== heldGroup);
    const held = items.filter((it) => it.groupId === heldGroup);
    if (train.every((it) => it.label === train[0].label)) {
      // Degenerate train fold (single class) — record and skip scoring this fold.
      foldRows.push({ held_group: heldGroup, n_held: held.length, skipped: 1, reason: "single-class train fold" });
      continue;
    }
    const st = fitStandardizer(train.map((it) => it.features));
    const Xtr = train.map((it) => applyStandardizer(it.features, st));
    const ytr = train.map((it) => it.label);
    const Xhe = held.map((it) => applyStandardizer(it.features, st));

    let trainScores;
    let heldGroupScores;
    if (tier === "small") {
      const model = trainLogistic(Xtr, ytr, classes.small);
      paramCount = model.paramCount;
      trainScores = predictLogistic(model, Xtr);
      heldGroupScores = predictLogistic(model, Xhe);
    } else {
      // medium: average the sigmoid outputs across the frozen seed ensemble.
      const seeds = classes.medium.seeds;
      const trainMat = seeds.map((seed) => {
        const model = trainMlp(Xtr, ytr, { ...classes.medium, seed });
        paramCount = model.paramCount;
        return { model };
      });
      const avg = (mat, X) => {
        const preds = mat.map(({ model }) => predictMlp(model, X));
        return X.map((_, i) => preds.reduce((s, p) => s + p[i], 0) / preds.length);
      };
      trainScores = avg(trainMat, Xtr);
      heldGroupScores = avg(trainMat, Xhe);
    }
    const threshold = selectThresholdTrainOnly(trainScores, ytr);
    for (let i = 0; i < held.length; i += 1) {
      heldScores.push(heldGroupScores[i]);
      heldLabels.push(held[i].label);
      heldPreds.push(heldGroupScores[i] >= threshold ? 1 : 0);
    }
    foldRows.push({
      held_group: heldGroup,
      n_held: held.length,
      train_threshold: threshold,
      held_positive: held.filter((it) => it.label === 1).length,
      skipped: 0,
    });
  }
  let baCorrect = 0;
  for (let i = 0; i < heldPreds.length; i += 1) if (heldPreds[i] === heldLabels[i]) baCorrect += 1;
  // Balanced accuracy from concatenated held-out predicted labels.
  let tp = 0; let fn = 0; let tn = 0; let fp = 0;
  for (let i = 0; i < heldPreds.length; i += 1) {
    if (heldLabels[i] === 1) { if (heldPreds[i] === 1) tp += 1; else fn += 1; }
    else { if (heldPreds[i] === 0) tn += 1; else fp += 1; }
  }
  const tpr = tp + fn > 0 ? tp / (tp + fn) : 0;
  const tnr = tn + fp > 0 ? tn / (tn + fp) : 0;
  return {
    auroc: auroc(heldScores, heldLabels),
    balanced_accuracy: (tpr + tnr) / 2,
    raw_accuracy: heldPreds.length ? baCorrect / heldPreds.length : null,
    n_scored: heldScores.length,
    n_groups: groups.length,
    param_count: paramCount,
    confusion: { tp, fn, tn, fp },
    folds: foldRows,
  };
}

// Leave-one-policy-group-out scalar regression for old_basin_pref (diagnostic
// unless its independence floor is met). Reuses the MLP trunk with a linear
// readout for medium; small tier uses ridge-like logistic-free linear fit.
export function leaveOnePolicyGroupOutScalar(items, tier, classes) {
  const groups = Array.from(new Set(items.map((it) => it.groupId)));
  const preds = [];
  const targets = [];
  for (const heldGroup of groups) {
    const train = items.filter((it) => it.groupId !== heldGroup);
    const held = items.filter((it) => it.groupId === heldGroup);
    const st = fitStandardizer(train.map((it) => it.features));
    const Xtr = train.map((it) => applyStandardizer(it.features, st));
    const ytr = train.map((it) => it.target);
    const Xhe = held.map((it) => applyStandardizer(it.features, st));
    // Closed-form ridge regression (deterministic) for both tiers' scalar
    // diagnostic; this target is diagnostic-only in v0 so a linear probe is the
    // honest minimal estimator rather than an over-parameterized fit.
    const model = ridge(Xtr, ytr, tier === "small" ? classes.small.l2 : classes.medium.l2);
    for (let i = 0; i < Xhe.length; i += 1) {
      preds.push(predictRidge(model, Xhe[i]));
      targets.push(held[i].target);
    }
  }
  return { mae: mae(preds, targets), n_scored: preds.length, n_groups: groups.length };
}

// Simple ridge regression via normal equations with L2 on weights (not bias).
function ridge(X, y, lambda) {
  const n = X.length;
  const dim = X[0].length;
  // augment with bias column
  const A = X.map((r) => [...r, 1]);
  const p = dim + 1;
  // Build A^T A + lambda*I (no penalty on bias) and A^T y
  const ata = Array.from({ length: p }, () => new Array(p).fill(0));
  const aty = new Array(p).fill(0);
  for (let i = 0; i < n; i += 1) {
    for (let a = 0; a < p; a += 1) {
      aty[a] += A[i][a] * y[i];
      for (let b = 0; b < p; b += 1) ata[a][b] += A[i][a] * A[i][b];
    }
  }
  for (let d = 0; d < dim; d += 1) ata[d][d] += lambda * n;
  const w = solveLinear(ata, aty);
  return { w };
}

function predictRidge(model, x) {
  let s = model.w[model.w.length - 1];
  for (let d = 0; d < x.length; d += 1) s += model.w[d] * x[d];
  return s;
}

// Gaussian elimination with partial pivoting.
function solveLinear(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col += 1) {
    let piv = col;
    for (let r = col + 1; r < n; r += 1) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    [M[col], M[piv]] = [M[piv], M[col]];
    const d = M[col][col] || 1e-12;
    for (let r = 0; r < n; r += 1) {
      if (r === col) continue;
      const f = M[r][col] / d;
      for (let c = col; c <= n; c += 1) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / (M[i][i] || 1e-12));
}

export { paramCountMlp };
