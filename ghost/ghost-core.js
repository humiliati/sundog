// Ghost Phase 2 toy closure workbench core.
//
// Pure logic for the internal reader toy at ghost/workbench.html and the Node
// acceptance tests in scripts/ghost-workbench-tests.mjs.
//
// This is not a tiling theorem probe. It compares a periodic symbolic stripe
// with a Fibonacci substitution stripe and keeps the Wang/SFT undecidability
// cliff out of the metric ladder.

export const DEFAULT_SYSTEM_ID = "periodic4";
export const SYSTEM_IDS = ["periodic4", "fibonacci"];
export const PERIODIC_MOTIF = ["A", "B", "C", "D"];
export const DEFAULT_PERIODIC_LENGTH = 192;
export const DEFAULT_FIB_LEVEL = 11;

const lenMemo = new Map();

export function isSystemId(id) {
  return SYSTEM_IDS.includes(id);
}

export function fibonacciSymbolLength(symbol, level) {
  const key = `${symbol}:${level}`;
  if (lenMemo.has(key)) return lenMemo.get(key);
  let value;
  if (level === 0) value = 1;
  else if (symbol === "A") {
    value = fibonacciSymbolLength("A", level - 1) + fibonacciSymbolLength("B", level - 1);
  } else if (symbol === "B") {
    value = fibonacciSymbolLength("A", level - 1);
  } else {
    throw new Error(`unknown Fibonacci symbol: ${symbol}`);
  }
  lenMemo.set(key, value);
  return value;
}

function expandFibonacciSymbol(symbol, level, start, intervals, symbols) {
  const length = fibonacciSymbolLength(symbol, level);
  const end = start + length;
  intervals.push({ symbol, level, start, end, length });
  if (level === 0) {
    symbols[start] = symbol;
    return end;
  }
  if (symbol === "A") {
    const mid = expandFibonacciSymbol("A", level - 1, start, intervals, symbols);
    return expandFibonacciSymbol("B", level - 1, mid, intervals, symbols);
  }
  return expandFibonacciSymbol("A", level - 1, start, intervals, symbols);
}

export function makePeriodicSystem(length = DEFAULT_PERIODIC_LENGTH) {
  const symbols = Array.from({ length }, (_, i) => PERIODIC_MOTIF[i % PERIODIC_MOTIF.length]);
  return {
    id: "periodic4",
    label: "Periodic control",
    type: "periodic",
    symbols,
    length,
    motif: [...PERIODIC_MOTIF],
    globalPeriod: PERIODIC_MOTIF.length,
    maxAncestryLevel: 0,
  };
}

export function makeFibonacciSystem(level = DEFAULT_FIB_LEVEL) {
  const intervals = [];
  const symbols = [];
  const length = fibonacciSymbolLength("A", level);
  expandFibonacciSymbol("A", level, 0, intervals, symbols);
  return {
    id: "fibonacci",
    label: "Fibonacci substitution",
    type: "substitution",
    symbols,
    length,
    rootLevel: level,
    maxAncestryLevel: level,
    intervals,
  };
}

export function makeSystem(id = DEFAULT_SYSTEM_ID) {
  if (id === "periodic4") return makePeriodicSystem();
  if (id === "fibonacci") return makeFibonacciSystem();
  throw new Error(`unknown Ghost system: ${id}`);
}

export function clampInt(value, min, max) {
  const n = Number.isFinite(Number(value)) ? Math.trunc(Number(value)) : min;
  return Math.max(min, Math.min(max, n));
}

export function windowBounds(system, center, radius) {
  const safeRadius = clampInt(radius, 1, Math.max(1, Math.floor(system.length / 2)));
  const safeCenter = clampInt(center, 0, system.length - 1);
  const start = Math.max(0, safeCenter - safeRadius);
  const end = Math.min(system.length, safeCenter + safeRadius + 1);
  return { center: safeCenter, radius: safeRadius, start, end, length: end - start };
}

export function visibleBounds(system, center, radius, pad = 34) {
  const w = windowBounds(system, center, radius);
  const extra = Math.max(pad, radius + 8);
  const start = Math.max(0, w.center - extra);
  const end = Math.min(system.length, w.center + extra + 1);
  return { start, end, length: end - start };
}

export function symbolCounts(symbols) {
  const counts = {};
  for (const s of symbols) counts[s] = (counts[s] || 0) + 1;
  return counts;
}

export function hasPeriod(symbols, period) {
  if (!Number.isInteger(period) || period < 1 || period > Math.floor(symbols.length / 2)) return false;
  for (let i = period; i < symbols.length; i++) {
    if (symbols[i] !== symbols[i - period]) return false;
  }
  return true;
}

export function periodCandidates(symbols, maxPeriod = 16) {
  const cap = Math.min(maxPeriod, Math.floor(symbols.length / 2));
  const hits = [];
  for (let p = 1; p <= cap; p++) {
    if (hasPeriod(symbols, p)) hits.push(p);
  }
  return hits;
}

export function intervalsAtLevel(system, level) {
  if (system.type !== "substitution") return [];
  const safeLevel = clampInt(level, 0, system.maxAncestryLevel);
  return system.intervals
    .filter((interval) => interval.level === safeLevel)
    .sort((a, b) => a.start - b.start);
}

export function intervalsPartition(system, level) {
  if (system.type !== "substitution") return false;
  const intervals = intervalsAtLevel(system, level);
  if (!intervals.length) return false;
  if (intervals[0].start !== 0) return false;
  for (let i = 1; i < intervals.length; i++) {
    if (intervals[i - 1].end !== intervals[i].start) return false;
  }
  return intervals[intervals.length - 1].end === system.length;
}

export function ancestryStats(system, start, end, level) {
  if (system.type !== "substitution") return null;
  const safeLevel = clampInt(level, 0, system.maxAncestryLevel);
  const intervals = intervalsAtLevel(system, safeLevel).filter(
    (interval) => interval.end > start && interval.start < end,
  );
  const boundaries = [];
  for (const interval of intervals) {
    if (interval.start > start && interval.start < end) boundaries.push(interval.start);
    if (interval.end > start && interval.end < end) boundaries.push(interval.end);
  }
  const uniqueBoundaries = [...new Set(boundaries)].sort((a, b) => a - b);
  const containing = intervals.find((interval) => interval.start <= start && interval.end >= end) || null;
  return {
    level: safeLevel,
    intersectingBlocks: intervals.length,
    interiorBoundaries: uniqueBoundaries.length,
    boundaryPositions: uniqueBoundaries,
    containingBlockLength: containing ? containing.length : null,
    containingSymbol: containing ? containing.symbol : null,
    partitionOk: intervalsPartition(system, safeLevel),
  };
}

export function analyzeWindow(system, center, radius, ancestryLevel = 3) {
  const bounds = windowBounds(system, center, radius);
  const symbols = system.symbols.slice(bounds.start, bounds.end);
  const candidates = periodCandidates(symbols, 16);
  const counts = symbolCounts(symbols);
  const periodicClosed =
    system.type === "periodic" &&
    bounds.length >= system.globalPeriod * 2 &&
    candidates.includes(system.globalPeriod) &&
    candidates.every((p) => p >= system.globalPeriod);
  const ancestry = ancestryStats(system, bounds.start, bounds.end, ancestryLevel);

  let regime;
  let verdict;
  if (system.type === "periodic") {
    regime = "bounded repeat-cell closure";
    verdict = periodicClosed
      ? "repeat cell captured inside this window"
      : "window is still too small or ambiguous for the repeat-cell certificate";
  } else {
    regime = "bounded substitution recognition";
    verdict = candidates.length
      ? "local repeat candidates are finite-window artifacts, not a global period"
      : "no repeat-cell certificate; ancestry is substitutional, not translational";
  }

  return {
    systemId: system.id,
    systemLabel: system.label,
    regime,
    verdict,
    window: bounds,
    counts,
    periodCandidates: candidates,
    periodicClosed,
    ancestry,
    cliff: {
      label: "computability cliff",
      active: false,
      note:
        "General Wang/SFT extension can encode computation. This toy does not measure that regime.",
    },
  };
}

export function exportReaderAnalysis(system, center, radius, ancestryLevel = 3) {
  const analysis = analyzeWindow(system, center, radius, ancestryLevel);
  return {
    systemId: analysis.systemId,
    regime: analysis.regime,
    window: analysis.window,
    counts: analysis.counts,
    periodCandidates: analysis.periodCandidates,
    ancestry: analysis.ancestry,
    cliff: analysis.cliff,
  };
}

