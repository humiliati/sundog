const SUN = Object.freeze({ x: 500, y: 500 });
const HALO_22_RADIUS = 220;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function readCssNumber(style, name, fallback = 0) {
  const raw = style.getPropertyValue(name);
  const num = Number.parseFloat(raw);
  return Number.isFinite(num) ? num : fallback;
}

function normalize2(x, y) {
  const mag = Math.hypot(x, y);
  if (mag <= 1e-9) return { x: 1, y: 0 };
  return { x: x / mag, y: y / mag };
}

function ellipseImplicitAxisAligned(point, ellipse) {
  const dx = point.x - ellipse.cx;
  const dy = point.y - ellipse.cy;
  return (dx * dx) / (ellipse.rx * ellipse.rx) + (dy * dy) / (ellipse.ry * ellipse.ry) - 1;
}

function ellipseTangentAxisAligned(point, ellipse) {
  const dx = point.x - ellipse.cx;
  const dy = point.y - ellipse.cy;
  const dFdx = (2 * dx) / (ellipse.rx * ellipse.rx);
  const dFdy = (2 * dy) / (ellipse.ry * ellipse.ry);
  return normalize2(-dFdy, dFdx);
}

function refineRootBisection(thetaA, thetaB, fA, fB, fTheta, maxIter = 28) {
  let a = thetaA;
  let b = thetaB;
  let fa = fA;
  let fb = fB;

  for (let i = 0; i < maxIter; i += 1) {
    const m = 0.5 * (a + b);
    const fm = fTheta(m);
    if (Math.abs(fm) < 1e-6) return m;
    if (fa * fm <= 0) {
      b = m;
      fb = fm;
    } else {
      a = m;
      fa = fm;
    }
  }
  return 0.5 * (a + b);
}

function circleEllipseIntersectionsAxisAligned(circle, ellipse, steps = 960) {
  const intersections = [];
  const twoPi = Math.PI * 2;

  const fTheta = (theta) => {
    const x = circle.cx + circle.r * Math.cos(theta);
    const y = circle.cy + circle.r * Math.sin(theta);
    return ellipseImplicitAxisAligned({ x, y }, ellipse);
  };

  let prevTheta = 0;
  let prevF = fTheta(prevTheta);
  for (let i = 1; i <= steps; i += 1) {
    const theta = (i / steps) * twoPi;
    const f = fTheta(theta);
    if (prevF === 0 || f === 0 || prevF * f < 0) {
      const rootTheta = refineRootBisection(prevTheta, theta, prevF, f, fTheta);
      intersections.push({
        x: circle.cx + circle.r * Math.cos(rootTheta),
        y: circle.cy + circle.r * Math.sin(rootTheta),
      });
    }
    prevTheta = theta;
    prevF = f;
  }

  // Deduplicate near-identical points (scan resolution can re-find the same root).
  const unique = [];
  const eps = 0.75;
  for (const p of intersections) {
    if (!unique.some((q) => Math.hypot(p.x - q.x, p.y - q.y) < eps)) unique.push(p);
  }
  return unique;
}

function pickLeftRightIntersections(points) {
  let left = null;
  let right = null;
  for (const p of points) {
    if (p.x < SUN.x) {
      if (!left || Math.abs(p.y - SUN.y) < Math.abs(left.y - SUN.y)) left = p;
    } else if (p.x > SUN.x) {
      if (!right || Math.abs(p.y - SUN.y) < Math.abs(right.y - SUN.y)) right = p;
    }
  }
  return { left, right };
}

function ellipseArcPathAxisAligned(ellipse, xMin, xMax, steps = 140) {
  const clampedMin = Math.max(ellipse.cx - ellipse.rx, xMin);
  const clampedMax = Math.min(ellipse.cx + ellipse.rx, xMax);
  const dx = clampedMax - clampedMin;
  if (dx <= 1e-6) return "";

  const points = [];
  for (let i = 0; i <= steps; i += 1) {
    const x = clampedMin + (dx * i) / steps;
    const u = (x - ellipse.cx) / ellipse.rx;
    const inside = 1 - u * u;
    const y = ellipse.cy + ellipse.ry * Math.sqrt(Math.max(0, inside)); // lower half in screen coords
    points.push({ x, y });
  }

  const parts = [];
  parts.push(`M ${points[0].x.toFixed(2)} ${points[0].y.toFixed(2)}`);
  for (let i = 1; i < points.length; i += 1) {
    parts.push(`L ${points[i].x.toFixed(2)} ${points[i].y.toFixed(2)}`);
  }
  return parts.join(" ");
}

function applyCompassRays(svg, compassLen) {
  const sunGroup = svg.querySelector("#layer-sun");
  if (!sunGroup) return;

  sunGroup.querySelectorAll(".ray").forEach((line) => {
    if (!line.dataset.basisX2) {
      line.dataset.basisX1 = line.getAttribute("x1");
      line.dataset.basisY1 = line.getAttribute("y1");
      line.dataset.basisX2 = line.getAttribute("x2");
      line.dataset.basisY2 = line.getAttribute("y2");
    }
    const x1 = Number.parseFloat(line.dataset.basisX1);
    const y1 = Number.parseFloat(line.dataset.basisY1);
    const x2 = Number.parseFloat(line.dataset.basisX2);
    const y2 = Number.parseFloat(line.dataset.basisY2);
    line.setAttribute("x2", (x1 + (x2 - x1) * compassLen).toFixed(2));
    line.setAttribute("y2", (y1 + (y2 - y1) * compassLen).toFixed(2));
  });

  const sunCore = svg.querySelector("#sun-core");
  if (sunCore) sunCore.setAttribute("r", (9 * Math.sqrt(compassLen)).toFixed(2));
}

function applySecondaryHalos(svg, overlapBias) {
  const secondaries = svg.querySelectorAll("#layer-secondary-halos circle");
  if (secondaries.length < 2) return;

  const offset = (overlapBias - 0.5) * 160;
  const lx = 320 + offset;
  const rx = 680 - offset;
  const cy = 320 + offset * 0.5;
  secondaries[0].setAttribute("cx", lx.toFixed(2));
  secondaries[0].setAttribute("cy", cy.toFixed(2));
  secondaries[1].setAttribute("cx", rx.toFixed(2));
  secondaries[1].setAttribute("cy", cy.toFixed(2));
}

function applyCza(svg, czaCurve) {
  const czaPrimary = svg.querySelector("#cza-primary");
  if (!czaPrimary) return;
  const apexY = 240 - 160 * (czaCurve - 0.85);
  czaPrimary.setAttribute("d", `M 200 240 Q 500 ${apexY.toFixed(2)} 800 240`);
}

function applyPillarLegacy(svg, pillarLen) {
  const pillar = svg.querySelector("#pillar-line");
  if (!pillar) return;
  const pillarUp = 360 * pillarLen;
  const pillarDown = 90 * pillarLen;
  pillar.setAttribute("y1", (SUN.y - pillarUp).toFixed(2));
  pillar.setAttribute("y2", (SUN.y + pillarDown).toFixed(2));
}

function applyPillarLens(svg, pillarLen) {
  const lens = svg.querySelector("#pillar-lens");
  if (!lens) return;

  const r = 90 + 190 * clamp(pillarLen, 0, 1);
  const d = 2 * r * (1 - 0.13); // slightly overlapped circles -> lens
  const half = d / 2;
  const h = Math.sqrt(Math.max(0, r * r - half * half));
  const yTop = SUN.y - h;
  const yBottom = SUN.y + h;

  lens.setAttribute(
    "d",
    [
      `M ${SUN.x.toFixed(2)} ${yTop.toFixed(2)}`,
      `A ${r.toFixed(2)} ${r.toFixed(2)} 0 0 1 ${SUN.x.toFixed(2)} ${yBottom.toFixed(2)}`,
      `A ${r.toFixed(2)} ${r.toFixed(2)} 0 0 0 ${SUN.x.toFixed(2)} ${yTop.toFixed(2)}`,
      "Z",
    ].join(" ")
  );
}

function applyDaggersFromParhelicEllipse(svg, daggerLen, parhelicEllipse) {
  const group = svg.querySelector(".layer-parhelia");
  if (!group) return;
  const daggers = group.querySelectorAll(".dagger");
  if (daggers.length < 2) return;

  const candidates = circleEllipseIntersectionsAxisAligned(
    { cx: SUN.x, cy: SUN.y, r: HALO_22_RADIUS },
    parhelicEllipse
  );
  const { left, right } = pickLeftRightIntersections(candidates);
  if (!left || !right) return;

  const applyOne = (dagger, point) => {
    const streak = dagger.querySelector(".dagger-streak");
    const core = dagger.querySelector(".dagger-core");
    if (!streak || !core) return;

    let tangent = ellipseTangentAxisAligned(point, parhelicEllipse);
    const fromSun = normalize2(point.x - SUN.x, point.y - SUN.y);
    if (tangent.x * fromSun.x + tangent.y * fromSun.y < 0) tangent = { x: -tangent.x, y: -tangent.y };

    const outward = 80 * daggerLen;
    const inward = 55 * daggerLen;
    const out = { x: point.x + tangent.x * outward, y: point.y + tangent.y * outward };
    const inn = { x: point.x - tangent.x * inward, y: point.y - tangent.y * inward };

    streak.setAttribute("x1", out.x.toFixed(2));
    streak.setAttribute("y1", out.y.toFixed(2));
    streak.setAttribute("x2", inn.x.toFixed(2));
    streak.setAttribute("y2", inn.y.toFixed(2));

    core.setAttribute("cx", point.x.toFixed(2));
    core.setAttribute("cy", point.y.toFixed(2));
  };

  applyOne(daggers[0], left);
  applyOne(daggers[1], right);
}

function applyParhelicLegacy(svg, parhelicCurvature) {
  const parhelic = svg.querySelector("#parhelic-path");
  if (!parhelic) return;

  const parhelicEndY = SUN.y - 200 * parhelicCurvature;
  const parhelicCtrlY = SUN.y + 200 * parhelicCurvature;
  parhelic.setAttribute(
    "d",
    `M 0 ${parhelicEndY.toFixed(2)} Q 500 ${parhelicCtrlY.toFixed(2)} 1000 ${parhelicEndY.toFixed(2)}`
  );
}

function applyParhelicHaloScaffold(svg, altitudeDeg, parhelicCurvature) {
  const parhelic = svg.querySelector("#parhelic-path");
  if (!parhelic) return null;

  const altNorm = clamp(altitudeDeg / 60, 0, 1);
  const curvature = clamp(parhelicCurvature, 0, 1);

  // Parhelic scaffold: an ellipse whose bottom point is the sun (tangent horizontal).
  // The ellipse is intentionally "halo-first": it governs both the arc AND (via
  // intersection with the 22° halo) the parhelia locations.
  const ry = 210 + 240 * curvature + 60 * altNorm;
  const rx = 560 + 90 * (1 - altNorm) + 160 * (1 - curvature);
  const ellipse = {
    cx: SUN.x,
    cy: SUN.y - ry,
    rx,
    ry,
  };

  const d = ellipseArcPathAxisAligned(ellipse, 0, 1000);
  if (d) parhelic.setAttribute("d", d);
  return ellipse;
}

function applyGeometryLegacy(svg, rootStyle) {
  const pillarLen = readCssNumber(rootStyle, "--sun-pillar-length", 0.65);
  const daggerLen = readCssNumber(rootStyle, "--parhelia-dagger-length", 1);
  const compassLen = readCssNumber(rootStyle, "--compass-ray-length", 1);
  const czaCurve = readCssNumber(rootStyle, "--cza-curvature", 0.85);
  const overlapBias = readCssNumber(rootStyle, "--ring-overlap-bias", 0.5);
  const parhelicCurvature = readCssNumber(rootStyle, "--parhelic-curvature", 0.66);

  applyPillarLegacy(svg, pillarLen);
  applyParhelicLegacy(svg, parhelicCurvature);
  applyCompassRays(svg, compassLen);
  applyCza(svg, czaCurve);
  applySecondaryHalos(svg, overlapBias);

  // Daggers ride the legacy Bezier arc by sampling y on the curve at fixed x.
  const parhelicY = (x, curvature) => {
    const A = SUN.y - 200 * curvature;
    const B = SUN.y + 200 * curvature;
    const t = x / 1000;
    const omt = 1 - t;
    return omt * omt * A + 2 * t * omt * B + t * t * A;
  };

  const parheliaGroup = svg.querySelector(".layer-parhelia");
  const daggers = parheliaGroup?.querySelectorAll(".dagger") ?? [];
  if (daggers.length >= 2) {
    const leftHiltX = 280;
    const rightHiltX = 720;
    const leftHiltY = parhelicY(leftHiltX, parhelicCurvature);
    const rightHiltY = parhelicY(rightHiltX, parhelicCurvature);

    const leftStreak = daggers[0].querySelector(".dagger-streak");
    const leftCore = daggers[0].querySelector(".dagger-core");
    const lx1 = leftHiltX - 60 * daggerLen;
    const lx2 = leftHiltX + 40 * daggerLen;
    leftStreak?.setAttribute("x1", lx1.toFixed(2));
    leftStreak?.setAttribute("y1", parhelicY(lx1, parhelicCurvature).toFixed(2));
    leftStreak?.setAttribute("x2", lx2.toFixed(2));
    leftStreak?.setAttribute("y2", parhelicY(lx2, parhelicCurvature).toFixed(2));
    leftCore?.setAttribute("cx", leftHiltX.toFixed(2));
    leftCore?.setAttribute("cy", leftHiltY.toFixed(2));

    const rightStreak = daggers[1].querySelector(".dagger-streak");
    const rightCore = daggers[1].querySelector(".dagger-core");
    const rx1 = rightHiltX + 60 * daggerLen;
    const rx2 = rightHiltX - 40 * daggerLen;
    rightStreak?.setAttribute("x1", rx1.toFixed(2));
    rightStreak?.setAttribute("y1", parhelicY(rx1, parhelicCurvature).toFixed(2));
    rightStreak?.setAttribute("x2", rx2.toFixed(2));
    rightStreak?.setAttribute("y2", parhelicY(rx2, parhelicCurvature).toFixed(2));
    rightCore?.setAttribute("cx", rightHiltX.toFixed(2));
    rightCore?.setAttribute("cy", rightHiltY.toFixed(2));
  }
}

function applyGeometryHaloScaffold(svg, rootStyle) {
  const altitudeDeg = readCssNumber(rootStyle, "--sun-altitude", 18);
  const pillarLen = readCssNumber(rootStyle, "--sun-pillar-length", 0.65);
  const daggerLen = readCssNumber(rootStyle, "--parhelia-dagger-length", 1);
  const compassLen = readCssNumber(rootStyle, "--compass-ray-length", 1);
  const czaCurve = readCssNumber(rootStyle, "--cza-curvature", 0.85);
  const overlapBias = readCssNumber(rootStyle, "--ring-overlap-bias", 0.5);
  const parhelicCurvature = readCssNumber(rootStyle, "--parhelic-curvature", 0.66);

  // 1) Halo/ellipse-first parhelic scaffold.
  const parhelicEllipse = applyParhelicHaloScaffold(svg, altitudeDeg, parhelicCurvature);

  // 2) Derive daggers from the interaction of the parhelic scaffold and the 22° halo.
  if (parhelicEllipse) applyDaggersFromParhelicEllipse(svg, daggerLen, parhelicEllipse);

  // 3) Model the north/south structure as a halo-contact lens, not a straight line.
  applyPillarLens(svg, pillarLen);

  // Shared behaviors.
  applyCompassRays(svg, compassLen);
  applyCza(svg, czaCurve);
  applySecondaryHalos(svg, overlapBias);

  // Keep the legacy pillar line updated (it is hidden in halo mode via CSS).
  applyPillarLegacy(svg, pillarLen);
}

export function applyParhelionGeometry({ svg, rootStyle, model }) {
  if (!svg || !rootStyle) return;
  svg.dataset.geometryModel = model;

  if (model === "halo_scaffold") {
    applyGeometryHaloScaffold(svg, rootStyle);
  } else {
    applyGeometryLegacy(svg, rootStyle);
  }
}

