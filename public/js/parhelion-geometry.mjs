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

// 120 steps × ≤4 real roots is comfortable; halving the workhorse step count from
// the original 960 cuts per-slider work ~8× without changing root quality, since
// bisection refines each crossing to 1e-6 anyway. Important for Phase 4 idle
// scintillation where this fires every animation frame.
function circleEllipseIntersectionsAxisAligned(circle, ellipse, steps = 120) {
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
  //
  // KNOWN GAP: the rx/ry blends below are placeholder hand-fit constants chosen to
  // approximate the legacy dagger placement at default sliders, NOT yet derived from
  // sun-altitude or any halo angular radius. The doc's "fewer ad hoc placements"
  // claim is not yet earned for this model. Replacing these with a true halo
  // (single-radius circle) is what the `halo_governed` model attempts. See
  // SUNDOG_V_GEOMETRY.md → "Outstanding gaps from issue #15".
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

  // The legacy pillar-line is hidden via CSS in this mode, so we deliberately
  // skip recomputing it on every slider event.
}

// --- halo_governed (v2, mockup-faithful) -------------------------------------
//
// Daggers are FIXED at the parhelic-circle position on the 22° halo — i.e.
// (500-220, 500) and (500+220, 500). They are the anchor of the central
// scaffold, not a derived intersection.
//
// Two virtual halos centered AT the daggers (radius = 220 + small slack) are
// the "two large halos" of the issue's proposal: catalysts/spawn of the
// daggers, never rendered. Their vesica IS the pillar.
//
// The parhelic arc is the unique circle through (left dagger, sun-apex,
// right dagger). At zero curvature it degenerates to a horizontal line. At
// positive curvature `--parhelic-curvature`, the apex sits `h` above the sun
// where h scales linearly with curvature, and the arc is the upper portion
// of the resulting circle.

const HALO_22_RADIUS_FOR_GOVERNED = HALO_22_RADIUS;

function daggerPointsHorizontal() {
  return {
    left: { x: SUN.x - HALO_22_RADIUS_FOR_GOVERNED, y: SUN.y },
    right: { x: SUN.x + HALO_22_RADIUS_FOR_GOVERNED, y: SUN.y },
  };
}

function applyParhelicCircleThroughDaggersAndSun(svg, parhelicCurvature) {
  const parhelic = svg.querySelector("#parhelic-path");
  if (!parhelic) return;

  // Map curvature ∈ [0, 1] to apex-BELOW-sun depth d ∈ [0, 200]. The parhelic
  // circle in the Troels Nielsen photograph smiles (apex below sun), so this
  // function draws the LOWER arc of a circle whose center sits ABOVE the
  // daggers — apex dips below the sun-line.
  const c = clamp(parhelicCurvature, 0, 1);
  const d = 200 * c;

  if (d < 0.5) {
    parhelic.setAttribute(
      "d",
      `M ${(SUN.x - HALO_22_RADIUS_FOR_GOVERNED).toFixed(2)} ${SUN.y.toFixed(2)} ` +
        `L ${(SUN.x + HALO_22_RADIUS_FOR_GOVERNED).toFixed(2)} ${SUN.y.toFixed(2)}`
    );
    return;
  }

  // Unique circle through (SUN.x ± 220, SUN.y) and (SUN.x, SUN.y + d).
  // Center on x = SUN.x. Let u = SUN.y - cy (positive: center above daggers).
  //   halfChord² + u² = (u + d)²  →  u = (halfChord² - d²) / (2d)
  const halfChord = HALO_22_RADIUS_FOR_GOVERNED;
  const u_ = (halfChord * halfChord - d * d) / (2 * d);
  const cy = SUN.y - u_;
  const r = Math.hypot(halfChord, u_);

  const xMin = Math.max(SUN.x - r, 0);
  const xMax = Math.min(SUN.x + r, 1000);
  const dx = xMax - xMin;
  if (dx <= 1e-6) return;
  const steps = 140;
  const parts = [];
  for (let i = 0; i <= steps; i += 1) {
    const x = xMin + (dx * i) / steps;
    const uu = (x - SUN.x) / r;
    const inside = 1 - uu * uu;
    // LOWER arc: y = cy + r·√(1-u²). Apex at (SUN.x, cy + r) = (SUN.x, SUN.y + d).
    const y = cy + r * Math.sqrt(Math.max(0, inside));
    parts.push(`${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`);
  }
  parhelic.setAttribute("d", parts.join(" "));
}

function applyDaggersFromGoverningHalo(svg, daggerLen, daggerPoints) {
  const group = svg.querySelector(".layer-parhelia");
  if (!group || !daggerPoints) return;
  const daggers = group.querySelectorAll(".dagger");
  if (daggers.length < 2) return;

  const applyOne = (dagger, point) => {
    const streak = dagger.querySelector(".dagger-streak");
    const core = dagger.querySelector(".dagger-core");
    if (!streak || !core) return;

    // Streak direction = tangent to the 22° halo at the dagger point (i.e.,
    // perpendicular to the radial line from sun to dagger). This is the
    // physically motivated choice: dagger streaks lie ALONG the halo, not
    // along the parhelic arc tangent.
    const radial = normalize2(point.x - SUN.x, point.y - SUN.y);
    let tangent = { x: -radial.y, y: radial.x };
    // Orient outward (away from sun on the streak's outer end).
    if (tangent.x * radial.x + tangent.y * radial.y < 0) {
      tangent = { x: -tangent.x, y: -tangent.y };
    }

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

  applyOne(daggers[0], daggerPoints.left);
  applyOne(daggers[1], daggerPoints.right);
}

function applyPillarFromTwoHalos(svg, daggerPoints, pillarLen) {
  const lens = svg.querySelector("#pillar-lens");
  if (!lens || !daggerPoints) return;

  // Two halos of radius (sun-to-dagger distance) + slack, centered AT the
  // daggers. They overlap in a tall thin vesica at the sun. This ties the
  // pillar's geometry to the governing halo via the daggers — no independent
  // pillar-radius constant.
  const dxSD = daggerPoints.left.x - SUN.x;
  const dySD = daggerPoints.left.y - SUN.y;
  const sunToDagger = Math.hypot(dxSD, dySD);
  const slack = 5 + 30 * clamp(pillarLen, 0, 1);
  const r = sunToDagger + slack;

  // Lens tips: along the perpendicular bisector of the two dagger centers,
  // passing through the sun (since the two daggers are mirror-symmetric across
  // x=SUN.x at the same y). Centers' midpoint y = daggers' y = SUN.y + dy_d
  // where dy_d is small and negative (daggers above sun). Perpendicular
  // bisector is the vertical line x = SUN.x. Tips at (SUN.x, midY ± h) where
  // h = sqrt(r² - dx_d²), dx_d = horizontal distance from midpoint to a
  // dagger center.
  const midY = daggerPoints.left.y;
  const halfChord = Math.abs(dxSD); // horizontal half-distance between dagger centers
  const inside = r * r - halfChord * halfChord;
  if (inside <= 0) {
    lens.setAttribute("d", "");
    return;
  }
  const h = Math.sqrt(inside);
  const yTop = midY - h;
  const yBottom = midY + h;

  // Two-arc lens path. First arc (sweep=1) bulges right — the boundary that
  // belongs to the LEFT halo (centered at the left dagger). Second arc
  // (sweep=0) bulges left — boundary of the RIGHT halo.
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

// --- halo_atlas: multi-primitive layered model -------------------------------
//
// Each visible halo / arc is a first-class circle (or upper-arc-of-circle).
// Daggers and pillar are derived from intersections / vesicas. Sun altitude
// finally binds: parhelion offset = R_22 / cos(altitude). Calibration against
// the Troels Nielsen photograph showed the parhelia sit at ~24° from the sun
// (not 22° on the halo edge), implying sun altitude ≈ 25° — this is the
// Phase 3 binding the doc has had on its roadmap.
//
// Primitives in this mode:
//   - 22° halo (sun-centered, existing SVG circle)
//   - 46° halo (sun-centered, existing SVG circle)
//   - CZA as full upper-arc of a real circle (replaces the buggy applyCza
//     for this mode; primary + secondary arcs share a (cx, cy) family,
//     bell-fill is the band between them)
//   - Supralateral arc (NEW): upper arc of a circle tangent to the 46° halo
//     at its top, curving up
//   - Parry-family / lateral vocabulary primitives: optional overlays for the
//     suncave Parry arc, Parry supralateral shoulders, and infralateral arcs
//     called out by the rich calibration references
//   - Daggers: at (SUN.x ± R_22/cos(h), SUN.y + beltOffset)
//   - Pillar: vesica of two halos centered at the (now altitude-derived)
//     daggers
//   - Parhelic arc: unique circle through dagger-sun-dagger

function daggerPointsFromSunAltitude(altitudeDeg, parhelicYOffsetR22 = 0) {
  const hRad = (clamp(altitudeDeg, 0, 80) * Math.PI) / 180;
  const offset = HALO_22_RADIUS / Math.cos(hRad);
  const y = SUN.y + HALO_22_RADIUS * parhelicYOffsetR22;
  return {
    left: { x: SUN.x - offset, y },
    right: { x: SUN.x + offset, y },
  };
}

function applyParhelicCircleGeneralized(svg, daggerPoints, parhelicCurvature) {
  // Draws the LOWER arc of a circle whose center sits above the daggers, so
  // the apex dips BELOW the sun (smile shape — what the Troels Nielsen photo
  // shows). Half-chord between daggers depends on sun altitude.
  const parhelic = svg.querySelector("#parhelic-path");
  if (!parhelic) return;

  const c = clamp(parhelicCurvature, 0, 1);
  const d = 200 * c;
  const halfChord = daggerPoints.right.x - SUN.x;
  const beltY = daggerPoints.left.y;

  if (d < 0.5) {
    parhelic.setAttribute(
      "d",
      "M " + daggerPoints.left.x.toFixed(2) + " " + beltY.toFixed(2) + " L " +
        daggerPoints.right.x.toFixed(2) + " " + beltY.toFixed(2)
    );
    return;
  }

  const u_ = (halfChord * halfChord - d * d) / (2 * d);
  const cy = beltY - u_;
  const r = Math.hypot(halfChord, u_);

  const xMin = Math.max(SUN.x - r, 0);
  const xMax = Math.min(SUN.x + r, 1000);
  const dx = xMax - xMin;
  if (dx <= 1e-6) return;
  const steps = 140;
  const parts = [];
  for (let i = 0; i <= steps; i += 1) {
    const x = xMin + (dx * i) / steps;
    const uu = (x - SUN.x) / r;
    const inside = 1 - uu * uu;
    const y = cy + r * Math.sqrt(Math.max(0, inside));
    parts.push((i === 0 ? "M " : "L ") + x.toFixed(2) + " " + y.toFixed(2));
  }
  parhelic.setAttribute("d", parts.join(" "));
}

function upperArcPath(cx, cy, r, xClipMin, xClipMax, yClipMax) {
  const xMin = Math.max(cx - r, xClipMin);
  const xMax = Math.min(cx + r, xClipMax);
  const dx = xMax - xMin;
  if (dx <= 1e-6) return "";
  const steps = 160;
  const parts = [];
  let started = false;
  for (let i = 0; i <= steps; i += 1) {
    const x = xMin + (dx * i) / steps;
    const u = (x - cx) / r;
    const inside = 1 - u * u;
    if (inside < 0) continue;
    const y = cy - r * Math.sqrt(inside);
    if (y > yClipMax) continue;
    parts.push((started ? "L " : "M ") + x.toFixed(2) + " " + y.toFixed(2));
    started = true;
  }
  return parts.join(" ");
}

function circleBranchPath(cx, cy, r, xClipMin, xClipMax, yClipMin, yClipMax, branch = "upper", steps = 160) {
  const xMin = Math.max(cx - r, xClipMin);
  const xMax = Math.min(cx + r, xClipMax);
  const dx = xMax - xMin;
  if (dx <= 1e-6) return "";
  const parts = [];
  let started = false;
  for (let i = 0; i <= steps; i += 1) {
    const x = xMin + (dx * i) / steps;
    const u = (x - cx) / r;
    const inside = 1 - u * u;
    if (inside < 0) continue;
    const root = r * Math.sqrt(inside);
    const y = branch === "lower" ? cy + root : cy - root;
    if (y < yClipMin || y > yClipMax) continue;
    parts.push((started ? "L " : "M ") + x.toFixed(2) + " " + y.toFixed(2));
    started = true;
  }
  return parts.join(" ");
}

function polarArcPath(cx, cy, r, startDeg, endDeg, yClipMin, yClipMax, steps = 80) {
  const parts = [];
  let started = false;
  for (let i = 0; i <= steps; i += 1) {
    const t = i / steps;
    const theta = ((startDeg + (endDeg - startDeg) * t) * Math.PI) / 180;
    const x = cx + r * Math.cos(theta);
    const y = cy + r * Math.sin(theta);
    if (x < 0 || x > 1000 || y < yClipMin || y > yClipMax) continue;
    parts.push((started ? "L " : "M ") + x.toFixed(2) + " " + y.toFixed(2));
    started = true;
  }
  return parts.join(" ");
}

// Atmospheric optics constraint: the CZA is TANGENT TO THE 46° HALO at its
// top. So the CZA apex y is anchored at (SUN.y - R_46), and --cza-curvature
// offsets from that anchor (default 0.85 → exactly tangent; lower values
// drop the apex toward the endpoint, higher values push it above-canvas).
// Calibration vs Troels Nielsen DR: anchored apex y=60 → photo y=66, vs
// observed (402, 65) — 1 px residual (was 15 px with the unanchored formula).
function czaApexFromCurvature(czaCurve) {
  const HALO_46_RADIUS = 440; // current workbench R_46
  const anchoredApexY = SUN.y - HALO_46_RADIUS; // = 60
  const offset = (0.85 - clamp(czaCurve, 0.4, 1.4)) * 200;
  return anchoredApexY + offset;
}

function czaCircleFromApex(apexY, endpointY, halfWidth) {
  const num = halfWidth * halfWidth + endpointY * endpointY - apexY * apexY;
  const den = 2 * (endpointY - apexY);
  if (Math.abs(den) < 1e-6) return null;
  const cy = num / den;
  const r = Math.abs(apexY - cy);
  return { cx: SUN.x, cy: cy, r: r };
}

function applyCzaFullRing(svg, czaCurve) {
  const primary = svg.querySelector("#cza-primary");
  const secondary = svg.querySelector(".cza-secondary");
  const bellFill = svg.querySelector(".cza-bell-fill");
  if (!primary) return;

  const primaryApexY = czaApexFromCurvature(czaCurve);
  const primaryCircle = czaCircleFromApex(primaryApexY, 240, 300);
  if (primaryCircle) {
    const d = upperArcPath(primaryCircle.cx, primaryCircle.cy, primaryCircle.r, 0, 1000, 240);
    if (d) primary.setAttribute("d", d);
  }

  const secondaryApexY = primaryApexY + 120;
  const secondaryCircle = czaCircleFromApex(secondaryApexY, 300, 240);
  if (secondary && secondaryCircle) {
    const d = upperArcPath(secondaryCircle.cx, secondaryCircle.cy, secondaryCircle.r, 0, 1000, 300);
    if (d) secondary.setAttribute("d", d);
  }

  if (bellFill && primaryCircle && secondaryCircle) {
    const primForward = upperArcPath(primaryCircle.cx, primaryCircle.cy, primaryCircle.r, 200, 800, 240);
    const xMinS = Math.max(secondaryCircle.cx - secondaryCircle.r, 260);
    const xMaxS = Math.min(secondaryCircle.cx + secondaryCircle.r, 740);
    const stepsS = 120;
    const dxS = xMaxS - xMinS;
    let secReversed = "";
    if (dxS > 1e-6) {
      const pts = [];
      for (let i = 0; i <= stepsS; i += 1) {
        const x = xMaxS - (dxS * i) / stepsS;
        const u = (x - secondaryCircle.cx) / secondaryCircle.r;
        const inside = 1 - u * u;
        if (inside < 0) continue;
        const y = secondaryCircle.cy - secondaryCircle.r * Math.sqrt(inside);
        pts.push("L " + x.toFixed(2) + " " + y.toFixed(2));
      }
      secReversed = pts.join(" ");
    }
    if (primForward && secReversed) {
      bellFill.setAttribute("d", primForward + " " + secReversed + " Z");
    }
  }
}

function applySupralateralArc(svg, intensity) {
  const supra = svg.querySelector("#supralateral-path");
  if (!supra) return;
  if (intensity <= 0.001) {
    supra.setAttribute("d", "");
    return;
  }
  const tangentY = SUN.y - 440;
  const R_supra = 400;
  const cx = SUN.x;
  const cy = tangentY - R_supra;
  const xMin = Math.max(cx - R_supra, 0);
  const xMax = Math.min(cx + R_supra, 1000);
  const dx = xMax - xMin;
  if (dx <= 1e-6) return;
  const steps = 160;
  const parts = [];
  let started = false;
  for (let i = 0; i <= steps; i += 1) {
    const x = xMin + (dx * i) / steps;
    const u = (x - cx) / R_supra;
    const inside = 1 - u * u;
    if (inside < 0) continue;
    const y = cy + R_supra * Math.sqrt(inside);
    if (y < 0 || y > tangentY + 5) continue;
    parts.push((started ? "L " : "M ") + x.toFixed(2) + " " + y.toFixed(2));
    started = true;
  }
  supra.setAttribute("d", parts.join(" "));
}

function applyLowerTangentArc(svg, intensity) {
  // Mirror of applyUpperTangentArc. Tangent to the 22° halo at its BOTTOM
  // point (SUN.x, SUN.y + 220) — curves DOWN from there on either side.
  // Geometric mirror: circle whose center sits R_lta directly BELOW the
  // tangent point; visible portion is the UPPER arc of that circle, which
  // tops out at the tangent and falls on either side. Atmospheric-optics
  // counterpart to the upper tangent arc that the vocabulary references in
  // docs/calibration/1 explicitly name.
  const arc = svg.querySelector("#lower-tangent-path");
  if (!arc) return;
  if (intensity <= 0.001) {
    arc.setAttribute("d", "");
    return;
  }
  const tangentY = SUN.y + HALO_22_RADIUS; // 720
  const R_lta = 200;
  const cx = SUN.x;
  const cy = tangentY + R_lta;
  const xMin = Math.max(cx - R_lta, 0);
  const xMax = Math.min(cx + R_lta, 1000);
  const dx = xMax - xMin;
  if (dx <= 1e-6) return;
  const steps = 160;
  const parts = [];
  let started = false;
  for (let i = 0; i <= steps; i += 1) {
    const x = xMin + (dx * i) / steps;
    const u = (x - cx) / R_lta;
    const inside = 1 - u * u;
    if (inside < 0) continue;
    // Upper arc of circle: y = cy - r*sqrt(1-u²) — peaks at the tangent
    // point (y = tangentY) when u=0, drops on either side as |u| grows.
    const y = cy - R_lta * Math.sqrt(inside);
    if (y < tangentY - 5 || y > 800) continue;
    parts.push((started ? "L " : "M ") + x.toFixed(2) + " " + y.toFixed(2));
    started = true;
  }
  arc.setAttribute("d", parts.join(" "));
}

function applyUpperTangentArc(svg, intensity) {
  // Tangent to the 22° halo at its TOP point (SUN.x, SUN.y - 220). Curves UP
  // from there — the "eyelid above the 22° halo" feature distinct from the
  // CZA. Geometrically: circle whose center sits R_uta directly ABOVE the
  // tangent point; visible portion is the LOWER arc of that circle, which
  // bottoms out at the tangent and rises on either side.
  const arc = svg.querySelector("#upper-tangent-path");
  if (!arc) return;
  if (intensity <= 0.001) {
    arc.setAttribute("d", "");
    return;
  }
  const tangentY = SUN.y - HALO_22_RADIUS; // 280
  const R_uta = 200;
  const cx = SUN.x;
  const cy = tangentY - R_uta;
  const xMin = Math.max(cx - R_uta, 0);
  const xMax = Math.min(cx + R_uta, 1000);
  const dx = xMax - xMin;
  if (dx <= 1e-6) return;
  const steps = 160;
  const parts = [];
  let started = false;
  for (let i = 0; i <= steps; i += 1) {
    const x = xMin + (dx * i) / steps;
    const u = (x - cx) / R_uta;
    const inside = 1 - u * u;
    if (inside < 0) continue;
    const y = cy + R_uta * Math.sqrt(inside);
    if (y < 0 || y > tangentY + 5) continue;
    parts.push((started ? "L " : "M ") + x.toFixed(2) + " " + y.toFixed(2));
    started = true;
  }
  arc.setAttribute("d", parts.join(" "));
}

function applySuncaveParryArc(svg, intensity) {
  // Parry-orientation companion above the upper tangent arc. "Suncave" means
  // the arc's ends bend back toward the sun; in screen coordinates that is a
  // shallow cap whose center sits above the 22° halo top and whose shoulders
  // fall toward the sun. This is an atlas primitive, not a HaloSim raytrace.
  const arc = svg.querySelector("#suncave-parry-path");
  if (!arc) return;
  if (intensity <= 0.001) {
    arc.setAttribute("d", "");
    return;
  }
  const apexY = SUN.y - HALO_22_RADIUS - 36;
  const endpointY = SUN.y - HALO_22_RADIUS + 24;
  const halfWidth = 210;
  const circle = czaCircleFromApex(apexY, endpointY, halfWidth);
  if (!circle) return;
  const d = circleBranchPath(
    circle.cx,
    circle.cy,
    circle.r,
    SUN.x - halfWidth,
    SUN.x + halfWidth,
    0,
    endpointY + 8,
    "upper",
    120
  );
  arc.setAttribute("d", d);
}

function applyParrySupralateralArcs(svg, intensity) {
  // Rare Parry-oriented shoulders riding the upper-lateral supralateral
  // family. Drawn as two short peripheral segments just outside the 46° halo.
  const arc = svg.querySelector("#parry-supralateral-path");
  if (!arc) return;
  if (intensity <= 0.001) {
    arc.setAttribute("d", "");
    return;
  }
  const r = 500;
  const left = polarArcPath(SUN.x, SUN.y, r, 206, 248, 0, SUN.y - 10);
  const right = polarArcPath(SUN.x, SUN.y, r, 292, 334, 0, SUN.y - 10);
  arc.setAttribute("d", [left, right].filter(Boolean).join(" "));
}

function applyInfralateralArcs(svg, intensity) {
  // Lower lateral tangent arcs: paired peripheral arcs outside the 46° halo
  // and below the parhelic circle. At low/mid sun elevations they read as
  // left/right side arcs rather than a full circle.
  const arc = svg.querySelector("#infralateral-path");
  if (!arc) return;
  if (intensity <= 0.001) {
    arc.setAttribute("d", "");
    return;
  }
  const r = 500;
  const left = polarArcPath(SUN.x, SUN.y, r, 142, 160, SUN.y + 5, 800);
  const right = polarArcPath(SUN.x, SUN.y, r, 20, 38, SUN.y + 5, 800);
  arc.setAttribute("d", [left, right].filter(Boolean).join(" "));
}

function applyGeometryHaloAtlas(svg, rootStyle) {
  const sunAltitude = readCssNumber(rootStyle, "--sun-altitude", 18);
  const pillarLen = readCssNumber(rootStyle, "--sun-pillar-length", 0.65);
  const daggerLen = readCssNumber(rootStyle, "--parhelia-dagger-length", 1);
  const compassLen = readCssNumber(rootStyle, "--compass-ray-length", 1);
  const czaCurve = readCssNumber(rootStyle, "--cza-curvature", 0.85);
  const overlapBias = readCssNumber(rootStyle, "--ring-overlap-bias", 0.5);
  const parhelicCurvature = readCssNumber(rootStyle, "--parhelic-curvature", 0.66);
  const parhelicYOffsetR22 = readCssNumber(rootStyle, "--parhelic-y-offset-r22", -0.05);
  const supralateralIntensity = readCssNumber(rootStyle, "--supralateral-intensity", 0);
  const upperTangentIntensity = readCssNumber(rootStyle, "--upper-tangent-intensity", 0);
  const lowerTangentIntensity = readCssNumber(rootStyle, "--lower-tangent-intensity", 0);
  const suncaveParryIntensity = readCssNumber(rootStyle, "--suncave-parry-intensity", 0);
  const parrySupralateralIntensity = readCssNumber(rootStyle, "--parry-supralateral-intensity", 0);
  const infralateralIntensity = readCssNumber(rootStyle, "--infralateral-intensity", 0);

  const daggerPoints = daggerPointsFromSunAltitude(sunAltitude, parhelicYOffsetR22);

  applyParhelicCircleGeneralized(svg, daggerPoints, parhelicCurvature);
  applyDaggersFromGoverningHalo(svg, daggerLen, daggerPoints);
  applyPillarFromTwoHalos(svg, daggerPoints, pillarLen);
  applyCzaFullRing(svg, czaCurve);
  applySupralateralArc(svg, supralateralIntensity);
  applyUpperTangentArc(svg, upperTangentIntensity);
  applyLowerTangentArc(svg, lowerTangentIntensity);
  applySuncaveParryArc(svg, suncaveParryIntensity);
  applyParrySupralateralArcs(svg, parrySupralateralIntensity);
  applyInfralateralArcs(svg, infralateralIntensity);

  applyCompassRays(svg, compassLen);
  applySecondaryHalos(svg, overlapBias);
}

function applyGeometryHaloGoverned(svg, rootStyle) {
  const pillarLen = readCssNumber(rootStyle, "--sun-pillar-length", 0.65);
  const daggerLen = readCssNumber(rootStyle, "--parhelia-dagger-length", 1);
  const compassLen = readCssNumber(rootStyle, "--compass-ray-length", 1);
  const czaCurve = readCssNumber(rootStyle, "--cza-curvature", 0.85);
  const overlapBias = readCssNumber(rootStyle, "--ring-overlap-bias", 0.5);
  const parhelicCurvature = readCssNumber(rootStyle, "--parhelic-curvature", 0.66);

  const daggerPoints = daggerPointsHorizontal();

  applyParhelicCircleThroughDaggersAndSun(svg, parhelicCurvature);
  applyDaggersFromGoverningHalo(svg, daggerLen, daggerPoints);
  applyPillarFromTwoHalos(svg, daggerPoints, pillarLen);

  applyCompassRays(svg, compassLen);
  applyCza(svg, czaCurve);
  applySecondaryHalos(svg, overlapBias);
}

export function applyParhelionGeometry({ svg, rootStyle, model }) {
  if (!svg || !rootStyle) return;
  svg.dataset.geometryModel = model;

  if (model === "halo_atlas") {
    applyGeometryHaloAtlas(svg, rootStyle);
  } else if (model === "halo_governed") {
    applyGeometryHaloGoverned(svg, rootStyle);
  } else if (model === "halo_scaffold") {
    applyGeometryHaloScaffold(svg, rootStyle);
  } else {
    applyGeometryLegacy(svg, rootStyle);
  }
}
