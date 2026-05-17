import { phase3 } from "./parhelion-geometry.mjs";

const atlas = document.querySelector("[data-hero-atlas]");

if (atlas) {
  initHomeHero(atlas);
}

function initHomeHero(root) {
  const svg = root.querySelector("svg");
  const readout = document.querySelector("[data-hero-altitude-readout]");
  const parts = {
    parhelic: svg?.querySelector("#hero-parhelic-path"),
    czaPrimary: svg?.querySelector("#hero-cza-primary"),
    czaSecondary: svg?.querySelector("#hero-cza-secondary"),
    upperTangent: svg?.querySelector("#hero-upper-tangent-path"),
    lowerTangent: svg?.querySelector("#hero-lower-tangent-path"),
    left: svg?.querySelector("#hero-left-parhelion"),
    right: svg?.querySelector("#hero-right-parhelion"),
    pillar: svg?.querySelector("#hero-sun-pillar"),
  };

  if (!svg || Object.values(parts).some((node) => !node)) {
    return;
  }

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const start = performance.now();

  const render = (now) => {
    const altitudeDeg = reducedMotion ? 25 : altitudeAt(now - start);
    updatePose({ svg, readout, parts, altitudeDeg });

    if (!reducedMotion) {
      window.requestAnimationFrame(render);
    }
  };

  render(start);
  svg.dataset.heroMotionReady = "true";
}

function altitudeAt(elapsedMs) {
  const min = 18.6;
  const max = 33.4;
  const period = 26000;
  const t = (elapsedMs % period) / period;
  const eased = 0.5 - 0.5 * Math.cos(t * Math.PI * 2);
  return min + (max - min) * eased;
}

function updatePose({ svg, readout, parts, altitudeDeg }) {
  const sun = { x: 500, y: 500 };
  const r22 = phase3.HALO_22_RADIUS;
  const beltY = sun.y - r22 * 0.05;
  const offset = phase3.daggerOffset(altitudeDeg);
  const leftX = sun.x - offset;
  const rightX = sun.x + offset;
  const curvature = phase3.parhelicCurvature(altitudeDeg);

  parts.left.setAttribute("transform", `translate(${leftX.toFixed(2)}, ${beltY.toFixed(2)})`);
  parts.right.setAttribute("transform", `translate(${rightX.toFixed(2)}, ${beltY.toFixed(2)})`);
  parts.parhelic.setAttribute("d", parhelicPath({ sun, offset, beltY, curvature }));

  const czaVisible = phase3.czaVisible(altitudeDeg);
  setCza({ parts, visible: czaVisible });

  const locus = phase3.tangentArcLocus(altitudeDeg, 0.1, 42);
  setTangent({ parts, locus, altitudeDeg });

  const pillarOpacity = 0.1 + 0.08 * (1 - Math.min(1, Math.abs(altitudeDeg - 25) / 12));
  parts.pillar.style.opacity = pillarOpacity.toFixed(2);

  svg.dataset.heroAltitudeDeg = altitudeDeg.toFixed(2);
  svg.dataset.heroCzaState = czaVisible ? "visible" : "cutoff";
  svg.dataset.heroTangentState = locus ? "visible" : "merged";

  if (readout) {
    const czaText = czaVisible ? "CZA in-window" : "CZA cutoff";
    const tangentText = locus ? "tangent in-window" : "tangent merged";
    readout.textContent = `h = ${altitudeDeg.toFixed(1)} deg | parhelion offset = ${(offset / r22).toFixed(2)} R22 | ${czaText} | ${tangentText}`;
  }
}

function setCza({ parts, visible }) {
  if (!visible) {
    parts.czaPrimary.setAttribute("d", "");
    parts.czaSecondary.setAttribute("d", "");
    parts.czaPrimary.style.opacity = "0";
    parts.czaSecondary.style.opacity = "0";
    return;
  }

  parts.czaPrimary.setAttribute("d", czaArcPath(60, 240, 300));
  parts.czaSecondary.setAttribute("d", czaArcPath(180, 300, 240));
  parts.czaPrimary.style.opacity = "0.92";
  parts.czaSecondary.style.opacity = "0.55";
}

function czaArcPath(apexY, endpointY, halfWidth) {
  const sunX = 500;
  const numerator = halfWidth * halfWidth + endpointY * endpointY - apexY * apexY;
  const denominator = 2 * (endpointY - apexY);
  if (Math.abs(denominator) < 1e-6) return "";

  const cy = numerator / denominator;
  const r = Math.abs(apexY - cy);
  const xMin = sunX - halfWidth;
  const xMax = sunX + halfWidth;
  const steps = 120;
  const points = [];

  for (let i = 0; i <= steps; i += 1) {
    const x = xMin + ((xMax - xMin) * i) / steps;
    const u = (x - sunX) / r;
    const inside = 1 - u * u;
    if (inside < 0) continue;
    const y = cy - r * Math.sqrt(inside);
    points.push(`${points.length ? "L" : "M"} ${x.toFixed(2)} ${y.toFixed(2)}`);
  }

  return points.join(" ");
}

function setTangent({ parts, locus, altitudeDeg }) {
  if (!locus) {
    parts.upperTangent.setAttribute("d", "");
    parts.lowerTangent.setAttribute("d", "");
    parts.upperTangent.style.opacity = "0";
    parts.lowerTangent.style.opacity = "0";
    return;
  }

  parts.upperTangent.setAttribute("d", tangentPath(locus, "upper"));
  parts.lowerTangent.setAttribute("d", tangentPath(locus, "lower"));

  const fadeToMerge = Math.max(0, Math.min(1, (29 - altitudeDeg) / 3));
  parts.upperTangent.style.opacity = (0.18 + 0.58 * fadeToMerge).toFixed(2);
  parts.lowerTangent.style.opacity = (0.08 + 0.22 * fadeToMerge).toFixed(2);
}

function tangentPath(locus, branch) {
  const pxPerDeg = phase3.HALO_22_RADIUS / 22;
  const sign = branch === "lower" ? 1 : -1;
  const points = [];

  for (const { psiDeg, radialDeg } of locus) {
    const psi = (psiDeg * Math.PI) / 180;
    const rPx = radialDeg * pxPerDeg;
    const x = 500 + rPx * Math.sin(psi);
    const y = 500 + sign * rPx * Math.cos(psi);
    if (x < 0 || x > 1000 || y < 0 || y > 800) {
      continue;
    }
    points.push(`${points.length ? "L" : "M"} ${x.toFixed(2)} ${y.toFixed(2)}`);
  }

  return points.join(" ");
}

function parhelicPath({ sun, offset, beltY, curvature }) {
  const d = 200 * Math.max(0, Math.min(1, curvature));
  if (d < 0.5) {
    return `M ${(sun.x - offset).toFixed(2)} ${beltY.toFixed(2)} L ${(sun.x + offset).toFixed(2)} ${beltY.toFixed(2)}`;
  }

  const u = (offset * offset - d * d) / (2 * d);
  const cy = beltY - u;
  const r = Math.hypot(offset, u);
  const xMin = Math.max(sun.x - r, 0);
  const xMax = Math.min(sun.x + r, 1000);
  const steps = 120;
  const points = [];

  for (let i = 0; i <= steps; i += 1) {
    const x = xMin + ((xMax - xMin) * i) / steps;
    const dx = (x - sun.x) / r;
    const inside = 1 - dx * dx;
    if (inside < 0) continue;
    const y = cy + r * Math.sqrt(inside);
    points.push(`${points.length ? "L" : "M"} ${x.toFixed(2)} ${y.toFixed(2)}`);
  }

  return points.join(" ");
}
