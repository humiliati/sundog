import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { deflateSync } from "node:zlib";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const ICON_DIR = resolve(ROOT, "public", "icons");
const PUBLIC_DIR = resolve(ROOT, "public");
const promoteProduction = process.argv.includes("--promote-production");

const COLORS = {
  skyInner: "#2c5f7f",
  skyMid: "#1a3a52",
  skyOuter: "#102436",
  ice: "#7da2b8",
  gold: "#f4c430",
  goldDeep: "#b97812",
  sunWhite: "#fff4b2",
  redInside: "#ff7a45",
  blueOutside: "#79bde8",
  ink: "#102436",
};

const GEOMETRY = {
  viewBox: 1024,
  sun: { x: 512, y: 512, radius: 112, coreRadius: 42 },
  halo22Radius: 258,
  halo46Radius: 346,
  sunAltitudeDeg: 18,
  // Decorative flat belt through the sun. The earlier -0.05 R22 belt-tilt
  // rule was falsified (Spearman rho ~= 0.086) and is no longer asserted —
  // see docs/PHASE10_ATTACK_ROADMAP.md. The valid promoted handle is the
  // parhelion *horizontal* offset = R22 / cos(h), computed below.
  parhelicYOffsetR22: 0,
};

const beltY =
  GEOMETRY.sun.y + GEOMETRY.halo22Radius * GEOMETRY.parhelicYOffsetR22;
const parhelionOffset =
  GEOMETRY.halo22Radius / Math.cos((GEOMETRY.sunAltitudeDeg * Math.PI) / 180);
const leftParhelionX = GEOMETRY.sun.x - parhelionOffset;
const rightParhelionX = GEOMETRY.sun.x + parhelionOffset;

const PATHS = {
  parhelicBelt: `M 166 ${round1(beltY)} C 340 ${round1(
    beltY - 18,
  )} 684 ${round1(beltY - 18)} 858 ${round1(beltY)}`,
  upperTangent: "M 276 392 C 356 292 668 292 748 392",
  cza: "M 328 276 C 420 232 604 232 696 276",
};

const GENERATED_FILES = [
  "public/icons/sundog-character-mark.svg",
  "public/icons/sundog-character-mark.transparent.svg",
  "public/icons/sundog-character-mark.animated.svg",
  "public/icons/sundog-character-mark.layers.json",
  "public/icons/sundog-character-favicon-16.png",
  "public/icons/sundog-character-favicon-32.png",
  "public/icons/sundog-character-favicon-48.png",
  "public/icons/sundog-character-icon-180.png",
  "public/icons/sundog-character-icon-192.png",
  "public/icons/sundog-character-icon-512.png",
  "public/icons/sundog-character-transparent-512.png",
];

const PRODUCTION_FILES = [
  "public/favicon.svg",
  "public/favicon.ico",
  "public/apple-touch-icon.png",
  "public/icons/icon-16.png",
  "public/icons/icon-32.png",
  "public/icons/icon-48.png",
  "public/icons/icon-180.png",
  "public/icons/icon-192.png",
  "public/icons/icon-512.png",
  "public/icons/maskable-192.png",
  "public/icons/maskable-512.png",
];

function round1(value) {
  return Math.round(value * 10) / 10;
}

function hexToRgb(hex) {
  const normalized = hex.replace("#", "");
  return [
    parseInt(normalized.slice(0, 2), 16),
    parseInt(normalized.slice(2, 4), 16),
    parseInt(normalized.slice(4, 6), 16),
  ];
}

function mixColor(left, right, t) {
  const clamped = Math.max(0, Math.min(1, t));
  return left.map((channel, index) =>
    Math.round(channel + (right[index] - channel) * clamped),
  );
}

function svgDefinitions(animated) {
  const style = animated
    ? `
    <style>
      .sun-body {
        animation: sun-breathe 4.8s ease-in-out infinite;
        transform-box: fill-box;
        transform-origin: center;
      }
      .parhelic-belt {
        stroke-dasharray: 44 22;
        animation: belt-shimmer 5.4s linear infinite;
      }
      .parhelion-left {
        animation: glint-left 3.8s ease-in-out infinite;
        transform-box: fill-box;
        transform-origin: center;
      }
      .parhelion-right {
        animation: glint-right 3.8s ease-in-out infinite;
        transform-box: fill-box;
        transform-origin: center;
      }
      .upper-eyelid {
        animation: eyelid-wink 8s ease-in-out infinite;
        transform-box: fill-box;
        transform-origin: center;
      }
      @keyframes sun-breathe {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.88; transform: scale(1.035); }
      }
      @keyframes belt-shimmer {
        from { stroke-dashoffset: 0; opacity: 0.54; }
        50% { opacity: 0.82; }
        to { stroke-dashoffset: -264; opacity: 0.54; }
      }
      @keyframes glint-left {
        0%, 100% { opacity: 0.74; transform: translateX(0); }
        48% { opacity: 1; transform: translateX(-8px); }
      }
      @keyframes glint-right {
        0%, 100% { opacity: 0.74; transform: translateX(0); }
        48% { opacity: 1; transform: translateX(8px); }
      }
      @keyframes eyelid-wink {
        0%, 76%, 100% { opacity: 0.92; transform: translateY(0) scaleY(1); }
        82% { opacity: 0.62; transform: translateY(14px) scaleY(0.68); }
        88% { opacity: 0.92; transform: translateY(0) scaleY(1); }
      }
      @media (prefers-reduced-motion: reduce) {
        * { animation: none !important; }
        .parhelic-belt { stroke-dasharray: none; }
      }
    </style>`
    : "";

  return `
  <defs>${style}
    <radialGradient id="sky" cx="50%" cy="42%" r="72%">
      <stop offset="0%" stop-color="${COLORS.skyInner}"/>
      <stop offset="58%" stop-color="${COLORS.skyMid}"/>
      <stop offset="100%" stop-color="${COLORS.skyOuter}"/>
    </radialGradient>
    <radialGradient id="sun" cx="42%" cy="34%" r="68%">
      <stop offset="0%" stop-color="${COLORS.sunWhite}"/>
      <stop offset="42%" stop-color="${COLORS.gold}"/>
      <stop offset="100%" stop-color="${COLORS.goldDeep}"/>
    </radialGradient>
    <linearGradient id="parhelion-left-gradient" x1="0%" x2="100%">
      <stop offset="0%" stop-color="${COLORS.blueOutside}"/>
      <stop offset="52%" stop-color="${COLORS.gold}"/>
      <stop offset="100%" stop-color="${COLORS.redInside}"/>
    </linearGradient>
    <linearGradient id="parhelion-right-gradient" x1="0%" x2="100%">
      <stop offset="0%" stop-color="${COLORS.redInside}"/>
      <stop offset="48%" stop-color="${COLORS.gold}"/>
      <stop offset="100%" stop-color="${COLORS.blueOutside}"/>
    </linearGradient>
    <filter id="soft-glow" x="-45%" y="-45%" width="190%" height="190%">
      <feGaussianBlur stdDeviation="18" result="blur"/>
      <feColorMatrix in="blur" type="matrix" values="1 0 0 0 0.96  0 1 0 0 0.77  0 0 1 0 0.19  0 0 0 0.68 0"/>
      <feMerge>
        <feMergeNode/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>`;
}

function glintPath(cx, cy) {
  const left = round1(cx - 90);
  const right = round1(cx + 90);
  const top = round1(cy - 43);
  const bottom = round1(cy + 43);
  return `M ${left} ${round1(cy)} L ${round1(cx)} ${top} L ${right} ${round1(
    cy,
  )} L ${round1(cx)} ${bottom} Z`;
}

function svgMarkup({ animated = false, transparent = false } = {}) {
  const title = animated
    ? "Animated Sundog character mark"
    : transparent
      ? "Transparent Sundog character mark"
      : "Sundog character mark";
  const desc = animated
    ? "A halo-based Sundog mark with idle shimmer, parhelion glints, and a reduced-motion fallback."
    : "A characterized Sundog halo mark generated from the Phase 10 Halo Atlas shape language.";
  const background = transparent
    ? ""
    : `  <rect data-layer="field.background" width="1024" height="1024" rx="220" fill="url(#sky)"/>`;

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" role="img" aria-labelledby="title desc">
  <title id="title">${title}</title>
  <desc id="desc">${desc}</desc>
${svgDefinitions(animated)}
${background}
  <g data-layer="field.halo-46">
    <circle cx="512" cy="512" r="346" fill="none" stroke="${COLORS.ice}" stroke-width="26" opacity="0.46"/>
  </g>
  <g data-layer="core.halo-22">
    <circle cx="512" cy="512" r="258" fill="none" stroke="${COLORS.gold}" stroke-width="18" opacity="0.66"/>
  </g>
  <g data-layer="core.parhelic-belt">
    <path class="parhelic-belt" d="${PATHS.parhelicBelt}" fill="none" stroke="${COLORS.gold}" stroke-width="25" stroke-linecap="round" opacity="0.7"/>
  </g>
  <g data-layer="core.upper-tangent">
    <path class="upper-eyelid" d="${PATHS.upperTangent}" fill="none" stroke="${COLORS.sunWhite}" stroke-width="28" stroke-linecap="round" opacity="0.82"/>
  </g>
  <g data-layer="optional.circumzenithal-arc">
    <path d="${PATHS.cza}" fill="none" stroke="${COLORS.ice}" stroke-width="18" stroke-linecap="round" opacity="0.44"/>
  </g>
  <g data-layer="core.sun" class="sun-body">
    <circle cx="512" cy="512" r="132" fill="${COLORS.gold}" opacity="0.15"/>
    <circle cx="512" cy="512" r="112" fill="url(#sun)" filter="url(#soft-glow)"/>
    <circle cx="512" cy="512" r="42" fill="${COLORS.sunWhite}" opacity="0.86"/>
  </g>
  <g data-layer="core.parhelia">
    <path class="parhelion-left" d="${glintPath(
      leftParhelionX,
      beltY,
    )}" fill="url(#parhelion-left-gradient)" opacity="0.94"/>
    <path class="parhelion-right" d="${glintPath(
      rightParhelionX,
      beltY,
    )}" fill="url(#parhelion-right-gradient)" opacity="0.94"/>
  </g>
</svg>
`;
}

function layerManifest() {
  return {
    generatedBy: "scripts/generate-sundog-logo-toolkit.mjs",
    roadmapPhase: "SUNDOG_V_GEOMETRY.md Phase 11",
    calibrationBasis: {
      phase10Images: ["p2", "p7", "p13"],
      promotedHandle:
        "parhelion horizontal offset = R22 / cos(h); see /h-of-x",
      parhelicBeltVerticalOffset:
        "decorative (flat through sun); the earlier -0.05 R22 belt-tilt rule was falsified and is not asserted",
      notes: "docs/PHASE10_ATTACK_ROADMAP.md, docs/MESA_CROSSOVER_NOTE.md",
    },
    geometry: {
      viewBox: GEOMETRY.viewBox,
      sun: GEOMETRY.sun,
      halo22Radius: GEOMETRY.halo22Radius,
      halo46Radius: GEOMETRY.halo46Radius,
      sunAltitudeDeg: GEOMETRY.sunAltitudeDeg,
      parhelicBeltY: round1(beltY),
      parhelionOffset: round1(parhelionOffset),
      leftParhelionX: round1(leftParhelionX),
      rightParhelionX: round1(rightParhelionX),
      paths: PATHS,
    },
    protectedLayers: [
      {
        id: "core.sun",
        rule: "Keep the sun centered inside the 22 deg halo. Scale as a group only.",
      },
      {
        id: "core.halo-22",
        rule: "This is the iris. Do not decenter it from the sun.",
      },
      {
        id: "core.parhelia",
        rule: "Glints sit on the parhelic belt at the sun-altitude-derived parhelion offset.",
      },
      {
        id: "core.parhelic-belt",
        rule: "Flat decorative band through the sun. Do NOT reintroduce a calibrated vertical offset; the earlier -0.05 R22 belt-tilt rule was falsified (see docs/PHASE10_ATTACK_ROADMAP.md).",
      },
      {
        id: "core.upper-tangent",
        rule: "Treat as the eyelid gesture. It can blink/reveal but should not detach from the halo system.",
      },
    ],
    optionalLayers: [
      {
        id: "optional.circumzenithal-arc",
        use: "Large or animated marks only.",
      },
      {
        id: "optional.suncave-parry",
        use: "Annotation and education only until stronger multi-photo support.",
      },
      {
        id: "optional.parry-supralateral",
        use: "Annotation only; not a default logo feature.",
      },
      {
        id: "optional.infralateral",
        use: "Rich-display variant only; omit from small marks.",
      },
    ],
    colorTokens: COLORS,
    motionStates: {
      idle: "Slow sun breathing plus belt shimmer.",
      activeReveal: "Draw the parhelic belt, then light the parhelia.",
      hover: "Increase parhelion opacity and slide glints outward by 6-10 px.",
      labelCallout: "Pulse one layer at a time using data-layer ids.",
      reducedMotion: "Use the static SVG; no opacity cycling or translation.",
    },
    smallSizeRules: {
      "16-32px": "Sun, 22 deg halo, and parhelia only.",
      "48-180px": "Add 46 deg halo and a simplified upper tangent.",
      "192px+": "Full default mark, including parhelic belt and optional CZA if contrast survives.",
    },
    outputs: GENERATED_FILES,
  };
}

class Raster {
  constructor(size, { transparent = false } = {}) {
    this.size = size;
    this.sample =
      size <= 48
        ? 4
        : size <= 192
          ? 3
          : 2;
    this.width = size * this.sample;
    this.height = size * this.sample;
    this.scale = this.width / GEOMETRY.viewBox;
    this.transparent = transparent;
    this.pixels = new Uint8ClampedArray(this.width * this.height * 4);
  }

  toView(px) {
    return (px + 0.5) / this.scale;
  }

  blend(ix, iy, color, alpha) {
    if (
      ix < 0 ||
      iy < 0 ||
      ix >= this.width ||
      iy >= this.height ||
      alpha <= 0
    ) {
      return;
    }
    const offset = (iy * this.width + ix) * 4;
    const dstA = this.pixels[offset + 3] / 255;
    const srcA = Math.max(0, Math.min(1, alpha));
    const outA = srcA + dstA * (1 - srcA);
    if (outA <= 0) {
      return;
    }
    this.pixels[offset] = Math.round(
      (color[0] * srcA + this.pixels[offset] * dstA * (1 - srcA)) / outA,
    );
    this.pixels[offset + 1] = Math.round(
      (color[1] * srcA + this.pixels[offset + 1] * dstA * (1 - srcA)) / outA,
    );
    this.pixels[offset + 2] = Math.round(
      (color[2] * srcA + this.pixels[offset + 2] * dstA * (1 - srcA)) / outA,
    );
    this.pixels[offset + 3] = Math.round(outA * 255);
  }

  forBox(x0, y0, x1, y1, fn) {
    const minX = Math.max(0, Math.floor(x0 * this.scale) - 2);
    const minY = Math.max(0, Math.floor(y0 * this.scale) - 2);
    const maxX = Math.min(this.width - 1, Math.ceil(x1 * this.scale) + 2);
    const maxY = Math.min(this.height - 1, Math.ceil(y1 * this.scale) + 2);
    for (let py = minY; py <= maxY; py += 1) {
      const y = this.toView(py);
      for (let px = minX; px <= maxX; px += 1) {
        fn(px, py, this.toView(px), y);
      }
    }
  }

  drawBackground() {
    if (this.transparent) {
      return;
    }
    const inner = hexToRgb(COLORS.skyInner);
    const mid = hexToRgb(COLORS.skyMid);
    const outer = hexToRgb(COLORS.skyOuter);
    const radius = GEOMETRY.viewBox * 0.72;
    const cx = GEOMETRY.viewBox * 0.5;
    const cy = GEOMETRY.viewBox * 0.42;
    const cornerRadius = 220;
    for (let py = 0; py < this.height; py += 1) {
      const y = this.toView(py);
      for (let px = 0; px < this.width; px += 1) {
        const x = this.toView(px);
        if (!insideRoundedRect(x, y, cornerRadius)) {
          continue;
        }
        const d = Math.hypot(x - cx, y - cy) / radius;
        const color =
          d < 0.58
            ? mixColor(inner, mid, d / 0.58)
            : mixColor(mid, outer, (d - 0.58) / 0.42);
        this.blend(px, py, color, 1);
      }
    }
  }

  drawStrokedCircle(cx, cy, radius, width, color, alpha) {
    const half = width / 2;
    this.forBox(
      cx - radius - half,
      cy - radius - half,
      cx + radius + half,
      cy + radius + half,
      (px, py, x, y) => {
        if (Math.abs(Math.hypot(x - cx, y - cy) - radius) <= half) {
          this.blend(px, py, color, alpha);
        }
      },
    );
  }

  drawLine(x1, y1, x2, y2, width, color, alpha) {
    const half = width / 2;
    const minX = Math.min(x1, x2) - half;
    const maxX = Math.max(x1, x2) + half;
    const minY = Math.min(y1, y2) - half;
    const maxY = Math.max(y1, y2) + half;
    this.forBox(minX, minY, maxX, maxY, (px, py, x, y) => {
      if (distanceToSegment(x, y, x1, y1, x2, y2) <= half) {
        this.blend(px, py, color, alpha);
      }
    });
  }

  drawQuadratic(p0, p1, p2, width, color, alpha, steps = 160) {
    let previous = p0;
    for (let i = 1; i <= steps; i += 1) {
      const t = i / steps;
      const current = quadraticPoint(p0, p1, p2, t);
      this.drawLine(previous.x, previous.y, current.x, current.y, width, color, alpha);
      previous = current;
    }
  }

  drawArc(cx, cy, radius, startDeg, endDeg, width, color, alpha, steps = 160) {
    let previous = pointOnCircle(cx, cy, radius, startDeg);
    for (let i = 1; i <= steps; i += 1) {
      const angle = startDeg + ((endDeg - startDeg) * i) / steps;
      const current = pointOnCircle(cx, cy, radius, angle);
      this.drawLine(previous.x, previous.y, current.x, current.y, width, color, alpha);
      previous = current;
    }
  }

  drawRadialCircle(cx, cy, radius, stops, alpha = 1) {
    this.forBox(cx - radius, cy - radius, cx + radius, cy + radius, (px, py, x, y) => {
      const t = Math.hypot(x - cx, y - cy) / radius;
      if (t > 1) {
        return;
      }
      const { color, opacity } = sampleStops(stops, t);
      this.blend(px, py, color, opacity * alpha);
    });
  }

  drawPolygon(points, colorAt, alpha = 1) {
    const xs = points.map((point) => point.x);
    const ys = points.map((point) => point.y);
    this.forBox(
      Math.min(...xs),
      Math.min(...ys),
      Math.max(...xs),
      Math.max(...ys),
      (px, py, x, y) => {
        if (pointInPolygon(x, y, points)) {
          this.blend(px, py, colorAt(x, y), alpha);
        }
      },
    );
  }

  downsample() {
    const output = new Uint8ClampedArray(this.size * this.size * 4);
    const samples = this.sample * this.sample;
    for (let y = 0; y < this.size; y += 1) {
      for (let x = 0; x < this.size; x += 1) {
        let alphaSum = 0;
        let redSum = 0;
        let greenSum = 0;
        let blueSum = 0;
        for (let sy = 0; sy < this.sample; sy += 1) {
          for (let sx = 0; sx < this.sample; sx += 1) {
            const source =
              ((y * this.sample + sy) * this.width + (x * this.sample + sx)) *
              4;
            const alpha = this.pixels[source + 3] / 255;
            alphaSum += alpha;
            redSum += this.pixels[source] * alpha;
            greenSum += this.pixels[source + 1] * alpha;
            blueSum += this.pixels[source + 2] * alpha;
          }
        }
        const target = (y * this.size + x) * 4;
        const outAlpha = alphaSum / samples;
        if (alphaSum > 0) {
          output[target] = Math.round(redSum / alphaSum);
          output[target + 1] = Math.round(greenSum / alphaSum);
          output[target + 2] = Math.round(blueSum / alphaSum);
          output[target + 3] = Math.round(outAlpha * 255);
        }
      }
    }
    return output;
  }
}

function insideRoundedRect(x, y, radius) {
  const max = GEOMETRY.viewBox - radius;
  const dx = x < radius ? radius - x : x > max ? x - max : 0;
  const dy = y < radius ? radius - y : y > max ? y - max : 0;
  return dx * dx + dy * dy <= radius * radius;
}

function distanceToSegment(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  if (dx === 0 && dy === 0) {
    return Math.hypot(px - x1, py - y1);
  }
  const t = Math.max(
    0,
    Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)),
  );
  return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
}

function quadraticPoint(p0, p1, p2, t) {
  const mt = 1 - t;
  return {
    x: mt * mt * p0.x + 2 * mt * t * p1.x + t * t * p2.x,
    y: mt * mt * p0.y + 2 * mt * t * p1.y + t * t * p2.y,
  };
}

function pointOnCircle(cx, cy, radius, angleDeg) {
  const angle = (angleDeg * Math.PI) / 180;
  return {
    x: cx + Math.cos(angle) * radius,
    y: cy + Math.sin(angle) * radius,
  };
}

function sampleStops(stops, t) {
  for (let index = 0; index < stops.length - 1; index += 1) {
    const left = stops[index];
    const right = stops[index + 1];
    if (t >= left.offset && t <= right.offset) {
      const span = right.offset - left.offset || 1;
      return {
        color: mixColor(left.color, right.color, (t - left.offset) / span),
        opacity: left.opacity + (right.opacity - left.opacity) * ((t - left.offset) / span),
      };
    }
  }
  return stops[stops.length - 1];
}

function pointInPolygon(x, y, points) {
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i, i += 1) {
    const xi = points[i].x;
    const yi = points[i].y;
    const xj = points[j].x;
    const yj = points[j].y;
    const intersect =
      yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (intersect) {
      inside = !inside;
    }
  }
  return inside;
}

function parhelionPoints(cx, cy) {
  return [
    { x: cx - 90, y: cy },
    { x: cx, y: cy - 43 },
    { x: cx + 90, y: cy },
    { x: cx, y: cy + 43 },
  ];
}

function drawMark(size, { transparent = false } = {}) {
  const raster = new Raster(size, { transparent });
  const ice = hexToRgb(COLORS.ice);
  const gold = hexToRgb(COLORS.gold);
  const sunWhite = hexToRgb(COLORS.sunWhite);
  const redInside = hexToRgb(COLORS.redInside);
  const blueOutside = hexToRgb(COLORS.blueOutside);
  const goldDeep = hexToRgb(COLORS.goldDeep);

  raster.drawBackground();
  raster.drawStrokedCircle(512, 512, 346, 26, ice, transparent ? 0.36 : 0.46);
  raster.drawStrokedCircle(512, 512, 258, 18, gold, transparent ? 0.72 : 0.66);
  raster.drawQuadratic(
    { x: 166, y: beltY },
    { x: 512, y: beltY - 28 },
    { x: 858, y: beltY },
    25,
    gold,
    transparent ? 0.76 : 0.7,
  );
  raster.drawQuadratic(
    { x: 276, y: 392 },
    { x: 512, y: 254 },
    { x: 748, y: 392 },
    28,
    sunWhite,
    transparent ? 0.82 : 0.76,
  );
  if (size >= 96) {
    raster.drawQuadratic(
      { x: 328, y: 276 },
      { x: 512, y: 204 },
      { x: 696, y: 276 },
      18,
      ice,
      transparent ? 0.4 : 0.34,
      100,
    );
  }
  raster.drawRadialCircle(512, 512, 162, [
    { offset: 0, color: gold, opacity: 0.22 },
    { offset: 0.72, color: gold, opacity: 0.08 },
    { offset: 1, color: gold, opacity: 0 },
  ]);
  raster.drawRadialCircle(512, 512, 112, [
    { offset: 0, color: sunWhite, opacity: 1 },
    { offset: 0.42, color: gold, opacity: 1 },
    { offset: 1, color: goldDeep, opacity: 1 },
  ]);
  raster.drawRadialCircle(512, 512, 42, [
    { offset: 0, color: sunWhite, opacity: 0.94 },
    { offset: 1, color: sunWhite, opacity: 0.78 },
  ]);

  raster.drawPolygon(parhelionPoints(leftParhelionX, beltY), (x) => {
    const t = (x - (leftParhelionX - 90)) / 180;
    return t < 0.52
      ? mixColor(blueOutside, gold, t / 0.52)
      : mixColor(gold, redInside, (t - 0.52) / 0.48);
  }, 0.94);
  raster.drawPolygon(parhelionPoints(rightParhelionX, beltY), (x) => {
    const t = (x - (rightParhelionX - 90)) / 180;
    return t < 0.48
      ? mixColor(redInside, gold, t / 0.48)
      : mixColor(gold, blueOutside, (t - 0.48) / 0.52);
  }, 0.94);

  if (size >= 48) {
    raster.drawArc(512, 512, 346, 180, 360, 12, sunWhite, 0.22, 120);
  }

  return raster.downsample();
}

const CRC_TABLE = new Uint32Array(256).map((_, index) => {
  let c = index;
  for (let bit = 0; bit < 8; bit += 1) {
    c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  }
  return c >>> 0;
});

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data = Buffer.alloc(0)) {
  const typeBuffer = Buffer.from(type, "ascii");
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length, 0);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([typeBuffer, data])), 0);
  return Buffer.concat([length, typeBuffer, data, crc]);
}

function encodePng(width, height, rgba) {
  const signature = Buffer.from([
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
  ]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 6;
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;

  const raw = Buffer.alloc((width * 4 + 1) * height);
  for (let row = 0; row < height; row += 1) {
    const rawOffset = row * (width * 4 + 1);
    raw[rawOffset] = 0;
    Buffer.from(rgba.buffer, row * width * 4, width * 4).copy(raw, rawOffset + 1);
  }

  return Buffer.concat([
    signature,
    pngChunk("IHDR", ihdr),
    pngChunk("IDAT", deflateSync(raw, { level: 9 })),
    pngChunk("IEND"),
  ]);
}

function pngBuffer(size, options = {}) {
  const rgba = drawMark(size, options);
  return encodePng(size, size, rgba);
}

async function writePng(path, size, options = {}) {
  await writeFile(path, pngBuffer(size, options));
}

function encodeIco(images) {
  const header = Buffer.alloc(6);
  header.writeUInt16LE(0, 0);
  header.writeUInt16LE(1, 2);
  header.writeUInt16LE(images.length, 4);

  const entries = [];
  let offset = header.length + images.length * 16;
  for (const image of images) {
    const entry = Buffer.alloc(16);
    entry[0] = image.size >= 256 ? 0 : image.size;
    entry[1] = image.size >= 256 ? 0 : image.size;
    entry[2] = 0;
    entry[3] = 0;
    entry.writeUInt16LE(1, 4);
    entry.writeUInt16LE(32, 6);
    entry.writeUInt32LE(image.buffer.length, 8);
    entry.writeUInt32LE(offset, 12);
    entries.push(entry);
    offset += image.buffer.length;
  }

  return Buffer.concat([header, ...entries, ...images.map((image) => image.buffer)]);
}

async function writeProductionAssets() {
  const png16 = pngBuffer(16);
  const png32 = pngBuffer(32);
  const png48 = pngBuffer(48);
  const png180 = pngBuffer(180);
  const png192 = pngBuffer(192);
  const png512 = pngBuffer(512);

  await writeFile(resolve(PUBLIC_DIR, "favicon.svg"), svgMarkup(), "utf8");
  await writeFile(
    resolve(PUBLIC_DIR, "favicon.ico"),
    encodeIco([
      { size: 16, buffer: png16 },
      { size: 32, buffer: png32 },
      { size: 48, buffer: png48 },
    ]),
  );
  await writeFile(resolve(PUBLIC_DIR, "apple-touch-icon.png"), png180);

  await writeFile(resolve(ICON_DIR, "icon-16.png"), png16);
  await writeFile(resolve(ICON_DIR, "icon-32.png"), png32);
  await writeFile(resolve(ICON_DIR, "icon-48.png"), png48);
  await writeFile(resolve(ICON_DIR, "icon-180.png"), png180);
  await writeFile(resolve(ICON_DIR, "icon-192.png"), png192);
  await writeFile(resolve(ICON_DIR, "icon-512.png"), png512);
  await writeFile(resolve(ICON_DIR, "maskable-192.png"), png192);
  await writeFile(resolve(ICON_DIR, "maskable-512.png"), png512);
}

async function main() {
  await mkdir(ICON_DIR, { recursive: true });

  await writeFile(
    resolve(ICON_DIR, "sundog-character-mark.svg"),
    svgMarkup(),
    "utf8",
  );
  await writeFile(
    resolve(ICON_DIR, "sundog-character-mark.transparent.svg"),
    svgMarkup({ transparent: true }),
    "utf8",
  );
  await writeFile(
    resolve(ICON_DIR, "sundog-character-mark.animated.svg"),
    svgMarkup({ animated: true }),
    "utf8",
  );
  await writeFile(
    resolve(ICON_DIR, "sundog-character-mark.layers.json"),
    `${JSON.stringify(layerManifest(), null, 2)}\n`,
    "utf8",
  );

  await writePng(resolve(ICON_DIR, "sundog-character-favicon-16.png"), 16);
  await writePng(resolve(ICON_DIR, "sundog-character-favicon-32.png"), 32);
  await writePng(resolve(ICON_DIR, "sundog-character-favicon-48.png"), 48);
  await writePng(resolve(ICON_DIR, "sundog-character-icon-180.png"), 180);
  await writePng(resolve(ICON_DIR, "sundog-character-icon-192.png"), 192);
  await writePng(resolve(ICON_DIR, "sundog-character-icon-512.png"), 512);
  await writePng(
    resolve(ICON_DIR, "sundog-character-transparent-512.png"),
    512,
    { transparent: true },
  );

  for (const file of GENERATED_FILES) {
    console.log(`generated ${file}`);
  }

  if (promoteProduction) {
    await writeProductionAssets();
    for (const file of PRODUCTION_FILES) {
      console.log(`promoted ${file}`);
    }
  }
}

await main();
