#!/usr/bin/env python3
"""Patch OG, Twitter, and JSON-LD meta blocks into the seven public-share pages.

Idempotent: if a `<!-- OG-BLOCK-START -->` / `<!-- OG-BLOCK-END -->` sentinel
pair already exists, the block between them is replaced. Otherwise the block is
inserted right after the page's `</title>`.

For sundog.html (which has existing free-floating og: tags), the existing
og:title / og:description / og:image / og:url / og:type / twitter:card lines
are stripped before insertion to prevent duplicates.
"""

from __future__ import annotations
import pathlib, re, json, sys

REPO = pathlib.Path("/sessions/gallant-vigilant-newton/mnt/Dev/sundog")

# Per-page configuration.
PAGES = {
    "index.html": {
        "url": "https://sundog.cc/",
        "og_image": "https://sundog.cc/og/home.png",
        "og_title": "Sundog — Alignment Without Sight",
        "og_description": "Halo geometry from the math, and traceability discipline carried into control, agents, and workbenches. An independent applied research lab.",
        "og_image_alt": "Sundog research lab landing card: 'Alignment without sight.' over a halo diagram with parhelia.",
        "schema_type": "WebSite",
        "schema_headline": "Sundog — Alignment Without Sight",
        "schema_about": ["Sundog Research Lab", "Indirect inference", "Hidden-state control"],
    },
    "about.html": {
        "url": "https://sundog.cc/about",
        "og_image": "https://sundog.cc/og/about.png",
        "og_title": "About Sundog Research Lab",
        "og_description": "A traceability harness for indirect-inference alignment. Hidden-state control, route fidelity, and explicit failure boundaries.",
        "og_image_alt": "Sundog About page card: 'A traceability harness for indirect inference.' over the evidence-tier ladder.",
        "schema_type": "AboutPage",
        "schema_headline": "About Sundog Research Lab",
        "schema_about": ["Sundog Research Lab", "Research traceability", "Evidence tiers"],
    },
    "alignment.html": {
        "url": "https://sundog.cc/alignment",
        "og_image": "https://sundog.cc/og/alignment.png",
        "og_title": "Alignment & Bayes — Sundog Research Lab",
        "og_description": "Route fidelity under partial observation. Bayes turns evidence into belief; Sundog turns response into control. A claim-hygiene comparator.",
        "og_image_alt": "Sundog Alignment card: a Bayes posterior heatmap next to a Sundog signal trace.",
        "schema_type": "Article",
        "schema_headline": "Alignment & Bayes — Sundog",
        "schema_about": ["Alignment", "Bayesian inference", "Partial observability"],
    },
    "balance.html": {
        "url": "https://sundog.cc/balance",
        "og_image": "https://sundog.cc/og/balance.png",
        "og_title": "Sundog Balance — Cart-Pole with the Body Angle Denied",
        "og_description": "Shadow-derived cart-pole balance under partial observation. Operating envelope mapped; failure boundary named.",
        "og_image_alt": "Sundog Balance card: a cart-pole diagram with the pole's shadow projected on a wall.",
        "schema_type": "TechArticle",
        "schema_headline": "Sundog Balance Workbench",
        "schema_about": ["Cart-pole control", "Partial observability", "Operating envelope"],
    },
    "threebody.html": {
        "url": "https://sundog.cc/threebody",
        "og_image": "https://sundog.cc/og/threebody.png",
        "og_title": "Sundog Three-Body — Local Signals Survive the Near-Escape Pocket",
        "og_description": "Indirect signatures in chaotic three-body dynamics. A bounded operating envelope, not the whole regime.",
        "og_image_alt": "Sundog Three-Body card: orbital traces with a dashed escape trajectory.",
        "schema_type": "TechArticle",
        "schema_headline": "Sundog Three-Body Workbench",
        "schema_about": ["Three-body problem", "Chaotic dynamics", "Operating envelope"],
    },
    "mines.html": {
        "url": "https://sundog.cc/mines",
        "og_image": "https://sundog.cc/og/mines.png",
        "og_title": "Sundog Pressure Mines — Pressure Maps Become Decisions",
        "og_description": "A noisy pressure field buys safe progress inside a narrow mapped pocket — and fails cleanly past it. A minesweeper-style workbench.",
        "og_image_alt": "Sundog Pressure Mines card: a 6x6 pressure-shaded grid with a safe-pocket marker and a revealed mine.",
        "schema_type": "TechArticle",
        "schema_headline": "Sundog Pressure Mines Workbench",
        "schema_about": ["Pressure mapping", "Hidden-state inference", "Operating envelope"],
    },
    "sundog.html": {
        "url": "https://sundog.cc/sundog",
        "og_image": "https://sundog.cc/og/atlas.png",
        "og_title": "Sundog · The Geometry of a Sun Dog",
        "og_description": "Every named arc in a halo display, calibrated to photographs, with the math visible at every step. Interactive atlas.",
        "og_image_alt": "Sundog Atlas card: a parhelion sun with 22° halo, parhelic circle, parhelia, and circumzenithal arc.",
        "schema_type": "TechArticle",
        "schema_headline": "Sundog · The Geometry of a Sun Dog",
        "schema_about": ["Solar halo", "Parhelion", "Atmospheric optics", "Sundog atlas"],
    },
}


def render_block(cfg: dict) -> str:
    """Render the OG + Twitter + JSON-LD block for one page."""
    schema = {
        "@context": "https://schema.org",
        "@type": cfg["schema_type"],
        "headline": cfg["schema_headline"],
        "description": cfg["og_description"],
        "image": cfg["og_image"],
        "url": cfg["url"],
        "publisher": {
            "@type": "Organization",
            "name": "Stellar Aqua LLC",
            "url": "https://sundog.cc",
        },
        "inLanguage": "en",
        "isAccessibleForFree": True,
        "about": [{"@type": "Thing", "name": a} for a in cfg["schema_about"]],
    }
    if cfg["schema_type"] == "Article" or cfg["schema_type"] == "TechArticle":
        schema["author"] = {"@type": "Organization", "name": "Sundog Research Lab", "url": "https://sundog.cc"}
        schema["mainEntityOfPage"] = {"@type": "WebPage", "@id": cfg["url"]}
    json_ld = json.dumps(schema, indent=2)

    return f"""    <!-- OG-BLOCK-START (managed by public/og/_patch_meta.py) -->
    <meta property="og:type" content="{'website' if cfg['schema_type']=='WebSite' else 'article'}">
    <meta property="og:site_name" content="Sundog">
    <meta property="og:title" content="{cfg['og_title']}">
    <meta property="og:description" content="{cfg['og_description']}">
    <meta property="og:image" content="{cfg['og_image']}">
    <meta property="og:image:width" content="1200">
    <meta property="og:image:height" content="630">
    <meta property="og:image:alt" content="{cfg['og_image_alt']}">
    <meta property="og:url" content="{cfg['url']}">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="{cfg['og_title']}">
    <meta name="twitter:description" content="{cfg['og_description']}">
    <meta name="twitter:image" content="{cfg['og_image']}">
    <meta name="twitter:image:alt" content="{cfg['og_image_alt']}">
    <script type="application/ld+json">
{json_ld}
    </script>
    <!-- OG-BLOCK-END -->
"""


def patch_html(path: pathlib.Path, cfg: dict) -> str:
    html = path.read_text()

    # 1) Strip any existing free-floating og: / twitter: meta tags or JSON-LD
    #    so we don't duplicate. This is targeted at sundog.html which already
    #    has some og: tags before this script runs.
    html = re.sub(
        r'^\s*<meta\s+property="og:[^"]+"\s+content="[^"]*"\s*/?>\s*\n',
        '',
        html, flags=re.MULTILINE,
    )
    html = re.sub(
        r'^\s*<meta\s+name="twitter:[^"]+"\s+content="[^"]*"\s*/?>\s*\n',
        '',
        html, flags=re.MULTILINE,
    )

    block = render_block(cfg)

    # 2) Replace existing sentinel block if present, else insert after </title>
    # Use lambda replacement so backslashes in JSON aren't treated as regex
    # template escapes.
    if "<!-- OG-BLOCK-START" in html:
        html = re.sub(
            r"    <!-- OG-BLOCK-START .*?<!-- OG-BLOCK-END -->\n",
            lambda m: block,
            html, flags=re.DOTALL,
        )
        action = "replaced"
    else:
        new_html, count = re.subn(
            r"(</title>\n)",
            lambda m: m.group(1) + block,
            html, count=1,
        )
        if count == 0:
            raise RuntimeError(f"{path.name}: no </title> found")
        html = new_html
        action = "inserted"

    path.write_text(html)
    return action


def main():
    for filename, cfg in PAGES.items():
        path = REPO / filename
        if not path.exists():
            print(f"  SKIP (missing): {filename}")
            continue
        action = patch_html(path, cfg)
        print(f"  {action:<8} {filename}")


if __name__ == "__main__":
    main()
