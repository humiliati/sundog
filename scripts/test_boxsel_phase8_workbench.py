#!/usr/bin/env python
"""Frozen test for the BoxSEL Phase-8 workbench data/page start.

Run: python scripts/test_boxsel_phase8_workbench.py
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, "scripts")
import boxsel_phase8_workbench_data as wb

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("(1) workbench data is sourced from the Phase-7b receipt:")
data = wb.build_data()
check("schema and version are named",
      data["schemaVersion"] == 1 and data["workbenchDataVersion"] == "phase8_boxsel_workbench_v0")
check("source paths point at the Phase-7b run receipt",
      data["sourceManifest"].endswith("phase7b_false_closure_run/manifest.json")
      and data["sourceResultNote"].endswith("PHASE7B_FALSE_CLOSURE_RUN.md"))
check("boundary text blocks real-KG/product overclaim",
      "Not a real-KG" in data["boundary"] and "not an Ask Sundog product claim" in data["boundary"])
check("summary preserves the Phase-7b pass metrics",
      data["summary"]["status"] == "PASS_PREREG_GATE"
      and data["summary"]["detectorAcceptedFalseClosures"] == 0
      and data["summary"]["baselineAcceptedFalseClosures"] == 16)

print("(2) interval layers support the Phase-8 visualization:")
cases = data["cases"]
stable = [case for case in cases if case["family"] == "stable_pmp_pressure_variants"]
helly = [case for case in cases if case["family"] == "helly_threshold_variants_v2"]
support = [case for case in cases if case["family"] == "support_floor_variants"]
controls = [case for case in cases if case["role"] == "acceptance_control"]
loss = [case for case in cases if case["role"] == "loss_control"]
check("case counts match the locked corpus",
      len(cases) == 28 and len(stable) == 8 and len(helly) == 4 and len(support) == 4)
check("stable PMP cases have sampled, pressure, and exact intervals",
      all(case["sample"] and case["pressure"] and case["exact"] for case in stable))
check("stable PMP pressure interval moves below the sampled interval",
      all(case["pressure"]["lower"] < case["sample"]["lower"] for case in stable))
check("Helly cases carry the exact box-attainable closed-form layer",
      all(case["box"] and case["middleKind"] == "box_attainable" for case in helly))
check("support cases carry pressure and low-support traces",
      all(case["pressure"] and case["trace"]["supportFloor"] <= data["thresholds"]["support_floor"] for case in support))
check("controls and loss cases preserve decisions",
      all(case["decision"] == "accept" for case in controls)
      and all(case["decision"] == "abstain" for case in loss))

print("(3) generated public data round-trips:")
written = wb.write_data()
path = ROOT / "public" / "data" / "boxsel-phase8-workbench.json"
loaded = json.loads(path.read_text(encoding="utf-8"))
check("public data file is written",
      path.exists() and loaded["workbenchDataVersion"] == written["workbenchDataVersion"])
check("public data cases match build_data cases",
      [case["caseId"] for case in loaded["cases"]] == [case["caseId"] for case in cases])

print("(4) page and route registration are review-only:")
html_path = ROOT / "boxsel.html"
html = html_path.read_text(encoding="utf-8") if html_path.exists() else ""
check("boxsel.html exists and loads the generated data",
      html_path.exists() and "/data/boxsel-phase8-workbench.json" in html)
check("page is marked noindex and carries the toy boundary",
      'name="robots"' in html and "noindex" in html and "Toy micro-SEL boundary" in html)
check("page has the expected control surfaces",
      all(token in html for token in ("case-list", "pressure-slider", "dimension-slider", "loss-slider")))
site_pages = json.loads((ROOT / "site-pages.json").read_text(encoding="utf-8"))
check("site-pages registers the review-only root HTML",
      any(page["entry"] == "boxsel.html" and page["kind"] == "workbench" for page in site_pages["pages"]))
seo = (ROOT / "docs" / "site" / "SEO_AND_SOCIAL_READINESS_ROADMAP.md").read_text(encoding="utf-8")
check("SEO roadmap records the review-only BoxSEL row",
      "| `/boxsel` | D |" in seo and "Phase-8 BoxSEL" in seo)

print(f"\n{'ALL PASS -- Phase-8 BoxSEL workbench data/page start is wired and review-only' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
