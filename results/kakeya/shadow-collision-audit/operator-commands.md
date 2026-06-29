# Kakeya Shadow Collision Audit - Operator Commands

```powershell
node scripts/kakeya-shadow-collision-audit.mjs --q 5 --max-size 6 --max-states 300000
```

Primary outputs:

- `results\kakeya\shadow-collision-audit\manifest.json`
- `results\kakeya\shadow-collision-audit\signature-summary.csv`
- `results\kakeya\shadow-collision-audit\structured-line-extension-summary.csv`
- `results\kakeya\shadow-collision-audit\witnesses.json`

This audit measures the registered direction-shadow only. It does not expose
point membership as the primary signature, does not search for extremal Kakeya
sets, and does not make a Euclidean Kakeya claim.
