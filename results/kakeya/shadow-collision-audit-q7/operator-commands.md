# Kakeya Shadow Collision Audit - Operator Commands

```powershell
node scripts/kakeya-shadow-collision-audit.mjs --q 7 --max-size 6 --max-states 300000
```

Primary outputs:

- `results\kakeya\shadow-collision-audit-q7\manifest.json`
- `results\kakeya\shadow-collision-audit-q7\signature-summary.csv`
- `results\kakeya\shadow-collision-audit-q7\structured-line-extension-summary.csv`
- `results\kakeya\shadow-collision-audit-q7\witnesses.json`

This audit measures the registered direction-shadow only. It does not expose
point membership as the primary signature, does not search for extremal Kakeya
sets, and does not make a Euclidean Kakeya claim.
