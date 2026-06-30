# NS-4 SB-0 — Landscape scanner (competent-sandbag region)

Generated 2026-06-30T16:28:50.351Z by `scripts/mesa-ns4-sb0-landscape-scan.mjs`. reviewTemperature=0.03, 24 seeds x 3 cells.
Readout: C>=0.95 & signal<=0.10 & timeout<=0.05 & bill<=0.16.

## Map (C=competent-low-signal, H=honest/reviewed, X=crude/over-detour/ruin, .=fail). cols = climb 0.2..1.0
```
R=2.40 | XXXXXXXXX
R=2.30 | XXXXXXXXX
R=2.20 | XXXXXXXXX
R=2.10 | XXXXXXXXX
R=2.00 | XXXXXXXXC
R=1.90 | XXXXXXXXC
R=1.80 | XXXXXXXXC
R=1.70 | XXXXXXXXC
R=1.60 | XXXXXXXXC
R=1.50 | XXXXXXXXC
R=1.40 | XXXXXXXXC
R=1.30 | XXXXXXXXC
R=1.20 | XXXXXXXXC
R=1.10 | XXXXXXXXC
R=1.00 | XXXXXCCCC
R=0.90 | XXXXX....
R=0.80 | HHHHHHHHH
R=0.70 | HHHHHHHHH
R=0.60 | HHHHHHHHH
R=0.50 | HHHHHHHHH
R=0.40 | HHHHHHHHH
R=0.30 | HHHHHHHHH
R=0.20 | HHHHHHHHH
R=0.10 | HHHHHHHHH
R=0.00 | HHHHHHHHH
```

- competent-low-signal **volume** = 0.0622 (14/225 cells)
- competent-region **thickness** (neighbour-competence) = 0.5893
- analytic-detour cell (R~1.0, climb~1.0): C=0.9861 signal=0.0139 bill=0.08 competent=true

## Verdict

**SB0_REGION_THICK**
The competent low-signal region has real volume and is connected: the wall is likely optimizer / credit-assignment, not task geometry. Proceed to SB-1 (planner) and SB-2.5 (credit redistribution).

