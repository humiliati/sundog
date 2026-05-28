import { spawnSync } from "node:child_process";

const python = process.env.SUNDOG_PYTHON ?? "python";
const child = spawnSync(python, ["docs/prereg/arc/phase3_decoder_v2.py", ...process.argv.slice(2)], {
  stdio: "inherit",
  shell: false
});

process.exit(child.status ?? 1);
