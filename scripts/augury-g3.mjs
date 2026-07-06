import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";

const venv = "./.venv-augury/Scripts/python.exe";
const python = process.env.SUNDOG_PYTHON ?? (existsSync(venv) ? venv : "python");
const child = spawnSync(python, ["docs/prereg/augury/augury_g3.py", ...process.argv.slice(2)], {
  stdio: "inherit",
  shell: false
});

process.exit(child.status ?? 1);
