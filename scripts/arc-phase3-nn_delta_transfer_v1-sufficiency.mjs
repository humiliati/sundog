import { spawnSync } from "node:child_process";
import { resolve } from "node:path";

const argv = process.argv.slice(2);
const outDir = resolve(valueAfter(argv, "--out") ?? "results/arc/phase3-sufficiency-nn_delta_transfer_v1");
const lodoManifest = resolve(valueAfter(argv, "--lodo-manifest") ?? `${outDir}/manifest.json`);

const lodo = spawnSync(process.execPath, ["scripts/arc-phase3-nn_delta_transfer_v1-lodo.mjs", ...argv], {
  stdio: "inherit",
  shell: false
});
if (lodo.status !== 0) {
  process.exit(lodo.status ?? 1);
}

const pttestArgs = argv.includes("--lodo-manifest")
  ? argv
  : [...argv, "--lodo-manifest", lodoManifest];

const pttest = spawnSync(process.execPath, ["scripts/arc-phase3-nn_delta_transfer_v1-pttest.mjs", ...pttestArgs], {
  stdio: "inherit",
  shell: false
});
process.exit(pttest.status ?? 1);

function valueAfter(args, flag) {
  const index = args.indexOf(flag);
  return index >= 0 ? args[index + 1] : null;
}
