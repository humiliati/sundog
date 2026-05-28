import { parseArgs, printUsage, runPttest } from "./lib/arc-phase3-nn_delta_transfer_v1-core.mjs";

try {
  const args = parseArgs(process.argv.slice(2), { requireLodoManifest: true });
  if (args.help || args.missingRequired || args.missingLodoManifest) {
    printUsage(
      "scripts/arc-phase3-nn_delta_transfer_v1-pttest.mjs",
      " --lodo-manifest <results/arc/phase3-sufficiency-nn_delta_transfer_v1/manifest.json>"
    );
    process.exit(args.help ? 0 : 2);
  }
  const result = await runPttest(args, process.argv.slice(2));
  console.log(`ARC Phase 3 nn_delta_transfer_v1 public-training test: ${result.manifest.pttestInstanceCount} instance(s)`);
  console.log(`Updated ${result.manifest.outDir}`);
} catch (err) {
  console.error(err.message);
  process.exit(1);
}
