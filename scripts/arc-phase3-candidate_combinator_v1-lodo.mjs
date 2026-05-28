import { parseArgs, printUsage, runLodo } from "./lib/arc-phase3-candidate_combinator_v1-core.mjs";

try {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || args.missingRequired) {
    printUsage("scripts/arc-phase3-candidate_combinator_v1-lodo.mjs");
    process.exit(args.help ? 0 : 2);
  }
  const result = await runLodo(args, process.argv.slice(2));
  console.log(`ARC Phase 3 candidate_combinator_v1 LODO: ${result.manifest.lodoInstanceCount} instance(s)`);
  console.log(`Wrote ${result.manifest.outDir}`);
} catch (err) {
  console.error(err.message);
  process.exit(1);
}
