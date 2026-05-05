import { cp, mkdir } from "node:fs/promises";
import { join } from "node:path";

const root = process.cwd();
const dist = join(root, "dist");

await mkdir(dist, { recursive: true });
await cp(join(root, "README.md"), join(dist, "README.md"));
await cp(join(root, "docs"), join(dist, "docs"), { recursive: true });
