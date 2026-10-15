import { cp, lstat, mkdir, rename, rm } from "node:fs/promises";
import { join, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const source = join(root, "dist");
const destination = join(root, "docs");
const staging = join(root, `.docs-next-${process.pid}`);

if (destination !== resolve(root, "docs")) {
  throw new Error(`refusing unexpected fallback destination: ${destination}`);
}

try {
  if ((await lstat(destination)).isSymbolicLink()) {
    throw new Error(`refusing to replace symlinked fallback directory: ${destination}`);
  }
} catch (error) {
  if (error?.code !== "ENOENT") throw error;
}

await rm(staging, { recursive: true, force: true });
try {
  await mkdir(staging, { recursive: true });
  await cp(source, staging, { recursive: true, preserveTimestamps: false });

  await rm(destination, { recursive: true, force: true });
  await rename(staging, destination);
} catch (error) {
  await rm(staging, { recursive: true, force: true });
  throw error;
}

console.log("Synchronized dist/ to docs/ with the same static email contact path.");
