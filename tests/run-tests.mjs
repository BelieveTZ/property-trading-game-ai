import { existsSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";

const root = resolve(import.meta.dirname, "..");
const argumentsSet = new Set(process.argv.slice(2));
const runJavaScript = !argumentsSet.has("--python-only");
const runPython = !argumentsSet.has("--js-only");

function run(command, args) {
  const result = spawnSync(command, args, {
    cwd: root,
    stdio: "inherit",
    shell: false,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

if (runJavaScript) {
  const tests = readdirSync(join(root, "tests"))
    .filter((name) => name.endsWith(".test.mjs"))
    .map((name) => join(root, "tests", name));
  run(process.execPath, ["--test", ...tests]);
}

if (runPython) {
  const candidates = [
    process.env.VIRTUAL_ENV
      ? join(process.env.VIRTUAL_ENV, process.platform === "win32" ? "Scripts/python.exe" : "bin/python")
      : "",
    join(root, ".venv-rl", process.platform === "win32" ? "Scripts/python.exe" : "bin/python"),
    join(root, "work", "site-source-v12", ".venv-rl", process.platform === "win32" ? "Scripts/python.exe" : "bin/python"),
  ].filter(Boolean);
  const python = candidates.find(existsSync);
  if (!python) {
    throw new Error("Python training environment is missing; run training/zero_knowledge/setup-training.ps1 first.");
  }
  run(python, ["-m", "unittest", "discover", "-s", "tests", "-p", "*_test.py", "-v"]);
}
