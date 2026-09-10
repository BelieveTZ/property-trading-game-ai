import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  calculateRent,
  createEmptyGenome,
  createHeuristic,
  playGame,
  tryBuild,
} from "../training/league-trainer.mjs";

test("the zero-knowledge setup script is portable across Windows user profiles", async () => {
  const source = await readFile(
    new URL("../training/zero_knowledge/setup-training.ps1", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(source, /C:\\Users\\/i);
  assert.match(source, /param\s*\(/i);
  assert.match(source, /PythonPath/i);
});

test("importing the classic league trainer does not start a training run", () => {
  const moduleUrl = new URL("../training/league-trainer.mjs", import.meta.url).href;
  const result = spawnSync(
    process.execPath,
    [
      "--input-type=module",
      "--eval",
      `await import(${JSON.stringify(moduleUrl)}); console.log("imported")`,
    ],
    { encoding: "utf8", timeout: 1500 },
  );
  assert.equal(result.error?.code, undefined);
  assert.equal(result.status, 0);
  assert.equal(result.stdout.trim(), "imported");
});

test("classic league rent and seeded simulations are deterministic", () => {
  const state = {
    houses: new Uint8Array(40),
    groupCounts: new Uint8Array(20),
  };
  state.groupCounts[0] = 2;
  state.groupCounts[8] = 2;
  state.groupCounts[9] = 2;
  assert.equal(calculateRent(state, 1, 0, 7), 4);
  assert.equal(calculateRent(state, 5, 0, 7), 50);
  assert.equal(calculateRent(state, 12, 0, 7), 70);

  const brains = Array.from({ length: 4 }, () => createHeuristic());
  assert.deepEqual(playGame(brains, 0, 20260724, 120), playGame(brains, 0, 20260724, 120));
});

test("classic league building cannot exceed the bank supply", () => {
  const brain = createEmptyGenome();
  brain.outputBias[2] = 1;
  brain.outputBias[4] = -1;
  const state = {
    cash: new Int32Array([5000, 1500]),
    alive: new Uint8Array([1, 1]),
    owners: new Int8Array(40).fill(-1),
    houses: new Uint8Array(40),
    mortgaged: new Uint8Array(40),
    groupCounts: new Uint8Array(20),
    features: new Float64Array(9),
  };
  state.owners[1] = 0;
  state.owners[3] = 0;
  state.groupCounts[0] = 2;
  for (const tile of [5, 6, 8, 9, 11, 13, 14, 15]) {
    state.houses[tile] = 4;
  }

  tryBuild(state, 0, brain, 1, 100);
  assert.equal(state.houses[1], 0);
  assert.equal(state.cash[0], 5000);
});
