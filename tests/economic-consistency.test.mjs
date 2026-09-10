import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  auctionBidCeiling,
  createEmptyGenome,
  wantsToBuy,
} from "../training/league-trainer.mjs";

const trainer = await readFile(
  new URL("../training/league-trainer.mjs", import.meta.url),
  "utf8",
);
const advisor = await readFile(
  new URL("../app/ai-advisor.mjs", import.meta.url),
  "utf8",
);

test("supplies the shared feature schema to purchase and auction decisions", () => {
  assert.match(advisor, /const features = decisionFeatures\(/);
  assert.match(advisor, /return encodeLeagueFeatures\(/);
  assert.match(trainer, /encodeLeagueFeatures\(/);
  assert.match(advisor, /\(features\[feature\] \?\? 0\)/);
});

test("buys directly whenever the same policy would protect face value at auction", () => {
  const brain = createEmptyGenome();
  brain.outputBias[0] = -1;
  brain.outputBias[1] = 1;
  brain.outputBias[4] = -1;
  const state = {
    cash: new Int32Array([1500, 1500]),
    alive: new Uint8Array([1, 1]),
    groupCounts: new Uint8Array(20),
    features: new Float64Array(9),
  };
  const reserve = 75;
  assert.ok(
    auctionBidCeiling(brain, state.features, 1500, 60, false, reserve) >= 60,
  );
  assert.equal(wantsToBuy(state, 0, 1, brain, 0, 100), true);
  assert.match(
    advisor,
    /score > -0\.05 \|\| maximumBid >= \(tile\.price \?\? 0\)/,
  );
  assert.match(advisor, /evaluateModel\(model, features, LEAGUE_OUTPUT\.auctionBid\)/);
});

test("marks retrained models with the economic-consistency algorithm", () => {
  assert.match(
    trainer,
    /version: "league-neuroevolution-v5-marginal-trade-valuation"/,
  );
  assert.match(trainer, /purchase-auction economic consistency/);
});
