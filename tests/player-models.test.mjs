import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function readModel(name) {
  const url = new URL(`../app/${name}`, import.meta.url);
  return JSON.parse(await readFile(url, "utf8"));
}

test("ships independently trained 3-player, 4-player, and 5-player models", async () => {
  const [three, four, five] = await Promise.all([
    readModel("pretrained-model-3p.json"),
    readModel("pretrained-model.json"),
    readModel("pretrained-model-5p.json"),
  ]);

  assert.equal(three.playerCount, 3);
  assert.equal(four.playerCount, 4);
  assert.equal(five.playerCount, 5);

  const models = [three, four, five];
  for (const model of models) {
    assert.ok(model.trainingGames > 0);
    assert.ok(model.generations >= model.stoppingStandard.minimumGenerations);
    assert.equal(model.achievedStandard.passed, true);
    assert.ok(
      model.achievedStandard.winRateLowerBound >=
        model.stoppingStandard.targetWinRateLowerBound,
    );
    assert.ok(
      model.averageRankVsHeuristic <=
        model.stoppingStandard.targetAverageRank,
    );
    assert.ok(model.winRateVsHeuristic > 1 / model.playerCount);
    assert.equal(model.inputCount, 9);
    assert.equal(model.outputCount, 6);
    assert.equal(
      model.version,
      "league-neuroevolution-v5-marginal-trade-valuation",
    );
    assert.match(model.algorithm, /economic consistency/i);
    assert.equal(model.featureNames.at(-1), "tradeWindow");
    assert.equal(model.actionNames.at(-1), "trade");
  }

  assert.equal(new Set(models.map((model) => model.seed)).size, 3);
  assert.equal(
    new Set(
      models.map(
        (model) => model.stoppingStandard.targetWinRateLowerBound,
      ),
    ).size,
    3,
  );
});
