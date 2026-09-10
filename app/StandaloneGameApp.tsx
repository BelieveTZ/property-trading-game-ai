"use client";

import {
  ChangeEvent,
  CSSProperties,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { CARD_DECKS } from "./card-catalog.mjs";
import { DEED_DETAILS, TILES } from "./board-catalog.mjs";
import { chooseAutomaticAction } from "./standalone-policy.mjs";
import {
  applyStandaloneAction,
  createTradeDraft,
  createStandaloneSession,
  legalManagementActions,
  legalStandaloneActions,
  readStandaloneSession,
  recommendStandaloneAction,
  standaloneActorId,
  writeStandaloneSession,
} from "./standalone-session.mjs";

declare global {
  interface Document {
    readonly modelContext?: {
      registerTool(tool: {
        name: string;
        title: string;
        description: string;
        inputSchema: object;
        annotations: { readOnlyHint: boolean; untrustedContentHint: boolean };
        execute(input: unknown): unknown;
      }, options?: { signal?: AbortSignal }): void | Promise<void>;
    };
  }
}

const SAVE_KEY = "property-trading-game-ai-standalone-v2";
const LEGACY_SAVE_KEYS = [
  "property-trading-game-ai-standalone-v1",
  "property-trading-game-ai-session-v4",
];
const DEFAULT_POLICY = {
  liquidity: 0.95,
  roi: 1.25,
  monopoly: 1.5,
  pressure: 0.8,
  reserve: 240,
};

type Session = Omit<ReturnType<typeof createStandaloneSession>, "negotiation" | "tradeAttemptsThisTurn"> & {
  negotiation: null | {
    open: boolean;
    proposerId: number;
    responderId: number;
    counterCount: number;
    draft: TradeDraft;
  };
  tradeAttemptsThisTurn: number[];
};
type TradeDraft = {
  fromId: number;
  toId: number;
  fromCash: number;
  toCash: number;
  fromCards: number;
  toCards: number;
  fromPropertyIds: number[];
  toPropertyIds: number[];
};
type DeedDetail =
  | { kind: "property"; rents: number[]; mortgage: number }
  | { kind: "station"; rents: number[]; mortgage: number }
  | { kind: "utility"; mortgage: number };

function money(value: number) {
  return `¤${Math.round(value).toLocaleString("zh-CN")}`;
}

function tilePosition(id: number): CSSProperties {
  if (id <= 10) return { gridRow: 11, gridColumn: 11 - id };
  if (id <= 20) return { gridRow: 21 - id, gridColumn: 1 };
  if (id <= 30) return { gridRow: 1, gridColumn: id - 19 };
  return { gridRow: id - 29, gridColumn: 11 };
}

export default function StandaloneGameApp() {
  const [session, setSession] = useState<Session | null>(null);
  const [playerCount, setPlayerCount] = useState(4);
  const [adviceEnabled, setAdviceEnabled] = useState(true);
  const [selectedTileId, setSelectedTileId] = useState<number | null>(null);
  const [managementOpen, setManagementOpen] = useState(false);
  const [tradeEditor, setTradeEditor] = useState<{
    mode: "propose" | "counter";
    draft: TradeDraft;
  } | null>(null);
  const [auctionOffer, setAuctionOffer] = useState({ bid: -1, amount: 1 });
  const [spectatePaused, setSpectatePaused] = useState(false);
  const [ready, setReady] = useState(false);
  const importRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let restored: Session | null = null;
    try {
      const keys = [SAVE_KEY, ...LEGACY_SAVE_KEYS];
      const saved = keys.map((key) => localStorage.getItem(key)).find(Boolean);
      if (saved) restored = readStandaloneSession(saved) as Session;
    } catch {
      localStorage.removeItem(SAVE_KEY);
    }
    if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
    const timer = window.setTimeout(() => {
      setSession(restored);
      setReady(true);
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    const context = document.modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    void Promise.resolve(context.registerTool({
      name: "start_property_trading_game",
      title: "开始地产交易牌局",
      description: "开始一局新的原创地产交易游戏，并在页面中显示该牌局。",
      inputSchema: {
        type: "object",
        properties: {
          mode: { type: "string", enum: ["play", "spectate"] },
          playerCount: { type: "integer", enum: [3, 4, 5] },
          seed: { type: "integer", minimum: 1 },
        },
        required: ["mode", "playerCount", "seed"],
        additionalProperties: false,
      },
      annotations: { readOnlyHint: false, untrustedContentHint: false },
      execute(input) {
        const candidate = input as { mode?: unknown; playerCount?: unknown; seed?: unknown };
        if (
          !["play", "spectate"].includes(String(candidate.mode)) ||
          ![3, 4, 5].includes(Number(candidate.playerCount)) ||
          !Number.isInteger(candidate.seed) ||
          Number(candidate.seed) < 1
        ) {
          throw new Error("mode, playerCount or seed is invalid");
        }
        const created = createStandaloneSession({
          mode: candidate.mode as "play" | "spectate",
          playerCount: Number(candidate.playerCount),
          seed: Number(candidate.seed),
          policy: DEFAULT_POLICY,
        });
        setSession(created);
        setSpectatePaused(false);
        return {
          status: "started",
          mode: created.mode,
          playerCount: created.game.players.length,
          seed: created.runtime.seed,
        };
      },
    }, { signal: lifecycle.signal })).catch(() => {});
    return () => lifecycle.abort();
  }, []);

  useEffect(() => {
    if (!ready || !session) return;
    localStorage.setItem(SAVE_KEY, writeStandaloneSession(session));
  }, [ready, session]);

  const actorId = session ? standaloneActorId(session) : null;
  const actor = session?.game.players.find(
    (player: { id: number }) => player.id === actorId,
  );
  const legal = useMemo(
    () => (session ? legalStandaloneActions(session) : []),
    [session],
  );
  const automatedTurn = Boolean(
    session &&
      !session.game.gameOver &&
      (session.mode === "spectate" || actorId !== session.humanPlayerId),
  );

  useEffect(() => {
    if (!session || !automatedTurn || (session.mode === "spectate" && spectatePaused)) return;
    const action = chooseAutomaticAction(session);
    if (!action) return;
    const timer = window.setTimeout(() => {
      const result = applyStandaloneAction(session, action);
      if (result.ok) setSession(result.session);
    }, 420);
    return () => window.clearTimeout(timer);
  }, [automatedTurn, session, spectatePaused]);

  function start(mode: "play" | "spectate") {
    setSession(createStandaloneSession({
      mode,
      playerCount,
      seed: Date.now(),
      policy: DEFAULT_POLICY,
    }));
    setSpectatePaused(false);
  }

  function act(action: Record<string, unknown>) {
    if (!session) return false;
    const result = applyStandaloneAction(session, action);
    if (!result.ok) return false;
    setSession(result.session);
    return true;
  }

  function exportReplay() {
    if (!session) return;
    const blob = new Blob([writeStandaloneSession(session, { space: 2 })], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `property-trading-game-${session.runtime.seed}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function importReplay(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      setSession(readStandaloneSession(await file.text()));
    } catch {
      window.alert("回放文件无效，牌局未被替换。");
    }
  }

  function resetGame() {
    localStorage.removeItem(SAVE_KEY);
    setSession(null);
  }

  if (!ready) return <main className="standalone-loading">正在恢复本机牌局…</main>;

  if (!session) {
    return (
      <main className="standalone-start">
        <section className="start-card" aria-labelledby="start-title">
          <div className="start-mark" aria-hidden="true">¤</div>
          <p>澄湾市地产交易局</p>
          <h1 id="start-title">地产交易游戏AI</h1>
          <div className="player-count-picker" aria-label="玩家人数">
            {[3, 4, 5].map((count) => (
              <button key={count} className={playerCount === count ? "active" : ""} onClick={() => setPlayerCount(count)}>
                {count} 人
              </button>
            ))}
          </div>
          <div className="start-actions">
            <button className="start-primary" onClick={() => start("play")}>开始游玩</button>
            <button onClick={() => start("spectate")}>观看对局</button>
          </div>
          <button className="start-import" onClick={() => importRef.current?.click()}>导入回放</button>
          <input ref={importRef} hidden type="file" accept="application/json,.json" onChange={importReplay} />
        </section>
      </main>
    );
  }

  const activePlayer = session.game.players.find(
    (player: { id: number }) => player.id === session.game.activePlayerId,
  );
  const winner = session.game.players.find(
    (player: { id: number }) => player.id === session.game.winnerId,
  );
  const pendingTile = session.game.pending?.tileId === undefined
    ? null
    : TILES[session.game.pending.tileId];
  const selectedTile = selectedTileId === null ? null : TILES[selectedTileId];
  const selectedDeed = selectedTile
    ? DEED_DETAILS[selectedTile.id] as DeedDetail | undefined
    : undefined;
  const ownerByTile = new Map(
    session.game.properties.map((property: { tileId: number }) => [property.tileId, property]),
  );
  const selectedProperty = selectedTile
    ? ownerByTile.get(selectedTile.id) as { mortgaged: boolean } | undefined
    : undefined;
  const management = legalManagementActions(session, actorId);
  const auctionAmount = auctionOffer.bid === session.auction.currentBid
    ? auctionOffer.amount
    : session.auction.currentBid + 1;
  const suggested = recommendStandaloneAction(session);
  const advice = suggested?.kind === "buy"
    ? `建议买入 ${pendingTile?.name}`
    : suggested?.kind === "roll"
      ? "掷骰子继续本回合"
      : suggested?.kind === "auction-bid"
        ? `可以从 ${money(session.auction.currentBid + 1)} 继续竞价`
        : suggested?.kind === "manage" && session.game.debt
          ? "先出售建筑或抵押地契筹集资金"
          : suggested?.kind === "confirm"
            ? "完成当前结算"
            : "结束回合前可以管理资产或交易";

  return (
    <main className="standalone-shell">
      <header className="standalone-header">
        <div><span>澄湾市</span><strong>地产交易游戏AI</strong></div>
        <div className="header-round">第 {session.game.round} 轮</div>
        <div className="header-tools">
          <label><input type="checkbox" checked={adviceEnabled} onChange={(event) => setAdviceEnabled(event.target.checked)} /> 实时建议</label>
          {session.mode === "spectate" && (
            <button onClick={() => setSpectatePaused((value) => !value)}>
              {spectatePaused ? "继续观战" : "暂停观战"}
            </button>
          )}
          <button onClick={exportReplay}>导出回放</button>
          <button onClick={() => importRef.current?.click()}>导入回放</button>
          <input ref={importRef} hidden type="file" accept="application/json,.json" onChange={importReplay} />
          <button onClick={resetGame}>新牌局</button>
        </div>
      </header>

      <section className="standalone-layout">
        <section className="game-stage" aria-label="地产交易棋盘">
          <div className="standalone-players">
            {session.game.players.map((player: { id: number; name: string; color: string; cash: number; bankrupt: boolean }) => (
              <article key={player.id} className={`${player.id === actorId ? "active" : ""} ${player.bankrupt ? "bankrupt" : ""}`}>
                <i style={{ background: player.color }}>{player.name.slice(0, 1)}</i>
                <div><strong>{player.name}</strong><span>{money(player.cash)}</span></div>
              </article>
            ))}
          </div>

          <div className="standalone-board">
            {TILES.map((tile: { id: number; name: string; short: string; type: string; color?: string }) => {
              const property = ownerByTile.get(tile.id) as { ownerId: number | null; houses: number; mortgaged: boolean } | undefined;
              const owner = session.game.players.find((player: { id: number }) => player.id === property?.ownerId);
              const occupants = session.game.players.filter((player: { position: number; bankrupt: boolean }) => player.position === tile.id && !player.bankrupt);
              return (
                <button key={tile.id} className={`standalone-tile tile-${tile.type}`} style={tilePosition(tile.id)} onClick={() => setSelectedTileId(tile.id)} aria-label={tile.name}>
                  {tile.color && <span className="standalone-band" style={{ background: tile.color }} />}
                  <strong>{tile.short}</strong>
                  {owner && <em style={{ background: owner.color }} aria-label={`属于${owner.name}`} />}
                  {property && property.houses > 0 && <small className={property.houses === 5 ? "hotel" : "houses"}>{property.houses === 5 ? "◆" : "■".repeat(property.houses)}</small>}
                  {property?.mortgaged && <b aria-label="已抵押">×</b>}
                  {!!occupants.length && (
                    <span className="standalone-tokens">
                      {occupants.map((player: { id: number; name: string; color: string }) => (
                        <i key={player.id} style={{ background: player.color }}>{player.name.slice(0, 1)}</i>
                      ))}
                    </span>
                  )}
                </button>
              );
            })}
            <div className="standalone-center">
              <span>PROPERTY TRADING</span>
              <strong>澄湾市</strong>
              <p>每一次选择都会改变城市的归属</p>
            </div>
          </div>
        </section>

        <aside className="standalone-panel">
          <div className="turn-owner">
            <span>{automatedTurn ? "AI 正在行动" : "轮到你"}</span>
            <h2>{actor?.name ?? activePlayer?.name}</h2>
            <p>{TILES[activePlayer?.position ?? 0].name} · {money(actor?.cash ?? 0)}</p>
          </div>

          {adviceEnabled && session.mode === "play" && !automatedTurn && (
            <div className="standalone-advice"><span>策略建议</span><strong>{advice}</strong></div>
          )}

          <section className="standalone-action-card">
            {session.game.gameOver ? (
              <div className="game-over-summary" role="status">
                <span>牌局结束</span>
                <strong>{winner?.name ?? "无人"} 获胜</strong>
                <button className="standalone-main-action" onClick={resetGame}>返回主界面</button>
              </div>
            ) : automatedTurn ? (
              <p className="thinking">{spectatePaused ? "观战已暂停" : "正在评估局面…"}</p>
            ) : (
              <>
                {legal.includes("roll") && <button className="standalone-main-action" onClick={() => act({ kind: "roll" })}>掷骰子</button>}
                {legal.includes("use-pass") && <button onClick={() => act({ kind: "use-pass" })}>使用暂留通行证</button>}
                {legal.includes("pay-fee") && <button onClick={() => act({ kind: "pay-fee" })}>支付 ¤50 后离开</button>}
                {legal.includes("buy") && <button className="standalone-main-action" onClick={() => act({ kind: "buy" })}>买入 {pendingTile?.name}</button>}
                {legal.includes("decline") && <button onClick={() => act({ kind: "decline" })}>放弃并开始拍卖</button>}
                {legal.includes("confirm") && <button className="standalone-main-action" onClick={() => act({ kind: "confirm" })}>确认结算</button>}
                {legal.includes("auction-bid") && (
                  <div className="auction-controls">
                    <p>当前出价 <strong>{money(session.auction.currentBid)}</strong></p>
                    <label>你的出价<input type="number" min={session.auction.currentBid + 1} max={actor?.cash ?? 0} value={auctionAmount} onChange={(event) => setAuctionOffer({ bid: session.auction.currentBid, amount: Number(event.target.value) })} /></label>
                    <button className="standalone-main-action" onClick={() => act({ kind: "auction-bid", amount: auctionAmount })}>出价</button>
                    <button onClick={() => act({ kind: "auction-pass" })}>退出拍卖</button>
                  </div>
                )}
                {legal.includes("manage") && <button onClick={() => setManagementOpen(true)}>资产管理</button>}
                {legal.includes("trade-propose") && session.game.players.filter((player: { bankrupt: boolean }) => !player.bankrupt).length > 1 && (
                  <button onClick={() => {
                    const other = session.game.players.find((player: { id: number; bankrupt: boolean }) => player.id !== actorId && !player.bankrupt)!;
                    setTradeEditor({ mode: "propose", draft: createTradeDraft(actorId!, other.id) });
                  }}>发起交易</button>
                )}
                {legal.includes("trade-accept") && (
                  <div className="negotiation-actions">
                    <p>{session.game.players.find((player: { id: number }) => player.id === session.negotiation!.proposerId)?.name} 提出了报价</p>
                    <button className="standalone-main-action" onClick={() => act({ kind: "trade-accept" })}>接受报价</button>
                    <button onClick={() => act({ kind: "trade-reject" })}>拒绝报价</button>
                    {legal.includes("trade-counter") && <button onClick={() => setTradeEditor({ mode: "counter", draft: structuredClone(session.negotiation!.draft) })}>提出反报价</button>}
                  </div>
                )}
                {legal.includes("end-turn") && <button className="standalone-main-action" onClick={() => act({ kind: "end-turn" })}>结束回合</button>}
                {legal.includes("bankrupt") && <button className="danger-action" onClick={() => act({ kind: "bankrupt" })}>宣布破产</button>}
              </>
            )}
            {session.game.lastDice && <div className="last-roll"><span>{session.game.lastDice[0]}</span><span>{session.game.lastDice[1]}</span></div>}
          </section>

          <section className="standalone-history">
            <h3>牌局记录</h3>
            <ol>{session.game.log.slice(0, 12).map((entry: string, index: number) => <li key={`${entry}-${index}`}>{entry}</li>)}</ol>
          </section>
        </aside>
      </section>

      {managementOpen && (
        <div className="standalone-modal" onMouseDown={() => setManagementOpen(false)}>
          <section className="management-dialog" role="dialog" aria-modal="true" aria-label="资产管理" onMouseDown={(event) => event.stopPropagation()}>
            <button aria-label="关闭" onClick={() => setManagementOpen(false)}>×</button>
            <span>本回合</span><h2>资产管理</h2>
            {!management.length && <p>当前没有可执行的资产操作。</p>}
            <div className="management-list">
              {session.game.properties.filter((property: { ownerId: number | null }) => property.ownerId === actorId).map((property: { tileId: number; houses: number; mortgaged: boolean }) => (
                <article key={property.tileId}>
                  <div><strong>{TILES[property.tileId].name}</strong><span>{property.mortgaged ? "已抵押" : property.houses === 5 ? "酒店" : `${property.houses} 栋房屋`}</span></div>
                  {management.filter((action: { tileId: number }) => action.tileId === property.tileId).map((action: { kind: string; tileId: number }) => (
                    <button key={action.kind} onClick={() => act({ kind: "manage", actions: [action] })}>
                      {{ build: "建造", "sell-building": "出售建筑", mortgage: "抵押", unmortgage: "赎回" }[action.kind]}
                    </button>
                  ))}
                </article>
              ))}
            </div>
          </section>
        </div>
      )}

      {tradeEditor && (
        <div className="standalone-modal" onMouseDown={() => setTradeEditor(null)}>
          <section className="trade-dialog" role="dialog" aria-modal="true" aria-label={tradeEditor.mode === "counter" ? "提出反报价" : "发起交易"} onMouseDown={(event) => event.stopPropagation()}>
            <button aria-label="关闭" onClick={() => setTradeEditor(null)}>×</button>
            <span>{tradeEditor.mode === "counter" ? `第 ${(session.negotiation?.counterCount ?? 0) + 1} 次反报价` : "双边协商"}</span>
            <h2>{tradeEditor.mode === "counter" ? "提出反报价" : "发起交易"}</h2>
            <label>交易对象
              <select value={tradeEditor.draft.toId} disabled={tradeEditor.mode === "counter"} onChange={(event) => setTradeEditor({ ...tradeEditor, draft: createTradeDraft(tradeEditor.draft.fromId, Number(event.target.value)) })}>
                {session.game.players.filter((player: { id: number; bankrupt: boolean }) => player.id !== tradeEditor.draft.fromId && !player.bankrupt).map((player: { id: number; name: string }) => <option key={player.id} value={player.id}>{player.name}</option>)}
              </select>
            </label>
            <div className="trade-columns">
              {(["from", "to"] as const).map((side) => {
                const playerId = side === "from" ? tradeEditor.draft.fromId : tradeEditor.draft.toId;
                const player = session.game.players.find((candidate: { id: number }) => candidate.id === playerId)!;
                const cashKey = side === "from" ? "fromCash" : "toCash";
                const cardsKey = side === "from" ? "fromCards" : "toCards";
                const propertyKey = side === "from" ? "fromPropertyIds" : "toPropertyIds";
                return (
                  <fieldset key={side}><legend>{player.name} 提供</legend>
                    <label>资金<input type="number" min="0" max={player.cash} value={tradeEditor.draft[cashKey]} onChange={(event) => setTradeEditor({ ...tradeEditor, draft: { ...tradeEditor.draft, [cashKey]: Number(event.target.value) } })} /></label>
                    <label>通行证<input type="number" min="0" max={player.jailFreeCards} value={tradeEditor.draft[cardsKey]} onChange={(event) => setTradeEditor({ ...tradeEditor, draft: { ...tradeEditor.draft, [cardsKey]: Number(event.target.value) } })} /></label>
                    {session.game.properties.filter((property: { ownerId: number | null; houses: number }) => property.ownerId === playerId && property.houses === 0).map((property: { tileId: number }) => (
                      <label className="trade-deed" key={property.tileId}><input type="checkbox" checked={tradeEditor.draft[propertyKey].includes(property.tileId)} onChange={(event) => {
                        const ids = event.target.checked ? [...tradeEditor.draft[propertyKey], property.tileId] : tradeEditor.draft[propertyKey].filter((id) => id !== property.tileId);
                        setTradeEditor({ ...tradeEditor, draft: { ...tradeEditor.draft, [propertyKey]: ids } });
                      }} /> {TILES[property.tileId].name}</label>
                    ))}
                  </fieldset>
                );
              })}
            </div>
            <button className="standalone-main-action" onClick={() => {
              const kind = tradeEditor.mode === "counter" ? "trade-counter" : "trade-propose";
              if (act({ kind, draft: tradeEditor.draft })) setTradeEditor(null);
            }}>{tradeEditor.mode === "counter" ? "发送反报价" : "发送报价"}</button>
          </section>
        </div>
      )}

      {selectedTile && (
        <div className="standalone-modal" onMouseDown={() => setSelectedTileId(null)}>
          <section className="deed-dialog" role="dialog" aria-modal="true" aria-label={selectedTile.name} onMouseDown={(event) => event.stopPropagation()}>
            <button aria-label="关闭" onClick={() => setSelectedTileId(null)}>×</button>
            <span>{selectedTile.type === "property" ? "城市地契" : selectedTile.type === "chance" ? "转机" : selectedTile.type === "community" ? "城市基金" : "棋盘位置"}</span>
            <h2>{selectedTile.name}</h2>
            {selectedProperty?.mortgaged && <b className="deed-mortgage-cross" aria-label="已抵押">×</b>}
            {'price' in selectedTile && <strong>地块售价 {money(Number(selectedTile.price ?? 0))}</strong>}
            {selectedDeed?.kind === "property" && <dl>{selectedDeed.rents.map((rent: number, index: number) => <div key={index}><dt>{index === 0 ? "基础租金" : index === 5 ? "酒店租金" : `${index} 栋房屋`}</dt><dd>{money(rent)}</dd></div>)}</dl>}
            {selectedDeed && <p>抵押价值 {money(selectedDeed.mortgage)}</p>}
            {selectedTile.type === "chance" && <div className="card-overview">{CARD_DECKS.chance.map((card: { id: string; label: string }) => <article key={card.id}>{card.label}</article>)}</div>}
            {selectedTile.type === "community" && <div className="card-overview">{CARD_DECKS.community.map((card: { id: string; label: string }) => <article key={card.id}>{card.label}</article>)}</div>}
            {selectedTile.type === "tax" && <div className="tax-face"><strong>{money(selectedTile.id === 4 ? 200 : 100)}</strong></div>}
          </section>
        </div>
      )}
    </main>
  );
}
