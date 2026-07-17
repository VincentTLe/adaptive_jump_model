"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const marketNames = { us: "United States", de: "Germany", jp: "Japan" };
  const modelNames = { fixed_jm: "Fixed JM", hmm: "HMM" };
  const speedIntervals = { 0.5: 1200, 1: 600, 2: 300, 5: 120 };
  const number = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
  const replay = {
    eventCache: null, marketCache: new Map(), storyCache: new Map(), request: null,
    loadVersion: 0, events: [], jobId: null, market: "us", model: "fixed_jm",
    delay: 1, marketData: null, storyData: null, marketRows: new Map(), marketIndexes: new Map(),
    speed: 1, start: 0, index: 0, auditIndex: 0, timer: null,
  };

  function text(id, value) { $(id).textContent = value; }
  function pause() { window.clearInterval(replay.timer); replay.timer = null; }
  function tradeText(state) {
    if (state === 0) return "Cash · 0";
    if (state === 1) return "Market · 1";
    return "Unavailable";
  }
  function value(numberValue) {
    if (numberValue === null || numberValue === "") return "--";
    const numeric = Number(numberValue);
    return Number.isFinite(numeric) ? number.format(numeric) : "--";
  }
  function percent(numberValue) {
    const numeric = Number(numberValue);
    return Number.isFinite(numeric) ? `${number.format(numeric * 100)}%` : "--";
  }
  function money(numberValue) {
    const numeric = Number(numberValue);
    return Number.isFinite(numeric) ? `$${number.format(numeric)}` : "--";
  }
  function setFiltersDisabled(disabled) {
    ["replay-market", "replay-model", "replay-delay", "replay-speed"].forEach((id) => { $(id).disabled = disabled; });
  }
  function cached(cache, key, path) {
    if (!cache.has(key)) {
      const pending = replay.request(path).catch((error) => { cache.delete(key); throw error; });
      cache.set(key, pending);
    }
    return cache.get(key);
  }

  function renderAudit() {
    const count = replay.events.length;
    replay.auditIndex = Math.max(0, Math.min(replay.auditIndex, Math.max(0, count - 1)));
    const event = replay.events[replay.auditIndex];
    text("replay-kind", event ? `${event.kind} · ${event.stage}` : "No event selected");
    text("replay-time", event?.time_utc || "--");
    text("replay-payload", JSON.stringify(event || {}, null, 2));
    text("audit-count", `${count.toLocaleString()} append-only events`);
    $("audit-range").max = Math.max(0, count - 1);
    $("audit-range").value = replay.auditIndex;
    text("audit-position", count ? `${replay.auditIndex + 1} / ${count}` : "0 / 0");
  }

  function setFrameRow(row) {
    const target = $("replay-frame-row");
    target.replaceChildren();
    [
      row.date, value(row.close), percent(row.dd_10), value(row.sortino_20),
      value(row.sortino_60), tradeText(row.signal), tradeText(row.position),
      percent(row.strategy_return), `${value(row.transaction_cost * 10_000)} bps`,
      money(row.strategy_wealth_100), money(row.buy_hold_wealth_100),
    ].forEach((item) => {
      const cell = document.createElement("td");
      cell.textContent = item;
      target.append(cell);
    });
  }

  function renderFrame() {
    const data = replay.storyData;
    if (!data?.rows.length) return;
    replay.index = Math.max(replay.start, Math.min(replay.index, data.rows.length - 1));
    const storyRow = data.rows[replay.index];
    const row = { ...(replay.marketRows.get(storyRow.date) || {}), ...storyRow };
    const first = Math.max(0, replay.index - 259);
    const storyRows = new Map(
      data.rows.slice(first, replay.index + 1).map((item) => [item.date, item]),
    );
    const marketIndex = replay.marketIndexes.get(storyRow.date);
    const marketRows = marketIndex === undefined ? [] : replay.marketData.rows.slice(
      Math.max(0, marketIndex - 259), marketIndex + 1,
    );
    const visible = marketRows.length
      ? marketRows.map((item) => ({ ...item, ...(storyRows.get(item.date) || {}) }))
      : [...storyRows.values()].map((item) => ({
        ...(replay.marketRows.get(item.date) || {}), ...item,
      }));
    window.MonitorCharts.story({
      rows: visible, currentDate: row.date,
      candlesAvailable: replay.marketData.quality.candles_available,
      volumeAvailable: replay.marketData.quality.volume_available,
      frameDuration: speedIntervals[replay.speed],
    });
    text("replay-summary-date", row.date);
    text("replay-summary-close", value(row.close));
    text("replay-summary-return", percent(row.equity_simple));
    text("replay-summary-range", `${value(row.low)} – ${value(row.high)}`);
    text("replay-summary-volume", replay.marketData.quality.volume_available ? value(row.volume) : "Unavailable");
    const input = data.model === "fixed_jm"
      ? `DD-10 ${percent(row.dd_10)} · S20 ${value(row.sortino_20)} · S60 ${value(row.sortino_60)}`
      : `Total return ${percent(row.equity_simple)}`;
    const traded = Number(row.one_way_turnover) > 0;
    const position = traded
      ? `${row.position === 1 ? "Enter market" : "Move to cash"} · ${value(row.transaction_cost * 10_000)} bps`
      : `${tradeText(row.position)} · hold`;
    text("decision-input", input);
    text("decision-signal", tradeText(row.signal));
    text("decision-delay", `Applies at t+${data.protocol.effective_return_offset}`);
    text("decision-position", position);
    text("decision-outcome", `${percent(row.strategy_return)} · ${money(row.strategy_wealth_100)}`);
    text("replay-context", `${modelNames[data.model]} · past ${visible.length} market days · ${storyRows.size} ${storyRows.size === 1 ? "strategy day" : "strategy days"}`);
    text("replay-feature-context", data.model === "fixed_jm" ? "Inputs used by Fixed JM" : "Context only · HMM fits returns");
    setFrameRow(row);
    $("replay-range").min = replay.start;
    $("replay-range").max = data.rows.length - 1;
    $("replay-range").value = replay.index;
    text("replay-position", `${row.date} · ${replay.index - replay.start + 1} / ${data.rows.length - replay.start}`);
  }

  function unavailable(message) {
    replay.marketData = null;
    replay.storyData = null;
    replay.marketRows = new Map();
    replay.marketIndexes = new Map();
    window.MonitorCharts.story({ rows: [] });
    text("replay-chart-fallback", message);
    text("replay-feature-fallback", message);
    text("replay-source", message);
    text("replay-position", "Strategy data unavailable");
    ["replay-summary-date", "replay-summary-close", "replay-summary-return", "replay-summary-range", "replay-summary-volume"].forEach((id) => text(id, "--"));
    ["decision-input", "decision-signal", "decision-delay", "decision-position", "decision-outcome"].forEach((id) => text(id, "Unavailable"));
    const cell = document.createElement("td");
    cell.colSpan = 11;
    cell.textContent = message;
    $("replay-frame-row").replaceChildren(cell);
  }

  async function loadSelection() {
    pause();
    const version = ++replay.loadVersion;
    const { jobId, market, model, delay } = replay;
    const marketKey = `${jobId}:${market}`;
    const storyKey = `${marketKey}:${model}:${delay}`;
    text("replay-chart-fallback", "Loading verified market and strategy…");
    text("replay-feature-fallback", "Loading verified causal features…");
    try {
      const [marketData, storyData] = await Promise.all([
        cached(replay.marketCache, marketKey, `/api/jobs/${jobId}/markets/${market}/ohlcv`),
        cached(replay.storyCache, storyKey, `/api/jobs/${jobId}/markets/${market}/story?model=${model}&delay=${delay}`),
      ]);
      if (version !== replay.loadVersion) return;
      if (marketData.run_id !== storyData.run_id || storyData.market !== market) {
        throw new Error("Verified market and strategy identities do not match.");
      }
      replay.marketData = marketData;
      replay.storyData = storyData;
      replay.marketRows = new Map(marketData.rows.map((row) => [row.date, row]));
      replay.marketIndexes = new Map(marketData.rows.map((row, index) => [row.date, index]));
      replay.start = 0;
      replay.index = replay.start;
      const source = `${marketData.source.source_id} · ${marketData.source.provider}`;
      const quality = `${Number(marketData.quality.distinct_ohlc_rows || 0).toLocaleString()} distinct candles · ${marketData.quality.volume_available ? "volume available" : "no volume"}`;
      text("replay-source", `${marketNames[market]} · ${source} · strategy ${storyData.coverage.first_date} to ${storyData.coverage.last_date} · ${quality}`);
      renderFrame();
    } catch (error) {
      if (version !== replay.loadVersion) return;
      unavailable(error instanceof Error ? error.message : String(error));
    }
  }

  async function load(jobId) {
    pause();
    const version = ++replay.loadVersion;
    replay.jobId = jobId;
    setFiltersDisabled(true);
    if (!replay.eventCache.has(jobId)) {
      text("audit-position", "Loading events");
      const payload = await replay.request(`/api/jobs/${jobId}/events`);
      replay.eventCache.set(jobId, payload.events);
    }
    if (version !== replay.loadVersion) return;
    replay.events = replay.eventCache.get(jobId);
    replay.auditIndex = 0;
    renderAudit();
    const selectedJob = $("replay-job").selectedOptions[0];
    if (selectedJob?.dataset.status !== "succeeded") {
      unavailable("Market replay opens after the job completes and its artifact is verified.");
      return;
    }
    const verified = replay.events.find(
      (event) => event.kind === "artifact_verified" && event.stage === "verification",
    );
    if (
      verified?.payload.status !== "complete"
      || typeof verified.payload.run_id !== "string"
      || !verified.payload.run_id.startsWith("fixed-baselines-")
    ) {
      unavailable(
        "Runtime audit is available, but market replay requires a completed fixed-baseline artifact.",
      );
      return;
    }
    const eventMarket = replay.events.find((event) => marketNames[event.market])?.market;
    replay.market = eventMarket || "us";
    $("replay-market").value = replay.market;
    $("replay-model").value = replay.model;
    $("replay-delay").value = replay.delay;
    $("replay-speed").value = replay.speed;
    setFiltersDisabled(false);
    await loadSelection();
  }

  function startPlayback() {
    if (replay.timer || !replay.storyData?.rows.length) return;
    const last = replay.storyData.rows.length - 1;
    if (replay.index >= last) return;
    replay.timer = window.setInterval(() => {
      replay.index += 1;
      renderFrame();
      if (replay.index >= last) pause();
    }, speedIntervals[replay.speed]);
  }

  function act(action) {
    if (action === "pause") pause();
    if (action === "reset") { pause(); replay.index = replay.start; }
    if (action === "previous") { pause(); replay.index -= 1; }
    if (action === "next") { pause(); replay.index += 1; }
    if (action === "play") startPlayback();
    renderFrame();
  }

  function init(eventCache, request, showError) {
    replay.eventCache = eventCache;
    replay.request = request;
    $("replay-job").addEventListener("change", (event) => load(event.target.value).catch(showError));
    $("replay-market").addEventListener("change", (event) => { replay.market = event.target.value; loadSelection(); });
    $("replay-model").addEventListener("change", (event) => { replay.model = event.target.value; loadSelection(); });
    $("replay-delay").addEventListener("change", (event) => { replay.delay = Number(event.target.value); loadSelection(); });
    $("replay-speed").addEventListener("change", (event) => {
      const playing = Boolean(replay.timer);
      pause();
      replay.speed = Number(event.target.value);
      if (playing) startPlayback();
    });
    $("replay-range").addEventListener("input", (event) => { pause(); replay.index = Number(event.target.value); renderFrame(); });
    $("audit-range").addEventListener("input", (event) => { replay.auditIndex = Number(event.target.value); renderAudit(); });
    document.querySelectorAll("[data-replay]").forEach((control) => {
      control.addEventListener("click", () => act(control.dataset.replay));
    });
  }

  window.MonitorReplay = { init, load, pause };
})();
