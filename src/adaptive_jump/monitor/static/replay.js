"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const marketNames = { us: "United States", de: "Germany", jp: "Japan" };
  const number = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
  const replay = {
    eventCache: null, marketCache: new Map(), request: null, loadVersion: 0,
    events: [], jobId: null, market: "us", marketData: null, models: null,
    candidate: null, start: 0, index: 0, auditIndex: 0, timer: null,
  };

  function text(id, value) { $(id).textContent = value; }
  function pause() { window.clearInterval(replay.timer); replay.timer = null; }
  function stateText(state) {
    if (state === 0) return "0 · low";
    if (state === 1) return "1 · high";
    return "Unavailable";
  }
  function value(numberValue) {
    if (numberValue === null || numberValue === "") return "--";
    const numeric = Number(numberValue);
    return Number.isFinite(numeric) ? number.format(numeric) : "--";
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

  function modelData(events, market) {
    const hmm = new Map();
    const jm = new Map();
    const candidates = new Set();
    events.forEach((event) => {
      if (event.kind !== "terminal_state" || event.market !== market || !event.date) return;
      if (event.model === "hmm" && [0, 1].includes(event.payload.state)) {
        hmm.set(event.date, event.payload.state);
      }
      if (event.model === "fixed_jm" && Array.isArray(event.payload.states)) {
        const states = new Map();
        event.payload.states.forEach((row) => {
          const candidate = Number(row.candidate);
          if (Number.isFinite(candidate) && [0, 1].includes(row.state)) {
            candidates.add(candidate);
            states.set(candidate, row.state);
          }
        });
        jm.set(event.date, states);
      }
    });
    const selected = events.slice().reverse().find((event) =>
      event.kind === "selected_signal" && event.market === market
      && event.model === "fixed_jm" && event.delay === 1);
    return { hmm, jm, candidates: [...candidates].sort((a, b) => a - b), preferred: Number(selected?.payload.selected_candidate) };
  }

  function renderCandidates() {
    const select = $("replay-candidate");
    select.replaceChildren();
    const candidates = replay.models?.candidates || [];
    if (!candidates.length) {
      const option = document.createElement("option");
      option.textContent = "Unavailable";
      select.append(option);
      select.disabled = true;
      replay.candidate = null;
      return;
    }
    candidates.forEach((candidate) => {
      const option = document.createElement("option");
      option.value = candidate;
      option.textContent = `λ = ${candidate}`;
      select.append(option);
    });
    replay.candidate = candidates.includes(replay.models.preferred)
      ? replay.models.preferred : candidates[0];
    select.value = replay.candidate;
    select.disabled = false;
  }

  function setFrameRow(row, hmm, jm, volumeAvailable) {
    const target = $("replay-frame-row");
    target.replaceChildren();
    [row.date, row.open, row.high, row.low, row.close, volumeAvailable ? row.volume : null, stateText(hmm), stateText(jm)]
      .forEach((item, index) => {
        const cell = document.createElement("td");
        cell.textContent = index === 0 || index > 5 ? item : value(item);
        target.append(cell);
      });
  }

  function renderFrame() {
    const data = replay.marketData;
    if (!data?.rows.length) return;
    replay.index = Math.max(replay.start, Math.min(replay.index, data.rows.length - 1));
    const row = data.rows[replay.index];
    const first = Math.max(0, replay.index - 259);
    const visible = data.rows.slice(first, replay.index + 1);
    const hmm = replay.models.hmm.get(row.date);
    const jm = replay.models.jm.get(row.date)?.get(replay.candidate);
    window.MonitorCharts.market({
      rows: visible, currentDate: row.date,
      candlesAvailable: data.quality.candles_available,
      volumeAvailable: data.quality.volume_available,
      hmm: visible.map((item) => replay.models.hmm.get(item.date) ?? null),
      jm: visible.map((item) => replay.models.jm.get(item.date)?.get(replay.candidate) ?? null),
      jmLabel: replay.candidate === null ? "Fixed JM" : `Fixed JM λ=${replay.candidate}`,
    });
    text("replay-summary-date", row.date);
    text("replay-summary-hmm", stateText(hmm));
    text("replay-summary-jm", stateText(jm));
    text("replay-context", `Trailing ${visible.length} rows · HMM shading · JM λ held fixed`);
    setFrameRow(row, hmm, jm, data.quality.volume_available);
    $("replay-range").min = replay.start;
    $("replay-range").max = data.rows.length - 1;
    $("replay-range").value = replay.index;
    text("replay-position", `${row.date} · ${replay.index - replay.start + 1} / ${data.rows.length - replay.start}`);
  }

  function unavailable(message) {
    replay.marketData = null;
    replay.models = null;
    renderCandidates();
    window.MonitorCharts.market({ rows: [] });
    text("replay-chart-fallback", message);
    text("replay-source", message);
    text("replay-position", "Market data unavailable");
    ["replay-summary-source", "replay-summary-date", "replay-summary-quality"].forEach((id) => text(id, "--"));
    text("replay-summary-hmm", "Unavailable");
    text("replay-summary-jm", "Unavailable");
    const cell = document.createElement("td");
    cell.colSpan = 8;
    cell.textContent = message;
    $("replay-frame-row").replaceChildren(cell);
  }

  async function loadMarket(market) {
    pause();
    replay.market = market;
    const jobId = replay.jobId;
    const key = `${replay.jobId}:${market}`;
    text("replay-chart-fallback", "Loading verified market source…");
    try {
      if (!replay.marketCache.has(key)) {
        const path = `/api/jobs/${jobId}/markets/${market}/ohlcv`;
        replay.marketCache.set(key, await replay.request(path));
      }
      const data = replay.marketCache.get(key);
      if (replay.jobId !== jobId || replay.market !== market) return;
      replay.marketData = data;
      replay.models = modelData(replay.events, market);
      renderCandidates();
      const modelDates = [...replay.models.hmm.keys(), ...replay.models.jm.keys()].sort();
      const firstModel = modelDates[0];
      const firstIndex = firstModel ? data.rows.findIndex((row) => row.date >= firstModel) : 0;
      replay.start = Math.max(0, (firstIndex < 0 ? 0 : firstIndex) - 60);
      replay.index = replay.start;
      const source = `${data.source.source_id} · ${data.source.provider}`;
      text("replay-source", `${marketNames[market]} · ${source} · ${data.coverage.first_date} to ${data.coverage.last_date}`);
      text("replay-summary-source", source);
      text("replay-summary-quality", `${Number(data.quality.distinct_ohlc_rows || 0).toLocaleString()} distinct candles · ${data.quality.volume_available ? "volume available" : "no volume"}`);
      renderFrame();
    } catch (error) {
      if (replay.jobId !== jobId || replay.market !== market) return;
      unavailable(error instanceof Error ? error.message : String(error));
    }
  }

  async function load(jobId) {
    pause();
    const version = ++replay.loadVersion;
    replay.jobId = jobId;
    $("replay-market").disabled = true;
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
    const eventMarket = replay.events.find((event) => marketNames[event.market])?.market;
    replay.market = eventMarket || "us";
    $("replay-market").value = replay.market;
    $("replay-market").disabled = false;
    await loadMarket(replay.market);
  }

  function act(action) {
    if (action === "pause") pause();
    if (action === "reset") { pause(); replay.index = replay.start; }
    if (action === "previous") { pause(); replay.index -= 1; }
    if (action === "next") { pause(); replay.index += 1; }
    if (action === "play" && !replay.timer && replay.marketData?.rows.length) {
      replay.timer = window.setInterval(() => {
        if (replay.index >= replay.marketData.rows.length - 1) pause();
        else replay.index = Math.min(replay.index + 5, replay.marketData.rows.length - 1);
        renderFrame();
      }, 120);
    }
    renderFrame();
  }

  function init(eventCache, request, showError) {
    replay.eventCache = eventCache;
    replay.request = request;
    $("replay-job").addEventListener("change", (event) => load(event.target.value).catch(showError));
    $("replay-market").addEventListener("change", (event) => loadMarket(event.target.value));
    $("replay-candidate").addEventListener("change", (event) => { replay.candidate = Number(event.target.value); renderFrame(); });
    $("replay-range").addEventListener("input", (event) => { pause(); replay.index = Number(event.target.value); renderFrame(); });
    $("audit-range").addEventListener("input", (event) => { replay.auditIndex = Number(event.target.value); renderAudit(); });
    document.querySelectorAll("[data-replay]").forEach((control) => {
      control.addEventListener("click", () => act(control.dataset.replay));
    });
  }

  window.MonitorReplay = { init, load, pause };
})();
