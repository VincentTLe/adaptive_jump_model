"use strict";

const $ = (id) => document.getElementById(id);
const terminalStatuses = new Set(["interrupted", "canceled", "succeeded", "failed"]);
const state = {
  session: null,
  studies: [],
  jobs: [],
  events: new Map(),
  currentJobId: null,
  source: null,
};
function setText(id, value) {
  $(id).textContent = value;
}
function formatNumber(value, digits = 3) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "--";
}
function formatDuration(seconds) {
  const safe = Math.max(0, Math.floor(Number(seconds) || 0));
  const hours = String(Math.floor(safe / 3600)).padStart(2, "0");
  const minutes = String(Math.floor((safe % 3600) / 60)).padStart(2, "0");
  return `${hours}:${minutes}:${String(safe % 60).padStart(2, "0")}`;
}
function showError(error) {
  const banner = $("error-banner");
  banner.textContent = error instanceof Error ? error.message : String(error);
  banner.hidden = false;
}
async function request(path, options = {}) {
  const headers = { Accept: "application/json", ...(options.headers || {}) };
  if (options.body) headers["Content-Type"] = "application/json";
  if (options.method && options.method !== "GET") {
    headers["X-CSRF-Token"] = state.session.csrf_token;
  }
  const response = await fetch(path, { ...options, headers, credentials: "same-origin" });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}
function activateView(name) {
  document.querySelectorAll("[data-view]").forEach((button) => {
    const active = button.dataset.view === name;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  document.querySelectorAll(".view").forEach((view) => {
    const active = view.id === `view-${name}`;
    view.classList.toggle("active", active);
    view.hidden = !active;
  });
  window.requestAnimationFrame(() => window.MonitorCharts?.resize());
  if (name !== "replay") window.MonitorReplay?.pause();
}
function button(label, title, action) {
  const control = document.createElement("button");
  control.type = "button";
  control.textContent = label;
  control.title = title;
  control.setAttribute("aria-label", title);
  control.addEventListener("click", action);
  return control;
}
async function mutate(path, body) {
  const options = { method: "POST" };
  if (body !== undefined) options.body = JSON.stringify(body);
  await request(path, options);
  await refreshJobs();
}
function renderStudyOptions() {
  const select = $("study-select");
  select.replaceChildren();
  if (!state.studies.length) {
    const option = document.createElement("option");
    option.textContent = "No FROZEN studies available";
    select.append(option);
  }
  state.studies.forEach((study) => {
    const option = document.createElement("option");
    option.value = study.study_id;
    option.textContent = `${study.study_id} (${study.cli_study})`;
    select.append(option);
  });
  const enabled = state.session.role === "owner" && state.studies.length > 0;
  select.disabled = !enabled;
  $("enqueue-form").querySelector("button").disabled = !enabled;
}
function renderQueue() {
  const list = $("queue-list");
  const queued = state.jobs
    .filter((job) => job.status === "queued")
    .sort((a, b) => a.queue_position - b.queue_position);
  const queuedIds = queued.map((job) => job.job_id);
  list.replaceChildren();
  setText("queue-count", `${queued.length} queued`);
  if (!state.jobs.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "No jobs recorded";
    list.append(empty);
    return;
  }
  const ordered = [...queued, ...state.jobs.filter((job) => job.status !== "queued").reverse()];
  ordered.forEach((job) => {
    const item = document.createElement("li");
    const name = document.createElement("div");
    name.className = "job-name";
    const title = document.createElement("strong");
    title.textContent = job.study_id;
    const identity = document.createElement("span");
    identity.textContent = `${job.job_id.slice(0, 10)} · attempt ${job.attempts}`;
    name.append(title, identity);
    const status = document.createElement("span");
    status.className = `job-status ${job.status === "failed" ? "negative" : ""}`;
    status.textContent = job.status.replaceAll("_", " ");
    const actions = document.createElement("div");
    actions.className = "job-actions";
    actions.append(button("Open", "Open job in Live view", () => selectJob(job.job_id, true)));
    if (state.session.role === "owner" && job.status === "queued") {
      const index = queuedIds.indexOf(job.job_id);
      const move = (offset) => {
        const ids = [...queuedIds];
        [ids[index], ids[index + offset]] = [ids[index + offset], ids[index]];
        mutate("/api/jobs/reorder", { job_ids: ids }).catch(showError);
      };
      const up = button("↑", "Move job earlier", () => move(-1));
      const down = button("↓", "Move job later", () => move(1));
      up.disabled = index === 0;
      down.disabled = index === queuedIds.length - 1;
      actions.append(up, down);
    }
    if (state.session.role === "owner" && ["queued", "running"].includes(job.status)) {
      actions.append(button("Cancel", "Cancel job", () => {
        if (window.confirm(`Cancel ${job.study_id}?`)) {
          mutate(`/api/jobs/${job.job_id}/cancel`).catch(showError);
        }
      }));
    }
    if (state.session.role === "owner" && job.status === "interrupted") {
      actions.append(button("Resume", "Resume interrupted job", () => {
        mutate(`/api/jobs/${job.job_id}/resume`).catch(showError);
      }));
    }
    item.append(name, status, actions);
    list.append(item);
  });
}
function renderReplayJobs() {
  const select = $("replay-job");
  const selected = select.value;
  select.replaceChildren();
  state.jobs.slice().reverse().forEach((job) => {
    const option = document.createElement("option");
    option.value = job.job_id;
    option.dataset.status = job.status;
    option.textContent = `${job.study_id} · ${job.status}`;
    select.append(option);
  });
  if (state.jobs.some((job) => job.job_id === selected)) select.value = selected;
}
function statusPill(job) {
  const pill = $("live-status");
  const value = job ? job.status : "idle";
  pill.textContent = value.replaceAll("_", " ");
  pill.className = `status-pill ${value === "running" ? "running" : value === "failed" ? "failed" : "idle"}`;
}
function renderStages(events) {
  const completed = new Set(events.filter((event) => event.kind === "stage_completed").map((event) => event.stage));
  const latest = events.slice().reverse().find((event) => event.stage !== "worker");
  document.querySelectorAll("#stage-track [data-stage]").forEach((item) => {
    item.className = completed.has(item.dataset.stage)
      ? "complete"
      : latest && latest.stage === item.dataset.stage ? "active" : "pending";
  });
}
function renderDecisions(events) {
  const body = $("state-body");
  body.replaceChildren();
  const rows = new Map();
  events.filter((event) => ["terminal_state", "selected_signal"].includes(event.kind)).forEach((event) => {
    const states = event.kind === "selected_signal"
      ? [{ candidate: event.payload.selected_candidate, state: 1 - event.payload.signal }]
      : event.payload.states || [{ candidate: "--", state: event.payload.state }];
    states.forEach((entry) => rows.set(`${event.market}:${event.model}:${entry.candidate}:${event.delay}`, { event, entry }));
  });
  if (!rows.size) {
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 5;
    cell.textContent = "No decision events";
    return;
  }
  [...rows.values()].slice(-30).forEach(({ event, entry }) => {
    const row = body.insertRow();
    const signal = Number(entry.state) === 0 ? "Risk asset" : "Cash";
    const scheduled = event.kind === "selected_signal"
      ? `t+${event.payload.effective_return_offset}` : "--";
    [event.model || event.stage, entry.candidate, entry.state, signal, scheduled]
      .forEach((value) => { row.insertCell().textContent = String(value); });
  });
}
function renderJournal(events) {
  const list = $("event-list");
  list.replaceChildren();
  events.slice(-60).reverse().forEach((event) => {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = `${event.sequence} · ${event.kind}`;
    const context = [event.stage, event.market, event.model, event.date].filter(Boolean).join(" · ");
    item.append(title, document.createTextNode(context || event.time_utc));
    list.append(item);
  });
  if (!events.length) {
    const empty = document.createElement("li");
    empty.textContent = "Waiting for the first runtime event.";
    list.append(empty);
  }
  setText("event-count", `${events.length} events`);
}
function renderLive(job, events) {
  statusPill(job);
  setText("summary-study", job ? job.study_id : "No active job");
  const latest = events.at(-1);
  const progress = events.slice().reverse().find((event) => event.completed !== null);
  const resource = events.slice().reverse().find((event) => event.kind === "resource_sample");
  setText("summary-stage", latest ? latest.stage : "Waiting");
  setText("summary-progress", progress ? `${progress.completed} / ${progress.total}` : "0 / 0");
  setText("summary-elapsed", formatDuration(latest?.elapsed_seconds));
  setText("summary-memory", resource ? `${formatNumber(resource.payload.rss_bytes / 1048576, 1)} MB` : "0 MB");
  const remaining = progress?.completed > 0
    ? (latest.elapsed_seconds / progress.completed) * (progress.total - progress.completed) : null;
  setText("eta", remaining === null ? "ETA unavailable" : `ETA ${formatDuration(remaining)}`);
  const featured = events.slice().reverse().find((event) => event.payload.features);
  const features = featured?.payload.features || {};
  setText("feature-date", featured?.date || "No model date");
  setText("feature-dd", formatNumber(features.dd_10));
  setText("feature-s20", formatNumber(features.sortino_20));
  setText("feature-s60", formatNumber(features.sortino_60));
  setText("feature-return", formatNumber(features.excess_return, 5));
  const selected = events.slice().reverse().find((event) => event.kind === "selection_checkpoint");
  setText("selected-candidate", selected ? `${selected.model} · ${selected.payload.selected_candidate}` : "Candidate unavailable");
  setText("resource-latest", resource ? `CPU ${formatNumber(resource.payload.cpu_percent, 1)}%` : "No samples");
  setText("resource-fallback", resource ? `Latest sample: CPU ${formatNumber(resource.payload.cpu_percent, 1)}%, RSS ${formatNumber(resource.payload.rss_bytes / 1048576, 1)} MB.` : "CPU and memory samples will appear here.");
  window.MonitorCharts?.resource(events);
  renderStages(events);
  renderDecisions(events);
  window.MonitorDiagnostics?.render(events);
  renderJournal(events);
}
function closeStream() {
  if (state.source) state.source.close();
  state.source = null;
}
function connectStream(job, after) {
  closeStream();
  if (!job || !["running", "cancel_requested"].includes(job.status)) return;
  const source = new EventSource(`/api/jobs/${job.job_id}/stream?after_sequence=${after}`);
  state.source = source;
  source.onopen = () => {
    $("connection").className = "status-dot online";
    setText("connection", "Live");
  };
  source.onerror = () => {
    $("connection").className = "status-dot offline";
    setText("connection", "Reconnecting");
  };
  source.addEventListener("research_event", (message) => {
    const event = JSON.parse(message.data);
    const events = state.events.get(job.job_id) || [];
    if (!events.some((item) => item.sequence === event.sequence)) events.push(event);
    state.events.set(job.job_id, events.sort((a, b) => a.sequence - b.sequence));
    if (state.currentJobId === job.job_id) renderLive(job, events);
  });
  source.addEventListener("stream_complete", () => {
    closeStream();
    refreshJobs().catch(showError);
  });
}
async function selectJob(jobId, openLive = false) {
  const job = state.jobs.find((item) => item.job_id === jobId);
  if (!job) return;
  state.currentJobId = jobId;
  const payload = await request(`/api/jobs/${jobId}/events`);
  state.events.set(jobId, payload.events);
  renderLive(job, payload.events);
  connectStream(job, payload.events.at(-1)?.sequence || 0);
  if (openLive) activateView("live");
}
async function refreshJobs() {
  state.jobs = (await request("/api/jobs")).jobs;
  renderQueue();
  renderReplayJobs();
  const active = state.jobs.find((job) => ["running", "cancel_requested"].includes(job.status));
  const current = state.jobs.find((job) => job.job_id === state.currentJobId);
  const target = active || current || state.jobs.at(-1);
  if (target && target.job_id !== state.currentJobId) await selectJob(target.job_id);
  else if (target) {
    renderLive(target, state.events.get(target.job_id) || []);
    if (terminalStatuses.has(target.status)) closeStream();
  } else renderLive(null, []);
}
async function init() {
  document.querySelectorAll("[data-view]").forEach((control) => {
    control.addEventListener("click", () => activateView(control.dataset.view));
  });
  window.MonitorReplay.init(state.events, request, showError);
  $("enqueue-form").addEventListener("submit", (event) => {
    event.preventDefault();
    mutate("/api/jobs", { study_id: $("study-select").value }).catch(showError);
  });
  try {
    const [session, studies] = await Promise.all([request("/api/session"), request("/api/studies")]);
    state.session = session;
    state.studies = studies.queueable;
    setText("identity", `${session.email} · ${session.role}`);
    $("connection").className = "status-dot online";
    setText("connection", "Ready");
    renderStudyOptions();
    await refreshJobs();
    window.MonitorEvidence?.init(request, showError);
    if (state.jobs.length) {
      window.MonitorReplay.load($("replay-job").value).catch(showError);
    }
    window.setInterval(() => refreshJobs().catch(showError), 3000);
  } catch (error) {
    $("connection").className = "status-dot offline";
    setText("connection", "Unavailable");
    showError(error);
  }
}

window.addEventListener("beforeunload", closeStream);
init();
