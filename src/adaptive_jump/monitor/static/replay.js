"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const replay = {
    eventCache: null,
    events: [],
    index: 0,
    request: null,
    showError: null,
    timer: null,
  };

  function text(id, value) {
    $(id).textContent = value;
  }

  function pause() {
    window.clearInterval(replay.timer);
    replay.timer = null;
  }

  function render() {
    const count = replay.events.length;
    replay.index = Math.max(0, Math.min(replay.index, Math.max(0, count - 1)));
    const event = replay.events[replay.index];
    text("replay-kind", event ? `${event.kind} · ${event.stage}` : "No event selected");
    text("replay-time", event?.time_utc || "--");
    text("replay-payload", JSON.stringify(event || {}, null, 2));
    $("replay-range").max = Math.max(0, count - 1);
    $("replay-range").value = replay.index;
    text("replay-position", count ? `${replay.index + 1} / ${count}` : "0 / 0");
  }

  async function load(jobId) {
    pause();
    if (!replay.eventCache.has(jobId)) {
      text("replay-position", "Loading events");
      const payload = await replay.request(`/api/jobs/${jobId}/events`);
      replay.eventCache.set(jobId, payload.events);
    }
    replay.events = replay.eventCache.get(jobId);
    replay.index = 0;
    render();
  }

  function act(action) {
    if (action === "pause") pause();
    if (action === "reset") { pause(); replay.index = 0; }
    if (action === "previous") { pause(); replay.index -= 1; }
    if (action === "next") { pause(); replay.index += 1; }
    if (action === "play" && !replay.timer && replay.events.length) {
      replay.timer = window.setInterval(() => {
        if (replay.index >= replay.events.length - 1) pause();
        else { replay.index += 1; render(); }
      }, 900);
    }
    render();
  }

  function init(eventCache, request, showError) {
    replay.eventCache = eventCache;
    replay.request = request;
    replay.showError = showError;
    $("replay-job").addEventListener("change", (event) => {
      load(event.target.value).catch(replay.showError);
    });
    $("replay-range").addEventListener("input", (event) => {
      pause();
      replay.index = Number(event.target.value);
      render();
    });
    document.querySelectorAll("[data-replay]").forEach((control) => {
      control.addEventListener("click", () => act(control.dataset.replay));
    });
  }

  window.MonitorReplay = { init, load, pause };
})();
