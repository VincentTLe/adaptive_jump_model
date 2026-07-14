"use strict";

(() => {
  const $ = (id) => document.getElementById(id);

  function emptyRow(body, columns, message) {
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = columns;
    cell.textContent = message;
  }

  function addCells(row, values) {
    values.forEach((value) => {
      row.insertCell().textContent = String(value ?? "--");
    });
  }

  function number(value, digits = 3) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toFixed(digits) : "--";
  }

  function renderSurface(events) {
    const body = $("cv-body");
    body.replaceChildren();
    const event = events.slice().reverse().find((item) => item.kind === "selection_checkpoint");
    if (!event) {
      emptyRow(body, 5, "No validation surface");
      $("selection-context").textContent = "No selection event";
      return;
    }
    const selected = Number(event.payload.selected_candidate);
    event.payload.cv_surface.forEach((candidate) => {
      const row = body.insertRow();
      const chosen = Number(candidate.candidate) === selected;
      if (chosen) row.className = "selected-row";
      addCells(row, [
        candidate.candidate,
        number(candidate.sharpe),
        candidate.valid_returns,
        candidate.eligible ? "Yes" : "No",
        chosen ? "Selected" : "",
      ]);
    });
    const context = [event.market?.toUpperCase(), event.model, `delay ${event.delay}`, event.date]
      .filter(Boolean).join(" · ");
    $("selection-context").textContent = context;
  }

  function renderBoundaries(events) {
    const body = $("live-boundary-body");
    body.replaceChildren();
    const latest = new Map();
    events.filter((event) => event.kind === "boundary_diagnostic").forEach((event) => {
      latest.set(`${event.market}:${event.model}:${event.delay}`, event);
    });
    latest.forEach((event) => {
      const row = body.insertRow();
      addCells(row, [
        `${event.market?.toUpperCase() || "--"} / ${event.model || "--"}`,
        event.delay,
        `${event.payload.selected_months} / ${event.payload.total_months}`,
      ]);
      const gate = row.insertCell();
      gate.textContent = event.payload.passed ? "Pass" : "Expand grid";
      gate.className = event.payload.passed ? "positive" : "negative";
    });
    if (!latest.size) emptyRow(body, 4, "No boundary events");
  }

  function render(events) {
    renderSurface(events);
    renderBoundaries(events);
  }

  window.MonitorDiagnostics = { render };
})();
