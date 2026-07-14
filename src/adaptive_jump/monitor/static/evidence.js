"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const evidenceByRun = new Map();
  const outcomesByRun = new Map();

  function text(id, value) {
    $(id).textContent = String(value ?? "--");
  }

  function cells(row, values) {
    values.forEach((value) => {
      const cell = row.insertCell();
      cell.textContent = String(value ?? "--");
    });
  }

  function renderBoundaries(evidence) {
    const body = $("boundary-body");
    body.replaceChildren();
    evidence.boundaries.forEach((boundary) => {
      const row = body.insertRow();
      cells(row, [boundary.market?.toUpperCase(), boundary.model, boundary.delay, boundary.upper_candidate, `${boundary.selected_months} / ${boundary.total_months}`]);
      const gate = row.insertCell();
      gate.textContent = boundary.passed ? "Pass" : "Expand grid";
      gate.className = boundary.passed ? "positive" : "negative";
    });
    if (!evidence.boundaries.length) {
      const row = body.insertRow();
      const cell = row.insertCell();
      cell.colSpan = 6;
      cell.textContent = "No verified boundary rows";
    }
  }

  function renderReceipt(evidence) {
    const receipt = evidence.verification;
    text("evidence-run-title", evidence.title);
    text("evidence-lock", evidence.metrics_opened ? "Verified outcomes open" : "Outcomes locked by protocol gate");
    text("receipt-status", evidence.status);
    text("receipt-files", `${receipt.inventory_files ?? "--"} files`);
    text("receipt-boundaries", `${receipt.boundary_rows ?? evidence.boundaries.length} rows`);
    text("receipt-metrics", evidence.metrics_opened ? `${receipt.metric_rows} rows open` : "Locked");
    text("receipt-difference", Number.isFinite(receipt.maximum_metric_absolute_difference) ? receipt.maximum_metric_absolute_difference.toExponential(2) : "--");
    renderBoundaries(evidence);
  }

  function selectRun(runId) {
    document.querySelectorAll("#evidence-runs button").forEach((button) => button.classList.toggle("active", button.dataset.runId === runId));
    const evidence = evidenceByRun.get(runId);
    if (evidence) renderReceipt(evidence);
  }

  function renderSelectors(runs) {
    const root = $("evidence-runs");
    root.replaceChildren();
    runs.forEach((run) => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.runId = run.run_id;
      button.textContent = run.title;
      button.disabled = !run.available;
      button.addEventListener("click", () => selectRun(run.run_id));
      root.append(button);
    });
    text("evidence-count", `${runs.length} runs`);
  }

  function renderComparison(runs) {
    const body = $("comparison-body");
    body.replaceChildren();
    let verified = 0;
    const metrics = [];
    runs.forEach((run) => {
      const evidence = evidenceByRun.get(run.run_id);
      const outcome = outcomesByRun.get(run.run_id);
      const row = body.insertRow();
      if (!evidence) {
        cells(row, [run.title, run.available ? "Verification failed" : "Unavailable", "--", "Not opened"]);
        return;
      }
      verified += 1;
      if (outcome) metrics.push(...outcome.metrics);
      cells(row, [run.title, evidence.status, outcome ? `${outcome.metrics.length} verified rows` : "Locked", outcome?.claim?.conclusion || "No outcome claim opened"]);
    });
    text("compare-status", `${verified} / ${runs.length} verified`);
    $("compare-status").className = `status-pill ${verified === runs.length ? "running" : "pending"}`;
    window.MonitorCharts?.comparison(metrics);
    text("comparison-fallback", metrics.length ? "Chart rendering is unavailable; verified values remain in the evidence response." : "No verified outcome metrics are open.");
  }

  async function init(request, showError) {
    try {
      const runs = (await request("/api/evidence")).runs;
      renderSelectors(runs);
      for (const run of runs.filter((item) => item.available)) {
        try {
          const evidence = await request(`/api/evidence/${run.run_id}`);
          evidenceByRun.set(run.run_id, evidence);
          if (evidence.metrics_opened) {
            outcomesByRun.set(run.run_id, await request(`/api/evidence/${run.run_id}/outcome`));
          }
        } catch (error) {
          showError(error);
        }
      }
      const first = runs.find((run) => evidenceByRun.has(run.run_id));
      if (first) selectRun(first.run_id);
      renderComparison(runs);
    } catch (error) {
      showError(error);
    }
  }

  window.MonitorEvidence = { init };
})();
