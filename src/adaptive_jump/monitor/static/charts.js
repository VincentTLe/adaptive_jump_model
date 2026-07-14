"use strict";

(() => {
  const instances = new Map();
  const pending = new Map();
  const colors = { green: "#27c187", blue: "#58a6ff", red: "#ff6b6b", amber: "#e7b84b", text: "#edf1f4", muted: "#94a0aa", line: "#2a323a" };
  const motion = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function chart(id) {
    if (!window.echarts) return null;
    if (!instances.has(id)) instances.set(id, window.echarts.init(document.getElementById(id), null, { renderer: "canvas" }));
    return instances.get(id);
  }

  function baseOption() {
    return {
      animationDuration: motion ? 250 : 0,
      backgroundColor: "transparent",
      textStyle: { color: colors.text },
      tooltip: { trigger: "axis", backgroundColor: "#171d23", borderColor: colors.line, textStyle: { color: colors.text } },
      grid: { left: 54, right: 54, top: 42, bottom: 38 },
      legend: { top: 10, textStyle: { color: colors.muted } },
    };
  }

  function render(id, option) {
    const root = document.getElementById(id);
    if (!root.offsetWidth || !root.offsetHeight) {
      pending.set(id, option);
      root.classList.remove("ready");
      return false;
    }
    const instance = chart(id);
    if (!instance) return false;
    try {
      instance.setOption(option, true);
      root.classList.add("ready");
      pending.delete(id);
      return true;
    } catch (_error) {
      document.getElementById(id).classList.remove("ready");
      return false;
    }
  }

  function empty(id) {
    pending.delete(id);
    document.getElementById(id).classList.remove("ready");
    instances.get(id)?.clear();
  }

  function resource(events) {
    const samples = events.filter((event) => event.kind === "resource_sample");
    if (!samples.length) return empty("resource-chart");
    render("resource-chart", {
      ...baseOption(),
      xAxis: { type: "category", data: samples.map((event) => event.sequence), axisLabel: { color: colors.muted }, axisLine: { lineStyle: { color: colors.line } } },
      yAxis: [
        { type: "value", name: "CPU %", axisLabel: { color: colors.muted }, splitLine: { lineStyle: { color: colors.line } } },
        { type: "value", name: "RSS MB", axisLabel: { color: colors.muted }, splitLine: { show: false } },
      ],
      series: [
        { name: "CPU %", type: "line", showSymbol: false, data: samples.map((event) => event.payload.cpu_percent), lineStyle: { color: colors.blue }, itemStyle: { color: colors.blue } },
        { name: "RSS MB", type: "line", yAxisIndex: 1, showSymbol: false, data: samples.map((event) => event.payload.rss_bytes / 1048576), lineStyle: { color: colors.green }, itemStyle: { color: colors.green } },
      ],
    });
  }

  function comparison(metrics) {
    const rows = metrics.filter((row) => Number(row.delay) === 1 && Number.isFinite(Number(row.sharpe)));
    if (!rows.length) return empty("comparison-chart");
    const markets = [...new Set(rows.map((row) => row.market))].sort();
    const models = ["buy_and_hold", "hmm", "fixed_jm"];
    const palette = [colors.amber, colors.red, colors.blue];
    render("comparison-chart", {
      ...baseOption(),
      xAxis: { type: "category", data: markets.map((market) => market.toUpperCase()), axisLabel: { color: colors.muted }, axisLine: { lineStyle: { color: colors.line } } },
      yAxis: { type: "value", name: "Sharpe", axisLabel: { color: colors.muted }, splitLine: { lineStyle: { color: colors.line } } },
      series: models.map((model, index) => ({
        name: model.replaceAll("_", " "), type: "bar", itemStyle: { color: palette[index] },
        data: markets.map((market) => {
          const row = rows.find((item) => item.market === market && item.model === model);
          return row ? Number(row.sharpe) : null;
        }),
      })),
    });
  }

  function resize() {
    pending.forEach((option, id) => render(id, option));
    instances.forEach((instance) => instance.resize());
  }

  window.addEventListener("resize", resize);
  window.MonitorCharts = { resource, comparison, resize };
})();
