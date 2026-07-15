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

  function highAreas(dates, states) {
    const areas = [];
    let start = null;
    states.forEach((state, index) => {
      if (state === 1 && start === null) start = index;
      if (start !== null && (state !== 1 || index === states.length - 1)) {
        const end = state === 1 && index === states.length - 1 ? index : index - 1;
        areas.push([{ xAxis: dates[start] }, { xAxis: dates[end] }]);
        start = null;
      }
    });
    return areas;
  }

  function finite(value) { return value !== null && value !== "" && Number.isFinite(Number(value)); }

  function market(view) {
    if (!view.rows.length) return empty("replay-market-chart");
    const dates = view.rows.map((row) => row.date);
    const volume = view.volumeAvailable;
    const laneAxis = volume ? 2 : 1;
    const grids = volume
      ? [{ left: 62, right: 42, top: 32, height: "50%" }, { left: 62, right: 42, top: "61%", height: "10%" }, { left: 62, right: 42, top: "78%", height: "13%" }]
      : [{ left: 62, right: 42, top: 32, height: "62%" }, { left: 62, right: 42, top: "78%", height: "13%" }];
    const categoryAxis = (gridIndex, labels) => ({
      type: "category", gridIndex, data: dates, boundaryGap: true,
      axisLabel: { color: colors.muted, show: labels, hideOverlap: true },
      axisLine: { lineStyle: { color: colors.line } }, axisTick: { show: false },
    });
    const xAxis = [categoryAxis(0, false)];
    const yAxis = [{ type: "value", scale: true, gridIndex: 0, axisLabel: { color: colors.muted }, splitLine: { lineStyle: { color: colors.line } } }];
    if (volume) {
      xAxis.push(categoryAxis(1, false));
      yAxis.push({ type: "value", gridIndex: 1, axisLabel: { color: colors.muted }, splitLine: { show: false } });
    }
    xAxis.push(categoryAxis(laneAxis, true));
    yAxis.push({ type: "category", gridIndex: laneAxis, data: [view.jmLabel, "HMM"], axisLabel: { color: colors.muted }, axisLine: { lineStyle: { color: colors.line } } });
    const candles = view.rows.map((row) => {
      const raw = [row.open, row.close, row.low, row.high];
      const values = raw.map(Number);
      return raw.every(finite) && new Set(values).size > 1 ? values : "-";
    });
    const series = [];
    if (view.candlesAvailable) series.push({
      name: "OHLC", type: "candlestick", xAxisIndex: 0, yAxisIndex: 0, data: candles,
      itemStyle: { color: colors.green, color0: colors.red, borderColor: colors.green, borderColor0: colors.red },
    });
    series.push({
      name: "Close", type: "line", xAxisIndex: 0, yAxisIndex: 0, showSymbol: false,
      connectNulls: false, data: view.rows.map((row) => finite(row.close) ? Number(row.close) : null),
      lineStyle: { color: colors.amber, width: 1.3 }, itemStyle: { color: colors.amber },
      markArea: { silent: true, itemStyle: { color: "rgba(255,107,107,0.10)" }, data: highAreas(dates, view.hmm) },
      markLine: { silent: true, symbol: ["none", "none"], label: { show: false }, lineStyle: { color: colors.text, type: "dashed" }, data: [{ xAxis: view.currentDate }] },
    });
    if (volume) series.push({
      name: "Volume", type: "bar", xAxisIndex: 1, yAxisIndex: 1,
      data: view.rows.map((row) => finite(row.volume) ? Number(row.volume) : null),
      itemStyle: { color: colors.muted },
    });
    const stateData = [];
    view.jm.forEach((state, index) => { if (state === 0 || state === 1) stateData.push([index, 0, state]); });
    view.hmm.forEach((state, index) => { if (state === 0 || state === 1) stateData.push([index, 1, state]); });
    series.push({
      name: "State", type: "heatmap", xAxisIndex: laneAxis, yAxisIndex: laneAxis, data: stateData,
      itemStyle: { color: (item) => item.value[2] === 1 ? colors.red : colors.blue, borderWidth: 0 },
    });
    render("replay-market-chart", {
      ...baseOption(), grid: grids, xAxis, yAxis, series, legend: { show: false },
      tooltip: { trigger: "axis", backgroundColor: "#171d23", borderColor: colors.line, textStyle: { color: colors.text } },
    });
  }

  function resize() {
    pending.forEach((option, id) => render(id, option));
    instances.forEach((instance) => instance.resize());
  }

  window.addEventListener("resize", resize);
  window.MonitorCharts = { resource, comparison, market, resize };
})();
