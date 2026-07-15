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

  function finite(value) { return value !== null && value !== "" && Number.isFinite(Number(value)); }

  function story(view) {
    if (!view.rows.length) {
      empty("replay-market-chart");
      empty("replay-feature-chart");
      return;
    }
    const dates = view.rows.map((row) => row.date);
    const storyPoints = view.rows.filter((row) => finite(row.strategy_wealth_100)).length;
    const showStorySymbols = storyPoints < 3;
    const volume = view.volumeAvailable;
    const wealthAxis = volume ? 2 : 1;
    const laneAxis = volume ? 3 : 2;
    const grids = volume
      ? [{ left: 68, right: 42, top: 34, height: "35%" }, { left: 68, right: 42, top: "44%", height: "8%" }, { left: 68, right: 42, top: "58%", height: "18%" }, { left: 68, right: 42, top: "82%", height: "11%" }]
      : [{ left: 68, right: 42, top: 34, height: "44%" }, { left: 68, right: 42, top: "57%", height: "19%" }, { left: 68, right: 42, top: "82%", height: "11%" }];
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
    xAxis.push(categoryAxis(wealthAxis, false));
    yAxis.push({ type: "value", name: "Wealth", scale: true, gridIndex: wealthAxis, axisLabel: { color: colors.muted }, splitLine: { lineStyle: { color: colors.line } } });
    xAxis.push(categoryAxis(laneAxis, true));
    yAxis.push({ type: "category", gridIndex: laneAxis, data: ["Position", "Signal"], axisLabel: { color: colors.muted }, axisLine: { lineStyle: { color: colors.line } } });
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
      markLine: { silent: true, symbol: ["none", "none"], label: { show: false }, lineStyle: { color: colors.text, type: "dashed" }, data: [{ xAxis: view.currentDate }] },
    });
    if (volume) series.push({
      name: "Volume", type: "bar", xAxisIndex: 1, yAxisIndex: 1,
      data: view.rows.map((row) => finite(row.volume) ? Number(row.volume) : null),
      itemStyle: { color: colors.muted },
    });
    series.push(
      {
        name: "Strategy", type: "line", xAxisIndex: wealthAxis, yAxisIndex: wealthAxis,
        showSymbol: showStorySymbols, connectNulls: false,
        data: view.rows.map((row) => finite(row.strategy_wealth_100) ? Number(row.strategy_wealth_100) : null),
        lineStyle: { color: colors.green, width: 2 }, itemStyle: { color: colors.green },
      },
      {
        name: "Buy & hold", type: "line", xAxisIndex: wealthAxis, yAxisIndex: wealthAxis,
        showSymbol: showStorySymbols, connectNulls: false,
        data: view.rows.map((row) => finite(row.buy_hold_wealth_100) ? Number(row.buy_hold_wealth_100) : null),
        lineStyle: { color: colors.amber, width: 1.5 }, itemStyle: { color: colors.amber },
      },
    );
    const stateData = [];
    view.rows.forEach((row, index) => {
      if (row.position === 0 || row.position === 1) stateData.push([index, 0, row.position]);
      if (row.signal === 0 || row.signal === 1) stateData.push([index, 1, row.signal]);
    });
    series.push({
      name: "State", type: "heatmap", xAxisIndex: laneAxis, yAxisIndex: laneAxis, data: stateData,
      itemStyle: { color: (item) => item.value[2] === 1 ? colors.green : colors.blue, borderWidth: 0 },
    });
    render("replay-market-chart", {
      ...baseOption(), grid: grids, xAxis, yAxis, series, legend: { show: false },
      tooltip: { trigger: "axis", backgroundColor: "#171d23", borderColor: colors.line, textStyle: { color: colors.text } },
    });
    render("replay-feature-chart", {
      ...baseOption(),
      grid: { left: 68, right: 55, top: 48, bottom: 38 },
      xAxis: { type: "category", data: dates, boundaryGap: false, axisLabel: { color: colors.muted, hideOverlap: true }, axisLine: { lineStyle: { color: colors.line } } },
      yAxis: [
        { type: "value", name: "DD-10", axisLabel: { color: colors.muted, formatter: (item) => `${(item * 100).toFixed(1)}%` }, splitLine: { lineStyle: { color: colors.line } } },
        { type: "value", name: "Sortino", axisLabel: { color: colors.muted }, splitLine: { show: false } },
      ],
      series: [
        { name: "DD-10", type: "line", showSymbol: showStorySymbols, connectNulls: false, data: view.rows.map((row) => row.dd_10), lineStyle: { color: colors.red }, itemStyle: { color: colors.red } },
        { name: "Sortino-20", type: "line", yAxisIndex: 1, showSymbol: showStorySymbols, connectNulls: false, data: view.rows.map((row) => row.sortino_20), lineStyle: { color: colors.blue }, itemStyle: { color: colors.blue } },
        { name: "Sortino-60", type: "line", yAxisIndex: 1, showSymbol: showStorySymbols, connectNulls: false, data: view.rows.map((row) => row.sortino_60), lineStyle: { color: colors.amber }, itemStyle: { color: colors.amber }, markLine: { silent: true, symbol: ["none", "none"], label: { show: false }, data: [{ yAxis: 0 }, { xAxis: view.currentDate }] } },
      ],
    });
  }

  function resize() {
    pending.forEach((option, id) => render(id, option));
    instances.forEach((instance) => instance.resize());
  }

  window.addEventListener("resize", resize);
  window.MonitorCharts = { resource, comparison, story, resize };
})();
