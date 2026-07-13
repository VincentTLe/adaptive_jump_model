"use strict";

document.documentElement.classList.add("js");

(function () {
  const palette = Object.freeze({
    cash: "#63d59a",
    low: "#75b8ff",
    high: "#ff8585",
    uncertainty: "#f0bd64",
    unavailable: "#aab3bf",
  });

  function money(value) {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  }

  function percent(value, digits = 1) {
    return `${(value * 100).toFixed(digits)}%`;
  }

  function updateRangeOutput(input) {
    const targetId = input.dataset.output;
    if (!targetId) return;
    const output = document.getElementById(targetId);
    if (!output) return;
    const value = Number(input.value);
    const scale = Number(input.dataset.scale || 1);
    const digits = Number(input.dataset.digits || 0);
    const suffix = input.dataset.singular
      ? (value === 1 ? input.dataset.singular : input.dataset.plural)
      : input.dataset.suffix || "";
    output.value = `${input.dataset.prefix || ""}${(value * scale).toFixed(digits)}${suffix}`;
  }

  function resetContainer(button) {
    const container = button.closest(".lab, .quiz");
    if (!container) return;
    const form = container.querySelector("form");
    if (form) {
      form.reset();
      form.querySelectorAll("input").forEach((input) => {
        updateRangeOutput(input);
        input.dispatchEvent(new Event("input", { bubbles: true }));
      });
    }
    container.querySelectorAll("[data-choice]").forEach((choice) => {
      choice.setAttribute("aria-pressed", "false");
    });
    const feedback = container.querySelector(".quiz-feedback");
    if (feedback) {
      feedback.textContent = "Choose an answer, then check the explanation.";
      feedback.className = "quiz-feedback";
    }
    container.dispatchEvent(new CustomEvent("course:reset", { bubbles: true }));
  }

  function chooseAnswer(button) {
    const quiz = button.closest(".quiz");
    if (!quiz) return;
    quiz.querySelectorAll("[data-choice]").forEach((choice) => {
      choice.setAttribute("aria-pressed", String(choice === button));
    });
    const feedback = quiz.querySelector(".quiz-feedback");
    if (!feedback) return;
    const correct = button.dataset.correct === "true";
    feedback.textContent = correct
      ? button.dataset.feedbackCorrect || "Correct."
      : button.dataset.feedbackIncorrect || "Not yet. Review the worked example and try again.";
    feedback.className = `quiz-feedback ${correct ? "correct" : "incorrect"}`;
  }

  function motionAllowed() {
    return !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  function mountStepper(root, options) {
    if (!root || !Number.isInteger(options.frameCount) || options.frameCount < 1) return null;
    const initialFrame = Math.max(0, Math.min(options.initialFrame || 0, options.frameCount - 1));
    const intervalMs = options.intervalMs || 900;
    const status = root.querySelector("[data-stepper-status]");
    const buttons = Object.fromEntries(
      [...root.querySelectorAll("[data-stepper-action]")].map((button) => [button.dataset.stepperAction, button]),
    );
    let frame = initialFrame;
    let timer = null;

    function updateControls() {
      const playing = timer !== null;
      root.dataset.playing = String(playing);
      if (buttons.play) buttons.play.disabled = playing;
      if (buttons.pause) buttons.pause.disabled = !playing;
      if (buttons.previous) buttons.previous.disabled = frame === 0;
      if (buttons.next) buttons.next.disabled = frame === options.frameCount - 1;
    }

    function stop() {
      if (timer !== null) window.clearInterval(timer);
      timer = null;
      updateControls();
    }

    function render() {
      root.dataset.frame = String(frame);
      options.render(frame);
      if (status) {
        status.textContent = options.describe
          ? options.describe(frame)
          : `Step ${frame + 1} of ${options.frameCount}`;
      }
      updateControls();
    }

    function setFrame(nextFrame) {
      frame = Math.max(0, Math.min(nextFrame, options.frameCount - 1));
      render();
      if (frame === options.frameCount - 1) stop();
    }

    function play() {
      if (timer !== null) return;
      if (frame === options.frameCount - 1) frame = initialFrame;
      render();
      timer = window.setInterval(() => setFrame(frame + 1), intervalMs);
      updateControls();
    }

    function reset() {
      stop();
      frame = initialFrame;
      render();
    }

    Object.entries(buttons).forEach(([action, button]) => {
      button.addEventListener("click", () => {
        if (action === "play") play();
        if (action === "pause") stop();
        if (action === "previous") {
          stop();
          setFrame(frame - 1);
        }
        if (action === "next") {
          stop();
          setFrame(frame + 1);
        }
        if (action === "reset") reset();
      });
    });
    (root.closest(".lab") || root).addEventListener("course:reset", reset);
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) stop();
    });
    render();
    return Object.freeze({ play, pause: stop, reset, setFrame });
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("input[type='range'][data-output]").forEach((input) => {
      updateRangeOutput(input);
      input.addEventListener("input", () => updateRangeOutput(input));
    });
    document.querySelectorAll("[data-reset]").forEach((button) => {
      button.addEventListener("click", () => resetContainer(button));
    });
    document.querySelectorAll("[data-choice]").forEach((button) => {
      button.addEventListener("click", () => chooseAnswer(button));
    });
  });

  window.Course = Object.freeze({ money, mountStepper, motionAllowed, palette, percent });
})();
