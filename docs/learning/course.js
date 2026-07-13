"use strict";

document.documentElement.classList.add("js");

(function () {
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
    output.value = `${input.dataset.prefix || ""}${(value * scale).toFixed(digits)}${input.dataset.suffix || ""}`;
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

  window.Course = Object.freeze({ money, percent });
})();
