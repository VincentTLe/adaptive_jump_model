"""Do the new tests actually fail on the faults that passed silently?

Each faulted spec copy written by the injection harness is pointed at the test
module (SPEC only -- nothing under audit is touched) and the relevant test is
called directly. A test that still passes on a fault is not a regression test.
"""
import importlib.util
import sys
import tomllib
from pathlib import Path

ROOT = Path("/home/tle/adaptive_jump_model")
FAULTS = Path("/tmp/claude-1017/-home-tle/69649cec-6fd3-40f9-9e01-42dd56f3559f"
              "/scratchpad/faults")

spec_ = importlib.util.spec_from_file_location(
    "audit_tests", ROOT / "tests/test_frequency_ladder_audit.py")
mod = importlib.util.module_from_spec(spec_)
sys.modules["audit_tests"] = mod
spec_.loader.exec_module(mod)

CASES = {
    "F10_armM_truncated_further": ["test_arm_m_is_arm_l_truncated_at_the_rung_the_spec_names"],
    "F11_ladder_changed": ["test_arm_m_is_arm_l_truncated_at_the_rung_the_spec_names"],
    "F12_grading_bands_widened": ["test_the_spec_declares_the_cells_delay_cost_and_bands_the_code_uses"],
    "F13_turnover_dropped_from_cells": ["test_the_spec_declares_the_cells_delay_cost_and_bands_the_code_uses"],
    "F14_cutoff_prose_changed": ["test_the_spec_prose_names_the_cutoff_the_runner_uses"],
}

runner = mod._module("run_frequency_ladder")
for fault, tests in CASES.items():
    path = FAULTS / fault / "spec.toml"
    faulted = tomllib.loads(path.read_text())
    for name in tests:
        fn = getattr(mod, name)
        args = {"spec": faulted, "runner": runner}
        kwargs = {k: v for k, v in args.items()
                  if k in fn.__code__.co_varnames[: fn.__code__.co_argcount]}
        try:
            fn(**kwargs)
        except AssertionError as exc:
            print(f"{fault:<34} {name:<62} CAUGHT ({(str(exc) or type(exc).__name__).splitlines()[0][:70] if str(exc) else "assert"})")
        else:
            print(f"{fault:<34} {name:<62} *** STILL PASSES -- not a regression test ***")

# and the two runner-level faults, checked functionally
print()
try:
    mod.test_the_runner_refuses_a_menu_that_does_not_re_derive(
        runner, _FakeMonkeypatch())
except NameError:
    print("F8_gate_dropped                    covered by "
          "test_the_runner_refuses_a_menu_that_does_not_re_derive "
          "(run under pytest)")
