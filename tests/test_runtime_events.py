from datetime import UTC, date, datetime

import pytest

from adaptive_jump.runtime.events import (
    EventError,
    ResearchEvent,
    RuntimeEvent,
    bind_event_context,
    emit_event,
    null_observer,
)


def test_runtime_event_round_trip_is_exact_and_json_safe() -> None:
    source_payload = {"states": [{"candidate": 5.0, "state": 1}]}
    event = ResearchEvent(
        kind="terminal_state",
        stage="fixed_jm",
        visibility="decision",
        market="us",
        model="fixed_jm",
        delay=1,
        date=date(2026, 7, 13),
        completed=4,
        total=10,
        payload=source_payload,
    )
    source_payload["changed_after_creation"] = True
    runtime = RuntimeEvent(
        job_id="job-001",
        sequence=7,
        time_utc=datetime(2026, 7, 13, 12, 30, tzinfo=UTC),
        elapsed_seconds=2.5,
        event=event,
    )

    document = runtime.to_dict()

    assert document["schema_version"] == 1
    assert document["time_utc"] == "2026-07-13T12:30:00Z"
    assert "changed_after_creation" not in document["payload"]
    assert RuntimeEvent.from_dict(document) == runtime
    document["payload"]["states"][0]["state"] = 0
    assert event.payload["states"][0]["state"] == 1


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"kind": "bad-kind", "stage": "hmm"}, "lower snake case"),
        (
            {"kind": "progress", "stage": "hmm", "completed": 2},
            "provided together",
        ),
        (
            {"kind": "progress", "stage": "hmm", "completed": 3, "total": 2},
            "inconsistent",
        ),
        (
            {"kind": "progress", "stage": "hmm", "payload": {"value": float("nan")}},
            "finite JSON",
        ),
    ],
)
def test_research_event_rejects_invalid_contract(
    values: dict[str, object], message: str
) -> None:
    with pytest.raises(EventError, match=message):
        ResearchEvent(**values)


def test_runtime_event_rejects_naive_time_and_wrong_schema() -> None:
    event = ResearchEvent("stage_started", "hmm")
    with pytest.raises(EventError, match="timezone-aware"):
        RuntimeEvent("job", 1, datetime(2026, 7, 13), 0, event)

    valid = RuntimeEvent("job", 1, datetime.now(UTC), 0, event).to_dict()
    valid["schema_version"] = True
    with pytest.raises(EventError, match="schema"):
        RuntimeEvent.from_dict(valid)


def test_emit_event_skips_construction_when_disabled() -> None:
    emit_event(None, kind="not-valid", stage="not-valid")
    emit_event(null_observer, kind="stage_started", stage="hmm")


def test_bound_event_context_is_explicit_and_cannot_be_overwritten() -> None:
    events = []
    observer = bind_event_context(events.append, market="us", model="hmm", delay=1)
    assert observer is not None

    observer(ResearchEvent("terminal_state", "hmm"))

    assert events[0].market == "us"
    assert events[0].model == "hmm"
    assert events[0].delay == 1
    with pytest.raises(EventError, match="conflicts"):
        observer(ResearchEvent("terminal_state", "hmm", market="jp"))
