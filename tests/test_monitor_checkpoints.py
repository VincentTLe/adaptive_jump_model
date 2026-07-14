import json
from pathlib import Path

import pandas as pd
import pytest

from adaptive_jump.monitor import checkpoints

IDENTITY = {"config_sha256": "a", "data_manifest_sha256": "b", "git_sha": "c"}


def test_checkpoint_round_trip_generations_and_clear(tmp_path: Path) -> None:
    stem = tmp_path / "run" / "hmm"
    first = pd.DataFrame({"state": [0.0, 1.0]})

    checkpoints.save_checkpoint(stem, first, kind="hmm", identity=IDENTITY)
    loaded = checkpoints.load_checkpoint(stem, kind="hmm", identity=IDENTITY)

    pd.testing.assert_frame_equal(loaded, first)
    assert len(list(stem.parent.glob("hmm.*.pkl"))) == 1

    second = first.assign(state=1.0)
    checkpoints.save_checkpoint(stem, second, kind="hmm", identity=IDENTITY)
    pd.testing.assert_frame_equal(
        checkpoints.load_checkpoint(stem, kind="hmm", identity=IDENTITY), second
    )
    assert len(list(stem.parent.glob("hmm.*.pkl"))) == 1

    checkpoints.clear_checkpoint(stem)
    assert checkpoints.load_checkpoint(stem, kind="hmm", identity=IDENTITY) is None
    assert not list(stem.parent.glob("hmm.*.pkl"))


def test_checkpoint_rejects_identity_kind_and_payload_tampering(tmp_path: Path) -> None:
    stem = tmp_path / "jm"
    checkpoints.save_checkpoint(
        stem, {"completed": 3}, kind="fixed_jm", identity=IDENTITY
    )

    with pytest.raises(checkpoints.CheckpointStoreError, match="identity mismatch"):
        checkpoints.load_checkpoint(stem, kind="hmm", identity=IDENTITY)
    changed = {**IDENTITY, "git_sha": "d"}
    with pytest.raises(checkpoints.CheckpointStoreError, match="identity mismatch"):
        checkpoints.load_checkpoint(stem, kind="fixed_jm", identity=changed)

    metadata = json.loads(stem.with_suffix(".json").read_text())
    payload = stem.parent / f"jm.{metadata['payload_sha256']}.pkl"
    payload.write_bytes(b"tampered")
    with pytest.raises(checkpoints.CheckpointStoreError, match="hash mismatch"):
        checkpoints.load_checkpoint(stem, kind="fixed_jm", identity=IDENTITY)


def test_interrupted_pointer_update_keeps_previous_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stem = tmp_path / "selection"
    checkpoints.save_checkpoint(stem, "old", kind="selection", identity=IDENTITY)
    original_atomic_write = checkpoints._atomic_write

    def interrupt_metadata(path: Path, payload: bytes) -> None:
        if path.suffix == ".json":
            raise OSError("simulated interruption")
        original_atomic_write(path, payload)

    monkeypatch.setattr(checkpoints, "_atomic_write", interrupt_metadata)
    with pytest.raises(OSError, match="simulated interruption"):
        checkpoints.save_checkpoint(stem, "new", kind="selection", identity=IDENTITY)

    assert (
        checkpoints.load_checkpoint(stem, kind="selection", identity=IDENTITY) == "old"
    )
    assert not list(stem.parent.glob("*.tmp"))
