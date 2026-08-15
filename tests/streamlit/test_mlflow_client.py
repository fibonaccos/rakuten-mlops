"""Tests for the MLflow file-store fallback."""

from pathlib import Path

from services.streamlit.clients.mlflow_client import read_local_runs


def _write_run(root: Path, experiment: str, run_id: str, start: int, accuracy: float) -> None:
    """Create a minimal MLflow file-store run on disk."""
    run_dir = root / experiment / run_id
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "params").mkdir()
    (run_dir / "tags").mkdir()

    (run_dir / "meta.yaml").write_text(
        "\n".join(
            [
                f"run_id: {run_id}",
                f"experiment_id: '{experiment}'",
                f"run_name: run-{run_id}",
                "status: FINISHED",
                f"start_time: {start}",
                f"end_time: {start + 12000}",
            ]
        ),
        encoding="utf-8",
    )
    # MLflow writes one "<timestamp> <value> <step>" line per record.
    (run_dir / "metrics" / "val_accuracy").write_text(
        f"{start} 0.5 0\n{start + 1000} {accuracy} 1\n", encoding="utf-8"
    )
    (run_dir / "params" / "seed").write_text("42", encoding="utf-8")
    (run_dir / "tags" / "mlflow.source.git.commit").write_text("abc1234def", encoding="utf-8")


def test_runs_are_read_from_the_file_store(tmp_path: Path) -> None:
    """Metrics, params, tags and timings are recovered without a server."""
    _write_run(tmp_path, "1", "aaa", start=1_700_000_000_000, accuracy=0.71)

    runs = read_local_runs(tmp_path)

    assert len(runs) == 1
    run = runs[0]
    assert run.run_name == "run-aaa"
    assert run.status == "FINISHED"
    assert run.metrics["val_accuracy"] == 0.71
    assert run.params["seed"] == "42"
    assert run.git_commit == "abc1234def"
    assert run.duration_s == 12.0


def test_runs_are_sorted_newest_first(tmp_path: Path) -> None:
    """The most recent run is the one shown at the top of the table."""
    _write_run(tmp_path, "1", "old", start=1_700_000_000_000, accuracy=0.60)
    _write_run(tmp_path, "1", "new", start=1_800_000_000_000, accuracy=0.75)

    runs = read_local_runs(tmp_path)

    assert [run.run_id for run in runs] == ["new", "old"]


def test_missing_store_returns_nothing(tmp_path: Path) -> None:
    """An absent mlruns directory is not an error."""
    assert read_local_runs(tmp_path / "absent") == []


def test_unreadable_run_is_skipped(tmp_path: Path) -> None:
    """A run without a run_id in its metadata is ignored."""
    broken = tmp_path / "1" / "broken"
    broken.mkdir(parents=True)
    (broken / "meta.yaml").write_text("lifecycle_stage: active\n", encoding="utf-8")

    assert read_local_runs(tmp_path) == []
