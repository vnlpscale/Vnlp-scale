from __future__ import annotations

import json

from vnlp_scale.cli import main


def test_cli_record_inspect_verify_run_and_plan(tmp_path, tiny_checkpoint, capsys):
    source, _, _ = tiny_checkpoint
    store = tmp_path / "store"

    assert (
        main(
            [
                "record",
                "--source",
                str(source),
                "--output",
                str(store),
                "--quality",
                "lossless",
                "--chunk-mib",
                "1",
                "--json",
            ]
        )
        == 0
    )
    record_payload = json.loads(capsys.readouterr().out)
    assert record_payload["summary"]["finalized"] is True

    assert main(["verify", "--store", str(store)]) == 0
    verify_payload = json.loads(capsys.readouterr().out)
    assert verify_payload["ok"] is True

    assert main(["inspect", "--store", str(store)]) == 0
    inspect_payload = json.loads(capsys.readouterr().out)
    assert inspect_payload["tensors"] > 0

    assert (
        main(
            [
                "run",
                "--store",
                str(store),
                "--prompt-ids",
                "1,2,3",
                "--max-new",
                "1",
                "--cache-mib",
                "0",
                "--json",
            ]
        )
        == 0
    )
    run_payload = json.loads(capsys.readouterr().out)
    assert len(run_payload["tokens"]) == 1

    assert (
        main(
            [
                "plan",
                "--total-params",
                "1T",
                "--active-params",
                "32B",
                "--bits",
                "1.5",
                "--layers",
                "120",
                "--bandwidth-gbps",
                "64",
                "--tflops",
                "80",
                "--vram-gb",
                "24",
                "--ram-gb",
                "128",
            ]
        )
        == 0
    )
    plan_payload = json.loads(capsys.readouterr().out)
    assert plan_payload["storage_gb"] == 187.5


def test_cli_reports_invalid_prompt_ids(capsys):
    # argparse owns this validation and exits before command execution.
    try:
        main(["run", "--store", "missing", "--prompt-ids", "not-an-int"])
    except SystemExit as exc:
        assert exc.code == 2
    assert "prompt IDs" in capsys.readouterr().err
