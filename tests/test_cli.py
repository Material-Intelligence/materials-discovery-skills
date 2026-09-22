"""The ``matdisc`` console script.

``--help`` has to work for every subcommand in an environment with none of the optional
backends installed, because that is the first thing anyone runs. The offline hull demo is
exercised here too: it is the one command that needs no credentials, no network, no GPU and
no checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from matdisc.cli import build_parser, main


def _subcommands() -> list[str]:
    """List the subcommands the parser offers.

    argparse exposes no public accessor for its subparsers, so the action list is scanned.
    Reading them from the parser rather than hard-coding them means a subcommand added later
    is covered by the ``--help`` sweep without anyone remembering to add it.

    Returns:
        The subcommand names, sorted.
    """
    for action in build_parser()._actions:  # noqa: SLF001 - no public accessor exists
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            return sorted(action.choices)
    raise AssertionError("the parser has no subcommands")


SUBCOMMANDS = _subcommands()
"""Every subcommand the parser offers."""


def test_the_expected_subcommands_exist() -> None:
    assert SUBCOMMANDS == [
        "cluster",
        "competing",
        "dft-inputs",
        "dft-submit",
        "finetune",
        "generate",
        "hull",
        "phonons",
        "pipeline",
        "screen",
    ]


def test_top_level_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--help"])
    assert exit_info.value.code == 0
    assert "matdisc" in capsys.readouterr().out


@pytest.mark.parametrize("command", SUBCOMMANDS)
def test_subcommand_help_exits_zero(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args([command, "--help"])
    assert exit_info.value.code == 0
    assert command in capsys.readouterr().out


def test_version_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    from matdisc import __version__

    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help_and_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_an_unknown_command_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["not-a-command"])
    assert exit_info.value.code != 0


class TestHullDemo:
    """``matdisc hull --demo``: the offline example."""

    def test_it_runs_and_reports_the_verdicts(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["hull", "--demo"]) == 0

        out = capsys.readouterr().out
        assert "e_above_hull" in out
        assert "toy-on-hull" in out
        assert "2 of 3 candidate(s) at or below the hull" in out

    def test_the_on_hull_candidate_reads_stable(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["hull", "--demo"])
        line = next(line for line in capsys.readouterr().out.splitlines() if "toy-on-hull" in line)
        assert "True" in line

    def test_it_can_write_the_table(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        import pandas as pd

        output = tmp_path / "hull.csv"
        assert main(["hull", "--demo", "-o", str(output)]) == 0
        capsys.readouterr()

        table = pd.read_csv(output)
        assert {"e_above_hull", "is_stable", "decomposition"} <= set(table.columns)
        assert len(table) == 3


class TestHullFromFiles:
    """``matdisc hull`` reading the two tables from disk."""

    def test_the_shipped_example_table(
        self, tmp_path: Path, example_competing_csv: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import pandas as pd

        table = pd.read_csv(example_competing_csv)
        candidates = table[table["material_id"] == "mp-8279"].rename(columns={"material_id": "id"})
        competing = table[table["material_id"] != "mp-8279"]

        candidates_path = tmp_path / "candidates.csv"
        competing_path = tmp_path / "competing.csv"
        candidates.to_csv(candidates_path, index=False)
        competing.to_csv(competing_path, index=False)

        status = main(
            [
                "hull",
                "--candidates",
                str(candidates_path),
                "--competing",
                str(competing_path),
                "--energy-column",
                "formation_energy_per_atom",
            ]
        )
        assert status == 0
        assert "mp-8279" in capsys.readouterr().out

    def test_missing_inputs_are_reported(self) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["hull"])
        assert "--demo" in str(exit_info.value)


class TestErrorReporting:
    """A failing command reports the reason and returns a non-zero status."""

    def test_a_missing_structure_directory(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        status = main(["dft-inputs", "-s", str(tmp_path / "absent"), "-o", str(tmp_path / "out")])
        assert status == 1
        assert "absent" in capsys.readouterr().err

    def test_competing_without_a_key(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        status = main(["competing", "-c", "Ba-Cd-P", "-o", str(tmp_path / "out")])
        assert status == 1
        assert "MP_API_KEY" in capsys.readouterr().err
