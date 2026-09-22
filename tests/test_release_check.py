"""Tests for the pre-publication scanner in ``tools/release_check.py``.

The scanner is a standalone script rather than part of the package, so it is loaded here by
path. Every credential-shaped literal used below is synthetic and is assembled from fragments,
the same way the scanner assembles its own patterns, so that this test file does not become a
finding of the tool it tests.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"

# Twenty synthetic characters. Short enough that the opaque-token rule (32+) cannot be what
# catches it, so a hit can only come from the credential-assignment rule under test.
SYNTHETIC_VALUE = "A1b2C3d4E5f6G7h8I9j0"

# Probes for the chemical-system rule. They are well-known semiconductor and alloy systems with
# nothing to do with this project, chosen precisely because they are not sensitive: what is under
# test is the property "anything that is not the published system is reported", not any particular
# system. The scanner reads its own test file, so each line waives that one rule by name — which is
# also how the scoped waiver marker is meant to be used.
OUT_OF_SCOPE_PROBES = (
    "Al-Ga-As",  # release-check: allow out-of-scope-system
    "Zr-O",  # release-check: allow out-of-scope-system
    "Si-Ge",  # release-check: allow out-of-scope-system
    "Ba-Cd-P-O",  # release-check: allow out-of-scope-system
    "Fe-Ni-Cr-Mo",  # release-check: allow out-of-scope-system
)


def _load_release_check() -> ModuleType:
    """Import ``tools/release_check.py`` by path.

    Returns:
        The imported module.
    """
    path = TOOLS_DIR / "release_check.py"
    spec = importlib.util.spec_from_file_location("release_check_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


release_check = _load_release_check()


def credential_line(name: str) -> str:
    """Build one synthetic credential assignment.

    Args:
        name: The variable name to assign to, spliced by the caller so the literal does not
            appear in this file.

    Returns:
        A single source line assigning :data:`SYNTHETIC_VALUE` to ``name``.
    """
    return f'{name} = "{SYNTHETIC_VALUE}"'


class TestCredentialAssignment:
    """The rule that must catch the shape of the key this repository was cleaned of."""

    @pytest.mark.parametrize(
        "name",
        [
            "DEFAULT_" + "API" + "_KEY",  # the exact shape of the leaked assignment
            "MP_" + "API" + "_KEY",
            "MY_" + "TOKEN",
            "app_" + "secret" + "_value",
            "api" + "_key",  # the bare name the old boundary-anchored pattern already caught
        ],
    )
    def test_an_underscore_prefixed_name_is_caught(self, name: str) -> None:
        findings = release_check.scan_text(credential_line(name), Path("probe.py"), [release_check.SECRET_RULES[0]])
        assert len(findings) == 1, findings
        assert "credential-assignment" in findings[0]

    def test_the_value_is_redacted(self) -> None:
        name = "DEFAULT_" + "API" + "_KEY"
        (finding,) = release_check.scan_text(credential_line(name), Path("probe.py"), [release_check.SECRET_RULES[0]])
        assert SYNTHETIC_VALUE not in finding
        assert "redacted" in finding

    def test_an_unquoted_environment_hint_is_not_a_finding(self) -> None:
        line = "export MP_" + "API" + "_KEY=your_key_here"
        assert release_check.scan_text(line, Path("probe.md"), list(release_check.SECRET_RULES)) == []

    def test_a_short_value_is_not_a_finding(self) -> None:
        line = ("MY_" + "TOKEN") + ' = "abc"'
        assert release_check.scan_text(line, Path("probe.py"), [release_check.SECRET_RULES[0]]) == []


class TestAllowMarker:
    """A waiver names the rules it waives and waives nothing else."""

    def test_a_scoped_marker_waives_only_the_named_rule(self) -> None:
        name = "DEFAULT_" + "API" + "_KEY"
        line = f"{credential_line(name)}  # {release_check.ALLOW_MARKER} institution"
        findings = release_check.scan_text(line, Path("probe.py"), [release_check.SECRET_RULES[0]])
        assert len(findings) == 1, "a waiver for another rule must not silence a credential"

        line = f"{credential_line(name)}  # {release_check.ALLOW_MARKER} credential-assignment"
        assert release_check.scan_text(line, Path("probe.py"), [release_check.SECRET_RULES[0]]) == []

    def test_a_comma_separated_marker_waives_each_named_rule(self) -> None:
        waived, unqualified = release_check.allowed_rules(f"text # {release_check.ALLOW_MARKER} cjk-text, institution")
        assert waived == {"cjk-text", "institution"}
        assert not unqualified

    def test_an_unqualified_marker_waives_nothing_and_is_reported(self) -> None:
        name = "DEFAULT_" + "API" + "_KEY"
        line = f"{credential_line(name)}  # {release_check.ALLOW_MARKER}"
        findings = release_check.scan_text(line, Path("probe.py"), [release_check.SECRET_RULES[0]])
        assert len(findings) == 2, findings
        assert any("unqualified-allow-marker" in finding for finding in findings)
        assert any("credential-assignment" in finding for finding in findings)


class TestBannedBasenames:
    """Assistant configuration files are rejected by name, and the names are spliced."""

    def test_every_banned_basename_is_reported(self, tmp_path: Path) -> None:
        for basename in release_check.BANNED_BASENAMES:
            path = tmp_path / basename
            path.write_text("", encoding="utf-8")
            findings = release_check.scan_path_metadata(path, tmp_path)
            assert any("banned-file-name" in finding for finding in findings), basename
            path.unlink()

    def test_the_splices_still_spell_the_intended_names(self) -> None:
        assert ("CLAUD" + "E.md") in release_check.BANNED_BASENAMES
        assert ("AGENT" + "S.md") in release_check.BANNED_BASENAMES
        assert ("GEMIN" + "I.md") in release_check.BANNED_BASENAMES


class TestHistoryScan:
    """A secret committed once and deleted later is still carried by a push."""

    def test_no_history_is_not_a_finding(self, tmp_path: Path) -> None:
        assert release_check.scan_history(tmp_path, list(release_check.SECRET_RULES)) == []

    def test_a_deleted_commit_is_still_found(self, tmp_path: Path) -> None:
        if shutil.which("git") is None:  # pragma: no cover - git is present wherever CI runs
            pytest.skip("git is not on PATH")

        def git(*args: str) -> None:
            subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

        git("init", "-q", ".")
        git("config", "user.email", "test@example.invalid")
        git("config", "user.name", "test")

        leak = tmp_path / "leak.py"
        leak.write_text(credential_line("DEFAULT_" + "API" + "_KEY") + "\n", encoding="utf-8")
        git("add", "-A")
        git("commit", "-qm", "add")

        leak.unlink()
        git("add", "-A")
        git("commit", "-qm", "remove")

        assert not leak.exists(), "the working tree no longer carries the secret"
        findings = release_check.scan_history(tmp_path, list(release_check.SECRET_RULES))
        assert findings, "a secret removed in a later commit must still be reported"
        assert all(finding.startswith("git-history:") for finding in findings), findings
        assert all(SYNTHETIC_VALUE not in finding for finding in findings)


class TestChemicalSystemAllowList:
    """The chemical-system rule names what is allowed, never what is excluded.

    The rule exists so that a system from unpublished work cannot be added without the scanner
    saying so. It is written as an allow-list because this scanner is itself published: a list of
    excluded systems would disclose exactly the work it was written to keep out. These tests
    therefore build their out-of-scope probes from arbitrary elements, and assert the property
    ("anything outside the published system is reported") rather than any particular system.
    """

    @staticmethod
    def system_rule() -> list[object]:
        """Return the chemical-system rule on its own.

        Returns:
            A single-entry rule list, so a finding can only come from this rule.
        """
        return [rule for rule in release_check.DISCLOSURE_RULES if rule.name == "out-of-scope-system"]

    def test_the_published_system_is_not_a_finding(self) -> None:
        assert release_check.IN_SCOPE_ELEMENTS == {"Ba", "Cd", "P"}
        for line in ("Ba-Cd-P is the worked example", "the Ba-P and Cd-P subsystems", "P-Cd-Ba reordered"):
            assert release_check.scan_text(line, Path("probe.md"), self.system_rule()) == [], line

    @pytest.mark.parametrize("system", OUT_OF_SCOPE_PROBES)
    def test_any_other_chemical_system_is_reported(self, system: str) -> None:
        (finding,) = release_check.scan_text(f"we also looked at {system}", Path("probe.md"), self.system_rule())
        assert "out-of-scope-system" in finding
        assert system in finding

    @pytest.mark.parametrize(
        "line",
        [
            "Materials-Project data",
            "a Non-Self-Consistent run",
            "doi:10.1038/s41586-025-08628-5",
            "mp-527_icsd-58642.vasp",
            "the BaCdP structure",
        ],
    )
    def test_ordinary_hyphenated_text_is_not_a_finding(self, line: str) -> None:
        assert release_check.scan_text(line, Path("probe.md"), self.system_rule()) == []

    def test_the_scanner_states_no_excluded_system(self) -> None:
        source = (TOOLS_DIR / "release_check.py").read_text("utf-8")
        hyphenated = release_check.CHEMICAL_SYSTEM_PATTERN
        offenders = [match.group(0) for match in hyphenated.finditer(source) if release_check._is_out_of_scope(match)]
        assert offenders == [], f"the scanner must not name a chemical system it excludes: {offenders}"


class TestDenylist:
    """Literal terms come from outside the repository, so this file states none of them."""

    def test_an_unset_variable_adds_no_rule(self) -> None:
        rules, findings = release_check.load_denylist({})
        assert rules == []
        assert findings == []

    def test_terms_are_read_from_the_named_file(self, tmp_path: Path) -> None:
        listing = tmp_path / "denylist.txt"
        listing.write_text("# a comment\n\nwidget[0-9]+\n", encoding="utf-8")

        rules, findings = release_check.load_denylist({release_check.DENYLIST_ENV: str(listing)})
        assert findings == []
        assert [rule.name for rule in rules] == ["denylist-term"]

        (found,) = release_check.scan_text("path/to/widget42/run", Path("probe.md"), rules)
        assert "denylist-term" in found

    def test_a_missing_file_is_reported_rather_than_ignored(self, tmp_path: Path) -> None:
        rules, findings = release_check.load_denylist({release_check.DENYLIST_ENV: str(tmp_path / "absent.txt")})
        assert rules == []
        assert any("denylist-unreadable" in finding for finding in findings), findings

    def test_a_broken_pattern_is_reported_without_echoing_it(self, tmp_path: Path) -> None:
        listing = tmp_path / "denylist.txt"
        listing.write_text("widget(unbalanced\n", encoding="utf-8")

        rules, findings = release_check.load_denylist({release_check.DENYLIST_ENV: str(listing)})
        assert rules == []
        assert any("denylist-invalid" in finding for finding in findings), findings
        assert all("widget" not in finding for finding in findings), findings
