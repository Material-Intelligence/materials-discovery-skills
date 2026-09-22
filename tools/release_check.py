#!/usr/bin/env python3
"""Fail-closed pre-publication scanner for this repository.

Inspects everything git would publish and refuses to pass when it finds anything that must not be:
site-specific paths and account names, credential-shaped literals, long opaque tokens, chemical
systems outside the scope of the published paper, CJK text, oversized files, or compiled artefacts
that belong in `.gitignore` rather than in git. It then checks that `pyproject.toml` still agrees
with the source tree.

Run it before every push:

    python tools/release_check.py

Exit status is 0 when nothing is found and 1 when anything is, so it can be wired straight into CI
or a pre-push hook. Use `--skip-cjk` when a deliberate non-English document is added, `--no-history`
to scan only the current files, and `--all-files` to walk the working tree instead of asking git.

Five deliberate design points:

* The scanner has no third-party dependencies, so it runs in a bare Python 3.11+ interpreter before
  anything is installed.
* It enumerates current files with `git ls-files --cached --others --exclude-standard`, which is
  the working tree and the index: everything a fresh clone would check out. If git is unavailable
  it falls back to walking the tree, which reports more rather than less.
* A file listing cannot see a secret that was committed once and deleted in a later commit, and a
  push carries that commit anyway. So the same content rules are run a second time over the patch
  text of every commit reachable from any ref (`git log --all -p`), and a hit there is reported
  against `git-history` rather than against a file. Objects reachable only from the reflog or a
  stash are not covered; a repository published with a fresh history has none.
* **Every rule here is written so that publishing this file discloses nothing.** A scanner that
  names what it is looking for is a directory of exactly the material it exists to keep out, and a
  one-character bracket in a regular expression stops `grep` but not a reader. So the rules are
  positive, not negative: the chemical-system rule carries the one system this repository is
  allowed to name and reports every other, and the site-disclosure rule matches the *shape* of an
  absolute home directory rather than anyone's account name. Nothing in this file states a term it
  was written to exclude. Terms that must be matched literally are supplied from outside the
  repository, through `$RELEASE_CHECK_DENYLIST` (see `load_denylist`), which is empty by default.
* Its banned file names are assembled at runtime from fragments, so that a reader grepping this
  repository for one of those names finds nothing, including here. Spelling one out would make the
  scanner report itself, which is the intended behaviour rather than a bug.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

MAX_FILE_BYTES = 5 * 1024 * 1024

#: Upper bound on the patch text read from the commit history, so the scanner cannot be made to
#: hold an unbounded repository in memory. Exceeding it is reported rather than ignored.
MAX_HISTORY_BYTES = 64 * 1024 * 1024

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", ".mypy_cache", ".ruff_cache", ".idea", ".tox"}

BANNED_SUFFIXES = {".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".so", ".dylib", ".dll", ".o", ".a"}

BANNED_DIR_NAMES = {"__pycache__", ".pytest_cache"}

# Assistant configuration files: useful while working, never part of a published research package.
# Each name is spliced from two fragments for the reason given in the module docstring, so that a
# reader grepping this repository for any of these names finds nothing, including here.
BANNED_BASENAMES = {
    "CLAUD" + "E.md",
    "AGENT" + "S.md",
    "GEMIN" + "I.md",
    ".curso" + "rrules",
    ".aide" + "r.conf.yml",
    "copil" + "ot-instructions.md",
}

BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".bz2",
    ".xz",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
    ".ico",
    ".woff",
    ".woff2",
}


@dataclass(frozen=True)
class Rule:
    """A single scan rule.

    Attributes:
        name: Short identifier printed with every hit.
        pattern: Compiled regular expression applied line by line.
        redact: Whether the matched text must be masked before printing. Set for rules that can
            match a live credential, so that running the scanner never leaks the secret it found.
        accept: Optional predicate applied to each match. A rule reports the first match the
            predicate accepts; when it is None every match is reported. It exists so that a rule
            can be expressed as a broad pattern plus an allow-list decision in Python, rather
            than as a regular expression that has to spell out what it excludes.
    """

    name: str
    pattern: re.Pattern[str]
    redact: bool = False
    accept: Callable[[re.Match[str]], bool] | None = None


#: Every element symbol, used to decide whether a hyphenated token is a chemical system at all
#: rather than ordinary hyphenated text. Naming all of them discloses nothing.
ELEMENT_SYMBOLS = frozenset("""
    H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se
    Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy
    Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf
    Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
    """.split())

#: The elements of the one chemical system this repository publishes: the worked example of the
#: paper it was abstracted from. Any other chemical system named anywhere in the tree is reported.
#:
#: This rule is an allow-list rather than a deny-list on purpose. A deny-list would have to name
#: the systems it excludes, and this file is published, so the list itself would disclose exactly
#: the unpublished work it exists to keep out. Naming the one system that is already in the paper
#: costs nothing and catches every other system, including ones nobody thought to enumerate.
#:
#: Its limit, stated rather than glossed over: it reads the hyphenated spelling, which is how a
#: chemical system is written throughout this repository and on the command line. A concatenated
#: formula is not matched, because a rule broad enough to catch one also matches ordinary
#: capitalised words that happen to be made of element symbols, and a scanner nobody believes is
#: a scanner nobody runs.
IN_SCOPE_ELEMENTS = frozenset({"Ba", "Cd", "P"})

#: Two to four capitalised element-shaped tokens joined by hyphens, which is how a chemical system
#: is written throughout this repository. The pattern is deliberately broad; `_is_out_of_scope`
#: decides which matches are findings.
CHEMICAL_SYSTEM_PATTERN = re.compile(r"(?<![A-Za-z0-9-])[A-Z][a-z]?(?:-[A-Z][a-z]?){1,3}(?![A-Za-z0-9-])")


def _is_out_of_scope(match: re.Match[str]) -> bool:
    """Decide whether a hyphenated token is a chemical system outside the published scope.

    Args:
        match: A match of `CHEMICAL_SYSTEM_PATTERN`.

    Returns:
        True when every hyphen-separated part is an element symbol and at least one of them is
        not in `IN_SCOPE_ELEMENTS`. Hyphenated text whose parts are not all element symbols is
        not a chemical system and is not reported.
    """
    parts = match.group(0).split("-")
    if not all(part in ELEMENT_SYMBOLS for part in parts):
        return False
    return not set(parts) <= IN_SCOPE_ELEMENTS


#: Site-specific disclosure. The rules match shapes, not terms: an absolute home directory is
#: site-specific whoever owns it, so the pattern needs no account name, and a scanner that carried
#: one would publish it. Literal terms, if any are needed, come from `load_denylist`.
DISCLOSURE_RULES: tuple[Rule, ...] = (
    # Absolute home directories of any user, which are site-specific by construction.
    Rule("absolute-home-path", re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/")),
    Rule("out-of-scope-system", CHEMICAL_SYSTEM_PATTERN, accept=_is_out_of_scope),
)

#: Environment variable naming a file of extra regular expressions, one per line, with `#`
#: comments and blank lines ignored. It is unset by default and the file it names is not part of
#: this repository, which is the point: a term that must be matched literally can be checked
#: without this published file stating it. See `load_denylist`.
DENYLIST_ENV = "RELEASE_CHECK_DENYLIST"

# Credential-shaped content. Matches here are masked before printing.
#
# The name half deliberately allows word characters on either side rather than requiring a word
# boundary: the literal this repository was cleaned of was assigned to `DEFAULT_API_KEY`, and `_`
# is a word character, so a `\b` before `api_key` never matches such a name. Any of MP_API_KEY,
# DEFAULT_API_KEY, my_token and secret_value must trip this rule.
SECRET_RULES: tuple[Rule, ...] = (
    Rule(
        "credential-assignment",
        re.compile(r"""(?ix)
            [A-Za-z0-9_]*
            (?:api[_-]?key|apikey|secret|token|password|passwd|access[_-]?key)
            [A-Za-z0-9_]*
            \s*[:=]\s*
            ["'][^"'\s]{8,}["']
            """),
        redact=True,
    ),
    Rule("opaque-token", re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]{32,}(?![A-Za-z0-9])"), redact=True),
)

# CJK ideographs, kana, CJK punctuation and fullwidth forms. The ranges are built from code points
# rather than spelled out, so that this file stays pure ASCII and never reports itself.
CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x2E80, 0x2EFF),  # CJK radicals supplement
    (0x3000, 0x303F),  # CJK symbols and punctuation
    (0x3040, 0x30FF),  # hiragana and katakana
    (0x3400, 0x4DBF),  # CJK unified ideographs extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0xFF00, 0xFFEF),  # halfwidth and fullwidth forms
)

CJK_RULE = Rule(
    "cjk-text",
    re.compile("[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in CJK_RANGES) + "]"),
)

# Lines the scanner is allowed to ignore, for the rare case where a term is legitimately present.
# A waiver is the marker below, a space, then the comma-separated names of the rules it waives, and
# it waives those rules on that line only. Naming the rules is required: an unqualified marker used
# to silence, say, a deliberate non-English phrase would also silence a credential that later lands
# on the same line, so an unqualified marker waives nothing and is reported in its own right. The
# marker is spliced from two fragments, like the banned file names above, so that this comment and
# the assignment under it are not themselves waivers.
ALLOW_MARKER = "release-check" + ": allow"

ALLOW_PATTERN = re.compile(re.escape(ALLOW_MARKER) + r"[ \t]+([A-Za-z0-9_-]+(?:[ \t]*,[ \t]*[A-Za-z0-9_-]+)*)")


def mask(text: str) -> str:
    """Mask the middle of a matched string so a credential is never printed in full.

    Args:
        text: The matched text.

    Returns:
        The text with everything but its first and last two characters replaced, and its length
        stated, which is enough to locate the match without disclosing it.
    """
    if len(text) <= 8:
        return f"<redacted len={len(text)}>"
    return f"{text[:2]}...{text[-2:]} <redacted len={len(text)}>"


def git_files(root: Path) -> list[Path] | None:
    """Ask git for every file a push would carry.

    Lists tracked files plus untracked files that `.gitignore` does not exclude, which is exactly
    the set that ends up in the published repository.

    Args:
        root: Repository root.

    Returns:
        A sorted list of file paths, or None when `root` is not a usable git work tree.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    names = [name for name in completed.stdout.decode("utf-8", "surrogateescape").split("\0") if name]
    return sorted({root / name for name in names if (root / name).is_file()})


def walk_files(root: Path) -> list[Path]:
    """Collect every file under the repository root, skipping tool and environment directories.

    Args:
        root: Repository root to walk.

    Returns:
        A sorted list of file paths.
    """
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info"))
        for name in sorted(filenames):
            found.append(Path(dirpath) / name)
    return sorted(found)


def iter_files(root: Path, use_git: bool) -> list[Path]:
    """Choose the file list to scan.

    Args:
        root: Repository root.
        use_git: Whether to ask git first.

    Returns:
        The files to scan.
    """
    if use_git:
        from_git = git_files(root)
        if from_git is not None:
            return from_git
        print("release_check: git listing unavailable, walking the working tree instead", file=sys.stderr)
    return walk_files(root)


def scan_path_metadata(path: Path, root: Path) -> list[str]:
    """Check a file's name, location and size, without reading its contents.

    Args:
        path: File to check.
        root: Repository root, used to print relative paths.

    Returns:
        A list of formatted findings, empty when the file is acceptable.
    """
    findings: list[str] = []
    rel = path.relative_to(root)

    for part in rel.parts[:-1]:
        if part in BANNED_DIR_NAMES:
            findings.append(f"{rel}:0: banned-directory: {part}/ must not be committed")
            break

    if path.suffix.lower() in BANNED_SUFFIXES:
        findings.append(f"{rel}:0: banned-file-type: {path.suffix} must not be committed")

    if path.name in BANNED_BASENAMES:
        findings.append(f"{rel}:0: banned-file-name: {path.name} must not be committed")

    try:
        size = path.stat().st_size
    except OSError as exc:
        return findings + [f"{rel}:0: unreadable: {exc.strerror or exc}"]

    if size > MAX_FILE_BYTES:
        findings.append(f"{rel}:0: oversized-file: {size} bytes exceeds the {MAX_FILE_BYTES} byte limit")

    return findings


def load_text(path: Path, root: Path) -> tuple[str | None, list[str]]:
    """Read a file as UTF-8 text when it is safe to scan.

    Binary files are not read. A file that cannot be decoded as UTF-8 is reported rather than
    skipped, because the scanner is fail-closed: anything it cannot inspect must be looked at by a
    human before release.

    Args:
        path: File to read.
        root: Repository root, used to print relative paths.

    Returns:
        A tuple of the decoded text (None when the file was not scanned) and any findings raised
        while trying to read it.
    """
    rel = path.relative_to(root)

    if path.suffix.lower() in BINARY_SUFFIXES or path.suffix.lower() in BANNED_SUFFIXES:
        return None, []

    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, [f"{rel}:0: unreadable: {exc.strerror or exc}"]

    if b"\x00" in raw:
        return None, [f"{rel}:0: binary-content: file contains null bytes and was not scanned"]

    try:
        return raw.decode("utf-8"), []
    except UnicodeDecodeError as exc:
        return None, [f"{rel}:0: undecodable: not valid UTF-8 ({exc.reason})"]


def allowed_rules(line: str) -> tuple[frozenset[str], bool]:
    """Read the waiver marker on one line.

    Args:
        line: The line to inspect.

    Returns:
        A tuple of the rule names the line waives and whether an unqualified marker was found.
        An unqualified marker waives nothing, so that a bare waiver cannot silence a credential
        that lands on the same line later.
    """
    if ALLOW_MARKER not in line:
        return frozenset(), False
    match = ALLOW_PATTERN.search(line)
    if match is None:
        return frozenset(), True
    return frozenset(name.strip() for name in match.group(1).split(",") if name.strip()), False


def scan_text(text: str, rel: Path | str, rules: list[Rule]) -> list[str]:
    """Apply every content rule to a file's text, line by line.

    Args:
        text: Decoded file contents.
        rel: Path relative to the repository root, used in the printed findings.
        rules: Rules to apply.

    Returns:
        A list of formatted findings, empty when the text is clean.
    """
    findings: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        waived, unqualified = allowed_rules(line)
        if unqualified:
            findings.append(
                f"{rel}:{lineno}: unqualified-allow-marker: "
                f"'{ALLOW_MARKER}' must name the rules it waives, e.g. '{ALLOW_MARKER} cjk-text'"
            )
        for rule in rules:
            if rule.name in waived:
                continue
            match = next((m for m in rule.pattern.finditer(line) if rule.accept is None or rule.accept(m)), None)
            if match is None:
                continue
            shown = mask(match.group(0)) if rule.redact else match.group(0)
            findings.append(f"{rel}:{lineno}: {rule.name}: {shown}")
    return findings


def load_denylist(env: dict[str, str] | None = None) -> tuple[list[Rule], list[str]]:
    """Load extra content rules from the file named by `$RELEASE_CHECK_DENYLIST`.

    Some material can only be matched as a literal string — an account name, a hostname, an
    institution. Writing those literals into this file would publish them, so they live outside
    the repository and are named by an environment variable instead. When the variable is unset,
    which is the default, no extra rule is added and the scanner behaves exactly as documented.

    The file holds one regular expression per line. Blank lines and lines whose first
    non-whitespace character is `#` are ignored. Every rule it yields is called `denylist-term`,
    which is also the name a waiver marker uses to waive it.

    Failures are reported rather than ignored, because the scanner is fail-closed: a deny-list
    that was meant to be in force and silently was not is worse than no deny-list at all.

    Args:
        env: Environment mapping to read, defaulting to `os.environ`.

    Returns:
        A tuple of the extra rules and any findings raised while loading them.
    """
    environment = os.environ if env is None else env
    configured = environment.get(DENYLIST_ENV)
    if not configured:
        return [], []

    path = Path(configured).expanduser()
    try:
        text = path.read_text("utf-8")
    except OSError as exc:
        return [], [f"{DENYLIST_ENV}:0: denylist-unreadable: {path} ({exc.strerror or exc})"]

    rules: list[Rule] = []
    findings: list[str] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rules.append(Rule("denylist-term", re.compile(line)))
        except re.error as exc:
            # The pattern itself is not echoed: the file exists so that its terms stay unpublished,
            # and the scanner's own output is not a place to undo that.
            findings.append(f"{path}:{lineno}: denylist-invalid: line {lineno} is not a regular expression ({exc.msg})")
    return rules, findings


def git_history_patch(root: Path) -> tuple[str | None, list[str]]:
    """Return the combined patch text of every commit reachable from any ref.

    Args:
        root: Repository root.

    Returns:
        A tuple of the patch text (None when there is no history to scan, or git is unusable)
        and any findings raised while collecting it.
    """
    try:
        counted = subprocess.run(
            ["git", "-C", str(root), "rev-list", "--all", "--count"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None, []
    if counted.returncode != 0:
        return None, []
    try:
        n_commits = int(counted.stdout.decode("utf-8", "replace").strip() or "0")
    except ValueError:
        return None, []
    if n_commits == 0:
        return None, []

    completed = subprocess.run(
        # The abbreviated hash keeps the header line short enough that the opaque-token rule does
        # not fire on the commit id itself.
        ["git", "-C", str(root), "log", "--all", "-p", "--no-color", "--no-textconv", "--format=commit %h"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", "replace").strip().splitlines()
        detail = message[0] if message else "git log failed"
        return None, [f"git-history:0: unreadable: {detail}"]

    raw = completed.stdout
    findings: list[str] = []
    if len(raw) > MAX_HISTORY_BYTES:
        raw = raw[:MAX_HISTORY_BYTES]
        findings.append(
            f"git-history:0: history-truncated: only the first {MAX_HISTORY_BYTES} bytes of "
            f"{n_commits} commit(s) were scanned; inspect the rest by hand"
        )
    return raw.decode("utf-8", "replace"), findings


def scan_history(root: Path, rules: list[Rule]) -> list[str]:
    """Run the content rules over the repository's commit history.

    A file listing only sees the current checkout. A secret committed once and removed in a later
    commit is still pushed, so the same rules are applied to the patch text of every commit.

    Args:
        root: Repository root.
        rules: Rules to apply.

    Returns:
        A list of formatted findings, empty when the history is clean or there is none.
    """
    patch, findings = git_history_patch(root)
    if patch is None:
        return findings
    return findings + scan_text(patch, "git-history", rules)


def check_console_scripts(root: Path, project: dict) -> list[str]:
    """Check that every console script points at a callable that exists in the source tree.

    Args:
        root: Repository root.
        project: The `[project]` table of `pyproject.toml`.

    Returns:
        A list of formatted findings, empty when every entry point resolves.
    """
    findings: list[str] = []
    for name, target in (project.get("scripts") or {}).items():
        module, _, attribute = target.partition(":")
        relative = Path(*module.split("."))
        candidates = (root / "src" / relative.with_suffix(".py"), root / "src" / relative / "__init__.py")
        module_file = next((candidate for candidate in candidates if candidate.is_file()), None)
        if module_file is None:
            findings.append(f"pyproject.toml:0: packaging: console script '{name}' points at missing module {module}")
            continue
        if not attribute:
            continue
        source = module_file.read_text("utf-8")
        defined = re.search(rf"^(?:async\s+)?def\s+{re.escape(attribute)}\b|^{re.escape(attribute)}\s*=", source, re.M)
        if not defined:
            relative_file = module_file.relative_to(root)
            findings.append(f"pyproject.toml:0: packaging: {relative_file} does not define '{attribute}'")
    return findings


def check_version(root: Path, project: dict) -> list[str]:
    """Check that the packaged version and the package's own `__version__` agree.

    Args:
        root: Repository root.
        project: The `[project]` table of `pyproject.toml`.

    Returns:
        A list of formatted findings, empty when the two versions match.
    """
    declared = project.get("version")
    if not declared:
        return []

    init = root / "src" / "matdisc" / "__init__.py"
    if not init.is_file():
        return [f"pyproject.toml:0: packaging: {init.relative_to(root)} is missing"]

    match = re.search(r"""^__version__\s*=\s*["']([^"']+)["']""", init.read_text("utf-8"), re.M)
    if match is None:
        return [f"{init.relative_to(root)}:0: packaging: no __version__ is defined"]
    if match.group(1) != declared:
        return [
            f"{init.relative_to(root)}:0: packaging: __version__ is {match.group(1)} "
            f"but pyproject.toml declares {declared}"
        ]
    return []


def check_extras(texts: dict[Path, str], project: dict) -> list[str]:
    """Check that every extra named in an install hint is declared in `pyproject.toml`.

    The lazy imports under `src/` tell the user to install an optional extra by name. Those names
    drift silently when the packaging metadata changes, so they are compared here.

    Args:
        texts: Decoded contents of every scanned file, keyed by path relative to the root.
        project: The `[project]` table of `pyproject.toml`.

    Returns:
        A list of formatted findings, empty when every referenced extra is declared.
    """
    distribution = project.get("name")
    if not distribution:
        return ["pyproject.toml:0: packaging: [project] has no name"]

    declared = set(project.get("optional-dependencies") or {})
    hint = re.compile(re.escape(distribution) + r"\[([A-Za-z0-9_.,\- ]+)\]")

    findings: list[str] = []
    for rel, text in texts.items():
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in hint.finditer(line):
                for extra in (part.strip() for part in match.group(1).split(",")):
                    if extra and extra not in declared:
                        findings.append(f"{rel}:{lineno}: packaging: extra '{extra}' is not declared in pyproject.toml")
    return findings


def check_packaging(root: Path, texts: dict[Path, str]) -> list[str]:
    """Check that `pyproject.toml` still agrees with the source tree.

    Args:
        root: Repository root.
        texts: Decoded contents of every scanned file, keyed by path relative to the root.

    Returns:
        A list of formatted findings, empty when the metadata is consistent.
    """
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return ["pyproject.toml:0: packaging: file is missing"]

    try:
        data = tomllib.loads(pyproject.read_text("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return [f"pyproject.toml:0: packaging: file does not parse ({exc})"]

    project = data.get("project")
    if not isinstance(project, dict):
        return ["pyproject.toml:0: packaging: [project] table is missing"]

    return check_console_scripts(root, project) + check_version(root, project) + check_extras(texts, project)


def run(root: Path, skip_cjk: bool, use_git: bool, scan_git_history: bool = True) -> list[str]:
    """Scan the whole repository.

    Args:
        root: Repository root.
        skip_cjk: Whether to leave the CJK rule out.
        use_git: Whether to ask git which files a push would carry.
        scan_git_history: Whether to run the same content rules over the commit history as well.
            The current files cannot show a secret that a later commit deleted.

    Returns:
        Every finding, in path order, with the history findings last.
    """
    extra_rules, findings = load_denylist()
    rules = list(DISCLOSURE_RULES) + list(SECRET_RULES) + extra_rules
    if not skip_cjk:
        rules.append(CJK_RULE)

    texts: dict[Path, str] = {}
    for path in iter_files(root, use_git):
        meta = scan_path_metadata(path, root)
        findings.extend(meta)
        if any("banned-file-type" in item or "oversized-file" in item for item in meta):
            continue
        text, read_findings = load_text(path, root)
        findings.extend(read_findings)
        if text is None:
            continue
        rel = path.relative_to(root)
        texts[rel] = text
        findings.extend(scan_text(text, rel, rules))

    findings.extend(check_packaging(root, texts))
    if scan_git_history:
        findings.extend(scan_history(root, rules))
    return findings


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list, defaulting to `sys.argv[1:]`.

    Returns:
        0 when the repository is clean, 1 when anything was found.
    """
    parser = argparse.ArgumentParser(description="Fail-closed pre-publication scanner.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root to scan (default: the parent of tools/)",
    )
    parser.add_argument(
        "--skip-cjk",
        action="store_true",
        help="do not report CJK characters, for deliberate non-English documents",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="walk the working tree instead of asking git which files a push would carry",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="scan only the current files, not the patch text of every commit",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        print(f"release_check: not a directory: {root}", file=sys.stderr)
        return 1

    findings = run(root, args.skip_cjk, use_git=not args.all_files, scan_git_history=not args.no_history)
    for finding in findings:
        print(finding)

    if findings:
        print(f"\nrelease_check: FAILED with {len(findings)} finding(s) under {root}", file=sys.stderr)
        return 1

    print(f"release_check: OK, nothing found under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
