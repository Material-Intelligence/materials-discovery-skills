"""Command-line interface for the ``matdisc`` console script.

Every subcommand is one stage of the workflow, plus ``pipeline`` which runs several of them
from a YAML configuration:

``generate``
    Sample candidate structures for a chemical system with MatterGen.
``cluster``
    Pick the representative structures that are worth a DFT calculation.
``dft-inputs``
    Write a VASP calculation directory per structure, with an optional submission script.
``dft-submit``
    Submit the scripts those directories hold.
``finetune``
    Turn finished DFT runs into a training set and, on request, fine-tune on it.
``competing``
    Harvest the competing phases of a chemical system from the Materials Project.
``screen``
    Relax a directory of candidates and place them on the hull of the competing phases: the
    one command that answers "which of these structures are stable?".
``hull``
    Place candidate energies, already computed, on the convex hull of those phases.
``phonons``
    Check structures for imaginary phonon frequencies.
``pipeline``
    Run the stages listed in a configuration file.

The optional backends (MatterGen, DeePMD-kit, dpdata, maml) are imported inside the
functions that use them, so ``--help`` works for every subcommand in an environment where
none of them is installed. ``matdisc hull --demo`` needs no backend, no credentials and no
network at all.

Results are printed to standard output; progress goes through the package logger to standard
error, at ``--verbose`` or ``--quiet`` level.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from matdisc import __version__

PROGRAM = "matdisc"

__all__ = ["build_parser", "main"]


def _echo(text: str = "") -> None:
    """Write one line of result output to standard output.

    Args:
        text: The line.
    """
    print(text)


def _report(result: Any) -> int:
    """Print a stage result and return the process exit status.

    Args:
        result: A :class:`~matdisc.pipeline.stages.StageResult`.

    Returns:
        ``0``; a stage that failed raises instead of returning.
    """
    _echo(f"{result.stage}: {result.status}")
    if result.message:
        _echo(f"  {result.message}")
    for key, value in sorted(result.stats.items()):
        _echo(f"  {key}: {value}")
    for key, value in sorted(result.artifacts.items()):
        _echo(f"  {key} -> {value}")
    return 0


def _slurm_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Collect the SLURM options a command was given.

    Args:
        args: The parsed arguments.

    Returns:
        A mapping accepted by :func:`matdisc.dft.slurm.settings_from_mapping`, holding only
        the options that were actually passed.
    """
    mapping: dict[str, Any] = {}
    for name, key in (
        ("job_name", "job_name"),
        ("partition", "partition"),
        ("nodes", "nodes"),
        ("ntasks", "ntasks"),
        ("walltime", "walltime_minutes"),
        ("account", "account"),
    ):
        value = getattr(args, name, None)
        if value is not None:
            mapping[key] = value
    return mapping


def _cmd_generate(args: argparse.Namespace) -> int:
    """Run the generation stage.

    Args:
        args: The parsed arguments.

    Returns:
        The process exit status.
    """
    from matdisc.pipeline.config import GenerationConfig, PipelineConfig
    from matdisc.pipeline.stages import run_generation

    config = PipelineConfig(
        name="generate",
        work_dir=args.outdir,
        generation=GenerationConfig(
            chemical_systems=args.chemsys,
            n_structures=args.n_structures,
            model_path=args.model_path,
            batch_size=args.batch_size,
            energy_above_hull=args.energy_above_hull,
            guidance_factor=args.guidance_factor,
            skip_existing=not args.no_skip_existing,
        ),
    )
    return _report(run_generation(config, output_dir=args.outdir))


def _cmd_cluster(args: argparse.Namespace) -> int:
    """Run the clustering stage.

    Args:
        args: The parsed arguments.

    Returns:
        The process exit status.
    """
    from matdisc.pipeline.config import ClusteringConfig, PipelineConfig
    from matdisc.pipeline.stages import run_clustering

    config = PipelineConfig(
        name="cluster",
        work_dir=args.outdir,
        clustering=ClusteringConfig(
            model_path=args.model_path,
            head=args.head,
            n_representatives=args.n_representatives,
            threshold=args.threshold,
            k_per_cluster=args.k_per_cluster,
        ),
    )
    return _report(run_clustering(config, args.structures, output_dir=args.outdir))


def _cmd_dft_inputs(args: argparse.Namespace) -> int:
    """Run the DFT input-generation stage.

    ``--kspacing 0`` (or any non-positive value) means "no ``KSPACING`` tag": the input set's own
    k-point grid is used and an explicit KPOINTS file is written instead.

    Args:
        args: The parsed arguments.

    Returns:
        The process exit status.
    """
    from matdisc.pipeline.config import DftConfig, PipelineConfig
    from matdisc.pipeline.stages import run_dft

    kspacing = args.kspacing if args.kspacing is not None and args.kspacing > 0 else None

    config = PipelineConfig(
        name="dft-inputs",
        work_dir=args.outdir,
        dft=DftConfig(
            kind=args.kind,
            kspacing=kspacing,
            encut_scale=args.encut_scale,
            magnetic=args.magnetic,
            write_slurm=not args.no_slurm,
            submit=False,
            slurm=_slurm_settings(args),
        ),
    )
    return _report(run_dft(config, args.structures, output_dir=args.outdir))


def _cmd_dft_submit(args: argparse.Namespace) -> int:
    """Submit the calculation directories below a root.

    Args:
        args: The parsed arguments.

    Returns:
        The process exit status: ``1`` when nothing was submitted.
    """
    from matdisc.dft.slurm import submit_directory

    submitted = submit_directory(args.root, script_name=args.script_name, dry_run=args.dry_run)
    if not submitted:
        _echo(f"No submission script named {args.script_name} was found below {args.root}.")
        return 1

    accepted = 0
    for name, job_id in sorted(submitted.items()):
        _echo(f"  {name}: {job_id or 'not submitted'}")
        accepted += 1 if job_id else 0
    if args.dry_run:
        _echo(f"Dry run: {len(submitted)} script(s) would be submitted.")
        return 0
    _echo(f"Submitted {accepted} of {len(submitted)} job(s).")
    return 0 if accepted else 1


def _cmd_finetune(args: argparse.Namespace) -> int:
    """Run the fine-tuning stage.

    Args:
        args: The parsed arguments.

    Returns:
        The process exit status.
    """
    from matdisc.pipeline.config import FinetuneConfig, PipelineConfig
    from matdisc.pipeline.stages import run_finetune

    config = PipelineConfig(
        name="finetune",
        work_dir=args.outdir,
        finetune=FinetuneConfig(
            pretrained_model=args.pretrained_model,
            model_branch=args.model_branch,
            numb_steps=args.numb_steps,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            exclude_patterns=args.exclude or [],
            run=args.run,
        ),
    )
    return _report(run_finetune(config, args.dft_dir, output_dir=args.outdir))


def _cmd_competing(args: argparse.Namespace) -> int:
    """Run the competing-phase stage.

    Args:
        args: The parsed arguments.

    Returns:
        The process exit status.
    """
    from matdisc.pipeline.config import CompetingConfig, PipelineConfig
    from matdisc.pipeline.stages import run_competing

    config = PipelineConfig(
        name="competing",
        work_dir=args.outdir,
        competing=CompetingConfig(
            chemsys=args.chemsys,
            include_unary=not args.no_unary,
            include_binary=not args.no_binary,
            include_higher_order=not args.no_higher_order,
            only_icsd=not args.no_icsd,
            only_stable=args.only_stable,
            energy_above_hull_max=args.max_energy_above_hull,
            unique_formula=not args.all_polymorphs,
            allow_partial=args.allow_partial,
            download_structures=not args.no_download,
        ),
    )
    return _report(run_competing(config, output_dir=args.outdir))


def _cmd_screen(args: argparse.Namespace) -> int:
    """Relax a directory of candidates and place them on the hull of the competing phases.

    Args:
        args: The parsed arguments.

    Returns:
        The process exit status.
    """
    from matdisc.pipeline.config import PipelineConfig, ScreeningConfig
    from matdisc.pipeline.stages import run_screening

    config = PipelineConfig(
        name="screen",
        work_dir=args.outdir,
        screening=ScreeningConfig(
            calculator=args.calculator,
            model_path=args.model_path,
            head=args.head,
            fmax=args.fmax,
            max_steps=args.max_steps,
            relax_cell=not args.no_relax_cell,
            tolerance=args.tolerance,
            competing_energy_source=args.competing_energy_source,
            allow_incomplete_hull=args.allow_incomplete_hull,
        ),
    )
    return _report(
        run_screening(
            config,
            structure_dir=args.structures,
            competing_csv=args.competing_csv,
            competing_structure_dir=args.competing_structures,
            output_dir=args.outdir,
        )
    )


def _cmd_hull(args: argparse.Namespace) -> int:
    """Place candidate energies on the convex hull of a set of competing phases.

    Args:
        args: The parsed arguments.

    Returns:
        The process exit status.

    Raises:
        SystemExit: If neither ``--demo`` nor both input tables were given.
    """
    import pandas as pd

    from matdisc.screening.hull import compute_e_above_hull, toy_system

    if args.demo:
        candidates, competing = toy_system()
        _echo("Offline demo: three Ba-Cd-P candidates against a small table of competing phases.")
    else:
        if not args.candidates or not args.competing:
            raise SystemExit("Pass --candidates and --competing, or --demo for the offline example.")
        candidates = pd.read_csv(args.candidates)
        competing = pd.read_csv(args.competing)

    result = compute_e_above_hull(
        candidates,
        competing,
        tolerance=args.tolerance,
        energy_column=args.energy_column,
        composition_column=args.composition_column,
        id_column=args.id_column,
    )

    columns = [args.id_column, args.composition_column, "e_above_hull", "is_stable", "decomposition"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        _echo(result[columns].to_string(index=False))
    _echo("")
    _echo(f"{int(result['is_stable'].sum())} of {len(result)} candidate(s) at or below the hull")
    _echo(f"(stable when e_above_hull <= {args.tolerance:g} eV/atom)")

    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out, index=False)
        _echo(f"Wrote {out}")
    return 0


def _cmd_phonons(args: argparse.Namespace) -> int:
    """Run the phonon stage.

    Args:
        args: The parsed arguments.

    Returns:
        The process exit status.
    """
    from matdisc.pipeline.config import PhononConfig, PipelineConfig, ScreeningConfig
    from matdisc.pipeline.stages import run_phonons

    config = PipelineConfig(
        name="phonons",
        work_dir=args.outdir,
        screening=ScreeningConfig(calculator=args.calculator, model_path=args.model_path, head=args.head),
        phonons=PhononConfig(
            threshold=args.threshold,
            min_cell_length=args.min_cell_length,
            supercell=args.supercell,
            delta=args.delta,
            relax_first=not args.no_relax,
            require_relaxation_converged=not args.allow_unconverged_relaxation,
            plot=not args.no_plot,
            max_structures=args.max_structures,
        ),
    )
    return _report(run_phonons(config, args.structures, output_dir=args.outdir))


def _cmd_pipeline(args: argparse.Namespace) -> int:
    """Write a configuration template, validate a run, or execute one.

    Args:
        args: The parsed arguments.

    Returns:
        The process exit status.
    """
    from matdisc.pipeline.config import load_config, write_config_template
    from matdisc.pipeline.data_interface import validate_pipeline
    from matdisc.pipeline.run import run_pipeline, stages_from

    if args.write_config:
        path = write_config_template(args.write_config, chemsys=args.chemsys)
        _echo(f"Wrote a configuration template to {path}")
        return 0

    if args.validate:
        checks = validate_pipeline(args.validate)
        for stage, check in checks.items():
            mark = "ok  " if check.valid else "-   "
            details = ", ".join(f"{key}={value}" for key, value in check.stats.items())
            _echo(f"{mark}{stage:<11} {details}")
            for message in check.messages:
                _echo(f"      {message}")
        return 0

    config = load_config(args.config, stages=args.stages, work_dir=args.work_dir)
    if args.resume_from:
        tail = stages_from(args.resume_from)
        config.stages = [stage for stage in config.stages if stage in tail]
        config.__post_init__()
        if not config.stages:
            raise SystemExit(f"No stage at or after {args.resume_from} is listed in {args.config}.")

    if args.dry_run:
        _echo(f"{config.name}: would run {', '.join(config.stages)} in {config.work_dir}")
        return 0

    result = run_pipeline(config)
    _echo("")
    for line in result.summary_lines():
        _echo(line)
    return 0 if result.status != "failed" else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the console script.

    Returns:
        The parser, with one subparser per stage.
    """
    # Imported here rather than at module level so that the defaults and the choices cannot
    # drift from the modules that implement them. (This is not a startup-cost optimisation:
    # main() builds the parser before it looks at argv, so every invocation pays for these.)
    from matdisc.common.calculators import DEFAULT_DP_HEAD, SUPPORTED_KINDS
    from matdisc.dft.inputs import DEFAULT_KSPACING, INPUT_SET_KINDS
    from matdisc.generation.mattergen import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_ENERGY_ABOVE_HULL,
        DEFAULT_GUIDANCE_FACTOR,
    )
    from matdisc.pipeline.config import COMPETING_ENERGY_SOURCES
    from matdisc.screening.hull import DEFAULT_TOLERANCE
    from matdisc.screening.phonons import DEFAULT_DELTA, DEFAULT_MIN_CELL_LENGTH, DEFAULT_THRESHOLD
    from matdisc.screening.relax import DEFAULT_FMAX, DEFAULT_MAX_STEPS

    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Generative design, machine-learning prescreening and DFT validation of inorganic materials.",
    )
    parser.add_argument("--version", action="version", version=f"{PROGRAM} {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log debug messages")
    parser.add_argument("-q", "--quiet", action="store_true", help="log warnings and errors only")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    generate = subparsers.add_parser("generate", help="sample candidate structures with MatterGen")
    generate.add_argument(
        "--chemsys",
        action="append",
        required=True,
        metavar="Ba-Cd-P",
        help="chemical system to generate; repeat for several",
    )
    generate.add_argument("-n", "--n-structures", type=int, default=24, help="structures per chemical system")
    generate.add_argument("-o", "--outdir", required=True, help="output directory")
    generate.add_argument("--model-path", help="MatterGen checkpoint directory ($MATTERGEN_MODEL_PATH)")
    generate.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="sampling batch size")
    generate.add_argument(
        "--energy-above-hull",
        type=float,
        default=DEFAULT_ENERGY_ABOVE_HULL,
        help="property-guidance target in eV/atom",
    )
    generate.add_argument("--guidance-factor", type=float, default=DEFAULT_GUIDANCE_FACTOR, help="guidance strength")
    generate.add_argument("--no-skip-existing", action="store_true", help="regenerate even if output exists")
    generate.set_defaults(func=_cmd_generate)

    cluster = subparsers.add_parser("cluster", help="select representative structures for DFT")
    cluster.add_argument("-s", "--structures", required=True, help="structure file or directory")
    cluster.add_argument("-n", "--n-representatives", type=int, default=10, help="structures to select")
    cluster.add_argument("-o", "--outdir", required=True, help="output directory")
    cluster.add_argument("--model-path", help="DeePMD-kit checkpoint ($DPA3_MODEL_PATH)")
    cluster.add_argument("--head", default=DEFAULT_DP_HEAD, help="model branch of a multi-task checkpoint")
    cluster.add_argument("--threshold", type=float, help="BIRCH threshold; tuned automatically when omitted")
    cluster.add_argument("--k-per-cluster", type=int, default=1, help="rows taken from each cluster")
    cluster.set_defaults(func=_cmd_cluster)

    dft_inputs = subparsers.add_parser("dft-inputs", help="write VASP inputs for a set of structures")
    dft_inputs.add_argument("-s", "--structures", required=True, help="structure file or directory")
    dft_inputs.add_argument("-o", "--outdir", required=True, help="output directory")
    dft_inputs.add_argument("--kind", default="relax", choices=INPUT_SET_KINDS, help="which input set to write")
    dft_inputs.add_argument(
        "--kspacing",
        type=float,
        default=DEFAULT_KSPACING,
        help="k-point spacing in 1/A written as KSPACING; pass 0 to write an explicit KPOINTS file instead",
    )
    dft_inputs.add_argument("--encut-scale", type=float, default=1.3, help="ENCUT as a multiple of max ENMAX")
    dft_inputs.add_argument("--magnetic", action="store_true", help="run spin-polarised")
    dft_inputs.add_argument("--no-slurm", action="store_true", help="do not write submission scripts")
    dft_inputs.add_argument("--job-name", help="SLURM job name")
    dft_inputs.add_argument("--partition", help="SLURM partition")
    dft_inputs.add_argument("--nodes", type=int, help="nodes per job")
    dft_inputs.add_argument("--ntasks", type=int, help="MPI tasks per job")
    dft_inputs.add_argument("--walltime", type=int, metavar="MINUTES", help="walltime in minutes")
    dft_inputs.add_argument("--account", help="SLURM account")
    dft_inputs.set_defaults(func=_cmd_dft_inputs)

    dft_submit = subparsers.add_parser("dft-submit", help="submit prepared calculation directories")
    dft_submit.add_argument("-r", "--root", required=True, help="directory holding the calculation directories")
    dft_submit.add_argument("--script-name", default="submit.slurm", help="submission script to look for")
    dft_submit.add_argument("--dry-run", action="store_true", help="report what would be submitted")
    dft_submit.set_defaults(func=_cmd_dft_submit)

    finetune = subparsers.add_parser("finetune", help="build a training set from DFT output and fine-tune")
    finetune.add_argument("-d", "--dft-dir", required=True, help="directory of finished DFT calculations")
    finetune.add_argument("-o", "--outdir", required=True, help="output directory")
    finetune.add_argument("--pretrained-model", help="checkpoint to fine-tune from ($DPA3_MODEL_PATH)")
    finetune.add_argument("--model-branch", default=DEFAULT_DP_HEAD, help="pretraining head to start from")
    finetune.add_argument("--numb-steps", type=int, default=100000, help="training steps")
    finetune.add_argument("--batch-size", type=int, default=4, help="training batch size")
    finetune.add_argument("--val-ratio", type=float, default=0.1, help="fraction held out for validation")
    finetune.add_argument(
        "--exclude",
        action="append",
        metavar="PATTERN",
        help="exclude systems whose name contains this; repeat for several",
    )
    finetune.add_argument(
        "--run",
        action="store_true",
        help="run the training command; without it only the dataset and config are written",
    )
    finetune.set_defaults(func=_cmd_finetune)

    competing = subparsers.add_parser("competing", help="harvest competing phases from the Materials Project")
    competing.add_argument("-c", "--chemsys", required=True, metavar="Ba-Cd-P", help="chemical system to search")
    competing.add_argument("-o", "--outdir", required=True, help="output directory")
    competing.add_argument("--no-unary", action="store_true", help="drop elemental phases")
    competing.add_argument("--no-binary", action="store_true", help="drop binary phases")
    competing.add_argument("--no-higher-order", action="store_true", help="drop ternary and higher phases")
    competing.add_argument("--no-icsd", action="store_true", help="keep materials without an ICSD cross-reference")
    competing.add_argument("--only-stable", action="store_true", help="keep only phases reported as stable")
    competing.add_argument(
        "--max-energy-above-hull",
        type=float,
        metavar="EV_PER_ATOM",
        help="drop phases further above the Materials Project hull than this",
    )
    competing.add_argument(
        "--all-polymorphs",
        action="store_true",
        help="keep every polymorph; by default one per formula is kept, the lowest in Materials Project energy",
    )
    competing.add_argument(
        "--allow-partial",
        action="store_true",
        help="accept a reference set in which a subsystem query failed",
    )
    competing.add_argument("--no-download", action="store_true", help="write the table only, no structure files")
    competing.set_defaults(func=_cmd_competing)

    screen = subparsers.add_parser("screen", help="relax candidates and place them on the hull of competing phases")
    screen.add_argument("-s", "--structures", required=True, help="candidate structure file or directory")
    screen.add_argument(
        "--competing-structures",
        help="directory of competing-phase structures, relaxed here with the same calculator",
    )
    screen.add_argument(
        "--competing-csv",
        help="competing-phase table; its energies are used only with --competing-energy-source table",
    )
    screen.add_argument("-o", "--outdir", required=True, help="output directory")
    screen.add_argument("--calculator", default="dp", choices=SUPPORTED_KINDS, help="calculator backend")
    screen.add_argument("--model-path", help="DeePMD-kit checkpoint ($DPA3_MODEL_PATH)")
    screen.add_argument("--head", default=DEFAULT_DP_HEAD, help="model branch of a multi-task checkpoint")
    screen.add_argument("--fmax", type=float, default=DEFAULT_FMAX, help="force convergence criterion in eV/A")
    screen.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="cap on optimizer steps")
    screen.add_argument("--no-relax-cell", action="store_true", help="relax the positions only, not the cell")
    screen.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="a candidate is stable when e_above_hull <= this, in eV/atom",
    )
    screen.add_argument(
        "--competing-energy-source",
        default="relax",
        choices=COMPETING_ENERGY_SOURCES,
        help="relax the competing structures here, or read their energies from the table",
    )
    screen.add_argument(
        "--allow-incomplete-hull",
        action="store_true",
        help="build the hull even when a competing phase could not be relaxed",
    )
    screen.set_defaults(func=_cmd_screen)

    hull = subparsers.add_parser("hull", help="place candidate energies on the convex hull")
    hull.add_argument("--candidates", help="CSV with id, composition and a per-atom energy column")
    hull.add_argument("--competing", help="CSV of competing phases, from the competing subcommand")
    hull.add_argument(
        "--demo", action="store_true", help="run the offline example instead: no files, no credentials, no network"
    )
    hull.add_argument("-o", "--output", help="write the result table here")
    hull.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="a candidate is stable when e_above_hull <= this, in eV/atom",
    )
    hull.add_argument("--energy-column", default="energy_per_atom", help="per-atom energy column of both tables")
    hull.add_argument("--composition-column", default="composition", help="formula column of both tables")
    hull.add_argument("--id-column", default="id", help="identifier column of the candidates")
    hull.set_defaults(func=_cmd_hull)

    phonons = subparsers.add_parser("phonons", help="check structures for imaginary phonon frequencies")
    phonons.add_argument("-s", "--structures", required=True, help="structure file or directory")
    phonons.add_argument("-o", "--outdir", required=True, help="output directory")
    phonons.add_argument("--calculator", default="dp", choices=SUPPORTED_KINDS, help="calculator backend")
    phonons.add_argument("--model-path", help="DeePMD-kit checkpoint ($DPA3_MODEL_PATH)")
    phonons.add_argument("--head", default=DEFAULT_DP_HEAD, help="model branch of a multi-task checkpoint")
    phonons.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="lowest frequency a stable structure may have, in THz",
    )
    phonons.add_argument(
        "--min-cell-length", type=float, default=DEFAULT_MIN_CELL_LENGTH, help="minimum supercell edge in A"
    )
    phonons.add_argument(
        "--supercell", type=int, nargs=3, metavar=("NA", "NB", "NC"), help="explicit supercell repetitions"
    )
    phonons.add_argument("--delta", type=float, default=DEFAULT_DELTA, help="finite displacement in A")
    phonons.add_argument("--no-relax", action="store_true", help="displace the structure as given")
    phonons.add_argument(
        "--allow-unconverged-relaxation",
        action="store_true",
        help="compute the spectrum even when the pre-relaxation did not reach its force criterion",
    )
    phonons.add_argument("--no-plot", action="store_true", help="write the data files without a plot")
    phonons.add_argument("--max-structures", type=int, help="check at most this many structures")
    phonons.set_defaults(func=_cmd_phonons)

    pipeline = subparsers.add_parser("pipeline", help="run the stages listed in a configuration file")
    source = pipeline.add_mutually_exclusive_group(required=True)
    source.add_argument("-c", "--config", help="YAML configuration to run")
    source.add_argument("--write-config", metavar="PATH", help="write a configuration template and exit")
    source.add_argument("--validate", metavar="WORK_DIR", help="report what each stage directory holds and exit")
    pipeline.add_argument(
        "--chemsys", default="Ba-Cd-P", metavar="Ba-Cd-P", help="chemical system to fill a written template with"
    )
    pipeline.add_argument(
        "--stages", nargs="+", metavar="STAGE", help="run these stages instead of the ones in the file"
    )
    pipeline.add_argument("--resume-from", metavar="STAGE", help="run the configured stages from this one onwards")
    pipeline.add_argument("--work-dir", help="working directory, overriding the file")
    pipeline.add_argument("--dry-run", action="store_true", help="report the stages that would run and exit")
    pipeline.set_defaults(func=_cmd_pipeline)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the console script.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        The process exit status: ``0`` on success, ``1`` on a reported failure and ``130``
        when interrupted.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 1

    from matdisc.common.logging import configure_logging

    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    configure_logging(level=level, stream=sys.stderr)

    try:
        return int(handler(args))
    except KeyboardInterrupt:
        print(f"{PROGRAM}: interrupted", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, KeyError, TypeError, OSError, ImportError) as error:
        if args.verbose:
            raise
        print(f"{PROGRAM}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())
