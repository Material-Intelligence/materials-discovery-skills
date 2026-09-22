"""Configuration for the end-to-end pipeline.

A run is described by one YAML file. The top level names the run, the working directory and
the stages to execute; every stage then has its own section, so no key is shared between two
stages and no stage can receive an argument meant for another one.

Sections are plain dataclasses that also implement :class:`~monty.json.MSONable`, so a
configuration round-trips through ``as_dict``/``from_dict`` and can be written next to the
results of the run it produced.

Credentials are never part of a configuration file. The Materials Project key is read from
``MP_API_KEY`` by :func:`matdisc.common.mp.get_api_key`, and the machine-learning potential
checkpoint falls back to ``$DPA3_MODEL_PATH`` when no path is given.

Path-valued fields (``work_dir``, ``model_path``, ``pretrained_model``, the ``*_dir`` inputs
and ``csv_file``) are expanded with ``$VAR``/``~`` when the configuration is loaded. No other
string is expanded, so shell variables that must survive into a generated job script -- such
as ``$VASP_CMD`` in the SLURM section -- are written out literally.

Example:
    ```yaml
    name: BaCdP
    work_dir: ./runs/BaCdP
    stages: [competing, screening]

    competing:
      chemsys: Ba-Cd-P
      only_icsd: true

    screening:
      structure_dir: ./candidates
      model_path: $DPA3_MODEL_PATH
    ```
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from monty.json import MSONable

from matdisc.clustering.sampling import DEFAULT_K_PER_CLUSTER
from matdisc.common.calculators import DEFAULT_DP_HEAD, SUPPORTED_KINDS
from matdisc.common.logging import get_logger
from matdisc.dft.inputs import DEFAULT_ENCUT_SCALE, DEFAULT_KSPACING, INPUT_SET_KINDS
from matdisc.generation.mattergen import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_ENERGY_ABOVE_HULL,
    DEFAULT_GUIDANCE_FACTOR,
)
from matdisc.screening.hull import DEFAULT_TOLERANCE
from matdisc.screening.phonons import DEFAULT_DELTA, DEFAULT_MIN_CELL_LENGTH
from matdisc.screening.phonons import DEFAULT_RELAX_FMAX as DEFAULT_PHONON_FMAX
from matdisc.screening.phonons import DEFAULT_RELAX_MAX_STEPS as DEFAULT_PHONON_MAX_STEPS
from matdisc.screening.phonons import DEFAULT_THRESHOLD as DEFAULT_PHONON_THRESHOLD
from matdisc.screening.relax import DEFAULT_FMAX, DEFAULT_MAX_STEPS, OPTIMIZERS

LOGGER = get_logger(__name__)

STAGE_ORDER = (
    "generation",
    "clustering",
    "dft",
    "finetune",
    "competing",
    "screening",
    "phonons",
)
"""The stages of a full run, in the order :func:`matdisc.pipeline.run.run_pipeline` executes them."""

COMPETING_ENERGY_SOURCES = ("relax", "table")
"""Where the competing-phase energies used by the hull come from; see :class:`ScreeningConfig`."""

__all__ = [
    "COMPETING_ENERGY_SOURCES",
    "STAGE_ORDER",
    "ClusteringConfig",
    "CompetingConfig",
    "DftConfig",
    "FinetuneConfig",
    "GenerationConfig",
    "PhononConfig",
    "PipelineConfig",
    "ScreeningConfig",
    "load_config",
    "write_config_template",
]


def _expand_path(value: str | os.PathLike[str] | None) -> str | None:
    """Expand ``$VAR`` and ``~`` in a path-valued configuration entry.

    Args:
        value: The raw entry, or ``None``.

    Returns:
        The expanded path, or ``None`` when nothing was given.
    """
    if value is None:
        return None
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    if "$" in expanded:
        LOGGER.warning("Path %r still contains an unresolved variable after expansion: %s", str(value), expanded)
    return expanded


class _Section(MSONable):
    """Loading, validation and serialisation shared by every configuration section."""

    @classmethod
    def section_name(cls) -> str:
        """Return the YAML key this section is read from.

        Returns:
            The lowercase section name, for example ``"generation"``.
        """
        return cls.__name__.removesuffix("Config").lower()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> Any:
        """Build a section from a plain mapping, rejecting unknown keys.

        Args:
            mapping: The section of the YAML document, or ``None`` for the defaults.

        Returns:
            An instance of the section class.

        Raises:
            ValueError: If the mapping holds a key the section does not define. A silently
                ignored key is a typo that changes the science without saying so.
        """
        data = {key: value for key, value in dict(mapping or {}).items() if not key.startswith("@")}
        known = {f.name for f in dataclass_fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"Unknown key(s) in the '{cls.section_name()}' section: {', '.join(unknown)}. "
                f"Known keys: {', '.join(sorted(known))}."
            )
        return cls(**data)

    def as_dict(self) -> dict[str, Any]:
        """Serialise the section, including the class markers Monty uses to decode it.

        Returns:
            A JSON-serialisable dictionary.
        """
        data: dict[str, Any] = {"@module": type(self).__module__, "@class": type(self).__name__}
        data.update(self.to_mapping())
        return data

    def to_mapping(self) -> dict[str, Any]:
        """Serialise the section as plain YAML-friendly data, without class markers.

        Returns:
            A dictionary of the section's fields.
        """
        out: dict[str, Any] = {}
        for f in dataclass_fields(self):
            value = getattr(self, f.name)
            out[f.name] = value.to_mapping() if isinstance(value, _Section) else value
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Any:
        """Rebuild a section from :meth:`as_dict` output.

        Args:
            d: The serialised section.

        Returns:
            An instance of the section class.
        """
        return cls.from_mapping(d)


@dataclass
class GenerationConfig(_Section):
    """Generative structure design with MatterGen.

    Attributes:
        chemical_systems: Hyphen-separated systems to generate, for example ``["Ba-Cd-P"]``.
        n_structures: Structures requested per chemical system.
        model_path: MatterGen checkpoint directory. ``None`` falls back to
            ``$MATTERGEN_MODEL_PATH``.
        batch_size: Sampling batch size handed to MatterGen.
        energy_above_hull: Property-guidance target in eV/atom, or ``None`` for unguided
            sampling. Defaults to
            :data:`matdisc.generation.mattergen.DEFAULT_ENERGY_ABOVE_HULL`, the same value the
            ``generate`` subcommand and :func:`matdisc.generation.mattergen.generate` use.
        guidance_factor: Classifier-free guidance strength used with that target.
        skip_existing: Reuse an output directory that already holds generated structures.
        structure_dir: Directory of structures to use instead of generating any. When it is
            set the stage reads that directory and MatterGen is never invoked.
    """

    chemical_systems: list[str] = field(default_factory=list)
    n_structures: int = 24
    model_path: str | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    energy_above_hull: float | None = DEFAULT_ENERGY_ABOVE_HULL
    guidance_factor: float = DEFAULT_GUIDANCE_FACTOR
    skip_existing: bool = True
    structure_dir: str | None = None

    def __post_init__(self) -> None:
        """Normalise the paths and validate the requested counts.

        Raises:
            ValueError: If ``n_structures`` is not positive.
        """
        self.model_path = _expand_path(self.model_path)
        self.structure_dir = _expand_path(self.structure_dir)
        self.chemical_systems = [str(system) for system in self.chemical_systems]
        if self.n_structures <= 0:
            raise ValueError(f"generation.n_structures must be positive, got {self.n_structures}.")


@dataclass
class ClusteringConfig(_Section):
    """Descriptor extraction and representative-subset selection.

    Attributes:
        model_path: DeePMD-kit checkpoint used for the descriptors. ``None`` falls back to
            ``$DPA3_MODEL_PATH``.
        head: Model branch of a multi-task checkpoint, or ``None`` for the model default.
        n_representatives: Number of structures to pass on to DFT.
        threshold: BIRCH clustering threshold. ``None`` tunes it to hit
            ``n_representatives``.
        k_per_cluster: Rows taken from each cluster.
        structure_dir: Structures to cluster, when the generation stage did not run.
    """

    model_path: str | None = None
    head: str | None = DEFAULT_DP_HEAD
    n_representatives: int = 10
    threshold: float | None = None
    k_per_cluster: int = DEFAULT_K_PER_CLUSTER
    structure_dir: str | None = None

    def __post_init__(self) -> None:
        """Normalise the paths and validate the requested subset size.

        Raises:
            ValueError: If ``n_representatives`` is not positive.
        """
        self.model_path = _expand_path(self.model_path)
        self.structure_dir = _expand_path(self.structure_dir)
        if self.n_representatives <= 0:
            raise ValueError(f"clustering.n_representatives must be positive, got {self.n_representatives}.")


@dataclass
class DftConfig(_Section):
    """VASP input generation and optional submission.

    Attributes:
        kind: Input set to write, one of :data:`matdisc.dft.inputs.INPUT_SET_KINDS`.
        kspacing: k-point spacing in A^-1, or ``None`` to write an explicit KPOINTS grid.
        encut_scale: ENCUT as a multiple of the largest ENMAX of the POTCARs.
        magnetic: Whether to run spin-polarised.
        user_incar_settings: Extra INCAR tags applied on top of the ported settings.
        write_slurm: Write a submission script into every calculation directory.
        submit: Submit those scripts with ``sbatch``.
        slurm: Resource request and environment, read into
            :class:`matdisc.dft.slurm.SlurmSettings`.
        structure_dir: Structures to prepare, when the clustering stage did not run.
    """

    kind: str = "relax"
    kspacing: float | None = DEFAULT_KSPACING
    encut_scale: float | None = DEFAULT_ENCUT_SCALE
    magnetic: bool = False
    user_incar_settings: dict[str, Any] = field(default_factory=dict)
    write_slurm: bool = True
    submit: bool = False
    slurm: dict[str, Any] = field(default_factory=dict)
    structure_dir: str | None = None

    def __post_init__(self) -> None:
        """Normalise the paths and validate the input-set kind.

        Raises:
            ValueError: If ``kind`` is not a known input set.
        """
        self.structure_dir = _expand_path(self.structure_dir)
        if self.kind not in INPUT_SET_KINDS:
            raise ValueError(f"dft.kind must be one of {', '.join(INPUT_SET_KINDS)}, got {self.kind!r}.")


@dataclass
class FinetuneConfig(_Section):
    """Dataset preparation and fine-tuning of the machine-learning potential.

    Attributes:
        pretrained_model: Checkpoint to fine-tune from. ``None`` falls back to
            ``$DPA3_MODEL_PATH``.
        model_branch: Pretraining head the fine-tuning starts from.
        numb_steps: Training steps.
        batch_size: Training batch size.
        val_ratio: Fraction of the converted systems held out for validation.
        exclude_patterns: Name fragments of the directory holding an OUTCAR that exclude it,
            as in :func:`matdisc.finetune.dataset.find_system_outcars`. With the flat layout
            the DFT stage writes, that directory is the calculation directory, so a fragment
            here drops a whole system.
        type_map: Element order of the dataset. ``None`` reads it from the converted data.
        run: Run the training command. When it is ``False`` the stage prepares the dataset
            and the configuration, reports the command to run, and returns no checkpoint.
        dft_dir: DFT output tree to convert, when the DFT stage did not run.
    """

    pretrained_model: str | None = None
    model_branch: str = DEFAULT_DP_HEAD
    numb_steps: int = 100000
    batch_size: int = 4
    val_ratio: float = 0.1
    exclude_patterns: list[str] = field(default_factory=list)
    type_map: list[str] | None = None
    run: bool = False
    dft_dir: str | None = None

    def __post_init__(self) -> None:
        """Normalise the paths and validate the validation split.

        Raises:
            ValueError: If ``val_ratio`` is outside ``[0, 1)``.
        """
        self.pretrained_model = _expand_path(self.pretrained_model)
        self.dft_dir = _expand_path(self.dft_dir)
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError(f"finetune.val_ratio must be in [0, 1), got {self.val_ratio}.")


@dataclass
class CompetingConfig(_Section):
    """Competing-phase harvest from the Materials Project.

    Attributes:
        chemsys: Chemical system to search, for example ``"Ba-Cd-P"``. ``None`` takes the
            first system of the generation section.
        include_unary: Keep elemental phases.
        include_binary: Keep binary phases.
        include_higher_order: Keep ternary and higher phases.
        only_icsd: Keep only materials cross-referenced to an ICSD entry.
        only_stable: Keep only phases the Materials Project reports as stable.
        energy_above_hull_max: Drop phases further above the Materials Project hull than
            this, in eV/atom, or ``None`` to keep them all.
        unique_formula: Keep one entry per reduced formula, the one with the lowest Materials
            Project formation energy. The polymorph is therefore chosen on Materials Project
            DFT energies, not on the calculator the screening stage relaxes with; set this to
            false to carry every polymorph into the hull.
        allow_partial: Accept a reference set in which a subsystem query failed. The hull
            then rests on fewer phases than the chemical system has, so the default is to
            stop instead.
        download_structures: Download a structure file for every phase found.
        csv_file: A competing-phase table to use instead of searching.
        structure_dir: A directory of competing-phase structures to use instead of
            downloading them.
    """

    chemsys: str | None = None
    include_unary: bool = True
    include_binary: bool = True
    include_higher_order: bool = True
    only_icsd: bool = True
    only_stable: bool = False
    energy_above_hull_max: float | None = None
    unique_formula: bool = True
    allow_partial: bool = False
    download_structures: bool = True
    csv_file: str | None = None
    structure_dir: str | None = None

    def __post_init__(self) -> None:
        """Normalise the path-valued entries."""
        self.csv_file = _expand_path(self.csv_file)
        self.structure_dir = _expand_path(self.structure_dir)


@dataclass
class ScreeningConfig(_Section):
    """Relaxation of the candidates and their placement on the convex hull.

    Candidates and competing phases must sit on one energy scale. With the default
    ``competing_energy_source="relax"`` the downloaded competing structures are relaxed with
    the same calculator as the candidates, so both sides are machine-learning-potential
    energies. ``"table"`` reads the competing energies from the harvested table instead,
    which is only correct when the candidate energies come from the same kind of calculation
    (Materials Project DFT energies, for example, must not be mixed with potential energies).

    Attributes:
        calculator: Calculator kind, one of
            :data:`matdisc.common.calculators.SUPPORTED_KINDS`.
        model_path: Checkpoint for the calculator. ``None`` uses the checkpoint the
            fine-tuning stage produced, and otherwise ``$DPA3_MODEL_PATH``.
        head: Model branch of a multi-task checkpoint.
        optimizer: ASE optimizer name.
        fmax: Force convergence criterion in eV/A.
        max_steps: Cap on optimizer steps.
        relax_cell: Relax the cell as well as the atomic positions.
        tolerance: A candidate is stable when ``e_above_hull <= tolerance``, in eV/atom.
        energy_column: Per-atom energy column read from both tables.
        competing_energy_source: ``"relax"`` or ``"table"``; see above.
        require_converged: Only a candidate whose relaxation reached ``fmax`` may be reported
            as stable. A structure still carrying large forces has an energy that belongs to
            no minimum, so with this set such candidates are excluded from the stable set and
            counted separately instead.
        allow_incomplete_hull: Build the hull even when a competing phase could not be
            relaxed or did not converge. A missing competing phase can only lower the hull,
            which moves every candidate toward stability, so the stage stops by default
            rather than report a verdict against a hull it knows is incomplete.
        structure_dir: Candidate structures, when the generation stage did not run.
    """

    calculator: str = "dp"
    model_path: str | None = None
    head: str | None = DEFAULT_DP_HEAD
    optimizer: str = "BFGS"
    fmax: float = DEFAULT_FMAX
    max_steps: int = DEFAULT_MAX_STEPS
    relax_cell: bool = True
    tolerance: float = DEFAULT_TOLERANCE
    energy_column: str = "energy_per_atom"
    competing_energy_source: str = "relax"
    require_converged: bool = True
    allow_incomplete_hull: bool = False
    structure_dir: str | None = None

    def __post_init__(self) -> None:
        """Normalise the paths and validate the calculator, optimizer and energy source.

        Raises:
            ValueError: If the calculator kind, the optimizer or the energy source is not
                one of the supported values.
        """
        self.model_path = _expand_path(self.model_path)
        self.structure_dir = _expand_path(self.structure_dir)
        if self.calculator not in SUPPORTED_KINDS:
            raise ValueError(
                f"screening.calculator must be one of {', '.join(SUPPORTED_KINDS)}, got {self.calculator!r}."
            )
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"screening.optimizer must be one of {', '.join(OPTIMIZERS)}, got {self.optimizer!r}.")
        if self.competing_energy_source not in COMPETING_ENERGY_SOURCES:
            raise ValueError(
                "screening.competing_energy_source must be one of "
                f"{', '.join(COMPETING_ENERGY_SOURCES)}, got {self.competing_energy_source!r}."
            )


@dataclass
class PhononConfig(_Section):
    """Dynamical-stability check of the hull survivors.

    Attributes:
        threshold: A structure is dynamically stable when its lowest frequency is above this
            value, in THz.
        min_cell_length: Minimum supercell edge length in A, used when ``supercell`` is
            ``None``.
        supercell: Explicit supercell repetitions, or ``None`` to derive them.
        delta: Finite displacement in A.
        relax_first: Relax the atomic positions before displacing them.
        fmax: Force criterion of that relaxation, in eV/A.
        max_steps: Step cap of that relaxation.
        require_relaxation_converged: Skip the spectrum when that relaxation hit the step cap
            without reaching ``fmax``. Displacing a geometry that is not stationary produces
            imaginary modes that belong to the leftover forces, not to the material.
        plot: Write a spectrum plot beside the data files.
        max_structures: Cap on how many hull survivors are checked, or ``None`` for all of
            them. Each check is expensive.
        structure_dir: Structures to check, when the screening stage did not run.
    """

    threshold: float = DEFAULT_PHONON_THRESHOLD
    min_cell_length: float = DEFAULT_MIN_CELL_LENGTH
    supercell: list[int] | None = None
    delta: float = DEFAULT_DELTA
    relax_first: bool = True
    fmax: float = DEFAULT_PHONON_FMAX
    max_steps: int = DEFAULT_PHONON_MAX_STEPS
    require_relaxation_converged: bool = True
    plot: bool = True
    max_structures: int | None = None
    structure_dir: str | None = None

    def __post_init__(self) -> None:
        """Normalise the paths and validate the supercell.

        Raises:
            ValueError: If ``supercell`` is given but is not three positive integers.
        """
        self.structure_dir = _expand_path(self.structure_dir)
        if self.supercell is not None:
            values = [int(v) for v in self.supercell]
            if len(values) != 3 or any(v < 1 for v in values):
                raise ValueError(f"phonons.supercell must be three positive integers, got {self.supercell}.")
            self.supercell = values


_SECTION_CLASSES: dict[str, type[_Section]] = {
    "generation": GenerationConfig,
    "clustering": ClusteringConfig,
    "dft": DftConfig,
    "finetune": FinetuneConfig,
    "competing": CompetingConfig,
    "screening": ScreeningConfig,
    "phonons": PhononConfig,
}


@dataclass
class PipelineConfig(_Section):
    """A complete run: what to call it, where to put it and which stages to execute.

    Attributes:
        name: Label for the run, used in log lines and in the result file.
        work_dir: Directory the stage outputs are written into.
        stages: Stages to execute, in :data:`STAGE_ORDER`. Listing fewer stages is how a run
            is resumed or restricted; a stage that is left out is not executed and its input
            must then be supplied through the following stage's ``*_dir``/``csv_file`` entry.
        generation: The generation section.
        clustering: The clustering section.
        dft: The DFT section.
        finetune: The fine-tuning section.
        competing: The competing-phase section.
        screening: The relaxation and convex-hull section.
        phonons: The phonon section.
    """

    name: str = "matdisc-run"
    work_dir: str = "./matdisc_run"
    stages: list[str] = field(default_factory=lambda: list(STAGE_ORDER))
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    dft: DftConfig = field(default_factory=DftConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    competing: CompetingConfig = field(default_factory=CompetingConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    phonons: PhononConfig = field(default_factory=PhononConfig)

    def __post_init__(self) -> None:
        """Normalise the working directory and validate the stage list.

        Raises:
            ValueError: If a stage name is unknown or listed twice.
        """
        self.work_dir = _expand_path(self.work_dir) or "./matdisc_run"
        names = [str(stage) for stage in self.stages]
        unknown = [stage for stage in names if stage not in STAGE_ORDER]
        if unknown:
            raise ValueError(f"Unknown stage(s): {', '.join(unknown)}. Known stages: {', '.join(STAGE_ORDER)}.")
        duplicates = sorted({stage for stage in names if names.count(stage) > 1})
        if duplicates:
            raise ValueError(f"Stage(s) listed more than once: {', '.join(duplicates)}.")
        self.stages = [stage for stage in STAGE_ORDER if stage in names]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "PipelineConfig":
        """Build a configuration from a parsed YAML document.

        Args:
            mapping: The document, or ``None`` for an all-default configuration.

        Returns:
            The configuration.

        Raises:
            ValueError: If the document holds an unknown top-level key, or a section is not
                a mapping.
        """
        data = {key: value for key, value in dict(mapping or {}).items() if not key.startswith("@")}
        known = {f.name for f in dataclass_fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown top-level key(s): {', '.join(unknown)}. Known keys: {', '.join(sorted(known))}.")

        sections: dict[str, Any] = {}
        for section_name, section_cls in _SECTION_CLASSES.items():
            raw = data.pop(section_name, None)
            if raw is not None and not isinstance(raw, Mapping):
                raise ValueError(f"Section '{section_name}' must be a mapping, got {type(raw).__name__}.")
            sections[section_name] = section_cls.from_mapping(raw)

        return cls(**data, **sections)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Read a configuration from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            The configuration.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the document is not a mapping, or holds an unknown key.
        """
        file_path = Path(path).expanduser()
        if not file_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        document = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        if document is not None and not isinstance(document, Mapping):
            raise ValueError(f"{file_path} must contain a YAML mapping, got {type(document).__name__}.")
        LOGGER.info("Read configuration %s", file_path)
        return cls.from_mapping(document)

    def to_yaml(self) -> str:
        """Render the configuration as a YAML document.

        Returns:
            The document, with the sections in :data:`STAGE_ORDER`.
        """
        return yaml.safe_dump(self.to_mapping(), sort_keys=False, default_flow_style=False)

    def write_yaml(self, path: str | Path) -> Path:
        """Write the configuration to a YAML file.

        Args:
            path: Destination path. Parent directories are created.

        Returns:
            The path written.
        """
        file_path = Path(path).expanduser().resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(self.to_yaml(), encoding="utf-8")
        LOGGER.info("Wrote configuration %s", file_path)
        return file_path

    def stage_dir(self, stage: str) -> Path:
        """Return the output directory of one stage.

        Args:
            stage: A stage name from :data:`STAGE_ORDER`.

        Returns:
            ``<work_dir>/<index>_<stage>``, numbered by position in :data:`STAGE_ORDER`.

        Raises:
            ValueError: If the stage name is unknown.
        """
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown stage {stage!r}. Known stages: {', '.join(STAGE_ORDER)}.")
        index = STAGE_ORDER.index(stage) + 1
        return Path(self.work_dir).expanduser().resolve() / f"{index:02d}_{stage}"


def load_config(path: str | Path, stages: Sequence[str] | None = None, work_dir: str | None = None) -> PipelineConfig:
    """Read a configuration file and apply the command-line overrides.

    Args:
        path: Path to the YAML file.
        stages: Stage names that replace the file's ``stages`` list, or ``None`` to keep it.
        work_dir: Working directory that replaces the file's ``work_dir``, or ``None``.

    Returns:
        The configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the document or an override is invalid.
    """
    config = PipelineConfig.from_yaml(path)
    if stages is not None:
        config.stages = list(stages)
    if work_dir is not None:
        config.work_dir = work_dir
    config.__post_init__()
    return config


def write_config_template(path: str | Path, chemsys: str = "Ba-Cd-P") -> Path:
    """Write a configuration file holding every key with its default value.

    Args:
        path: Destination path.
        chemsys: Chemical system to fill the generation and competing sections with.

    Returns:
        The path written.
    """
    config = PipelineConfig(
        name=chemsys,
        work_dir=f"./runs/{chemsys}",
        generation=GenerationConfig(chemical_systems=[chemsys]),
        competing=CompetingConfig(chemsys=chemsys),
    )
    return config.write_yaml(path)
