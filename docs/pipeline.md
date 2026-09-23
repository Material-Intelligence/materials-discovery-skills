# The pipeline, stage by stage

Inputs, outputs, commands and requirements of the seven stages, and how they map onto the
workflow of [arXiv:2606.10251](https://arxiv.org/abs/2606.10251).

## What the paper does, and what is here

The paper reports a high-throughput discovery workflow for metal phosphides that combines
generative structure design, machine-learning interatomic potentials and targeted DFT. Its
abstract describes four moving parts: candidate structures produced by ICSD-derived Wyckoff-site
substitution and by conditional generation with MatterGen; a DPA-3 machine-learning potential
fine-tuned on in-domain DFT labels; prescreening of thermodynamic and dynamical stability with
that potential; and DFT validation of what survives, including HSE06 band gaps. It reports 3,574
previously unreported stable phosphide structures, 196 of them semiconductors with HSE06 gaps
between 0 and 3.0 eV, and from those a shortlist of optoelectronic and thermoelectric candidates.

This package implements the parts of that loop that are procedure rather than result:

| Part of the paper's workflow | Here |
|---|---|
| Wyckoff-site substitution from ICSD prototypes | **Not implemented.** No substitution engine ships here |
| Conditional generation with MatterGen | Stage 1, `matdisc generate` |
| Choosing which candidates are worth DFT | Stage 2, `matdisc cluster` |
| DFT labels for fine-tuning, and DFT validation | Stage 3, `matdisc dft-inputs` / `dft-submit` |
| Domain fine-tuning of the DPA-3 potential | Stage 4, `matdisc finetune` |
| Reference phases for the thermodynamic screen | Stage 5, `matdisc competing` |
| Thermodynamic prescreening (convex hull) | Stage 6, `matdisc hull` and the `screening` stage |
| Dynamical prescreening (phonons) | Stage 7, `matdisc phonons` |
| The candidate lists, checkpoints and DFT output | **Not here.** None of the paper's data ships |

HSE06 band gaps correspond to `matdisc dft-inputs --kind hse-static`, the single-point hybrid run
on an already-relaxed geometry; it writes the inputs, and running them is your own VASP. The
separate `--kind hse-relax` is a hybrid *geometry optimisation*, which is a different and far more
expensive calculation.

## How a run is laid out

`matdisc pipeline -c config.yaml` runs the stages a YAML file lists, in this fixed order:

```
generation -> clustering -> dft -> finetune -> competing -> screening -> phonons
```

Each stage writes into `<work_dir>/<NN>_<stage>/`, numbered by that order, and the run writes
`<work_dir>/pipeline_result.json` holding every stage's status, statistics and artifact paths,
plus the configuration that produced them. A single-stage command (`matdisc cluster -o …`) writes
the same contents into the directory you name instead.

An input written in the configuration always wins. When a stage has no configured input it takes
the output of the stage that produced it earlier in the same run; when neither exists the run
stops with a message naming the configuration key and the stage that would have supplied it.
Nothing is guessed, no stage is silently skipped, and no stage falls back to a reduced behaviour
or writes placeholder files: a stage either does what its name says or stops the run.

Three commands do not run anything, which makes them useful for checking a setup:

```bash
matdisc pipeline --write-config config.yaml --chemsys Ba-Cd-P   # a template with every key
matdisc pipeline -c config.yaml --dry-run                       # which stages would run
matdisc pipeline --validate ./runs/BaCdP                        # what each stage directory holds
```

The last one prints one line per stage. After a run that only did the phonon check:

```
-   generation
      Output directory does not exist
-   clustering
      Output directory does not exist
...
ok  phonons     n_spectra=1
```

Resuming: `--resume-from screening` keeps the configured stages from that one onwards, and
`--stages phonons` runs exactly what you name. The DFT stage deliberately stops a run after it has
written inputs, because the calculations have to finish elsewhere before fine-tuning has anything
to train on; the message tells you which directory to point `finetune.dft_dir` at.

## Stage 1 — Structure generation

- **Skill:** `structure-generation` · **Command:** `matdisc generate --chemsys Ba-Cd-P -n 24 -o runs/BaCdP`
- **In:** one or more chemical systems, a MatterGen checkpoint (`--model-path`, else
  `$MATTERGEN_MODEL_PATH`).
- **Out:** `structures/` — one POSCAR-format file per generated structure.
- **Needs:** MatterGen installed with its `mattergen-generate` console script on `PATH`, and a GPU
  in practice.

MatterGen is driven as a subprocess, never imported. A run writes `generated_crystals_cif.zip`,
`generated_crystals.extxyz` and `generated_trajectories.zip`; structures are read back from the
CIF archive when it exists and from the extxyz file otherwise. Property guidance towards a target
energy above the hull is exposed as `--energy-above-hull` and `--guidance-factor`.

**The guidance target is 0.1 eV/atom by default** — the metastability window a generated structure
is asked to fall inside, with a guidance factor of 2.0 and a batch size of 24. The Python function
`matdisc.generation.mattergen.generate`, the `generate` subcommand and `generation.energy_above_hull`
in a configuration file all read the same constant (`DEFAULT_ENERGY_ABOVE_HULL`), so two candidate
sets produced through different entry points came from the same distribution. It is a sampling
target handed to MatterGen, not a stability criterion: what counts as stable is decided later by
the hull, with `screening.tolerance`.

Already have candidates? Set `generation.structure_dir` and the stage adopts them and reports
`reused` instead of running MatterGen.

## Stage 2 — Representative selection

- **Skill:** `descriptor-clustering` · **Command:** `matdisc cluster -s runs/BaCdP/structures -n 10 -o runs/BaCdP/selection`
- **In:** a structure file or a directory of them, a DPA-3 checkpoint (`--model-path`, else `$DPA3_MODEL_PATH`).
- **Out:** `selected/` with the chosen structures, and `selected.csv` recording the selection.
- **Needs:** DeePMD-kit for the descriptors, `maml` (and scikit-learn) for the sampling.

Every atom of every structure becomes one row of a feature matrix of per-atom descriptors taken
from the potential (`DeepPot.eval_descriptor`). DIRECT sampling — BIRCH clustering in a
PCA-reduced space, a fixed number of rows per cluster — picks rows, and the rows are mapped back
onto the structures they came from. The point is to spend DFT time on structures that are not
near-duplicates of each other. The element order comes from the model itself, so an element the
loaded model does not know about is reported by name rather than crashing in a list lookup.

## Stage 3 — DFT labelling and validation

- **Skill:** `dft-calculation` · **Commands:** `matdisc dft-inputs -s <structures> -o runs/BaCdP`,
  then `matdisc dft-submit -r runs/BaCdP/calculations`
- **In:** a structure file or a directory of them; `--kind relax|static|band|hse-relax|hse-static`.
  A file is taken as given; a directory is globbed for `*.cif`, `*.vasp`, `POSCAR*` and `CONTCAR*`.
  In a pipeline run the stage hands over the structures it has already read, so the files it
  validated are exactly the ones written — `*.xyz` and `*.extxyz` candidates included.
- **Out:** `calculations/<name>/` holding INCAR, POSCAR, KPOINTS where the set needs one, POTCAR
  (or `POTCAR.spec`) and `submit.slurm`.
- **Needs:** nothing to write the inputs. Real POTCARs need `$PMG_VASP_PSP_DIR`; running the jobs
  needs your own licensed VASP, reached through `$VASP_CMD`, and a SLURM cluster.

Inputs are built with `pymatgen.io.vasp.sets`, with these convergence settings: `EDIFF = 1e-05`,
`EDIFFG = -0.01` on the relaxations, `KSPACING = 0.189`, `ENCUT` at 1.3 × the largest `ENMAX` of
the POTCARs, and a line-mode k-path for `--kind band`. The functional is plain PBE with no
Hubbard U — pymatgen's Materials Project sets apply a +U only to oxides and fluorides, so a
phosphide INCAR carries no `LDAU` tag. Set anything else, a U included, with
`user_incar_settings` in the configuration.

`--kind band`, `--kind hse-relax` and `--kind hse-static` are continuation runs: their INCAR reads a
charge density (`ICHARG` 11, 1 and 1), so a `CHGCAR` from a converged `--kind static` run on the
same structure has to be copied in before they can start. The stage logs a warning and lists
`CHGCAR` under `requires`.

No POTCAR files are in this repository and none may be redistributed. Without a pseudopotential
directory the stage still writes everything else and leaves a `POTCAR.spec` listing the symbols the
run needs, so inputs can be prepared and inspected on a machine that has no VASP licence.

Finished calculations are read back with `matdisc.dft.outputs.collect_results`, which prefers
`vasprun.xml` and falls back to `OUTCAR` plus `CONTCAR`. Its `id`, `composition` and
`energy_per_atom` columns carry the names the hull stage expects — but not the same energy scale as
the Materials Project table stage 5 harvests, which comes from Materials Project settings and
corrections. Compete your own DFT energies against competing-phase energies computed the same way,
or convert both sides to formation energies with
`matdisc.screening.hull.formation_energy_per_atom`.

## Stage 4 — Potential fine-tuning

- **Skill:** `model-finetuning` · **Command:** `matdisc finetune -d <dft_dir> -o runs/BaCdP --run`
- **In:** a directory of finished DFT calculations; a pretrained checkpoint
  (`--pretrained-model`, else `$DPA3_MODEL_PATH`).
- **Out:** a `deepmd/npy` dataset split into training and validation systems, the DeePMD-kit JSON
  configuration, a SLURM script, and — with `--run` — the fine-tuned checkpoint.
- **Needs:** dpdata for the conversion, DeePMD-kit's `dp` command for the training, a GPU.

Without `--run` the stage prepares the dataset and the configuration and stops, reporting
`prepared`; it does not claim to have trained anything. With `--run` it invokes
`dp --pt train <config> --finetune <checkpoint> --model-branch <head>` and returns the checkpoint,
which the screening and phonon stages of the same run then use automatically unless the
configuration names a checkpoint of its own.

## Stage 5 — Competing phases

- **Skill:** `competing-phases` · **Command:** `matdisc competing -c Ba-Cd-P -o runs/BaCdP`
- **In:** a chemical system.
- **Out:** `competing_phases.csv` with columns `material_id`, `formula_pretty`, `composition`,
  `natoms`, `formation_energy_per_atom`, `energy_per_atom`, `is_stable_mp`, `icsd_ids`; and
  `structures/` with the relaxed Materials Project structure behind each id.
- **Needs:** network access and a Materials Project key in `$MP_API_KEY`. Without one the stage
  stops with a message saying where to get one.

A hull rests on every subsystem, so a Ba-Cd-P search queries Ba, Cd, P, Ba-Cd, Ba-P, Cd-P and
Ba-Cd-P — seven queries, because the Materials Project matches `chemsys` exactly. By default only
phases carrying an ICSD cross-reference are kept, which restricts the reference set to compounds
that have been made and characterised; `--no-icsd` keeps everything. That filter also lowers the
hull: an ICSD-only set is a subset of the Materials Project's own, so every `e_above_hull` measured
against it is a lower bound on the real one and a candidate can read stable that is not. Downloaded
structures are Materials Project data under CC BY 4.0; see [`data/README.md`](data/README.md).

Two more choices change the reference set, so both are on the command line and in the
configuration. **One polymorph per reduced formula is kept by default** — the one with the lowest
*Materials Project* formation energy, not the lowest energy under the calculator stage 6 relaxes
with, so the polymorph carried into the hull can differ from the one that calculator would pick;
`--all-polymorphs` (`competing.unique_formula: false`) keeps them all.
`--max-energy-above-hull` (`competing.energy_above_hull_max`) drops phases sitting further above
the Materials Project hull than the value given. A subsystem query that fails stops the search,
because its phases would be silently missing from the hull; `--allow-partial` accepts the gap and
records the missing subsystems in the stage result.

## Stage 6 — Relaxation and convex hull

- **Skill:** `stability-screening` · **Commands:**
  `matdisc screen -s <candidates> --competing-structures <dir> -o runs/BaCdP` for structures,
  `matdisc hull --candidates candidates.csv --competing competing_phases.csv -o hull.csv` for
  tables you already have, or the `screening` stage of `matdisc pipeline`
- **In:** candidate structures (`matdisc screen`, pipeline) or a candidate table (`matdisc hull`),
  plus the competing phases from stage 5.
- **Out:** `hull.csv` — the candidate table with `e_above_hull`, `hull_energy_per_atom`,
  `is_stable`, `decomposition` and the provenance columns `calculator`, `calculator_head`,
  `n_competing_used`, `n_competing_dropped` added — plus `candidates.csv`,
  `competing_relaxed.csv`, `relaxed/` and `stable/`.
- **Needs:** for `matdisc hull` on existing tables, nothing but the base install. For
  `matdisc screen` and the pipeline stage, a calculator: DeePMD-kit and a checkpoint, or
  `--calculator emt` for the elements EMT covers.

The pipeline stage relaxes the candidates **and** the competing phases with the same calculator,
so both sides of the hull come from one energy scale. That is the default for a reason: mixing
machine-learning candidate energies with Materials Project DFT energies compares two different
quantities. Setting `screening.competing_energy_source: table` reads the harvested energies
instead and warns that matching the two scales is then your responsibility.

The hull itself is `pymatgen.analysis.phase_diagram.PhaseDiagram`, built once from one `PDEntry`
per competing phase; each candidate is placed with `get_decomp_and_e_above_hull`. Every energy read
from either table is per-atom and is multiplied by the atom count of its reduced formula to get the
total energy a `PDEntry` expects — nothing is divided by an atom count twice. Formulas are parsed
by `pymatgen.core.Composition`, so `Ba(CdP)2` stays a five-atom formula unit.

**A candidate is stable when `e_above_hull <= tolerance`, default `1e-6` eV/atom**, settable with
`--tolerance` or `screening.tolerance`. The tolerance admits a phase sitting exactly on the hull
and absorbs the rounding of the hull solve; a strict `< 0` test would call an on-hull phase
unstable. A candidate whose elements the competing phases do not cover, or whose energy is
missing, gets `NaN` and a reason in the `decomposition` column rather than aborting the batch.

Two things the stage refuses to do quietly:

- **A competing phase that could not be relaxed, or that did not converge, stops the stage.** A
  hull missing one of its phases can only sit lower, which moves every candidate toward stability,
  so the numbers would be wrong in the flattering direction. `screening.allow_incomplete_hull`
  (`--allow-incomplete-hull`) builds the hull anyway and records
  `n_competing_requested`/`n_competing_used`/`n_competing_dropped` in the stage result and in
  `hull.csv`.
- **A candidate whose own relaxation hit the step cap is not reported as stable.** Its energy
  belongs to no minimum. It stays in `hull.csv` with its hull numbers and `converged = False`, is
  counted as `n_unconverged`, and is left out of `stable/`; `screening.require_converged: false`
  turns the gate off.

`stable/` holds the **relaxed** cells, copied from `relaxed/<id>/relaxed.vasp` — the geometries
the reported energies belong to, and the ones stage 7 then computes a spectrum for. A survivor
whose relaxed structure is missing from disk stops the stage rather than being replaced by the
input cell.

`matdisc hull --demo` runs the whole stage on a synthetic Ba-Cd-P system in about a second, with
no files, no credentials and no network.

## Stage 7 — Phonon check

- **Skill:** `stability-screening` · **Command:** `matdisc phonons -s <structures> -o runs/BaCdP`
- **In:** structures — in a pipeline run, the ones that came through stage 6.
- **Out:** `phonons.csv`, one row per structure with `success`, `min_frequency`, `is_stable`,
  `relaxation_converged`, `output_dir` and `error`; per-structure `band_structure.dat`, `dos.dat`,
  `band_path_info.txt`, `relaxed.vasp` and a spectrum plot; and `stable/` with the structures that
  passed, as the relaxed geometries the spectra were computed on.
- **Needs:** a calculator. `--calculator emt` runs offline for the elements ASE's EMT covers, which
  is enough to exercise the code path; anything real needs DeePMD-kit and a checkpoint.

Atoms are displaced in a supercell with `ase.phonons.Phonons`, forces are evaluated with the
calculator, and the lowest frequency along a high-symmetry band path decides the verdict: a
structure is dynamically stable when that frequency is above `--threshold`, `-0.1` THz by default,
which tolerates the small imaginary frequencies finite displacements leave at the zone centre.
The supercell is chosen by `--min-cell-length` (10 Å by default) unless `--supercell` names one.

The positions are relaxed before the displacements (`--no-relax` skips it; the cell is never
touched). **That relaxation has to converge**: a geometry that is not stationary carries imaginary
modes belonging to its leftover forces, not to the material, so by default a structure whose
pre-relaxation hit the step cap is reported as a failed check with
`error = "pre-relaxation did not converge…"` and `relaxation_converged = False` instead of being
called dynamically unstable. Raise `phonons.max_steps`, loosen `phonons.fmax`, or pass
`--allow-unconverged-relaxation` to compute the spectrum anyway.

This is the expensive stage: force evaluations scale with the supercell, and a 10 Å cell can hold a
few hundred atoms. Run it last, on what survived the hull, and cap it with `--max-structures`.

## One run from one file

```yaml
name: BaCdP
work_dir: ./runs/BaCdP
stages: [competing, screening, phonons]

competing:
  chemsys: Ba-Cd-P
  only_icsd: true

screening:
  structure_dir: ./candidates
  model_path: $DPA3_MODEL_PATH
  tolerance: 1.0e-06

phonons:
  threshold: -0.1
  max_structures: 20
```

`matdisc pipeline --write-config config.yaml --chemsys Ba-Cd-P` writes a template carrying every
key with its default. Credentials never belong in a configuration file: the Materials Project key
is read from `MP_API_KEY` at call time. Path-valued keys expand `$VAR` and `~` when the file is
loaded, which is why `model_path: $DPA3_MODEL_PATH` works; other strings are written out
literally, so `$VASP_CMD` survives into the generated job script.
