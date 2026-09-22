---
name: dft-calculation
description: Use when preparing VASP calculations for a set of structures - INCAR, KPOINTS, POSCAR and POTCAR through pymatgen input sets, plus a plain SLURM submission script - or when parsing finished runs back into a table.
---

# DFT calculation

Three steps, one module each:

- `matdisc.dft.inputs` writes a calculation directory per structure with
  `pymatgen.io.vasp.sets`.
- `matdisc.dft.slurm` writes a plain `sbatch` script and submits it.
- `matdisc.dft.outputs` reads finished runs with `pymatgen.io.vasp.outputs` into a table the
  convex-hull and fine-tuning stages consume.

Nothing here requires VASP to be installed to write inputs. The settings of the original
screening runs are carried over: `EDIFF = 1e-05`, `EDIFFG = -0.01` (a force criterion, on the
relaxations only), `KSPACING = 0.189`, `ENCUT = ceil(1.3 * max ENMAX)` read from the POTCARs,
and spin polarisation off.

**The functional is plain PBE, with no Hubbard U.** The original template asked for
`pbe ldau`, but pymatgen's Materials Project sets apply a +U only to oxides and fluorides, so
for a phosphide no `LDAU`, `LDAUU` or `LDAUJ` tag is written at all. Nothing here adds one
behind your back, and nothing here silently omits one you asked for: if your workflow does
apply a U to these systems, pass `LDAU`, `LDAUTYPE`, `LDAUL`, `LDAUU` and `LDAUJ` through
`user_incar_settings` and record which elements carry which U. A hull built from PBE energies
and one built from PBE+U energies are not comparable.

Five kinds are available:

| `--kind` | What it is | Moves ions | Starts from the directory as written |
|---|---|---|---|
| `relax` | PBE ionic relaxation | yes | yes |
| `static` | PBE self-consistent field run | no | yes |
| `band` | non-self-consistent PBE run along a k-path | no | no: `ICHARG = 11`, needs a CHGCAR |
| `hse-relax` | HSE06 ionic relaxation | yes | no: `ICHARG = 1`, needs a CHGCAR |
| `hse-static` | single-point HSE06 run on a fixed geometry | no | no: `ICHARG = 1`, needs a CHGCAR |

A hybrid **band gap** comes from `hse-static`, not from `hse-relax`. The two are not
interchangeable: `hse-relax` writes `NSW = 99`, `ISIF = 3` and `EDIFFG = -0.01` and relaxes the
cell under the hybrid functional, which on the `KSPACING = 0.189` mesh this package writes (a
7x7x7 grid for a 5 A cell) costs orders of magnitude more than the single point, which writes
`NSW = 0` and no `EDIFFG`.

## Prerequisites

- The package itself: `pip install -e .` from the repository root. `pymatgen` is a required
  dependency, so input generation needs no extra.
- **POTCAR files are never shipped here.** pymatgen looks them up in the directory named by
  `PMG_VASP_PSP_DIR` (set it in the environment or in `~/.pmgrc.yaml`):

  ```bash
  export PMG_VASP_PSP_DIR=/path/to/your/vasp/pseudopotentials
  ```

  Without it, INCAR, POSCAR and KPOINTS are still written and POTCAR is replaced by a
  `POTCAR.spec` file listing the symbols the run needs, so inputs can be prepared and
  inspected on a machine with no VASP licence.
- The generated script reads the VASP executable from `VASP_CMD` at run time and exits with a
  message if it is unset: `export VASP_CMD=/path/to/vasp_std`.
- Submission needs `sbatch` on `PATH`; `--dry-run` works anywhere.

## Usage

```bash
matdisc dft-inputs --help
matdisc dft-submit --help
```

Write one relaxation directory per structure, each with a submission script:

```bash
matdisc dft-inputs -s runs/clustering/selected -o runs/dft --kind relax \
    --job-name bacdp --ntasks 96 --walltime 600
```

Check what would be submitted, then submit:

```bash
matdisc dft-submit -r runs/dft/calculations --dry-run
matdisc dft-submit -r runs/dft/calculations
```

From Python:

```python
from matdisc.dft.inputs import write_vasp_inputs, write_batch_inputs
from matdisc.dft.slurm import SlurmSettings, write_slurm_script
from matdisc.dft.outputs import collect_results, check_calculation_status

write_vasp_inputs(structure, "runs/dft/BaCdP", kind="relax",
                  user_incar_settings={"NSW": 200})
write_slurm_script("runs/dft/BaCdP", SlurmSettings(ntasks=96, walltime_minutes=600))
table = collect_results("runs/dft/calculations", only_converged=True)
```

`user_incar_settings` is applied last, so it overrides everything above it; a value of `None`
removes a tag. `SlurmSettings.from_preset` starts from one of the `small`, `medium`, `large`
or `hse` resource shapes in `RESOURCE_PRESETS` - node, core and walltime requests, unrelated to
the `--kind` names above.

## Inputs and outputs

Input: a structure file or a directory of them. Each structure gets a subdirectory named
after its file stem.

`<outdir>/calculations/<stem>/`:

```
INCAR  POSCAR  POTCAR.spec (or POTCAR)  submit.slurm
```

`KSPACING` is written by default, so there is no KPOINTS file. A KPOINTS file appears instead
when the k-point grid comes from the input set: pass `--kspacing 0` on the command line, or
`kspacing=None` to `build_input_set`/`write_vasp_inputs`. `--kind band` and `--kind hse-static`
always carry their own k-point list and ignore `--kspacing`.

`collect_results` returns `id`, `composition` and `energy_per_atom` among its columns - the
column names `matdisc hull` reads for a candidate table. **The names line up; the energy scales
do not.** These are raw total energies from your own VASP, with your POTCARs, your ENCUT and no
+U, while the competing table harvested in stage 5 carries Materials Project energies from
Materials Project settings and corrections. Putting one straight against the other produces a
hull dominated by the offset between two different calculations. Either compete your own DFT
energies against competing-phase energies you computed the same way, or convert both sides to
formation energies against elemental references from those same settings -
`matdisc.screening.hull.formation_energy_per_atom` does that conversion.
`export_for_finetuning` writes energies, forces and stresses as JSON.
`check_calculation_status` reports `pending`, `running`, `completed`, `failed` or `unknown`
by reading the tail of `OUTCAR`, which stays cheap on the multi-gigabyte files a long
relaxation produces.

## Pitfalls

- **Without POTCARs, `ENCUT` is not rescaled.** The cutoff is `1.3 x max(ENMAX)` read from
  the pseudopotentials; when they are unavailable the input set's own `ENCUT` is kept and the
  substitution is logged. Inputs written this way are for inspection, not for production.
- **The submission script is deliberately bare.** Module loads and environment exports are
  empty by default because they are site-specific; add yours through `SlurmSettings(modules=
  [...], exports=[...])` or by editing the generated script. There is no hidden machine
  tuning in it.
- **`band`, `hse-relax` and `hse-static` are continuation runs.** Their INCAR sets `ICHARG` to
  read a charge density (11 for `band`, 1 for both hybrids), so the directory as written cannot
  start: run `--kind static` on the same structure first and copy its `CHGCAR` in. That is
  pymatgen's own default for these input sets, and how the Materials Project workflow runs a
  hybrid. `write_vasp_inputs` logs a warning saying so and lists `CHGCAR` under `requires` in
  the summary it returns.
- **`--kind band` ignores `--kspacing`** - a band-structure run follows a k-path, controlled
  by the line density instead. `--kind hse-static` ignores it too, for the explicit uniform
  k-point list it carries.
- **`--kind hse-relax` relaxes; it does not report a gap.** It is a hybrid geometry
  optimisation on the full k-mesh and is expensive accordingly. For a gap on a geometry you
  have already relaxed, use `--kind hse-static`.
- A structure file that cannot be read is logged and skipped, and the batch continues; the
  returned mapping tells you which ones were written.
- `matdisc dft-submit` exits non-zero when nothing was submitted, so it is safe to chain.

## What comes next

Finished relaxations feed `model-finetuning` (as training labels) and `stability-screening`
(as candidate energies for the hull).
