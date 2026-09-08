[![Documentation Status](https://app.readthedocs.org/projects/ligandparam/badge/?version=latest)](https://ligandparam.readthedocs.io/en/latest/?badge=latest)
[![Python](https://img.shields.io/badge/python->=3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# ligandparam

**Amber ligand parameterization, plus ffpopt torsion correction**

Code originally written by York Lab (Rutgers), then in collaboration with Cheatham Lab (Utah).

`ligandparam` parameterizes a nonstandard ligand or residue (Antechamber / Gaussian RESP / `parmchk2` / LEaP) into an Amber triplet: `mol2` + `lib` + `frcmod`. **ALPS** is the orchestrator you install from this repo: it binds independent `ligandparam`, `scission`, and `ffpopt` checkouts and runs the pipeline (`lig-getparam` → `lig-scission` / `lig-dihed-correct`) through APIs in one process. Two twist modes: **fragment** (default) and **whole-ligand**.

**Docs:** [ligandparam.readthedocs.io](https://ligandparam.readthedocs.io/en/latest/)
**Repo:** [github.com/piskulichz/ligandparam](https://github.com/piskulichz/ligandparam)

---

## Typical session

```bash
# 1) Charges, types, baseline frcmod / lib
lig-getparam -i chaps.mol2 -r CHA -d CHA3 -rn freeligand --net_charge 0 -n 10 -mem 32

# 2a) Default: fragment the ligand, twist each piece, merge DIHE back
lig-dihed-correct -d CHA3 -r CHA --label chaps --model xtb -n 44

# 2b) Alternative: twist the intact parent (no scission)
lig-dihed-correct -d CHA3 -r CHA --label chaps --model xtb -n 44 --whole-ligand
```

`--label` is the recipe file stem (`chaps` from `chaps.mol2`), not the residue name (`CHA`). The `.lib` is never rewritten; corrected torsions land in `{label}.dihed.frcmod` (use that with the original `.lib` in LEaP).

`--fast` and AFFDO extras (`--soft-dihed-restraint`, `--fit-full`, `--multi-centroid`, ...) are still accepted so old scripts keep working; independent `ffpopt-main` does not implement those presets, so ALPS warns and runs a plain York twist. Logs from ligandparam, scission, and ffpopt all land on the ALPS process stdout (each package keeps its own logger / `[twist]` prints).

---

## ffpopt: fragment vs whole-ligand

Both modes start from the same Amber triplet and the same `lig-dihed-correct` CLI. They differ in **what molecule is scanned**.

| | **Fragment (default)** | **Whole-ligand (`--whole-ligand`)** |
|--|------------------------|-------------------------------------|
| **What is scanned** | Scission caps; each rotatable bond in a small fragment | The intact parent ligand |
| **Why use it** | Cheaper HL opts; local environment around each torsion | Coupled rotors / bulky detergents that fragments distort |
| **CPU** | Cheap 1-D fragments share `-n` first; then each 3+ bond fragment takes the whole node. 1-2-bond fragments flatten bond x wavefront; larger fragments nest like whole-ligand | One parent job; nested bond x wavefront (e.g. 4 x 11 on 44 cores for 8 bonds) |
| **Bond batches** | 1–2 bonds: independent 1-D wavefronts. 3+ bonds: whole-ligand packing (`FFPOPT_WHOLE_MAX_BONDS_PER_TWIST`, default 8) | `FFPOPT_WHOLE_MAX_BONDS_PER_TWIST` (default 8) |
| **Output** | Merged parent `{label}.dihed.frcmod` | Parent `{label}.dihed.frcmod` (no fragment merge) |
| **Lib** | Unchanged | Unchanged |

### Fragment (default)

Scission cuts the parent at rotatable bonds, builds capped `parm7`/`rst7` pieces, runs ffpopt twist in each fragment directory, then merges fitted DIHE terms by atom type. Drop-mode iterations keep earlier survivors unless a later `itXX.frcmod` refits the same key.

```bash
lig-dihed-correct -d CHA3 -r CHA --label chaps --model xtb -n 44
# inspect cuts only:
lig-scission fragment -d CHA3 -r CHA --label chaps
```

Use this for typical drug-like ligands. Prefer `--model xtb` (tblite) or `--model aimnet2` when you do not have QDpi2.

### Whole-ligand

Skip scission. Discover rotatable bonds on the parent and twist them on the full molecule:

```bash
lig-dihed-correct -d CHA3 -r CHA --label chaps --model xtb -n 44 --whole-ligand
```

AFFDO-style extras from the old monorepo (`--soft-dihed-restraint`, `--multi-centroid`, `--fit-full`) are not in independent ffpopt-main; ALPS ignores them with a warning.

| Flag | Role |
|------|------|
| `--whole-ligand` | No scission; twist the parent |
| `--soft-dihed-restraint` | Harmonic dihedral spring (500 kcal/mol/rad^2, +/-0.5 deg) instead of a hard IC; k doubles up to 8000 if out of band. A lost well (`|dphi|` > 30 deg) fails the node. Once in-band, unconstrained hard IC is skipped; two-stage (`--fast`) does one restrained HL opt at the final k |
| `--multi-centroid N` | Extra ConfSearch starts; keep the smoothest HL profile. Centroid-0 + orig share one pool; extra starts only if Fourier RMSE exceeds `FFPOPT_CENTROID_FOURIER_MAX` (default 0.5 kcal). Costly on large ligands -- try 0 or 2 before 5 |
| `--fit-full --fit-backend jax` | Fit FC + phase + period + scee/scnb (default is barrier / FC-only). JAX extra: from the clone, `pip install -e '.[jax]'` (not PyPI `ligandparam[jax]`, which is 1.0.0) |
| `--boltzmann-charges` | Boltzmann-average charges over centroid mol2s (needs `--multi-centroid >= 2`) |
| `--maxiter 1` | Skip the second orig rescan round (default is 2) |

Whole-ligand opts are full-molecule XTB (or QDpi2) jobs. That is the dominant cost; the extras above add more of those jobs. Keep `--fast`. Leave `OMP_NUM_THREADS=1` (unset is fine): many concurrent 1-thread SCFs beat one fat SCF.

Python entry points:

```python
from alps.workflows import (
    run_fragmented_dihed_twist_workflow,
    run_whole_ligand_dihed_twist_workflow,
)
```

Env knobs (`FFPOPT_*`) live in the independent ffpopt tree. Overlay with `FFPOPT_DEFAULTS=/path.json`; `export FFPOPT_*=` still wins.

---

## Features

- **Recipe-based parameterization** (BCC, single-orientation RESP, multi-orientation RESP, DeepMD-assisted minimize)
- **Composable stages** to add, remove, or reorder steps
- **Gaussian** geometry + ESP / RESP; **Amber** Antechamber / `parmchk2` / LEaP
- **ffpopt fragment twist** (scission + per-fragment wavefront + DIHE merge)
- **ffpopt whole-ligand twist** (`--whole-ligand` plus optional soft restraint / multi-centroid / full fit)
- **Wavefront scans** that expand from seeds (coalesced pending jobs, von Neumann N-D neighbors, outlier rescue, node wall-clock, `--fast` presets; see docs *Wavefront scans*)
- **ASCII console** for Slurm latin-1 logs (`[wavefront]`, `[twist]`, `[affdo]` scopes)
- **CLI** for batch param, fragmentation, SMILES -> PDB, Sage conversion

---

## Requirements

### Python

- Python **>= 3.10**
- Core deps install with the package (`numpy<2` for ParmEd, `MDAnalysis`, `ParmEd`, `RDKit`, ...)
- Optional ML potentials: `pip install -e ".[ml-potentials]"` from the clone

### External tools

| Tool | Used for |
|------|----------|
| [AmberTools](https://ambermd.org/AmberTools.php) (`antechamber`, `parmchk2`, `tleap`) | Typing, frcmod / lib, fragment `parm7`/`rst7` |
| Gaussian (`g16` or compatible) | Optimization and RESP ESP |
| tblite (extra `[tblite]`) | `--model xtb` dihedral scans |
| AIMNet2 (extra `[aimnet]`) | `--model aimnet2` (Python 3.11-3.13 + PyTorch 2.8+) |

---

## Installation

```bash
# From the workspace that holds the four *-main trees:
mamba env create -f env.yaml   # or: conda env create -f env.yaml
conda activate ligandparam
pip install -e ligandparam-main
pip install -e scission-main
pip install -e ffpopt-main
pip install -e "./alps-main[dihed,tblite]"
```

From this directory alone (after companions are installed):

```bash
pip install -e ".[dihed,tblite]"
```

CLI names (`lig-getparam`, `lig-dihed-correct`, `lig-scission`) stay the same. If the sibling `*-main` folders exist beside this checkout, `import alps` also binds them without an extra `pip install`. Override with `ALPS_LIGANDPARAM_PATH`, `ALPS_SCISSION_PATH`, `ALPS_FFPOPT_PATH`.

`pip install 'ligandparam[jax]'` (no `-e`, no `.`) pulls **PyPI 1.0.0** and can uninstall a local tree. From this checkout:

```bash
pip install -e '.[jax]'          # or: conda install -c conda-forge jax jaxlib
python -m unittest tests.test_install_validation -v
python -m unittest tests.test_developer_regression -v
```

Other extras from the clone: `.[ml]`, `.[sage]`, `.[docs]`, `.[all]`. DeepMD on HPC: install TensorFlow from conda-forge, then `pip install -e ".[ml]"`.

Default install auto-binds sibling `ligandparam-main`, `scission-main`, and `ffpopt-main`. Independent checkouts:

```bash
export ALPS_FFPOPT_PATH=/path/to/ffpopt/src/python/lib
export ALPS_SCISSION_PATH=/path/to/scission
export ALPS_LIGANDPARAM_PATH=/path/to/ligandparam
```

Contract for those packages: [`docs/companions.rst`](docs/companions.rst).

---

## Parameterization recipes

```python
from pathlib import Path
from ligandparam.recipes import FreeLigand

recipe = FreeLigand(
    in_filename=Path("ligand.pdb"),
    cwd=Path("output"),
    net_charge=0,
    atom_type="gaff2",
    molname="LIG",
    logger="file",
    nproc=12,
    mem=8,
)
recipe.setup()
recipe.list_stages()
recipe.execute(dry_run=False, nproc=12, mem=8)
```

| Recipe | What it does |
|--------|----------------|
| `LazierLigand` | Antechamber charges (e.g. BCC) + LEaP |
| `LazyLigand` | Gaussian minimize + single-orientation RESP |
| `FreeLigand` | 28-point quaternion SO(3) sampling + multi-RESP |
| `DPLigand` / `DPFreeLigand` | DeepMD geometry + RESP / multi-RESP |
| `SQMLigand` | SQM / DeepMD-assisted minimize + RESP |

After `setup()` you can `remove_stage` / `insert_stage`. `FreeLigand` default orientation pack is `so3_n28`; `orientation_protocol="legacy_euler"` restores the old Euler grid. Both feed the same multi-RESP -> `parmchk2` -> LEaP path.

Recipes can append twist with `dihed_correct=True` (`dihed_delta`, `dihed_fragment_config`). Prefer the standalone `lig-dihed-correct` CLI after `lig-getparam`.

---

## Command-line tools

| Command | Purpose |
|---------|---------|
| `lig-getparam` | Run a parameterization recipe |
| `lig-dihed-correct` | ffpopt fragment or whole-ligand dihedral correction |
| `lig-scission` / `scission` | Fragment only, or merge fragment frcmods |
| `smiles-to-pdb` | SMILES -> 3D PDB |
| `lighfix` | Fix hydrogenation / bonding from PDB |
| `lig-to-sage` | mol2 toward OpenFF Sage (optional `[sage]`) |
| `ffpopt-specialty` | Quarantined sugar/pucker, JSON, animate tools |

```bash
lig-getparam --help
lig-dihed-correct --help
lig-scission --help
```

Re-run `lig-getparam` after walltime without repeating finished Gaussian work (default). Force all Gaussian jobs again with `-O`.

---

## Examples

Runnable trees under [`examples/`](examples/): `01_LazyLigand`, `02_FreeLigand`, `03_ModifySteps`, `04_FromSmiles`, `05_FromSDF`. Sphinx example 07 is the same-session `lig-getparam` then `lig-dihed-correct` narrative (fragment and whole-ligand).

---

## Documentation

- User guide and API: https://ligandparam.readthedocs.io/en/latest/
- Local: `pip install ".[docs]" && cd docs && make html`
- ffpopt glossary: [`ffpopt-main/src/python/lib/ffpopt`](ffpopt-main) (when the sibling tree is present)

---

## Project layout

```text
alps-main/               # this package (import alps)
  cli/                   # lig-getparam, lig-dihed-correct, lig-scission
  workflows/             # fragmented / whole-ligand twist (API glue)
  stages/
  tests/
ligandparam-main/        # independent ligandparam (beside this checkout)
scission-main/           # independent scission
ffpopt-main/             # independent ffpopt (York layout: src/python/lib/ffpopt)
```

---

## Contributing

1. Fork, branch, `pip install -e ".[docs]"`
2. `python -m unittest tests.test_developer_regression -v`
3. Keep stdout, comments, and docs ASCII (`+/-`, `deg`, `chi^2`, `->`)
4. Open a PR that says why the change is needed

Release: bump `version` in `pyproject.toml` and `__version__` in `__init__.py`, commit, `git tag 1.6.1 && git push origin --tags`.

---

## Authors

- [Zeke Piskulich (York Lab)](https://theory.rutgers.edu/profile.php?people_id=399)
- [German P. Barletta (York Lab)](https://theory.rutgers.edu/profile.php?people_id=407)
- [Timothy J. Giese (York Lab)](https://theory.rutgers.edu/profile.php?people_id=3)
- [Nate Levinzon (Cheatham Lab)](https://people.utah.edu/basic.hml?eid=273961099)

---

## License

MIT.
