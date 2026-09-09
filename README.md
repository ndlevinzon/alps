[![Python](https://img.shields.io/badge/python->=3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Amber Ligand Parameters (ALPs)

**Orchestrator for ligandparam, scission, and ffpopt**

ALPs does not reimplement parameterization, fragmentation, or torsion fitting.
It binds independent checkouts of those three tools and runs the pipeline in one
process:

1. `lig-getparam` — ligandparam recipes (charges, types, baseline Amber triplet)
2. `lig-scission` — optional inspect-only fragmentation
3. `lig-dihed-correct` — scission fragments (default) or whole-ligand twist, then ffpopt fit

Two twist modes: **fragment** (default) and **whole-ligand**.

**Repo:** [github.com/ndlevinzon/alps](https://github.com/ndlevinzon/alps)

Companion docs: [ligandparam](https://github.com/ndlevinzon/ligandparam),
[scission](https://github.com/ndlevinzon/scission),
[ffpopt](https://github.com/ndlevinzon/ffpopt).
The import contract is in [`docs/companions.rst`](docs/companions.rst).
The combined Sphinx API reference lives in the workspace:
[`../docs/`](../docs/index.rst).

---

## Typical session

```bash
# 1) Charges, types, baseline frcmod / lib
lig-getparam -i chaps.mol2 -r CHA -d CHA -rn freeligand --net_charge 0 -n 10 -mem 32

# 2a) Default: fragment the ligand, twist each piece, merge DIHE back
lig-dihed-correct -d CHA -r CHA --label chaps --model xtb -n 44

# 2a') Pfizer or WBO fragments (Stern et al.) before the scan
lig-dihed-correct -d CHA -r CHA --label chaps --model xtb -n 44 --strategy pfizer
lig-dihed-correct -d CHA -r CHA --label chaps --model xtb -n 44 --strategy wbo

# 2b) Alternative: twist the intact parent (no scission)
lig-dihed-correct -d CHA -r CHA --label chaps --model xtb -n 44 --whole-ligand
```

`--label` is the recipe file stem (`chaps` from `chaps.mol2`), not the residue
name (`CHA`). The `.lib` is never rewritten; corrected torsions land in
`{label}.dihed.frcmod` (use that with the original `.lib` in LEaP).

`--fast` and AFFDO extras (`--soft-dihed-restraint`, `--fit-full`,
`--multi-centroid`, ...) are still accepted so old scripts keep working.
Independent ffpopt does not implement those presets, so ALPS warns and runs a
plain York twist. Logs from ligandparam, scission, and ffpopt all land on the
ALPS process stdout.

---

## Fragment vs whole-ligand

Both modes start from the same Amber triplet and the same `lig-dihed-correct`
CLI. They differ in **what molecule is scanned**.

| | **Fragment (default)** | **Whole-ligand (`--whole-ligand`)** |
|--|------------------------|-------------------------------------|
| **What is scanned** | Scission caps; each rotatable bond in a small fragment | The intact parent ligand |
| **Why use it** | Cheaper HL opts; local environment around each torsion | Coupled rotors / bulky detergents that fragments distort |
| **Output** | Merged parent `{label}.dihed.frcmod` | Parent `{label}.dihed.frcmod` (no fragment merge) |
| **Lib** | Unchanged | Unchanged |

Inspect cuts only:

```bash
lig-scission fragment -d CHA3 -r CHA --label chaps
lig-scission fragment -d CHA3 -r CHA --label chaps --strategy pfizer
```

`--strategy` on `lig-dihed-correct` (and `lig-scission fragment`) chooses the
scission scheme **before** the ffpopt scan: `scission` (default), `pfizer`,
or `wbo`. YAML is `--fragment-config file.yaml`.

Python entry points:

```python
from alps.workflows import (
    run_fragmented_dihed_twist_workflow,
    run_whole_ligand_dihed_twist_workflow,
)
```

Env knobs (`FFPOPT_*`) live in the independent ffpopt tree. Overlay with
`FFPOPT_DEFAULTS=/path.json`; `export FFPOPT_*=` still wins.

---

## Installation

The full HPC / conda stack (Python 3.12, conda-forge AmberTools, all four
repos) lives in the workspace README: [`../README.md`](../README.md).

From this directory alone (after companions are already installed):

```bash
pip install -e ".[dihed,tblite]"
```

CLI names (`lig-getparam`, `lig-dihed-correct`, `lig-scission`) are ALPS
wrappers. ligandparam still ships its own `lig-getparam`; install ALPS last so
the orchestrator wins. `smiles-to-pdb`, `lighfix`, and `lig-to-sage` stay on
ligandparam. The `scission` console script is the scission package.

If sibling folders named `ligandparam`, `scission`, and `ffpopt` (or the older
`*-main` names) sit beside this checkout, `import alps` also binds them without
an extra `pip install`. Override with `ALPS_LIGANDPARAM_PATH`,
`ALPS_SCISSION_PATH`, `ALPS_FFPOPT_PATH`.

```bash
python -m unittest tests.test_install_validation -v
python -m unittest tests.test_developer_regression -v
```

The combined Sphinx docs (ALPS, LigandParam, Scission, FFPOPT) are in
the workspace repo: `pip install -r ../docs/requirements.txt`, then
`sphinx-build -b html ../docs ../docs/_build/html`.

---

## Command-line tools

| Command | Purpose |
|---------|---------|
| `lig-getparam` | Run a ligandparam parameterization recipe (ALPS banner + log tee) |
| `lig-dihed-correct` | ffpopt fragment or whole-ligand dihedral correction |
| `lig-scission` | Fragment or merge; `-d` / `-r` / `--label` resolve a getparam directory |

```bash
lig-getparam --help
lig-dihed-correct --help
lig-scission --help
```

---

## Project layout

```text
alps/                    # this package (import alps)
  cli/                   # lig-getparam, lig-dihed-correct, lig-scission
  workflows/             # fragmented / whole-ligand twist (API glue)
  stages/
  tests/
ligandparam/             # independent ligandparam (beside this checkout)
scission/                # independent scission
ffpopt/                  # independent ffpopt (York layout: src/python/lib/ffpopt)
```

---

## Contributing

1. Fork, branch, `pip install -e ".[dihed]"`
2. `python -m unittest tests.test_developer_regression -v`
3. Keep stdout, comments, and docs ASCII (`+/-`, `deg`, `chi^2`, `->`)
4. Open a PR that says why the change is needed

This package is one git repo. In the ALPS workspace umbrella, push this repo
first, then commit the updated submodule pointer in the workspace.

Release: bump `version` in `pyproject.toml` and `__version__` in `__init__.py`,
commit, `git tag 1.6.1 && git push origin --tags`.

---

## Authors

- [Zeke Piskulich (York Lab)](https://theory.rutgers.edu/profile.php?people_id=399)
- [German P. Barletta (York Lab)](https://theory.rutgers.edu/profile.php?people_id=407)
- [Timothy J. Giese (York Lab)](https://theory.rutgers.edu/profile.php?people_id=3)
- [Nate Levinzon (Cheatham Lab)](https://people.utah.edu/basic.hml?eid=273961099)

## License

MIT.
