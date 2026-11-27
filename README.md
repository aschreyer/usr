# USR: Ultrafast Shape Recognition

[![CI](https://github.com/aschreyer/usr/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/aschreyer/usr/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rd-usr)](https://pypi.org/project/rd-usr/)
![Python 3.12 and later](https://img.shields.io/badge/python-3.12%2B-3776ab)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy in strict mode](https://img.shields.io/badge/mypy-strict-2a6db2)](https://mypy-lang.org/)

**USR** is a Python implementation of _Ultrafast Shape Recognition_
(Ballester and Richards, 2007) on top of [RDKit](https://www.rdkit.org).
USR describes the shape of a 3D conformer with 12 numbers and compares two
shapes without aligning them, which keeps a screen of a large database
cheap.

- **Describe shapes.** Calculate the 12 USR moments of every conformer in an
  SD file or of any RDKit molecule.
- **Search by shape.** Score a moments database against query molecules and
  keep the best hits per query conformer.
- **Use it from Python or the shell.** The `usr` command writes plain CSV, so
  its output works with any tool that reads tables.

## How USR works

USR describes a conformer of $N$ atoms at positions
$\mathbf x_1, \dots, \mathbf x_N$ by its distances to four reference points:

- $\mathbf c_{\mathrm{ctd}} = \frac{1}{N} \sum_{k=1}^{N} \mathbf x_k$, the
  centroid;
- $\mathbf c_{\mathrm{cst}}$, the atom closest to $\mathbf c_{\mathrm{ctd}}$:
  $\mathbf x_j$ with $j = \arg\min_k \lVert \mathbf x_k - \mathbf c_{\mathrm{ctd}} \rVert$;
- $\mathbf c_{\mathrm{fct}}$, the atom farthest from $\mathbf c_{\mathrm{ctd}}$:
  $\mathbf x_j$ with $j = \arg\max_k \lVert \mathbf x_k - \mathbf c_{\mathrm{ctd}} \rVert$;
- $\mathbf c_{\mathrm{ftf}}$, the atom farthest from $\mathbf c_{\mathrm{fct}}$:
  $\mathbf x_j$ with $j = \arg\max_k \lVert \mathbf x_k - \mathbf c_{\mathrm{fct}} \rVert$.

Where atoms tie, the one with the lowest index $k$ is taken. For each reference
point $r$, the distances $d_k^r = \lVert \mathbf x_k - \mathbf c_r \rVert$ form a
distribution, summarised by its mean, its population standard deviation and the
cube root of its standardised skewness:

$$
\mu_1^r = \frac{1}{N} \sum_{k=1}^{N} d_k^r, \qquad
\mu_2^r = \sqrt{\frac{1}{N} \sum_{k=1}^{N} \left(d_k^r - \mu_1^r\right)^2}, \qquad
\mu_3^r = \sqrt[3]{\frac{1}{N} \sum_{k=1}^{N}
  \left(\frac{d_k^r - \mu_1^r}{\mu_2^r}\right)^3}
$$

The cube root is real, so $\mu_3^r$ keeps the sign of the skewness. Where
$\mu_2^r = 0$, as for a single atom, the skewness is undefined and $\mu_3^r$ is
NaN. The four distributions give the 12 moments `M1` to `M12`:

$$
M = \left(\mu_1^{\mathrm{ctd}}, \mu_2^{\mathrm{ctd}}, \mu_3^{\mathrm{ctd}},
  \mu_1^{\mathrm{cst}}, \dots, \mu_3^{\mathrm{ftf}}\right)
$$

The similarity of a query conformer $q$ and a database conformer $i$ is

$$
S_{qi} = \frac{1}{1 + \frac{1}{12} \sum_{l=1}^{12} \left| M_l^q - M_l^i \right|}
  \in (0, 1],
$$

1 for identical moments, falling towards 0 as they differ. USR ignores atom
types and needs 3D coordinates; a 2D drawing gives the moments of a flat shape.

> [!NOTE]
> The 2007 paper's text calls $\mu_2$ the variance and $\mu_3$ the skewness, and
> the values in its Figure 3 match the variance and the raw third central
> moment. The method's author has confirmed that the intended descriptors are
> the population standard deviation and the cube root of the standardised
> skewness, as above. This package computes those.

## Install

USR supports Python 3.12 and later. The distribution is `rd-usr` and the
import package is `usr`. RDKit installs from PyPI as a dependency.

```bash
uv add rd-usr     # or: pip install rd-usr
```

To install the current source from GitHub:

```bash
uv add git+https://github.com/aschreyer/usr
```

## Usage

### From the shell

Calculate the moments of every conformer in an SD file (plain or gzipped),
then screen that database with query molecules:

```bash
usr generate tests/data/drugs-3d.sdf -o drugs.csv
usr search tests/data/drugs-3d.sdf drugs.csv --top 2
```

```
Query,QueryConformer,Title,Conformer,Score
aspirin,0,aspirin,0,1.0
aspirin,0,caffeine,0,0.9256322791990178
caffeine,0,caffeine,0,1.0
caffeine,0,aspirin,0,0.9256322791990178
...
```

`usr generate` writes one row per SD record: `Title`, `Conformer` and the
moments `M1` to `M12`. `usr search` takes an SD file or a moments CSV as the
query and writes one row per hit, best first for each query conformer.

- Results go to stdout unless `-o/--out` names a file. Warnings, such as a
  skipped SD record that RDKit cannot read, go to stderr.
- Hydrogen atoms are removed before the moments are calculated. Pass
  `--explicit-h` to count them, to both commands alike.
- A wrong input exits with code 2 and a one-line message.

Run `usr --help` or `usr <command> --help` for every option.

### From Python

```python
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from usr import io
from usr.moments import get_usr_moment_table
from usr.search import screen


database = get_usr_moment_table(io.read_molecules("tests/data/drugs-3d.sdf"))

naproxen = Chem.AddHs(Chem.MolFromSmiles("COc1ccc2cc(ccc2c1)[C@H](C)C(=O)O"))
naproxen.SetProp("_Name", "naproxen")
rdDistGeom.EmbedMultipleConfs(naproxen, numConfs=1, randomSeed=42)
query = get_usr_moment_table([naproxen])

for hit in screen(query, database, top=3):
    print(f"{hit.title:<12} {hit.score:.3f}")
```

```
ibuprofen    0.852
diazepam     0.647
paracetamol  0.616
```

| Module | Holds |
|---|---|
| `usr.moments` | `get_usr_moments` for one molecule, `get_usr_moment_table` for many, `calc_usr_moments` for a `(conformers, atoms, 3)` coordinate array |
| `usr.similarity` | `calc_usr_sim` for one set of moments against many |
| `usr.search` | `screen`, which yields `Hit` records best first |
| `usr.io` | `read_molecules` for SD files; `read_moment_table`, `write_moment_table` and `write_hits` for CSV |

The library logs through [loguru](https://github.com/Delgan/loguru) and is
silent until an application calls `logger.enable("usr")`.

## Citation

If you use USR in published work, please cite the original method:

> Ballester, P. J. and Richards, W. G. (2007). Ultrafast shape recognition to
> search compound databases for similar molecular shapes. _J. Comput. Chem._,
> 28: 1711–1723. [doi:10.1002/jcc.20681](https://doi.org/10.1002/jcc.20681)

```bibtex
@article{ballester2007usr,
  author  = {Ballester, Pedro J. and Richards, W. Graham},
  title   = {Ultrafast shape recognition to search compound databases for
             similar molecular shapes},
  journal = {Journal of Computational Chemistry},
  volume  = {28},
  number  = {10},
  pages   = {1711--1723},
  year    = {2007},
  doi     = {10.1002/jcc.20681},
}
```

## Development

[uv](https://docs.astral.sh/uv/) is the only prerequisite. It installs the
Python in `.python-version` and every dependency from `uv.lock`.

```bash
git clone https://github.com/aschreyer/usr.git
cd usr
uv sync --locked
uvx pre-commit install   # optional: run the checks on each commit
```

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy src tests        # type-check, strict
uv run pytest --cov          # tests and doctests, with branch coverage
uv build                     # sdist and wheel under dist/
```

CI runs the same commands on Python 3.12 and 3.15. Pushing a `v*` tag
publishes that version to PyPI and creates its GitHub release.
[AGENTS.md](AGENTS.md) records the code and test conventions.

## Licence

MIT. See [LICENSE](LICENSE).
