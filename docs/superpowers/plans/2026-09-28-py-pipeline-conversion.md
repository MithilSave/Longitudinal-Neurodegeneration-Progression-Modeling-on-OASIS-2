# Notebook-to-.py Pipeline Conversion & Digital Twin Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the OASIS-2 pipeline from `OASIS2_Complete_Colab_Pipeline.ipynb` / `OASIS2_Colab.py` into a tested, importable `src/oasis2/` package, completing the 5 functions that never got ported (the digital twin's trajectory simulator + interpretability layer) and fixing the bugs found along the way.

**Architecture:** Split the existing 3549-line `OASIS2_Colab.py` (which already has 68 of 73 needed functions, loosely grouped under five internal `"""x.py — ..."""` docstring banners) into focused modules under `src/oasis2/`, one module per pipeline phase/model. Each module gets synthetic-data pytest tests (no real MRI data available locally — see spec §1). Two entirely new modules (`interpretability.py`, `digital_twin.py`) are written from the notebook's still-unconverted tail cells.

**Tech Stack:** Python 3.11, PyTorch 2.10, NumPy, pandas, scipy, networkx, statsmodels, scikit-learn, Captum (new dependency), pytest.

**Source-of-truth spec:** [docs/superpowers/specs/2026-09-28-py-pipeline-conversion-design.md](../specs/2026-09-28-py-pipeline-conversion-design.md)

---

## Conventions used in this plan

1. **Porting existing code:** `OASIS2_Colab.py` already contains correct, working source for 68 of the 73 target functions/classes. Where a task says "port lines A–B of `OASIS2_Colab.py` (anchored on `def foo` / `class Foo`) into `<file>`", that means: open `OASIS2_Colab.py`, copy that exact code, paste it into the new file, and apply only the explicitly listed adaptations below it (import fixes, global→parameter changes, bug fixes). Do not rewrite ported logic beyond what's listed — the goal is a faithful, testable relocation, not a rewrite. Line numbers are anchors, not guarantees — if they're off by a few lines because of blank lines, use the named `def`/`class` to find the right block.
2. **New code:** Two modules (`interpretability.py`, `digital_twin.py`) have no working source to port — they're written fresh from the notebook cells shown inline in each task, adapted from notebook-global style (`m1`, `test_ld`, `nf_in`) to explicit function parameters.
3. Every task ends with `pytest tests/ -q` passing and a commit.

---

### Task 1: Package scaffold, pytest config, synthetic fixtures

**Files:**
- Create: `pyproject.toml`
- Create: `src/oasis2/__init__.py`
- Create: `src/oasis2/data/__init__.py`
- Create: `src/oasis2/models/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `requirements.txt`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "oasis2"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Create empty `__init__.py` files**

```python
# src/oasis2/__init__.py
```

```python
# src/oasis2/data/__init__.py
```

```python
# src/oasis2/models/__init__.py
```

```python
# tests/__init__.py
```

- [ ] **Step 3: Write `tests/conftest.py` with synthetic fixtures used by every later test module**

```python
"""Shared synthetic-data fixtures for oasis2 tests.

No real MRI data is available in this environment (see
docs/superpowers/specs/2026-09-28-py-pipeline-conversion-design.md §1), so every
fixture here is a small, fully synthetic stand-in with the same shapes the real
data pipeline produces.
"""
import numpy as np
import pandas as pd
import pytest
import torch

N_REGIONS = 8
N_COV = 6


@pytest.fixture
def synthetic_connectome():
    """An (8, 8) symmetric non-negative adjacency matrix, zero diagonal."""
    rng = np.random.default_rng(0)
    A = rng.random((N_REGIONS, N_REGIONS))
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    return A


@pytest.fixture
def synthetic_demo_df():
    """Longitudinal demographics for 6 synthetic subjects, 2-4 visits each."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(6):
        sid = f"OAS2_{i:04d}"
        n_visits = rng.integers(2, 5)
        age0 = rng.uniform(65, 85)
        group = "Nondemented" if i % 3 == 0 else "Demented"
        cdr0 = 0.0 if group == "Nondemented" else rng.choice([0.5, 1.0])
        for v in range(n_visits):
            rows.append({
                "subject_id": sid,
                "session_id": f"{sid}_MR{v+1}",
                "Subject ID": sid,
                "MRI ID": f"{sid}_MR{v+1}",
                "Visit": v + 1,
                "MR Delay": v * 365,
                "Group": group,
                "M/F": "F" if i % 2 == 0 else "M",
                "Age": age0 + v,
                "EDUC": rng.uniform(8, 20),
                "SES": rng.integers(1, 5),
                "MMSE": max(0, 30 - v * rng.uniform(0, 2) - (5 if group == "Demented" else 0)),
                "CDR": min(2.0, cdr0 + v * 0.25 if group == "Demented" else 0.0),
                "eTIV": rng.uniform(1300, 1700),
                "nWBV": rng.uniform(0.65, 0.82) - v * 0.005,
                "ASF": rng.uniform(0.9, 1.3),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_regional_volumes(synthetic_demo_df):
    """One row per (subject, session) with N_REGIONS normalized volume columns."""
    rng = np.random.default_rng(2)
    df = synthetic_demo_df[["subject_id", "session_id"]].copy()
    for r in range(N_REGIONS):
        df[f"region_{r}_norm"] = rng.uniform(0.3, 1.0, size=len(df))
    return df


@pytest.fixture
def synthetic_batch():
    """A dict shaped like a `collate_longitudinal` batch: B=4 subjects, K=3 visits, N=8 regions, D=1."""
    B, K, N, D = 4, 3, N_REGIONS, 1
    g = torch.Generator().manual_seed(0)
    node_features = torch.rand(B, K, N, D, generator=g)
    times = torch.stack([torch.arange(K, dtype=torch.float32)] * B)
    covariates = torch.rand(B, N_COV, generator=g)
    mmse = torch.rand(B, K, generator=g) * 30
    cdr = torch.randint(0, 4, (B, K), generator=g)
    mask = torch.ones(B, K, dtype=torch.bool)
    adj = torch.rand(1, N, N, generator=g)
    adj = (adj + adj.transpose(1, 2)) / 2
    return {
        "node_features": node_features, "times": times, "covariates": covariates,
        "mmse": mmse, "cdr": cdr, "mask": mask, "adj": adj,
    }
```

- [ ] **Step 4: Write `requirements.txt`**

```
numpy
pandas
scipy
scikit-learn
networkx
statsmodels
torch
nibabel
nilearn
SimpleITK
tqdm
openpyxl
matplotlib
seaborn
captum
pytest
```

- [ ] **Step 5: Install the package in editable mode and verify pytest collects (0 tests, no errors)**

Run: `pip install -e . && pytest tests/ -q`
Expected: `no tests ran` (exit code 5) — no import errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt src/oasis2/__init__.py src/oasis2/data/__init__.py src/oasis2/models/__init__.py tests/__init__.py tests/conftest.py
git commit -m "chore: scaffold oasis2 package and synthetic test fixtures"
```

---

### Task 2: `config.py`, `logging_utils.py`, `io_utils.py`

**Files:**
- Create: `src/oasis2/config.py`
- Create: `src/oasis2/logging_utils.py`
- Create: `src/oasis2/io_utils.py`
- Test: `tests/test_config.py`
- Test: `tests/test_io_utils.py`

Source: `OASIS2_Colab.py` lines 41–70 (`PROJECT_ROOT`, Colab-only — rewrite, see Step 3), 96–145 (`is_colab`, `is_gpu_available`, `get_device`, `environment_summary`), 149–179 (`setup_logging`), 183–248 (`save_checkpoint`, `load_checkpoint`, `checkpoint_exists`, `save_json`, `load_json`, `z_score`).

`PROJECT_ROOT` in the original is hardcoded to `LOCAL_BASE = Path('/content/oasis2')`, a Colab-only global. This task replaces it with an environment-aware resolver so the package imports and runs identically locally and in Colab.

- [ ] **Step 1: Write `tests/test_config.py`**

```python
import os
from pathlib import Path

from oasis2 import config


def test_get_project_root_defaults_to_repo_root(monkeypatch):
    monkeypatch.delenv("OASIS2_PROJECT_ROOT", raising=False)
    root = config.get_project_root()
    assert isinstance(root, Path)
    assert root.exists()


def test_get_project_root_respects_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("OASIS2_PROJECT_ROOT", str(tmp_path))
    assert config.get_project_root() == tmp_path


def test_get_data_dir_creates_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("OASIS2_PROJECT_ROOT", str(tmp_path))
    d = config.get_data_dir("raw")
    assert d.exists()
    assert d == tmp_path / "data" / "raw"


def test_is_colab_false_outside_colab():
    assert config.is_colab() is False


def test_get_device_returns_torch_device():
    import torch
    assert isinstance(config.get_device(), torch.device)


def test_environment_summary_has_expected_keys():
    info = config.environment_summary()
    for key in ("platform", "python", "colab", "gpu_available", "project_root"):
        assert key in info
```

- [ ] **Step 2: Run tests to verify they fail (module doesn't exist yet)**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oasis2.config'`

- [ ] **Step 3: Write `src/oasis2/config.py`**

```python
"""Project paths and runtime-environment detection for the oasis2 package.

Locally, the project root is this repo's checkout. In Colab, it's the fast
local workspace mounted by colab_setup.bootstrap() (/content/oasis2). An
explicit OASIS2_PROJECT_ROOT env var overrides both, mainly for tests.
"""
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict


def is_colab() -> bool:
    """Return True if running in Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def get_project_root() -> Path:
    env_override = os.environ.get("OASIS2_PROJECT_ROOT")
    if env_override:
        return Path(env_override)
    if is_colab():
        return Path("/content/oasis2")
    # src/oasis2/config.py -> src/oasis2 -> src -> repo root
    return Path(__file__).resolve().parents[2]


def get_data_dir(sub: str = "") -> Path:
    d = get_project_root() / "data" / sub if sub else get_project_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_outputs_dir(sub: str = "") -> Path:
    d = get_project_root() / "outputs" / sub if sub else get_project_root() / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_models_dir() -> Path:
    d = get_project_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_gpu_available() -> bool:
    """Return True if a CUDA GPU is accessible."""
    import torch
    return torch.cuda.is_available()


def get_device():
    """Return a torch.device: cuda if available, else cpu."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def environment_summary() -> Dict[str, Any]:
    """Return a dict summarising the current runtime environment."""
    info = {
        "platform": platform.system(),
        "python": sys.version,
        "colab": is_colab(),
        "gpu_available": is_gpu_available(),
        "project_root": str(get_project_root()),
    }
    if info["gpu_available"]:
        try:
            import torch
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
        except Exception:
            pass
    return info
```

- [ ] **Step 4: Write `src/oasis2/logging_utils.py`** (ported verbatim from `OASIS2_Colab.py:152-179`, only the module docstring/imports are new)

```python
"""Logging setup for the oasis2 package."""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str = "oasis2",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configure and return a named logger that writes to stdout (and optionally a file)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s  %(name)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
```

- [ ] **Step 5: Write `src/oasis2/io_utils.py`** (ported from `OASIS2_Colab.py:186-248`; `directory` defaults now call `oasis2.config.get_outputs_dir` explicitly instead of relying on a same-file global)

```python
"""Checkpoint, JSON, and array helpers for the oasis2 package."""
import json
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np

from oasis2.config import get_outputs_dir


def save_checkpoint(data: Any, name: str, directory: Optional[Path] = None) -> Path:
    """Pickle-save `data` to `directory/<name>.pkl`. Default directory is outputs/checkpoints/."""
    if directory is None:
        directory = get_outputs_dir("checkpoints")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


def load_checkpoint(name: str, directory: Optional[Path] = None) -> Any:
    """Load and return a previously saved checkpoint. Raises FileNotFoundError if missing."""
    if directory is None:
        directory = get_outputs_dir("checkpoints")
    directory = Path(directory)
    path = directory / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def checkpoint_exists(name: str, directory: Optional[Path] = None) -> bool:
    if directory is None:
        directory = get_outputs_dir("checkpoints")
    return (Path(directory) / f"{name}.pkl").exists()


def save_json(data: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def z_score(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standardise array along given axis (zero mean, unit variance)."""
    mu = arr.mean(axis=axis, keepdims=True)
    sigma = arr.std(axis=axis, keepdims=True)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (arr - mu) / sigma
```

- [ ] **Step 6: Write `tests/test_io_utils.py`**

```python
import numpy as np
import pytest

from oasis2 import io_utils


def test_checkpoint_roundtrip(tmp_path):
    path = io_utils.save_checkpoint({"a": 1}, "mycp", directory=tmp_path)
    assert path.exists()
    assert io_utils.checkpoint_exists("mycp", directory=tmp_path)
    assert io_utils.load_checkpoint("mycp", directory=tmp_path) == {"a": 1}


def test_load_checkpoint_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        io_utils.load_checkpoint("nope", directory=tmp_path)


def test_json_roundtrip(tmp_path):
    p = tmp_path / "out.json"
    io_utils.save_json({"x": [1, 2, 3]}, p)
    assert io_utils.load_json(p) == {"x": [1, 2, 3]}


def test_z_score_zero_mean_unit_var():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = io_utils.z_score(arr)
    assert abs(z.mean()) < 1e-8
    assert abs(z.std() - 1.0) < 1e-8


def test_z_score_constant_column_no_div_by_zero():
    arr = np.array([5.0, 5.0, 5.0])
    z = io_utils.z_score(arr)
    assert np.all(z == 0.0)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_config.py tests/test_io_utils.py -v`
Expected: PASS (11 tests)

- [ ] **Step 8: Commit**

```bash
git add src/oasis2/config.py src/oasis2/logging_utils.py src/oasis2/io_utils.py tests/test_config.py tests/test_io_utils.py
git commit -m "feat: port config, logging, and io utility modules"
```

---

### Task 3: `connectome.py` — includes the missing `scipy.linalg` import bugfix

**Files:**
- Create: `src/oasis2/connectome.py`
- Test: `tests/test_connectome.py`

Source: `OASIS2_Colab.py` lines 2173–2342 (`CONNECTOME_REGIONS`, `N_REGIONS`, `build_synthetic_connectome`, `load_template_connectome`, `compute_laplacian`, `eigen_decompose_laplacian`, `ndm_propagate`).

**Confirmed bug to fix while porting:** `eigen_decompose_laplacian` (source line 2314) calls `scipy.linalg.eigh(L)`, but `OASIS2_Colab.py` never imports `scipy` or `scipy.linalg` before that line — the only `scipy` import in the whole file is `import scipy.stats` at line 2796, which comes *after* `eigen_decompose_laplacian` is defined and called (`run_phase3` calls it unconditionally as its first step, source line ~2720). Running the script as-is raises `NameError: name 'scipy' is not defined` the first time Phase 3 runs — the NDM baseline the rest of the digital twin depends on cannot execute at all today. Fix: add `import scipy.linalg` at the top of the new `connectome.py`.

- [ ] **Step 1: Write `tests/test_connectome.py`**

```python
import numpy as np
import pytest

from oasis2 import connectome


def test_build_synthetic_connectome_shape_and_symmetry():
    A = connectome.build_synthetic_connectome(n_regions=10, seed=1)
    assert A.shape == (10, 10)
    assert np.all(np.diag(A) == 0)
    assert np.all(A >= 0)


def test_load_template_connectome_falls_back_to_synthetic(tmp_path, monkeypatch):
    monkeypatch.setenv("OASIS2_PROJECT_ROOT", str(tmp_path))
    A, region_names = connectome.load_template_connectome(connectome_path=None)
    assert A.shape[0] == A.shape[1] == len(region_names)
    # falls back to N_REGIONS by default
    assert A.shape[0] == connectome.N_REGIONS


def test_compute_laplacian_is_symmetric_and_psd(synthetic_connectome):
    L = connectome.compute_laplacian(synthetic_connectome)
    assert np.allclose(L, L.T)
    eigvals = np.linalg.eigvalsh(L)
    assert eigvals.min() > -1e-8  # positive semi-definite


def test_eigen_decompose_laplacian_does_not_raise_nameerror(synthetic_connectome):
    """Regression test for the missing `import scipy.linalg` bug."""
    L = connectome.compute_laplacian(synthetic_connectome)
    eigenvalues, eigenvectors = connectome.eigen_decompose_laplacian(L)
    assert eigenvalues.shape == (8,)
    assert eigenvectors.shape == (8, 8)
    assert np.all(eigenvalues >= 0)


def test_ndm_propagate_identity_at_t_zero(synthetic_connectome):
    L = connectome.compute_laplacian(synthetic_connectome)
    eigenvalues, eigenvectors = connectome.eigen_decompose_laplacian(L)
    x0 = np.random.default_rng(0).random(8)
    x_t0 = connectome.ndm_propagate(x0, t=0.0, beta=0.5, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
    assert np.allclose(x_t0, x0, atol=1e-6)


def test_ndm_propagate_decays_toward_uniform(synthetic_connectome):
    """As t -> large, diffusion should shrink the spread of x(t) (high-freq eigenmodes decay)."""
    L = connectome.compute_laplacian(synthetic_connectome)
    eigenvalues, eigenvectors = connectome.eigen_decompose_laplacian(L)
    x0 = np.array([5.0, -5.0, 3.0, -3.0, 1.0, -1.0, 0.5, -0.5])
    x_t_small = connectome.ndm_propagate(x0, t=0.1, beta=0.5, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
    x_t_large = connectome.ndm_propagate(x0, t=50.0, beta=0.5, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
    assert x_t_large.std() < x_t_small.std()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_connectome.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oasis2.connectome'`

- [ ] **Step 3: Write `src/oasis2/connectome.py`**

Port lines 2173–2342 of `OASIS2_Colab.py` verbatim (`CONNECTOME_REGIONS`, `N_REGIONS`, `build_synthetic_connectome`, `load_template_connectome`, `compute_laplacian`, `eigen_decompose_laplacian`, `ndm_propagate`) into this file, then apply these adaptations:
- Add module docstring and these imports at the top: `import scipy.linalg`, `from pathlib import Path`, `from typing import List, Optional, Tuple`, `import numpy as np`, `import pandas as pd`, `from oasis2.config import get_data_dir`, `from oasis2.logging_utils import setup_logging`.
- Add `logger = setup_logging("connectome")` (the source implicitly relied on a module-level `logger` defined earlier in the flat script).
- In `load_template_connectome`, the call `get_data_dir("processed")` replaces the source's bare `get_data_dir("processed")` (same name, now an explicit import — no other change needed).

The resulting file must define exactly: `CONNECTOME_REGIONS`, `N_REGIONS`, `build_synthetic_connectome(n_regions=N_REGIONS, seed=42)`, `load_template_connectome(connectome_path=None)`, `compute_laplacian(A)`, `eigen_decompose_laplacian(L)`, `ndm_propagate(x0, t, beta, eigenvalues, eigenvectors)` — same signatures as the source.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_connectome.py -v`
Expected: PASS (6 tests). If `test_eigen_decompose_laplacian_does_not_raise_nameerror` still fails with NameError, the `import scipy.linalg` line is missing — add it.

- [ ] **Step 5: Commit**

```bash
git add src/oasis2/connectome.py tests/test_connectome.py
git commit -m "fix: port connectome module, add missing scipy.linalg import

eigen_decompose_laplacian called scipy.linalg.eigh without scipy ever
being imported before that point in OASIS2_Colab.py, so Phase 3 (and
everything downstream of it, including the digital twin) crashed with
NameError on every run."
```

---

### Task 4: `ndm.py`

**Files:**
- Create: `src/oasis2/ndm.py`
- Test: `tests/test_ndm.py`

Source: `OASIS2_Colab.py` lines 2349–2769: `compute_w_scores`, `fit_beta`, `fit_all_betas`, `predict_future_atrophy`, `validate_ndm`, `run_phase3`.

- [ ] **Step 1: Write `tests/test_ndm.py`**

```python
import numpy as np
import pandas as pd
import pytest

from oasis2 import connectome, ndm


def test_compute_w_scores_adds_wscore_columns(synthetic_regional_volumes, synthetic_demo_df):
    w = ndm.compute_w_scores(synthetic_regional_volumes, synthetic_demo_df)
    wscore_cols = [c for c in w.columns if c.endswith("_wscore")]
    assert len(wscore_cols) > 0
    assert not w[wscore_cols].isna().all().all()


def test_run_phase3_end_to_end_on_synthetic_data(tmp_path, monkeypatch, synthetic_regional_volumes, synthetic_demo_df):
    monkeypatch.setenv("OASIS2_PROJECT_ROOT", str(tmp_path))
    result = ndm.run_phase3(
        regional_volumes=synthetic_regional_volumes,
        demo_df=synthetic_demo_df,
        out_dir=tmp_path / "phase3",
    )
    assert "eigenvalues" in result and "eigenvectors" in result
    assert "w_scores" in result
    assert "beta_df" in result
    assert "validation" in result
    # regression check: must not raise NameError from connectome.eigen_decompose_laplacian
    assert result["eigenvalues"].shape[0] == result["eigenvectors"].shape[0]


def test_run_phase3_handles_empty_regional_volumes_gracefully(tmp_path, monkeypatch, synthetic_demo_df):
    monkeypatch.setenv("OASIS2_PROJECT_ROOT", str(tmp_path))
    empty = pd.DataFrame({"subject_id": synthetic_demo_df["subject_id"].unique()})
    result = ndm.run_phase3(regional_volumes=empty, demo_df=synthetic_demo_df, out_dir=tmp_path / "phase3")
    assert result["beta_df"].empty
    assert result["validation"]["n_subjects_validated"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ndm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oasis2.ndm'`

- [ ] **Step 3: Write `src/oasis2/ndm.py`**

Port lines 2349–2769 of `OASIS2_Colab.py` (`compute_w_scores`, `fit_beta`, `fit_all_betas`, `predict_future_atrophy`, `validate_ndm`, `run_phase3`) verbatim into this file, then apply:
- Add module docstring and imports: `import json`, `from pathlib import Path`, `from typing import Dict, List, Optional`, `import numpy as np`, `import pandas as pd`, `from oasis2.config import get_outputs_dir`, `from oasis2.logging_utils import setup_logging`, `from oasis2.connectome import load_template_connectome, compute_laplacian, eigen_decompose_laplacian, ndm_propagate`.
- Add `logger = setup_logging("ndm")`.
- `run_phase3`'s local `import json` (source line ~2714) can stay or move to the top-level import — keep the top-level one and delete the redundant local `import json` inside the function.

The resulting file must define exactly: `compute_w_scores(regional_volumes, demo_df, region_cols=None)`, `fit_beta(...)`, `fit_all_betas(...)`, `predict_future_atrophy(...)`, `validate_ndm(...)`, `run_phase3(regional_volumes, demo_df, connectome_path=None, out_dir=None)` — same signatures as source.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ndm.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/oasis2/ndm.py tests/test_ndm.py
git commit -m "feat: port NDM baseline module (w-scores, beta fitting, phase 3 pipeline)"
```

---

### Task 5: `data/datasets.py`

**Files:**
- Create: `src/oasis2/data/datasets.py`
- Test: `tests/test_datasets.py`

Source: `OASIS2_Colab.py` lines 1006–1200: `LongitudinalGraphDataset` (with nested `CDR_MAP`), `collate_longitudinal`, `build_dl_datasets`.

- [ ] **Step 1: Write `tests/test_datasets.py`**

```python
import torch

from oasis2.data.datasets import LongitudinalGraphDataset, collate_longitudinal, build_dl_datasets


def test_build_dl_datasets_returns_three_loaders_or_none(synthetic_demo_df, synthetic_regional_volumes, synthetic_connectome):
    from oasis2 import ndm
    w_scores = ndm.compute_w_scores(synthetic_regional_volumes, synthetic_demo_df)
    train_loader, val_loader, test_loader = build_dl_datasets(
        demo_df=synthetic_demo_df,
        w_scores=w_scores,
        adj_matrix=synthetic_connectome,
        region_names=[f"region_{i}" for i in range(synthetic_connectome.shape[0])],
        seed=42,
    )
    assert train_loader is not None
    batch = next(iter(train_loader))
    assert batch["node_features"].dim() == 4  # (B, K, N, D)
    assert batch["adj"].shape[-2:] == synthetic_connectome.shape


def test_collate_longitudinal_pads_to_max_k(synthetic_connectome):
    N = synthetic_connectome.shape[0]
    adj = torch.tensor(synthetic_connectome, dtype=torch.float32).unsqueeze(0)
    items = [
        {
            "node_features": torch.rand(k, N, 1),
            "times": torch.arange(k, dtype=torch.float32),
            "covariates": torch.rand(6),
            "mmse": torch.rand(k) * 30,
            "cdr": torch.randint(0, 4, (k,)),
            "adj": adj,
        }
        for k in (2, 4)
    ]
    batch = collate_longitudinal(items)
    assert batch["node_features"].shape[1] == 4  # padded to max K
    assert batch["mask"].shape == (2, 4)
    assert batch["mask"][0].tolist() == [True, True, False, False]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_datasets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oasis2.data.datasets'`

- [ ] **Step 3: Write `src/oasis2/data/datasets.py`**

Port lines 1006–1200 of `OASIS2_Colab.py` (`LongitudinalGraphDataset`, `collate_longitudinal`, `build_dl_datasets`) verbatim, then apply:
- Add module docstring and imports: `from typing import Dict, List, Optional, Tuple`, `import numpy as np`, `import pandas as pd`, `import torch`, `from torch.utils.data import Dataset, DataLoader`, `from oasis2.logging_utils import setup_logging`.
- Add `logger = setup_logging("datasets")` if the source references a module-level `logger` inside `build_dl_datasets` (check with `grep -n "logger\." OASIS2_Colab.py | awk -F: '$1>=1006 && $1<=1200'`); if it does not, skip this line.
- Verify the notebook's collate function name: the earlier `diff.txt`-era draft called it `collate_fn`, but the current `OASIS2_Colab.py` defines `collate_longitudinal` (confirm with `grep -n "^def collate" OASIS2_Colab.py`) — use whichever name is actually present in the current file; update the test imports in Step 1 to match if it differs.

The resulting file must define: `LongitudinalGraphDataset(Dataset)` (with `CDR_MAP = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}` as a class attribute), `collate_longitudinal(batch)`, `build_dl_datasets(demo_df, w_scores, adj_matrix, region_names, seed=42)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_datasets.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/oasis2/data/datasets.py tests/test_datasets.py
git commit -m "feat: port longitudinal graph dataset builder"
```

---

### Task 6: `models/layers.py` + `models/st_gnn_ode.py` (Model 1: ST-GNN-ODE)

**Files:**
- Create: `src/oasis2/models/layers.py`
- Create: `src/oasis2/models/st_gnn_ode.py`
- Test: `tests/test_st_gnn_ode.py`

Source: `OASIS2_Colab.py` lines 1204–1260 (`SinusoidalTimeEncoding`, `GraphConvLayer`, `GATLayer`) → `layers.py`; lines 1261–1474 (`ODEFunc`, `STGraphNeuralODE`, `train_st_gnn_ode`) → `st_gnn_ode.py`.

- [ ] **Step 1: Write `tests/test_st_gnn_ode.py`**

```python
import torch

from oasis2.models.st_gnn_ode import STGraphNeuralODE, train_st_gnn_ode


def test_forward_shapes(synthetic_batch):
    b = synthetic_batch
    n_feat = b["node_features"].shape[-1]
    n_cov = b["covariates"].shape[-1]
    model = STGraphNeuralODE(n_node_feat=n_feat, hidden_dim=16, n_cov=n_cov)
    recon, mmse_pred, cdr_logits = model(
        b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"]
    )
    B, K, N, D = b["node_features"].shape
    assert recon.shape == (B, K, N, D)
    assert mmse_pred.shape == (B, K)
    assert cdr_logits.shape == (B, K, 4)


def test_gradients_flow(synthetic_batch):
    b = synthetic_batch
    model = STGraphNeuralODE(n_node_feat=1, hidden_dim=16, n_cov=b["covariates"].shape[-1])
    recon, mmse_pred, _ = model(b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"])
    loss = recon.pow(2).mean() + mmse_pred.pow(2).mean()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(g == g for g in grad_norms)  # no NaNs


def test_train_st_gnn_ode_runs_for_a_few_epochs(synthetic_batch):
    from torch.utils.data import DataLoader, TensorDataset

    class ListLoader:
        """Minimal stand-in that yields the same synthetic batch dict repeatedly."""
        def __init__(self, batch, n_batches=3):
            self.batch, self.n = batch, n_batches
        def __iter__(self):
            return iter([self.batch] * self.n)
        def __len__(self):
            return self.n

    loader = ListLoader(synthetic_batch)
    model = train_st_gnn_ode(loader, loader, n_epochs=2, lr=1e-3, patience=10)
    assert model is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_st_gnn_ode.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oasis2.models.st_gnn_ode'`

- [ ] **Step 3: Write `src/oasis2/models/layers.py`**

Port lines 1204–1260 of `OASIS2_Colab.py` (`SinusoidalTimeEncoding`, `GraphConvLayer`, `GATLayer`) verbatim, then apply: add module docstring and `import math`, `import torch`, `import torch.nn as nn`, `import torch.nn.functional as F` at the top.

- [ ] **Step 4: Write `src/oasis2/models/st_gnn_ode.py`**

Port lines 1261–1474 of `OASIS2_Colab.py` (`ODEFunc`, `STGraphNeuralODE`, `train_st_gnn_ode`) verbatim, then apply:
- Add module docstring and imports: `import torch`, `import torch.nn as nn`, `import torch.nn.functional as F`, `from oasis2.config import get_device`, `from oasis2.logging_utils import setup_logging`, `from oasis2.models.layers import SinusoidalTimeEncoding, GraphConvLayer, GATLayer`.
- Add `logger = setup_logging("st_gnn_ode")` and `DEVICE = get_device()` at module scope (the source used a shared top-of-script `DEVICE` global — `train_st_gnn_ode` calls `.to(DEVICE)`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_st_gnn_ode.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add src/oasis2/models/layers.py src/oasis2/models/st_gnn_ode.py tests/test_st_gnn_ode.py
git commit -m "feat: port shared GNN layers and ST-GNN-ODE model (Model 1)"
```

---

### Task 7: `models/nd_vae.py` (Model 2: ND-VAE)

**Files:**
- Create: `src/oasis2/models/nd_vae.py`
- Test: `tests/test_nd_vae.py`

Source: `OASIS2_Colab.py` lines 1481–1718: `PhysicsConstrainedNDVAE` (with `encode`, `reparameterize`, `decode`, `forward`, `physics_loss` methods), `train_nd_vae`.

- [ ] **Step 1: Write `tests/test_nd_vae.py`**

```python
import torch

from oasis2.models.nd_vae import PhysicsConstrainedNDVAE, train_nd_vae


def test_forward_returns_five_tensors(synthetic_batch):
    b = synthetic_batch
    N = b["node_features"].shape[2]
    model = PhysicsConstrainedNDVAE(n_regions=N, n_node_feat=1, n_cov=b["covariates"].shape[-1], latent_dim=8, hidden_dim=16)
    recon, mmse_pred, cdr_logits, mu, logvar = model(
        b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"]
    )
    B, K, _, D = b["node_features"].shape
    assert recon.shape == (B, K, N, D)
    assert mu.shape == (B, 8) and logvar.shape == (B, 8)


def test_physics_loss_is_zero_for_single_visit(synthetic_batch):
    b = synthetic_batch
    N = b["node_features"].shape[2]
    model = PhysicsConstrainedNDVAE(n_regions=N, n_cov=b["covariates"].shape[-1], latent_dim=8, hidden_dim=16)
    recon = b["node_features"][:, :1]  # K=1
    loss = model.physics_loss(recon, b["times"][:, :1], b["adj"], b["mask"][:, :1])
    assert loss.item() == 0.0


def test_physics_loss_finite_and_nonnegative(synthetic_batch):
    b = synthetic_batch
    N = b["node_features"].shape[2]
    model = PhysicsConstrainedNDVAE(n_regions=N, n_cov=b["covariates"].shape[-1], latent_dim=8, hidden_dim=16)
    loss = model.physics_loss(b["node_features"], b["times"], b["adj"], b["mask"])
    assert loss.item() >= 0.0
    assert loss.item() == loss.item()  # not NaN


def test_train_nd_vae_runs(synthetic_batch):
    class ListLoader:
        def __init__(self, batch, n=3): self.batch, self.n = batch, n
        def __iter__(self): return iter([self.batch] * self.n)
        def __len__(self): return self.n

    loader = ListLoader(synthetic_batch)
    N = synthetic_batch["node_features"].shape[2]
    model = train_nd_vae(loader, loader, n_regions=N, n_epochs=2, lr=1e-3, patience=10)
    assert model is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nd_vae.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oasis2.models.nd_vae'`

- [ ] **Step 3: Write `src/oasis2/models/nd_vae.py`**

Port lines 1481–1718 of `OASIS2_Colab.py` (`PhysicsConstrainedNDVAE`, `train_nd_vae`) verbatim, then apply:
- Add module docstring and imports: `import torch`, `import torch.nn as nn`, `import torch.nn.functional as F`, `from oasis2.config import get_device`, `from oasis2.logging_utils import setup_logging`.
- Add `logger = setup_logging("nd_vae")` and `DEVICE = get_device()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_nd_vae.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/oasis2/models/nd_vae.py tests/test_nd_vae.py
git commit -m "feat: port physics-constrained ND-VAE model (Model 2)"
```

---

### Task 8: `models/tadm.py` (Model 3: TADM)

**Files:**
- Create: `src/oasis2/models/tadm.py`
- Test: `tests/test_tadm.py`

Source: `OASIS2_Colab.py` lines 1725–2000: `cosine_beta_schedule`, `DiffusionConditioner`, `DenoisingNet`, `TemporalDiffusionModel`, `train_tadm`.

- [ ] **Step 1: Write `tests/test_tadm.py`**

```python
import torch

from oasis2.models.tadm import cosine_beta_schedule, TemporalDiffusionModel, train_tadm


def test_cosine_beta_schedule_monotonic_and_bounded():
    betas = cosine_beta_schedule(timesteps=50)
    assert betas.shape == (50,)
    assert (betas >= 0).all() and (betas <= 1).all()


def test_temporal_diffusion_model_forward_shapes(synthetic_batch):
    b = synthetic_batch
    N = b["node_features"].shape[2]
    model = TemporalDiffusionModel(n_regions=N, n_node_feat=1, n_cov=b["covariates"].shape[-1], hidden_dim=16)
    out = model(b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"])
    assert isinstance(out, tuple)


def test_train_tadm_runs(synthetic_batch):
    class ListLoader:
        def __init__(self, batch, n=3): self.batch, self.n = batch, n
        def __iter__(self): return iter([self.batch] * self.n)
        def __len__(self): return self.n

    loader = ListLoader(synthetic_batch)
    N = synthetic_batch["node_features"].shape[2]
    model = train_tadm(loader, loader, n_regions=N, n_epochs=2, lr=1e-3, patience=10)
    assert model is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tadm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oasis2.models.tadm'`

- [ ] **Step 3: Write `src/oasis2/models/tadm.py`**

Port lines 1725–2000 of `OASIS2_Colab.py` (`cosine_beta_schedule`, `DiffusionConditioner`, `DenoisingNet`, `TemporalDiffusionModel`, `train_tadm`) verbatim, then apply:
- Add module docstring and imports: `import math`, `import torch`, `import torch.nn as nn`, `import torch.nn.functional as F`, `from oasis2.config import get_device`, `from oasis2.logging_utils import setup_logging`.
- Add `logger = setup_logging("tadm")` and `DEVICE = get_device()`.
- Before writing Step 2's test more precisely: open the source and confirm `TemporalDiffusionModel.forward`'s actual return arity (used by `evaluate_dl_model` in Task 9 — check with `grep -n "class TemporalDiffusionModel" -A 80 OASIS2_Colab.py`), and adjust `test_temporal_diffusion_model_forward_shapes` to unpack the exact number of returned tensors instead of the generic `isinstance(out, tuple)` check once known.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tadm.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/oasis2/models/tadm.py tests/test_tadm.py
git commit -m "feat: port temporally-aware trajectory diffusion model (Model 3)"
```

---

### Task 9: `models/benchmark.py`

**Files:**
- Create: `src/oasis2/models/benchmark.py`
- Test: `tests/test_benchmark.py`

Source: `OASIS2_Colab.py` lines 2007–2159: `evaluate_dl_model`, `run_benchmark`.

- [ ] **Step 1: Write `tests/test_benchmark.py`**

```python
from oasis2.models.benchmark import evaluate_dl_model, run_benchmark
from oasis2.models.st_gnn_ode import STGraphNeuralODE


def test_evaluate_dl_model_returns_metrics_dict(synthetic_batch):
    b = synthetic_batch
    N = b["node_features"].shape[2]
    model = STGraphNeuralODE(n_node_feat=1, hidden_dim=16, n_cov=b["covariates"].shape[-1])

    class ListLoader:
        def __init__(self, batch, n=2): self.batch, self.n = batch, n
        def __iter__(self): return iter([self.batch] * self.n)
        def __len__(self): return self.n

    metrics = evaluate_dl_model(model, ListLoader(b), "ST-GNN-ODE")
    assert isinstance(metrics, dict)
    assert len(metrics) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `src/oasis2/models/benchmark.py`**

Port lines 2007–2159 of `OASIS2_Colab.py` (`evaluate_dl_model`, `run_benchmark`) verbatim, then apply: add module docstring and imports `import numpy as np`, `import pandas as pd`, `import torch`, `from pathlib import Path`, `from typing import Dict, Optional`, `from oasis2.logging_utils import setup_logging`, `from oasis2.config import get_device, get_outputs_dir`; add `logger = setup_logging("benchmark")` and `DEVICE = get_device()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_benchmark.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/oasis2/models/benchmark.py tests/test_benchmark.py
git commit -m "feat: port cross-model benchmark runner"
```

---

### Task 10: `deficits.py`

**Files:**
- Create: `src/oasis2/deficits.py`
- Test: `tests/test_deficits.py`

Source: `OASIS2_Colab.py` lines 2804–3139: region→cognitive-domain atlas, `get_domain_weights`, `compute_deficit_scores`, `compute_all_deficit_scores`, `validate_deficit_predictions`, `compute_longitudinal_deltas`, `run_phase4`.

- [ ] **Step 1: Write `tests/test_deficits.py`**

```python
import pandas as pd

from oasis2 import ndm, deficits


def test_run_phase4_on_synthetic_ndm_results(tmp_path, monkeypatch, synthetic_regional_volumes, synthetic_demo_df):
    monkeypatch.setenv("OASIS2_PROJECT_ROOT", str(tmp_path))
    p3 = ndm.run_phase3(regional_volumes=synthetic_regional_volumes, demo_df=synthetic_demo_df, out_dir=tmp_path / "phase3")
    p3["region_names"] = [f"region_{i}" for i in range(p3["eigenvalues"].shape[0])]
    result = deficits.run_phase4(ndm_results=p3, demo_df=synthetic_demo_df, out_dir=tmp_path / "phase4")
    assert "deficit_scores" in result
    assert isinstance(result["deficit_scores"], pd.DataFrame)


def test_get_domain_weights_returns_nonempty_dict():
    weights = deficits.get_domain_weights()
    assert isinstance(weights, dict)
    assert len(weights) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_deficits.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `src/oasis2/deficits.py`**

Port lines 2804–3139 of `OASIS2_Colab.py` verbatim, then apply: add module docstring and imports `import json`, `from pathlib import Path`, `from typing import Dict, List, Optional, Tuple`, `import numpy as np`, `import pandas as pd`, `import scipy.stats`, `from oasis2.config import get_outputs_dir`, `from oasis2.logging_utils import setup_logging`; add `logger = setup_logging("deficits")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_deficits.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/oasis2/deficits.py tests/test_deficits.py
git commit -m "feat: port cognitive deficit mapping module (phase 4)"
```

---

### Task 11: `interpretability.py` — NEW module (was never ported)

**Files:**
- Create: `src/oasis2/interpretability.py`
- Test: `tests/test_interpretability.py`
- Modify: `requirements.txt` (already has `captum` from Task 1)

Source: notebook cell 188 of `OASIS2_Complete_Colab_Pipeline.ipynb` ("Phase 4/5: Interpretability Layer") — reproduced in full in the design spec §1. This cell was never ported; today there is no callable interpretability code outside the notebook at all. The notebook version relies on notebook-global variables (`m1`, `test_ld`, `nf_in`, `t_in`, `cov_in`, `adj_in`, `msk_in`, `DEVICE`) set by earlier cells — this task turns it into explicit, parameterized functions/classes.

- [ ] **Step 1: Write `tests/test_interpretability.py`**

```python
import torch

from oasis2.interpretability import NodePredictor, SingleInputWrapper, TemplateRenderer, compute_node_attribution
from oasis2.models.st_gnn_ode import STGraphNeuralODE


def test_node_predictor_returns_scalar_per_sample(synthetic_batch):
    b = synthetic_batch
    model = STGraphNeuralODE(n_node_feat=1, hidden_dim=16, n_cov=b["covariates"].shape[-1])
    wrapper = NodePredictor(model, target_node_idx=0)
    out = wrapper(b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"])
    assert out.shape == (b["node_features"].shape[0],)


def test_single_input_wrapper_handles_captum_batch_expansion(synthetic_batch):
    b = synthetic_batch
    model = STGraphNeuralODE(n_node_feat=1, hidden_dim=16, n_cov=b["covariates"].shape[-1])
    node_wrapper = NodePredictor(model, target_node_idx=0)
    si = SingleInputWrapper(node_wrapper, b["times"], b["covariates"], b["adj"], b["mask"])
    # Captum's IntegratedGradients internally repeats the input batch n_steps times
    expanded_nf = b["node_features"].repeat(5, 1, 1, 1)
    out = si(expanded_nf)
    assert out.shape == (b["node_features"].shape[0] * 5,)


def test_template_renderer_names_the_top_node():
    import numpy as np
    renderer = TemplateRenderer()
    attr = np.array([0.1, 0.9, -0.2, 0.05])
    report = renderer.render(attr)
    assert "Node 1" in report  # index of the max value


def test_compute_node_attribution_end_to_end(synthetic_batch):
    """Exercises the full Captum IntegratedGradients path on a tiny model."""
    b = synthetic_batch
    model = STGraphNeuralODE(n_node_feat=1, hidden_dim=8, n_cov=b["covariates"].shape[-1])
    attr_sum, report = compute_node_attribution(
        model, b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"], target_node_idx=0
    )
    assert attr_sum.shape == (b["node_features"].shape[2],)  # (N,)
    assert isinstance(report, str)
    assert "[CLINICIAN REPORT]" in report
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_interpretability.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oasis2.interpretability'`

- [ ] **Step 3: Write `src/oasis2/interpretability.py`**

```python
"""Interpretability layer for the digital twin's regional predictions.

Wraps a trained trajectory model (e.g. STGraphNeuralODE) so Captum's
IntegratedGradients can attribute a single node's predicted future value back
to the baseline input features. Ported from OASIS2_Complete_Colab_Pipeline.ipynb
cell 188 ("Phase 4/5: Interpretability Layer"), which previously only ran
inline inside the notebook against notebook-global variables (m1, test_ld,
nf_in, ...) and was never callable as a library function.
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from oasis2.logging_utils import setup_logging

logger = setup_logging("interpretability")


class NodePredictor(nn.Module):
    """Wraps a trajectory model to output a single node's next-visit prediction."""

    def __init__(self, model: nn.Module, target_node_idx: int = 0):
        super().__init__()
        self.model = model
        self.target_node_idx = target_node_idx

    def forward(self, nf, times, cov, adj, mask):
        recon, _, _ = self.model(nf, times, cov, adj, mask)
        if recon.shape[1] > 1:
            return recon[:, 1, self.target_node_idx, 0]
        return recon[:, 0, self.target_node_idx, 0]


class SingleInputWrapper(nn.Module):
    """Exposes a model that takes (nf, times, cov, adj, mask) as a single-input
    nf -> scalar function, repeating the fixed context tensors to match
    Captum's internal batch expansion (IntegratedGradients replicates the
    input along dim 0 by `n_steps`)."""

    def __init__(self, model: nn.Module, t, cov, adj, msk):
        super().__init__()
        self.model = model
        self.t, self.cov, self.adj, self.msk = t, cov, adj, msk

    def forward(self, nf):
        B_in = self.t.shape[0]
        B_nf = nf.shape[0]
        if B_nf > B_in:
            M = B_nf // B_in
            t = self.t.repeat(M, 1)
            cov = self.cov.repeat(M, 1)
            msk = self.msk.repeat(M, 1)
            return self.model(nf, t, cov, self.adj, msk)
        return self.model(nf, self.t, self.cov, self.adj, self.msk)


class TemplateRenderer:
    """Turns a per-region attribution array into a one-line clinician-facing report."""

    def render(self, attr_array: np.ndarray) -> str:
        top_node = int(np.argmax(attr_array))
        top_val = attr_array[top_node]
        return (
            f"[CLINICIAN REPORT] The model's prediction of future atrophy is most "
            f"strongly driven by the baseline state of Node {top_node} "
            f"(Attribution Score: {top_val:.4f})."
        )


def compute_node_attribution(
    model: nn.Module, nf, times, cov, adj, mask, target_node_idx: int = 0
) -> Tuple[np.ndarray, str]:
    """Run Integrated Gradients on `target_node_idx`'s prediction and render a report.

    Returns (attr_sum, report) where attr_sum has shape (N,) — the baseline-visit
    attribution for every region.
    """
    from captum.attr import IntegratedGradients

    node_wrapper = NodePredictor(model, target_node_idx=target_node_idx)
    single_input = SingleInputWrapper(node_wrapper, times, cov, adj, mask)
    ig = IntegratedGradients(single_input)

    baseline = torch.zeros_like(nf)
    attributions, _delta = ig.attribute(nf, baseline, return_convergence_delta=True)
    attr_sum = attributions[0, 0, :, 0].cpu().detach().numpy()

    renderer = TemplateRenderer()
    report = renderer.render(attr_sum)
    logger.info(report)
    return attr_sum, report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pip install captum && pytest tests/test_interpretability.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/oasis2/interpretability.py tests/test_interpretability.py
git commit -m "feat: port interpretability layer (Captum attribution) — previously notebook-only

NodePredictor, SingleInputWrapper, and TemplateRenderer existed only as
inline notebook cell 188 code that ran against notebook-global variables.
They were never callable as library functions; this adds a parameterized
compute_node_attribution() entry point with test coverage."
```

---

### Task 12: `digital_twin.py` — NEW module (was never ported) + `.train()`/`.eval()` bugfix

**Files:**
- Create: `src/oasis2/digital_twin.py`
- Test: `tests/test_digital_twin.py`

Source: notebook cell 190 of `OASIS2_Complete_Colab_Pipeline.ipynb` ("Phase 6: Trajectory Simulator & Counterfactuals" + "Phase 7: Advanced Validation Metrics") — reproduced in full in the design spec §1. This is the digital twin's actual forward-prediction code (autoregressive rollout, Monte-Carlo confidence bands, counterfactual scenarios) and it was never ported — today it only runs inline in the notebook.

**Confirmed bug to fix while porting:** the notebook's `generate_confidence_bands` calls `model.train()` to enable VAE sampling stochasticity, but never calls `model.eval()` afterward — any code that runs after it (e.g. a subsequent `evaluate_dl_model` call on the same model instance) would silently run in train mode, which changes GRU/dropout/BatchNorm behavior and produces different results than intended. Fix: wrap the sampling loop so the model is always restored to its original mode (`train` or `eval`) before returning, even if an exception occurs.

- [ ] **Step 1: Write `tests/test_digital_twin.py`**

```python
import numpy as np
import torch

from oasis2.digital_twin import simulate_rollout, generate_confidence_bands, apply_counterfactual
from oasis2.models.nd_vae import PhysicsConstrainedNDVAE


def _make_model(synthetic_batch):
    N = synthetic_batch["node_features"].shape[2]
    return PhysicsConstrainedNDVAE(n_regions=N, n_cov=synthetic_batch["covariates"].shape[-1], latent_dim=8, hidden_dim=16)


def test_simulate_rollout_produces_requested_steps(synthetic_batch):
    b = synthetic_batch
    model = _make_model(b)
    nf_init = b["node_features"][0:1, 0:1]  # single subject, single visit
    rollout = simulate_rollout(model, nf_init, b["covariates"][0:1], b["adj"], steps=3)
    assert rollout.shape[1] == 3  # 3 predicted future steps
    assert rollout.shape[0] == 1


def test_generate_confidence_bands_upper_bounds_lower(synthetic_batch):
    b = synthetic_batch
    model = _make_model(b)
    mean_pred, lower, upper = generate_confidence_bands(
        model, b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"], n_samples=5
    )
    assert (upper >= lower).all()
    assert mean_pred.shape == lower.shape == upper.shape


def test_generate_confidence_bands_restores_model_mode(synthetic_batch):
    """Regression test: the notebook version left the model in .train() mode."""
    b = synthetic_batch
    model = _make_model(b)
    model.eval()
    assert not model.training
    generate_confidence_bands(model, b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"], n_samples=3)
    assert not model.training, "model was left in train() mode after generate_confidence_bands"


def test_generate_confidence_bands_restores_mode_even_on_exception(synthetic_batch, monkeypatch):
    b = synthetic_batch
    model = _make_model(b)
    model.eval()

    original_forward = model.forward
    calls = {"n": 0}

    def flaky_forward(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated failure mid-sampling")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", flaky_forward)
    try:
        generate_confidence_bands(model, b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"], n_samples=5)
    except RuntimeError:
        pass
    assert not model.training, "model mode must be restored even if sampling raises"


def test_apply_counterfactual_changes_age_covariate(synthetic_batch):
    b = synthetic_batch
    model = _make_model(b)
    diff = apply_counterfactual(
        model, b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"],
        covariate_idx=0, delta=-0.1,
    )
    assert isinstance(diff, float)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_digital_twin.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oasis2.digital_twin'`

- [ ] **Step 3: Write `src/oasis2/digital_twin.py`**

```python
"""Digital twin forward-prediction layer: autoregressive trajectory rollout,
Monte-Carlo uncertainty bands, and counterfactual scenarios.

Ported from OASIS2_Complete_Colab_Pipeline.ipynb cell 190 ("Phase 6: Trajectory
Simulator & Counterfactuals" / "Phase 7: Advanced Validation Metrics"), which
previously only ran inline in the notebook against notebook-global variables
(m2, nf_in, t_in, ...) and was never callable as a library function — meaning
the project's stated core deliverable (predict which regions are affected
next, per implementation.md §1) could not run outside the notebook at all.
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from oasis2.logging_utils import setup_logging

logger = setup_logging("digital_twin")


def simulate_rollout(model: nn.Module, nf_init: torch.Tensor, cov: torch.Tensor, adj: torch.Tensor, steps: int = 3) -> torch.Tensor:
    """Autoregressive forward rollout: repeatedly predicts the next visit and
    feeds it back in as input. `nf_init` is (B, 1, N, D) — a single baseline visit.
    Returns (B, steps, N, D)."""
    model.eval()
    B, _, N, D = nf_init.shape
    current_nf = nf_init.clone()
    trajectories = []

    with torch.no_grad():
        for _step in range(steps):
            K = current_nf.shape[1]
            dummy_next = torch.zeros_like(current_nf[:, 0:1])
            nf_padded = torch.cat([current_nf, dummy_next], dim=1)
            t_padded = torch.arange(K + 1, dtype=torch.float32).unsqueeze(0).repeat(B, 1).to(nf_init.device)
            msk_padded = torch.ones((B, K + 1), dtype=torch.bool).to(nf_init.device)
            msk_padded[:, -1] = False

            recon, _, _, _, _ = model(nf_padded, t_padded, cov, adj, msk_padded)
            next_state = recon[:, -1:, :, :]
            trajectories.append(next_state)
            current_nf = torch.cat([current_nf, next_state], dim=1)

    return torch.cat(trajectories, dim=1)


def generate_confidence_bands(
    model: nn.Module, nf, times, cov, adj, mask, n_samples: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monte-Carlo sample the VAE's stochastic reparameterization n_samples times
    and return (mean, 5th percentile, 95th percentile) prediction arrays.

    The model must be put into train() mode to re-sample the VAE's latent
    noise on every forward pass (PhysicsConstrainedNDVAE.reparameterize always
    samples eps ~ N(0,1), but train() vs eval() affects GRU/dropout behavior
    the same way it does for any other torch module) — this function always
    restores the model's original mode before returning, including when
    sampling raises, so it can't leak train-mode into unrelated code that
    reuses the same model instance afterward.
    """
    was_training = model.training
    model.train()
    samples = []
    try:
        with torch.no_grad():
            for _ in range(n_samples):
                recon, _, _, _, _ = model(nf, times, cov, adj, mask)
                samples.append(recon.cpu().numpy())
    finally:
        model.train(was_training)

    samples = np.array(samples)
    mean_pred = np.mean(samples, axis=0)
    lower_bound = np.percentile(samples, 5, axis=0)
    upper_bound = np.percentile(samples, 95, axis=0)
    return mean_pred, lower_bound, upper_bound


def apply_counterfactual(
    model: nn.Module, nf, times, cov, adj, mask, covariate_idx: int = 0, delta: float = -0.1
) -> float:
    """Perturb one covariate (e.g. age, scaled by 100 in the source data — see
    implementation.md) by `delta` and return the average change in predicted
    regional volumes vs. the unperturbed mean prediction."""
    mean_pred, _, _ = generate_confidence_bands(model, nf, times, cov, adj, mask, n_samples=10)

    was_training = model.training
    model.eval()
    cov_cf = cov.clone()
    cov_cf[:, covariate_idx] += delta
    try:
        with torch.no_grad():
            recon_cf, _, _, _, _ = model(nf, times, cov_cf, adj, mask)
    finally:
        model.train(was_training)

    diff = (recon_cf.cpu().numpy() - mean_pred).mean()
    return float(diff)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_digital_twin.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/oasis2/digital_twin.py tests/test_digital_twin.py
git commit -m "feat: port digital twin trajectory simulator — previously notebook-only

simulate_rollout, generate_confidence_bands, and the counterfactual helper
existed only as inline notebook cell 190 code, never callable outside the
notebook. Also fixes generate_confidence_bands leaving the model in
.train() mode after returning (no matching .eval() in the source) by
restoring the model's original mode in a finally block, including when
sampling raises."
```

---

### Task 13: `data/ingest.py` + `data/synthseg.py`

**Files:**
- Create: `src/oasis2/data/ingest.py`
- Create: `src/oasis2/data/synthseg.py`
- Test: `tests/test_ingest.py`
- Test: `tests/test_synthseg.py`

Source: `OASIS2_Colab.py` lines 279–620 (`find_demographic_csv`, `load_demographic_csv`, `find_nifti_or_analyze`, `walk_oasis2_directory`, `_process_session`, `convert_analyze_to_nifti`, `batch_convert_analyze`, `data_quality_report`, `build_subjects_index`) → `data/ingest.py`; lines 622–965 (`synthseg_available`, `get_synthseg_threads`, `run_synthseg_single`, `run_synthseg_batch`, `aggregate_volumes`, `extract_fsl_seg_volumes`, `build_regional_volumes`, `spot_check_synthseg_output`) → `data/synthseg.py`.

No real MRI data is available locally, so these tests use `tmp_path`-based mock directory structures (empty placeholder files with the right names/extensions) rather than real scans — they check the directory-walking and file-discovery logic, not actual image processing.

- [ ] **Step 1: Write `tests/test_ingest.py`**

```python
from pathlib import Path

from oasis2.data.ingest import find_nifti_or_analyze, walk_oasis2_directory, load_demographic_csv


def test_find_nifti_or_analyze_finds_hdr_files(tmp_path):
    session_dir = tmp_path / "OAS2_0001_MR1" / "RAW"
    session_dir.mkdir(parents=True)
    (session_dir / "mpr-1.nifti.hdr").touch()
    (session_dir / "mpr-1.nifti.img").touch()
    found = find_nifti_or_analyze(session_dir)
    assert any(p.suffix == ".hdr" for p in found)


def test_walk_oasis2_directory_discovers_sessions(tmp_path):
    for subj, visit in [("0001", "MR1"), ("0001", "MR2"), ("0002", "MR1")]:
        d = tmp_path / f"OAS2_{subj}_{visit}" / "RAW"
        d.mkdir(parents=True)
        (d / "mpr-1.nifti.hdr").touch()
        (d / "mpr-1.nifti.img").touch()
    records = walk_oasis2_directory(tmp_path)
    assert len(records) == 3


def test_load_demographic_csv_reads_expected_columns(tmp_path):
    import pandas as pd
    p = tmp_path / "demo.csv"
    pd.DataFrame({"Subject ID": ["OAS2_0001"], "MRI ID": ["OAS2_0001_MR1"], "Age": [75]}).to_csv(p, index=False)
    df = load_demographic_csv(p)
    assert "Subject ID" in df.columns
    assert len(df) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `src/oasis2/data/ingest.py`**

Port lines 279–620 of `OASIS2_Colab.py` verbatim, then apply: add module docstring and imports `import re`, `from pathlib import Path`, `from typing import Dict, List, Optional, Tuple`, `import numpy as np`, `import pandas as pd`, `from tqdm.auto import tqdm`, `from oasis2.logging_utils import setup_logging`; add `logger = setup_logging("ingest")` (the source's own docstring names this virtual module `preprocessing.py` — the file's actual responsibilities, per its own header comment, are Phase 0 data audit + Phase 1 prep, which is why it's placed under `data/`).

- [ ] **Step 4: Write `tests/test_synthseg.py`**

```python
from oasis2.data.synthseg import synthseg_available, extract_fsl_seg_volumes


def test_synthseg_available_returns_bool():
    assert isinstance(synthseg_available(), bool)


def test_extract_fsl_seg_volumes_on_missing_dir_returns_empty(tmp_path):
    result = extract_fsl_seg_volumes(tmp_path / "does_not_exist")
    assert result == {}
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `pytest tests/test_synthseg.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 6: Write `src/oasis2/data/synthseg.py`**

Port lines 622–965 of `OASIS2_Colab.py` verbatim, then apply: add module docstring and imports `import os`, `import subprocess`, `import shutil`, `from pathlib import Path`, `from typing import Dict, List, Optional, Tuple`, `import numpy as np`, `import pandas as pd`, `from tqdm.auto import tqdm`, `from oasis2.logging_utils import setup_logging`, `from oasis2.config import get_data_dir`; add `logger = setup_logging("synthseg")`. If `extract_fsl_seg_volumes` doesn't already return `{}` for a nonexistent directory (check the source), add that guard as the first line of the function body.

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_ingest.py tests/test_synthseg.py -v`
Expected: PASS (5 tests)

- [ ] **Step 8: Commit**

```bash
git add src/oasis2/data/ingest.py src/oasis2/data/synthseg.py tests/test_ingest.py tests/test_synthseg.py
git commit -m "feat: port data ingest (phase 0/1) and SynthSeg wrapper modules"
```

---

### Task 14: `visualization.py`

**Files:**
- Create: `src/oasis2/visualization.py`
- Test: `tests/test_visualization.py`

Source: `OASIS2_Colab.py` lines 3291–3441 — currently inline script code (not wrapped in functions) that plots the Phase 0 dataset overview, Phase 3 NDM results, and Phase 4 deficit profiles using module-level globals (`demo_df`, `outputs_dir`, `beta_df`, `w_scores`, `eigenvalues`, `eigenvectors`, `deficit_df`). This task wraps each block into an explicit function.

- [ ] **Step 1: Write `tests/test_visualization.py`**

```python
import matplotlib
matplotlib.use("Agg")  # no display in CI/test environments

from oasis2 import visualization


def test_plot_phase0_overview_writes_png(tmp_path, synthetic_demo_df):
    out = visualization.plot_phase0_overview(synthetic_demo_df, tmp_path)
    assert out.exists()
    assert out.suffix == ".png"


def test_plot_phase4_deficit_profiles_writes_png(tmp_path):
    import pandas as pd
    import numpy as np
    deficit_df = pd.DataFrame(
        np.random.default_rng(0).random((10, 4)),
        columns=["memory", "language", "executive", "visuospatial"],
        index=[f"OAS2_{i:04d}" for i in range(10)],
    )
    out = visualization.plot_phase4_deficit_profiles(deficit_df, tmp_path)
    assert out.exists()


def test_plot_phase4_deficit_profiles_handles_empty_df(tmp_path):
    import pandas as pd
    out = visualization.plot_phase4_deficit_profiles(pd.DataFrame(), tmp_path)
    assert out is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_visualization.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `src/oasis2/visualization.py`**

```python
"""Plotting functions for pipeline outputs.

Wraps the inline plotting code that previously lived directly in the
OASIS2_Colab.py script tail (lines ~3291-3441), which referenced
script-level globals (demo_df, outputs_dir, beta_df, ...) instead of taking
explicit parameters, into standalone, individually-testable functions.
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oasis2.logging_utils import setup_logging

logger = setup_logging("visualization")


def plot_phase0_overview(demo_df: pd.DataFrame, outputs_dir: Path) -> Path:
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('OASIS-2 Dataset Overview', fontsize=14, fontweight='bold')
    if 'Group' in demo_df.columns:
        gc = demo_df.groupby('subject_id')['Group'].last().value_counts()
        axes[0].bar(gc.index, gc.values, color=['#4CAF50', '#F44336', '#FF9800'][:len(gc)])
        axes[0].set_title('Subjects by Group'); axes[0].set_ylabel('Count')
    if 'Age' in demo_df.columns:
        demo_df.groupby('subject_id')['Age'].first().hist(bins=15, ax=axes[1], color='#9C27B0', alpha=0.8)
        axes[1].set_title('Age Distribution'); axes[1].set_xlabel('Age')
    if 'CDR' in demo_df.columns:
        cc = demo_df['CDR'].value_counts().sort_index()
        axes[2].bar(cc.index.astype(str), cc.values, color='#2196F3')
        axes[2].set_title('CDR Score Distribution'); axes[2].set_xlabel('CDR')
    plt.tight_layout()
    out_path = outputs_dir / 'phase0_overview.png'
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_phase3_ndm_results(
    beta_df: pd.DataFrame, w_scores: pd.DataFrame, eigenvalues: np.ndarray,
    eigenvectors: np.ndarray, outputs_dir: Path,
) -> Optional[Path]:
    from oasis2.connectome import ndm_propagate

    if beta_df.empty or 'beta' not in beta_df.columns:
        return None
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Phase 3: Network Diffusion Model', fontsize=13, fontweight='bold')
    valid_b = beta_df['beta'].dropna()
    axes[0].hist(valid_b, bins=20, color='#9C27B0', alpha=0.85, edgecolor='white')
    axes[0].axvline(valid_b.median(), color='red', linestyle='--', label=f'Median beta={valid_b.median():.3f}')
    axes[0].set_xlabel('Diffusivity beta'); axes[0].set_ylabel('Count')
    axes[0].set_title('Fitted beta per subject'); axes[0].legend()

    top = beta_df.dropna(subset=['beta']).sort_values('beta', ascending=False).iloc[0]
    wscore_cols = [c for c in w_scores.columns if c.endswith('_wscore')]
    ws = w_scores[w_scores['subject_id'] == top['subject_id']]
    if not ws.empty and wscore_cols:
        x0 = ws.iloc[0][wscore_cols].values.astype(float)
        n = min(len(x0), len(eigenvalues))
        x5 = ndm_propagate(x0[:n], 5.0, top['beta'], eigenvalues[:n], eigenvectors[:n, :n])
        axes[1].bar(range(n), x5, color=['#F44336' if v > 0 else '#4CAF50' for v in x5], alpha=0.8)
        axes[1].axhline(0, color='black', linewidth=1)
        axes[1].set_xlabel('Brain Region Index'); axes[1].set_ylabel('Atrophy (W-score)')
        axes[1].set_title(f'5-yr Prediction — {top["subject_id"]} (beta={top["beta"]:.3f})')
    plt.tight_layout()
    out_path = outputs_dir / 'ndm_results.png'
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_phase4_deficit_profiles(deficit_df: pd.DataFrame, outputs_dir: Path) -> Optional[Path]:
    if deficit_df.empty:
        return None
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    mean_d = deficit_df.mean().sort_values(ascending=False)
    axes[0].bar(range(len(mean_d)), mean_d.values, color=plt.get_cmap('RdYlGn_r')(np.linspace(0.1, 0.9, len(mean_d))))
    axes[0].set_xticks(range(len(mean_d)))
    axes[0].set_xticklabels(mean_d.index, rotation=45, ha='right', fontsize=9)
    axes[0].set_ylabel('Mean deficit'); axes[0].set_title('Population-avg Deficit Profiles')
    if len(deficit_df) > 1:
        s = deficit_df.sample(min(20, len(deficit_df)), random_state=42)
        im = axes[1].imshow(s.values, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[1].set_yticks(range(len(s))); axes[1].set_yticklabels(s.index, fontsize=7)
        axes[1].set_xticks(range(len(deficit_df.columns)))
        axes[1].set_xticklabels(deficit_df.columns, rotation=45, ha='right', fontsize=8)
        plt.colorbar(im, ax=axes[1], label='Deficit score')
        axes[1].set_title('Per-subject deficit profiles (sample 20)')
    plt.tight_layout()
    out_path = outputs_dir / 'deficit_profiles.png'
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return out_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_visualization.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/oasis2/visualization.py tests/test_visualization.py
git commit -m "feat: extract plotting code into testable visualization module"
```

---

### Task 15: `colab_setup.py`, `pipeline.py`, `cli.py` — orchestration

**Files:**
- Create: `src/oasis2/colab_setup.py`
- Create: `src/oasis2/pipeline.py`
- Create: `src/oasis2/cli.py`
- Test: `tests/test_pipeline.py`

Source: `OASIS2_Colab.py` lines 1–39 (GPU check, Drive mount, pip installs) + 252–264 (`colab_drive_path`) + lines ~3162–3246 (Kaggle credentials/download) → `colab_setup.py` (Colab-only, not imported by default). Lines ~3140–3549 (the full linear driver: connectome load, Phase 0/1/3/4 calls, DL model training, benchmark, model saving, per-subject report) → `pipeline.py`, replacing script-level globals with an explicit `run_pipeline()` function that also calls the digital twin (`simulate_rollout`, `generate_confidence_bands`) and interpretability (`compute_node_attribution`) modules from Tasks 11–12 — **this wiring doesn't exist anywhere yet, including in the notebook's linear driver cell**, since those two modules were only ever exercised from their own standalone notebook cells.

- [ ] **Step 1: Write `tests/test_pipeline.py`** — full synthetic integration test exercising every module built in Tasks 2–14 together

```python
from oasis2 import pipeline


def test_run_pipeline_end_to_end_on_synthetic_data(tmp_path, monkeypatch, synthetic_demo_df, synthetic_regional_volumes, synthetic_connectome):
    monkeypatch.setenv("OASIS2_PROJECT_ROOT", str(tmp_path))
    import numpy as np
    (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
    np.save(str(tmp_path / "data" / "processed" / "template_connectome.npy"), synthetic_connectome)

    result = pipeline.run_pipeline(
        demo_df=synthetic_demo_df,
        regional_volumes=synthetic_regional_volumes,
        outputs_dir=tmp_path / "outputs",
        dl_epochs=1,  # keep the integration test fast
    )

    assert "phase3" in result and "phase4" in result
    assert "benchmark" in result
    assert "digital_twin" in result
    rollout = result["digital_twin"]["rollout"]
    assert rollout is not None
    bands = result["digital_twin"]["confidence_bands"]
    assert bands is not None
    mean_pred, lower, upper = bands
    assert (upper >= lower).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oasis2.pipeline'`

- [ ] **Step 3: Write `src/oasis2/colab_setup.py`**

```python
"""Colab-only environment bootstrap: GPU check, Drive mount, dependency install.

Import this module only when running in Colab (guarded by config.is_colab()).
Importing it outside Colab raises ImportError from the `google.colab` import,
which is intentional — it keeps the rest of the oasis2 package importable and
testable in any environment.
"""
import subprocess
import sys
from pathlib import Path

from oasis2.config import is_colab
from oasis2.logging_utils import setup_logging

logger = setup_logging("colab_setup")


def check_gpu() -> None:
    try:
        gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if gpu.returncode != 0:
            logger.warning('No GPU detected! Runtime > Change runtime type > T4 GPU. '
                            'Phase 1 (SynthSeg) will be ~12x slower on CPU.')
        else:
            lines = [l for l in gpu.stdout.split('\n') if any(k in l for k in ('GeForce', 'Tesla', 'T4', 'A100'))]
            logger.info(f"GPU: {lines[0].strip() if lines else 'detected'}")
    except FileNotFoundError:
        logger.warning('nvidia-smi not found. No GPU detected!')


def mount_drive() -> Path:
    if not is_colab():
        raise RuntimeError("mount_drive() only works inside Google Colab")
    from google.colab import drive
    drive.mount('/content/drive')
    drive_base = Path('/content/drive/MyDrive/oasis2_pipeline')
    drive_base.mkdir(parents=True, exist_ok=True)
    local_base = Path('/content/oasis2')
    for sub in ['data/raw', 'data/processed', 'outputs', 'models']:
        (local_base / sub).mkdir(parents=True, exist_ok=True)
    logger.info(f"Drive mounted at: {drive_base}")
    logger.info(f"Local workspace: {local_base}")
    return drive_base


def install_dependencies() -> None:
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q',
         'nibabel', 'nilearn', 'SimpleITK', 'networkx', 'statsmodels', 'kaggle', 'tqdm', 'openpyxl', 'captum'],
        check=True,
    )
    logger.info("Dependencies installed.")


def colab_drive_path(subpath: str = "") -> Path:
    if not is_colab():
        raise RuntimeError("colab_drive_path() only works inside Google Colab")
    base = Path("/content/drive/MyDrive/oasis2_pipeline")
    base.mkdir(parents=True, exist_ok=True)
    return base / subpath if subpath else base


def bootstrap() -> None:
    """Full Colab bootstrap: GPU check, Drive mount, dependency install."""
    check_gpu()
    mount_drive()
    install_dependencies()
```

- [ ] **Step 4: Write `src/oasis2/pipeline.py`**

```python
"""End-to-end pipeline orchestration: phases 0/1 (data), 3 (NDM), 4 (deficits),
DL model training + benchmark, and the digital twin layer (trajectory rollout,
confidence bands, interpretability).

This replaces the linear script that previously lived at the tail of
OASIS2_Colab.py (script-level globals like demo_df, outputs_dir, beta_df) with
an explicit run_pipeline() function. It is also the first place the digital
twin (oasis2.digital_twin) and interpretability (oasis2.interpretability)
modules are actually wired into the rest of the pipeline — previously they
only existed as standalone notebook cells that were never called from the
main driver.
"""
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from oasis2.config import get_device, get_outputs_dir
from oasis2.connectome import load_template_connectome
from oasis2.data.datasets import build_dl_datasets
from oasis2.deficits import run_phase4
from oasis2.digital_twin import generate_confidence_bands, simulate_rollout
from oasis2.interpretability import compute_node_attribution
from oasis2.logging_utils import setup_logging
from oasis2.models.benchmark import run_benchmark
from oasis2.models.nd_vae import train_nd_vae
from oasis2.models.st_gnn_ode import train_st_gnn_ode
from oasis2.models.tadm import train_tadm
from oasis2.ndm import run_phase3

logger = setup_logging("pipeline")


def run_pipeline(
    demo_df: pd.DataFrame,
    regional_volumes: pd.DataFrame,
    outputs_dir: Optional[Path] = None,
    dl_epochs: int = 50,
) -> Dict:
    if outputs_dir is None:
        outputs_dir = get_outputs_dir()
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    p3_results = run_phase3(regional_volumes=regional_volumes, demo_df=demo_df, out_dir=outputs_dir / "phase3")
    w_scores = p3_results["w_scores"]

    p3_results_with_names = dict(p3_results)
    p3_results_with_names["region_names"] = p3_results.get("region_names", [])
    p4_results = run_phase4(ndm_results=p3_results_with_names, demo_df=demo_df, out_dir=outputs_dir / "phase4")

    adj_matrix, region_names = load_template_connectome(None)
    n_regions = adj_matrix.shape[0]
    train_loader, val_loader, test_loader = build_dl_datasets(
        demo_df=demo_df, w_scores=w_scores, adj_matrix=adj_matrix, region_names=region_names, seed=42,
    )

    st_gnn_ode_model = nd_vae_model = tadm_model = None
    benchmark_df = pd.DataFrame()
    digital_twin_result = {"rollout": None, "confidence_bands": None, "attribution": None}

    if train_loader is not None:
        st_gnn_ode_model = train_st_gnn_ode(train_loader, val_loader, n_epochs=dl_epochs, lr=1e-3, patience=10)
        nd_vae_model = train_nd_vae(train_loader, val_loader, n_regions, n_epochs=dl_epochs, lr=1e-3, patience=10)
        tadm_model = train_tadm(train_loader, val_loader, n_regions, n_epochs=dl_epochs, lr=1e-3, patience=15)

        benchmark_df = run_benchmark(
            ndm_results=p3_results, st_gnn_ode_model=st_gnn_ode_model, nd_vae_model=nd_vae_model,
            tadm_model=tadm_model, test_loader=test_loader, n_regions=n_regions,
            out_dir=outputs_dir / "benchmark",
        )

        batch = next(iter(test_loader))
        if nd_vae_model is not None:
            rollout = simulate_rollout(
                nd_vae_model, batch["node_features"][0:1, 0:1], batch["covariates"][0:1], batch["adj"], steps=3,
            )
            bands = generate_confidence_bands(
                nd_vae_model, batch["node_features"], batch["times"], batch["covariates"], batch["adj"], batch["mask"],
            )
            digital_twin_result["rollout"] = rollout
            digital_twin_result["confidence_bands"] = bands
        if st_gnn_ode_model is not None:
            attr_sum, report = compute_node_attribution(
                st_gnn_ode_model, batch["node_features"], batch["times"], batch["covariates"],
                batch["adj"], batch["mask"], target_node_idx=0,
            )
            digital_twin_result["attribution"] = {"attr_sum": attr_sum, "report": report}

    return {
        "phase3": p3_results,
        "phase4": p4_results,
        "benchmark": benchmark_df,
        "digital_twin": digital_twin_result,
        "models": {"st_gnn_ode": st_gnn_ode_model, "nd_vae": nd_vae_model, "tadm": tadm_model},
    }
```

- [ ] **Step 5: Write `src/oasis2/cli.py`**

```python
"""Command-line entry point: `python -m oasis2 --demo-csv ... --regional-volumes-csv ...`."""
import argparse
from pathlib import Path

import pandas as pd

from oasis2.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OASIS-2 neurodegeneration pipeline.")
    parser.add_argument("--demo-csv", type=Path, required=True)
    parser.add_argument("--regional-volumes-csv", type=Path, required=True)
    parser.add_argument("--outputs-dir", type=Path, default=None)
    parser.add_argument("--dl-epochs", type=int, default=50)
    args = parser.parse_args()

    demo_df = pd.read_csv(args.demo_csv)
    regional_volumes = pd.read_csv(args.regional_volumes_csv)
    run_pipeline(demo_df, regional_volumes, outputs_dir=args.outputs_dir, dl_epochs=args.dl_epochs)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS (1 test, may take 10-30s due to DL training on synthetic data)

- [ ] **Step 7: Commit**

```bash
git add src/oasis2/colab_setup.py src/oasis2/pipeline.py src/oasis2/cli.py tests/test_pipeline.py
git commit -m "feat: add pipeline orchestration wiring the digital twin into the main driver

run_pipeline() is the first place simulate_rollout/generate_confidence_bands
and compute_node_attribution are actually called from the main driver logic
rather than only from their own standalone notebook cells."
```

---

### Task 16: Full test suite, README update, final review

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the entire test suite**

Run: `pytest tests/ -v`
Expected: PASS, all tests from Tasks 1-15 (approximately 45 tests)

- [ ] **Step 2: Update `README.md`** — add a "Running the .py pipeline" section after the existing "## Notebooks" section (source: `README.md:53-75`)

Add this content immediately before the `## Repository Structure` heading:

```markdown
## Running the Python Package

The pipeline is also available as a tested Python package under `src/oasis2/`, converted from the Colab notebook. It's the canonical, importable implementation; the notebooks remain as Colab/local-runnable artifacts.

```bash
pip install -e .
pip install -r requirements.txt
pytest tests/  # synthetic-data tests, no MRI scans required
python -m oasis2 --demo-csv data/raw/oasis_longitudinal_demographics-*.xlsx --regional-volumes-csv outputs/regional_volumes.csv
```

See [docs/superpowers/specs/2026-09-28-py-pipeline-conversion-design.md](docs/superpowers/specs/2026-09-28-py-pipeline-conversion-design.md) for the module layout.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document the src/oasis2 package as the canonical pipeline implementation"
```

---

## Post-implementation note for whoever executes this plan

Tasks 2-10 and 13-14 lean on the "port + adapt" convention (see top of this document) rather than reproducing every already-correct line of `OASIS2_Colab.py` here. While porting each one, actually diff it against the notebook cell it originally came from (see the design spec's function inventory) — the audit in the spec only confirmed presence/absence of the 73 top-level names, not line-by-line correctness of the 68 that already exist. If a port task's tests reveal a bug beyond the two already documented (the missing `scipy.linalg` import in Task 3, and the `.train()`/`.eval()` leak in Task 12), fix it in that task, note it in the commit message the same way, and add a regression test the way Task 3 and Task 12 do.
