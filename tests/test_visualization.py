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
