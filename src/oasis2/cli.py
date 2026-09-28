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
