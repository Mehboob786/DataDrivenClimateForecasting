from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a real climate dataset from Kaggle or NASA."
    )
    parser.add_argument(
        "--source",
        choices=["kaggle", "nasa"],
        default="kaggle",
        help="Dataset source.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save downloaded dataset.",
    )
    parser.add_argument(
        "--kaggle-dataset",
        type=str,
        default="berkeleyearth/climate-change-earth-surface-temperature-data",
        help="Kaggle dataset slug.",
    )
    parser.add_argument(
        "--kaggle-file",
        type=str,
        default="GlobalTemperatures.csv",
        help="Filename inside Kaggle dataset zip to extract.",
    )
    return parser.parse_args()


def fetch_from_kaggle(output_dir: Path, dataset: str, filename: str) -> Path:
    kaggle_cli = shutil.which("kaggle")
    if not kaggle_cli:
        raise RuntimeError(
            "Kaggle CLI not found. Install with `pip install kaggle` and configure API token."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        kaggle_cli,
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(output_dir),
        "--unzip",
        "-f",
        filename,
    ]
    subprocess.run(cmd, check=True)
    target = output_dir / filename
    if not target.exists():
        raise FileNotFoundError(f"Expected extracted file not found: {target}")
    return target


def fetch_from_nasa(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "GLB.Ts+dSST.csv"
    # NASA GISS global mean monthly/annual temperature anomalies.
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    urllib.request.urlretrieve(url, target)
    return target


def main() -> None:
    args = parse_args()
    if args.source == "kaggle":
        path = fetch_from_kaggle(args.output_dir, args.kaggle_dataset, args.kaggle_file)
    else:
        path = fetch_from_nasa(args.output_dir)
    print(f"Downloaded dataset: {path}")


if __name__ == "__main__":
    main()
