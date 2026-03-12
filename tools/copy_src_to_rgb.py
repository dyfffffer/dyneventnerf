#!/usr/bin/env python3
"""Copy files from src layout `r_xx_tttt.png` into `rgb/tttt/xx.png`.

Example:
  src/r_03_0123.png -> rgb/0123/03.png
"""

import argparse
import re
import shutil
from pathlib import Path

PATTERN = re.compile(r"^r_(\d+)_(\d+)\.(png|jpg|jpeg|JPG|JPEG|PNG)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reorganize images from src/r_xx_tttt.png into rgb/tttt/xx.png"
    )
    parser.add_argument("--src", type=Path, default=Path("src"), help="Source directory")
    parser.add_argument("--dst", type=Path, default=Path("rgb"), help="Destination rgb directory")
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print planned operations",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.src.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {args.src}")

    files = sorted([p for p in args.src.iterdir() if p.is_file()])
    matched = 0
    skipped = 0

    for src_file in files:
        m = PATTERN.match(src_file.name)
        if m is None:
            skipped += 1
            continue

        view_idx, time_idx, ext = m.group(1), m.group(2), m.group(3).lower()
        dst_dir = args.dst / time_idx
        dst_file = dst_dir / f"{view_idx}.{ext}"

        if dst_file.exists() and not args.overwrite:
            print(f"[skip-exists] {dst_file}")
            continue

        matched += 1
        op = "move" if args.move else "copy"
        print(f"[{op}] {src_file} -> {dst_file}")

        if args.dry_run:
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        if args.move:
            shutil.move(str(src_file), str(dst_file))
        else:
            shutil.copy2(src_file, dst_file)

    print(
        f"Done. matched={matched}, skipped_non_matching={skipped}, src_total={len(files)}"
    )


if __name__ == "__main__":
    main()
