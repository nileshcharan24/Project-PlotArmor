"""
Bundling script for Kaggle deployment.
Creates a ZIP of essential code for upload.
"""

import os
import shutil
from pathlib import Path


def main():
    # Create dist directory
    dist_dir = Path("dist")
    dist_dir.mkdir(exist_ok=True)

    # Files and folders to include
    items_to_zip = [
        "research",
        "requirements.txt",
        "project_context.md"
    ]

    # Create a temporary directory for the bundle
    bundle_dir = dist_dir / "plot_armor_bundle"
    bundle_dir.mkdir(exist_ok=True)

    # Copy items to bundle directory
    for item in items_to_zip:
        src = Path(item)
        dst = bundle_dir / src.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    # Create ZIP
    zip_path = dist_dir / "plot_armor_code"
    shutil.make_archive(zip_path, 'zip', bundle_dir)

    # Get ZIP size
    zip_file = zip_path.with_suffix('.zip')
    size_bytes = zip_file.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"Bundle created: {zip_file}")
    print(f"Size: {size_mb:.2f} MB")

    # Clean up temporary directory
    shutil.rmtree(bundle_dir)


if __name__ == "__main__":
    main()