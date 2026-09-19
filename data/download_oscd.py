"""
Download and extract the OSCD (Onera Satellite Change Detection) Dataset.

Data Source:
  The original ONERA HTTP direct links (onera.fr/sites/default/files/...) return HTTP 404
  due to a server-side infrastructure migration.
  This script downloads from the verified, byte-identical Hugging Face mirror:
  https://huggingface.co/datasets/hkristen/oscd

Per FINAL_ARCH_3.md (Part 13 Q4 & Part 19.1):
  - 24 total Sentinel-2 satellite change detection pairs at 10m resolution (imgs_*_rect).
  - Both Train Labels and Test Labels are ingested.
"""

import os
import sys
import shutil
import zipfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "oscd"

# Verified Hugging Face mirror endpoints (Total: ~513 MB)
OSCD_FILES = {
    "images": {
        "url": "https://huggingface.co/datasets/hkristen/oscd/resolve/main/Onera%20Satellite%20Change%20Detection%20dataset%20-%20Images.zip",
        "zip_name": "oscd_images.zip",
        "prefix": "Onera Satellite Change Detection dataset - Images"
    },
    "train_labels": {
        "url": "https://huggingface.co/datasets/hkristen/oscd/resolve/main/Onera%20Satellite%20Change%20Detection%20dataset%20-%20Train%20Labels.zip",
        "zip_name": "oscd_train_labels.zip",
        "prefix": "Onera Satellite Change Detection dataset - Train Labels"
    },
    "test_labels": {
        "url": "https://huggingface.co/datasets/hkristen/oscd/resolve/main/Onera%20Satellite%20Change%20Detection%20dataset%20-%20Test%20Labels.zip",
        "zip_name": "oscd_test_labels.zip",
        "prefix": "Onera Satellite Change Detection dataset - Test Labels"
    }
}

def download_file(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 10 * 1024:
        print(f"  [Cache] {dest.name} already exists ({dest.stat().st_size / 1e6:.2f} MB). Skipping download.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} -> {dest.name}...")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            sys.stdout.write(f"\r    {downloaded / 1e6:.1f} MB / {total_size / 1e6:.1f} MB ({percent:.1f}%)")
        else:
            sys.stdout.write(f"\r    {downloaded / 1e6:.1f} MB")
        sys.stdout.flush()

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, str(dest), reporthook=report_progress)
    print(f"\n    Finished {dest.name} ({dest.stat().st_size / 1e6:.2f} MB).")

def extract_zip_flatten(zip_path: Path, extract_to: Path, prefix: str):
    """
    Extract zip entries while stripping the top-level container directory
    so that city folders land directly into extract_to/<city>/
    """
    print(f"  Extracting {zip_path.name} to {extract_to}...")
    prefix_clean = prefix.rstrip('/') + '/'
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.infolist():
            # Skip the root directory itself
            if member.filename.rstrip('/') == prefix.rstrip('/'):
                continue
            
            # Strip the prefix if present
            if member.filename.startswith(prefix_clean):
                rel_path = member.filename[len(prefix_clean):]
            else:
                rel_path = member.filename
            
            if not rel_path:
                continue

            target_path = extract_to / rel_path
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
    print(f"    Extracted {zip_path.name}.")

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_zips_dir = DATA_DIR / "_zips"
    raw_zips_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 1: Downloading OSCD archives from verified Hugging Face mirror")
    print("=" * 70)
    
    for key, info in OSCD_FILES.items():
        zip_path = raw_zips_dir / info["zip_name"]
        download_file(info["url"], zip_path)

    print("\n" + "=" * 70)
    print("STEP 2: Extracting OSCD archives into unified directory structure")
    print("=" * 70)
    
    for key, info in OSCD_FILES.items():
        zip_path = raw_zips_dir / info["zip_name"]
        extract_zip_flatten(zip_path, DATA_DIR, info["prefix"])

    # Clean up temporary _zips folder
    print("  Cleaning up raw archive files...")
    shutil.rmtree(raw_zips_dir, ignore_errors=True)

    # Verification of downloaded cities
    cities = [d.name for d in DATA_DIR.iterdir() if d.is_dir() and (d / "imgs_1_rect").exists()]
    print("\n" + "=" * 70)
    print(f"OSCD DATASET ACQUISITION COMPLETE! Total verified cities: {len(cities)}")
    print(f"Location: {DATA_DIR.resolve()}")
    print("Cities found:", sorted(cities))
    print("=" * 70)

if __name__ == "__main__":
    main()
