#!/usr/bin/env python3
"""
Structural integrity check across all 24 OSCD cities.
Run this BEFORE building the manifest or staging to Kaggle.
Per FINAL_ARCH_3.md v3.2 (Part 28.2).
"""
import sys, json
from pathlib import Path
import numpy as np

# Robust import: prefer rasterio, fall back to tifffile / PIL
try:
    # pyrefly: ignore [missing-import]
    import rasterio
    def read_shape(p: Path):
        with rasterio.open(p) as s:
            return (s.height, s.width)
    def read_label_info(p: Path):
        with rasterio.open(p) as s:
            a = s.read(1)
        return a.shape, sorted(np.unique(a).tolist())
except ImportError:
    try:
        import tifffile
        from PIL import Image
        def read_shape(p: Path):
            if p.suffix.lower() in ('.tif', '.tiff'):
                a = tifffile.imread(str(p))
                return a.shape[:2] if a.ndim == 2 else a.shape[1:3]
            img = Image.open(str(p))
            return (img.height, img.width)
        def read_label_info(p: Path):
            if p.suffix.lower() in ('.tif', '.tiff'):
                a = tifffile.imread(str(p))
                if a.ndim == 3: a = a[0]
            else:
                a = np.array(Image.open(str(p)))
            return a.shape, sorted(np.unique(a).tolist())
    except ImportError:
        from PIL import Image
        def read_shape(p: Path):
            img = Image.open(str(p))
            return (img.height, img.width)
        def read_label_info(p: Path):
            a = np.array(Image.open(str(p)))
            return a.shape, sorted(np.unique(a).tolist())

BANDS_13 = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']

TRAIN_14 = ['abudhabi','aguasclaras','beihai','beirut','bercy','bordeaux',
            'cupertino','hongkong','mumbai','nantes','paris','pisa','rennes','saclay_e']
TEST_10  = ['brasilia','chongqing','dubai','lasvegas','milano','montpellier',
            'norcia','rio','saclay_w','valencia']


def find_band(d: Path, b: str):
    """Mirrors differ: B01.tif / B1.tif / b01.tif. Glob, do not hard-code."""
    for pat in (f'{b}.tif', f'{b.replace("B0","B")}.tif',
                f'{b.lower()}.tif', f'*{b}.tif'):
        hit = list(d.glob(pat))
        if hit:
            return hit[0]
    return None


def resolve_img_dir(city_root: Path, t: int):
    """Prefer imgs_t_rect (10 m, co-registered). Verify it is COMPLETE."""
    rect = city_root / f'imgs_{t}_rect'
    if rect.exists() and sum(find_band(rect, b) is not None for b in BANDS_13) >= 13:
        return rect, True
    plain = city_root / f'imgs_{t}'
    if plain.exists():
        return plain, False          # caller must resample 20 m / 60 m to 10 m
    return None, False


def verify(images_root: Path, train_lab: Path, test_lab: Path):
    report, ok = {}, True
    for city in TRAIN_14 + TEST_10:
        root = images_root / city
        r = {'city': city, 'split': 'train' if city in TRAIN_14 else 'test'}
        if not root.exists():
            r['error'] = 'MISSING CITY DIRECTORY'; report[city] = r; ok = False; continue

        for t in (1, 2):
            d, rect = resolve_img_dir(root, t)
            if d is None:
                r[f'imgs_{t}'] = 'MISSING'; ok = False; continue
            found = [b for b in BANDS_13 if find_band(d, b) is not None]
            r[f'imgs_{t}'] = {'dir': d.name, 'rect': rect, 'n_bands': len(found)}
            if len(found) < 13:
                r[f'imgs_{t}']['missing'] = [b for b in BANDS_13 if b not in found]
                ok = False
            bp = find_band(d, 'B04')
            if bp:
                r[f'imgs_{t}']['shape'] = read_shape(bp)

        lab_root = (train_lab if city in TRAIN_14 else test_lab) / city / 'cm'
        cand = list(lab_root.glob('*.tif')) + list(lab_root.glob('*.png'))
        if not cand:
            r['label'] = 'MISSING'; ok = False
        else:
            shape, vals = read_label_info(cand[0])
            r['label'] = {'file': cand[0].name, 'shape': shape, 'values': vals}
            # TRAP: original TIFFs use 1 = no-change, 2 = change. Not 0/1.
            if vals not in ([1,2],[0,1],[0,255],[1],[2]):
                r['label']['WARNING'] = f'unexpected label values {vals}'
            if 'shape' in r.get('imgs_1', {}) and shape != r['imgs_1']['shape']:
                r['label']['WARNING'] = (f'label {shape} != image '
                                         f'{r["imgs_1"]["shape"]} -- use _rect')
                ok = False
        report[city] = r

    print(json.dumps(report, indent=2, default=str))
    print(f'\n{"ALL CHECKS PASSED" if ok else "FAILURES PRESENT -- fix before staging"}')
    return ok, report


def resolve_subdirs(base: Path):
    """Find images, train labels, and test labels directories under base."""
    img = base / 'Images' if (base / 'Images').exists() else base / 'Onera Satellite Change Detection dataset - Images'
    tr = base / 'Train Labels' if (base / 'Train Labels').exists() else base / 'Onera Satellite Change Detection dataset - Train Labels'
    te = base / 'Test Labels' if (base / 'Test Labels').exists() else base / 'Onera Satellite Change Detection dataset - Test Labels'
    
    if not img.exists() and (base / 'rennes').exists():
        img, tr, te = base, base, base

    return img, tr, te


if __name__ == '__main__':
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "OSCD"
    img, tr, te = resolve_subdirs(base)
    print(f"Verifying OSCD at: {base}")
    print(f"  Images root:       {img}")
    print(f"  Train Labels root: {tr}")
    print(f"  Test Labels root:  {te}")
    ok, rep = verify(img, tr, te)
    out_rep = base / 'verify_report.json'
    json.dump(rep, open(out_rep, 'w'), indent=2, default=str)
    print(f"\nReport saved to: {out_rep}")
    sys.exit(0 if ok else 1)
