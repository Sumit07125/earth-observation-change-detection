#!/usr/bin/env python3
"""
OSCD -> 128x128, 17-channel patches matching the Maharashtra tensor layout.
stride 64 on train, 128 on val/test. 3:1 oversampling metadata for train only.
Per FINAL_ARCH_3.md v3.2 (Part 28.4).
"""
import sys
import json
from pathlib import Path
import numpy as np

try:
    import rasterio
    def read_band(p: Path) -> np.ndarray:
        with rasterio.open(p) as s:
            return s.read(1).astype(np.float32)
except ImportError:
    try:
        import tifffile
        def read_band(p: Path) -> np.ndarray:
            a = tifffile.imread(str(p))
            if a.ndim == 3: a = a[0]
            return a.astype(np.float32)
    except ImportError:
        from PIL import Image
        def read_band(p: Path) -> np.ndarray:
            return np.array(Image.open(str(p)), dtype=np.float32)

try:
    from data.verify_oscd import resolve_img_dir, find_band, resolve_subdirs
except ImportError:
    from verify_oscd import resolve_img_dir, find_band, resolve_subdirs

# drop B01 (60 m coastal) and B10 (60 m cirrus -- and B10 does not exist in the
# L2A product used for Maharashtra). The remaining 11 match OUR_S2_BANDS exactly.
KEEP = ['B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']
PATCH, EPS = 128, 1e-6


def load_city(city_root: Path, t: int):
    d, rect = resolve_img_dir(city_root, t)
    if d is None:
        raise FileNotFoundError(f"Missing images for {city_root.name} t{t}")
    out = []
    for b in KEEP:
        bp = find_band(d, b)
        if bp is None:
            raise FileNotFoundError(f"Missing band {b} in {d}")
        a = read_band(bp)
        out.append(a / 10000.0)         # L1C TOA DN -> reflectance
    return np.stack(out), rect


def to_17ch(raw11):
    B4, B8, B11 = raw11[2], raw11[6], raw11[9]
    ndvi = ((B8 - B4) / (B8 + B4 + EPS))[None]
    ndbi = ((B11 - B8) / (B11 + B8 + EPS))[None]
    sar  = np.zeros((3, *raw11.shape[1:]), np.float32)   # bypassed by g=1.0
    q    = np.ones((1, *raw11.shape[1:]), np.float32)    # OSCD is cloud-free
    return np.concatenate([raw11, ndvi, ndbi, sar, q]).astype(np.float32)


def load_label(lab_root: Path, city: str):
    cand = list((lab_root / city / 'cm').glob('*.tif')) + \
           list((lab_root / city / 'cm').glob('*.png'))
    if not cand:
        raise FileNotFoundError(f"No label found for {city} in {lab_root}")
    a = read_band(cand[0])
    vals = set(np.unique(a).tolist())
    if vals <= {1.0, 2.0}:
        a = a - 1.0          # original encoding: 1=no, 2=change
    elif vals <= {0.0, 255.0}:
        a = (a > 0.0).astype(np.float32)
    return a.astype(np.uint8)


def prepare(base: Path, splits: dict, out: Path):
    img_root, tr_root, te_root = resolve_subdirs(base)
    out.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PREPARING OSCD 17-CHANNEL 128x128 PATCHES (v3.2 Protocol)")
    print("=" * 70)

    for split, cities in (('train', splits['train']), ('val', splits['val']),
                          ('test', splits['test'])):
        lab_root = tr_root if split != 'test' else te_root
        stride = 64 if split == 'train' else PATCH
        X, Y, M = [], [], []
        print(f"\nProcessing split: {split.upper()} ({len(cities)} cities, stride={stride})...")
        
        for city in cities:
            r1, _ = load_city(img_root / city, 1)
            r2, _ = load_city(img_root / city, 2)
            x1, x2 = to_17ch(r1), to_17ch(r2)
            y = load_label(lab_root, city)
            H, W = y.shape
            city_patches = 0
            
            for r in range(0, max(H - PATCH, 0) + 1, stride):
                for c in range(0, max(W - PATCH, 0) + 1, stride):
                    if r + PATCH > H or c + PATCH > W:
                        continue
                    X.append(np.stack([x1[:, r:r+PATCH, c:c+PATCH],
                                       x2[:, r:r+PATCH, c:c+PATCH]]))
                    yy = y[r:r+PATCH, c:c+PATCH]
                    Y.append(yy)
                    M.append({'city': city, 'row': r, 'col': c,
                              'chg_frac': float(yy.mean())})
                    city_patches += 1
            print(f'  {city:12s}: {H}x{W}, {float(y.mean())*100:5.2f}% change, {city_patches} patches')

        X = np.clip(np.asarray(X) * 10000, -32768, 32767).astype(np.int16)
        np.save(out / f'oscd_{split}.npy', X)
        np.save(out / f'oscd_{split}_label.npy', np.asarray(Y, np.uint8))
        json.dump(M, open(out / f'oscd_{split}_meta.json', 'w'), indent=2)

        if split == 'train':
            # 3:1 oversampling of change-containing patches. TRAIN ONLY --
            # oversampling val or test would distort the metric.
            pos = [i for i, m in enumerate(M) if m['chg_frac'] > 0.01]
            neg = [i for i, m in enumerate(M) if m['chg_frac'] <= 0.01]
            json.dump({'sample_index': pos * 3 + neg,
                       'n_pos': len(pos), 'n_neg': len(neg)},
                      open(out / 'oscd_train_sampler.json', 'w'), indent=2)
            print(f'  -> Oversampler: {len(pos)} pos x3 + {len(neg)} neg = {len(pos)*3 + len(neg)} total items')
        print(f'  Saved {out / f"oscd_{split}.npy"}: shape {X.shape}')


if __name__ == '__main__':
    base_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/OSCD')
    manifest_path = Path('data/OSCD/splits/oscd_splits.json')
    out_dir = Path('data/OSCD/prepared')
    
    if not manifest_path.exists():
        from data.build_oscd_manifest import build
        build()
        
    prepare(base_dir, json.load(open(manifest_path)), out_dir)
