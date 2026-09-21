# ============================================================
# CELL 1 — ENVIRONMENT + DRIVE + ARCHIVE DISCOVERY
# ============================================================

!pip -q install rasterio kagglehub

from google.colab import drive
drive.mount("/content/drive")

import json
import tarfile
import shutil
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import rasterio

DRIVE_PROC = Path("/content/drive/MyDrive/geonexus_v3_processed")
ARCHIVE = DRIVE_PROC / "geonexus_v3_processed.tar"

VERIFY_ROOT = DRIVE_PROC / "verification"
CORRECTED_ROOT = VERIFY_ROOT / "corrected_masks"
VERIFIED_ROOT = VERIFY_ROOT / "verified"

LOCAL_ROOT = Path("/content/geonexus_mh_public_170")
LOCAL_QGIS = LOCAL_ROOT / "qgis"

for path in [LOCAL_ROOT, VERIFY_ROOT, CORRECTED_ROOT, VERIFIED_ROOT]:
    path.mkdir(parents=True, exist_ok=True)

SPLITS = [
    "MH_VAL",
    "MH_ADAPT",
    "MH_TEST_DRY_A",
    "MH_TEST_DRY_B",
    "MH_TEST_BLIND",
    "MH_TEST_MONSOON",
    "MH_TEST_VIDARBHA",
]

EXPECTED_COUNTS = {
    "MH_VAL": 30,
    "MH_ADAPT": 30,
    "MH_TEST_DRY_A": 40,
    "MH_TEST_DRY_B": 40,
    "MH_TEST_BLIND": 20,
    "MH_TEST_MONSOON": 30,
    "MH_TEST_VIDARBHA": 30,
}

RELEASE_SPLITS = {
    "MH_VAL": 30,
    "MH_ADAPT": 30,
    "MH_TEST_DRY_A": 40,
    "MH_TEST_DRY_B": 40,
    "MH_TEST_VIDARBHA": 30,
}

PATCH_SIZE = 128
ALLOWED_LABELS = set(range(7)) | {255}
KAGGLE_HANDLE = "sumit07125/geonexus-mh-v3"

print("=" * 80)
print("Geo-Nexus v3.2 — PUBLIC 170-REVIEWED RELEASE")
print("=" * 80)
print("Drive processed directory:", DRIVE_PROC)
print("Archive:", ARCHIVE)
print("Kaggle handle:", KAGGLE_HANDLE)

if not DRIVE_PROC.exists():
    raise FileNotFoundError(
        f"Processed Drive directory not found:\n{DRIVE_PROC}"
    )

LOOSE_QGIS = DRIVE_PROC / "qgis"

def safe_extract_qgis(archive_path: Path, output_dir: Path):
    selected = []
    with tarfile.open(archive_path, "r") as tar:
        for member in tar.getmembers():
            name = member.name.replace("\\", "/")
            if "/qgis/" in f"/{name}":
                selected.append(member)

        if not selected:
            raise RuntimeError(
                "No qgis/ entries were found in the processed archive."
            )

        for member in selected:
            name = member.name.replace("\\", "/")
            idx = name.find("qgis/")
            if idx < 0:
                continue

            relative = Path(name[idx + len("qgis/"):])

            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
            ):
                raise RuntimeError(
                    f"Unsafe archive path: {name}"
                )

            destination = (
                output_dir / relative
            ).resolve()

            destination.relative_to(
                output_dir.resolve()
            )

            if member.isdir():
                destination.mkdir(
                    parents=True,
                    exist_ok=True
                )
                continue

            if not member.isfile():
                continue

            destination.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            source = tar.extractfile(member)

            if source is None:
                raise RuntimeError(
                    f"Could not extract archive member: {name}"
                )

            with open(destination, "wb") as target:
                shutil.copyfileobj(source, target)

if LOCAL_QGIS.exists():
    shutil.rmtree(LOCAL_QGIS)

LOCAL_QGIS.mkdir(
    parents=True,
    exist_ok=True
)

if LOOSE_QGIS.exists():
    shutil.copytree(
        LOOSE_QGIS,
        LOCAL_QGIS,
        dirs_exist_ok=True
    )
    print("✓ Using loose qgis/ directory from Drive.")
elif ARCHIVE.exists():
    print("✓ Extracting qgis/ from the existing archive...")
    safe_extract_qgis(
        ARCHIVE,
        LOCAL_QGIS
    )
    print("✓ QGIS package extracted.")
else:
    raise FileNotFoundError(
        "Neither a loose qgis/ directory nor the processed archive exists."
    )

print("\nQGIS split inventory:")
for split, expected in EXPECTED_COUNTS.items():
    folder = LOCAL_QGIS / split
    if not folder.exists():
        raise RuntimeError(
            f"Missing split folder: {split}"
        )

    count = len(
        list(folder.iterdir())
    )

    print(
        f"{split:22s}: {count:4d} files"
    )

print("\n✓ Cell 1 PASS")


# ============================================================
# CELL 2 — INDEX AND HARD-CHECK THE ORIGINAL 220 PATCH POOL
# ============================================================

def patch_key(item):
    return f"{item['split']}/{item['stem']}"

PATCHES = {}

for split in SPLITS:
    folder = LOCAL_QGIS / split
    items = []

    for image_path in sorted(folder.glob("*.tif")):
        filename = image_path.name

        if filename.endswith("_ae.tif"):
            continue

        if filename.endswith("_autolabel.tif"):
            continue

        stem = image_path.stem

        autolabel = (
            folder / f"{stem}_autolabel.tif"
        )

        alphaearth = (
            folder / f"{stem}_ae.tif"
        )

        items.append({
            "split": split,
            "stem": stem,
            "image": image_path,
            "autolabel": autolabel if autolabel.exists() else None,
            "ae": alphaearth if alphaearth.exists() else None,
            "manual_only": not autolabel.exists(),
        })

    PATCHES[split] = items

total = 0
auto_total = 0
manual_total = 0

print("=" * 80)
print("ORIGINAL 220-PATCH INVENTORY")
print("=" * 80)

for split in SPLITS:
    expected = EXPECTED_COUNTS[split]
    actual = len(PATCHES[split])

    if actual != expected:
        raise RuntimeError(
            f"{split}: expected {expected}, found {actual}"
        )

    auto_count = sum(
        item["autolabel"] is not None
        for item in PATCHES[split]
    )

    manual_count = actual - auto_count

    print(
        f"{split:22s} "
        f"total={actual:3d} "
        f"auto={auto_count:3d} "
        f"manual_only={manual_count:3d}"
    )

    total += actual
    auto_total += auto_count
    manual_total += manual_count

if total != 220:
    raise RuntimeError(
        f"Original annotation pool must be exactly 220, found {total}"
    )

if auto_total != 170:
    raise RuntimeError(
        f"Expected exactly 170 auto-label patches, found {auto_total}"
    )

if manual_total != 50:
    raise RuntimeError(
        f"Expected exactly 50 manual-only patches, found {manual_total}"
    )

expected_manual_only = {
    "MH_TEST_BLIND": 20,
    "MH_TEST_MONSOON": 30,
}

for split, expected in expected_manual_only.items():
    actual = sum(
        item["manual_only"]
        for item in PATCHES[split]
    )
    if actual != expected:
        raise RuntimeError(
            f"{split}: expected {expected} manual-only, found {actual}"
        )

print("-" * 80)
print("TOTAL:", total)
print("AUTO-LABEL:", auto_total)
print("MANUAL-ONLY:", manual_total)

print("\n✓ Exact 220 pool PASS")
print("✓ 170 automatic-label patches PASS")
print("✓ 50 blind/monsoon manual-only patches PASS")


# ============================================================
# CELL 3 — BULK ACCEPT ALL 170 HUMAN-REVIEWED AUTO-LABELS
# ============================================================

# The user has already visually inspected all 220 exported pairs.
# Only the 170 patches with an authoritative _autolabel.tif can be accepted
# into this release. The 50 blind/monsoon patches are recorded as excluded,
# never as fabricated labels.

USER_CONFIRMED_VISUAL_REVIEW = True

if not USER_CONFIRMED_VISUAL_REVIEW:
    raise RuntimeError(
        "This cell may only be run after the visual review is complete."
    )

DECISION_FILE = (
    VERIFY_ROOT /
    "verification_decisions.json"
)

if DECISION_FILE.exists():
    try:
        decisions = json.loads(
            DECISION_FILE.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        decisions = {}
else:
    decisions = {}

now_utc = datetime.now(
    timezone.utc
).isoformat()

decision_counts = {
    "ACCEPT_AUTO": 0,
    "PRESERVED_CORRECTED": 0,
    "PRESERVED_MANUAL_VERIFIED": 0,
    "EXCLUDED_NO_HUMAN_LABEL": 0,
}

review_record = {
    "project": "Geo-Nexus v3.2",
    "release": "170-reviewed-weak-labels",
    "visual_review_confirmed": True,
    "confirmed_utc": now_utc,
    "original_annotation_pool": 220,
    "released_verified_patches": 170,
    "excluded_manual_only_patches": 50,
    "patches": {},
}

for split in SPLITS:
    for item in PATCHES[split]:
        key = patch_key(item)
        previous = decisions.get(key, {})
        previous_decision = previous.get(
            "decision",
            "UNREVIEWED"
        )

        review_record["patches"][key] = {
            "visually_checked": True,
            "auto_label_available": bool(
                item["autolabel"] is not None
            ),
            "confirmed_utc": now_utc,
        }

        if previous_decision == "CORRECTED":
            decision_counts["PRESERVED_CORRECTED"] += 1
            continue

        if previous_decision == "MANUAL_VERIFIED":
            decision_counts["PRESERVED_MANUAL_VERIFIED"] += 1
            continue

        if item["autolabel"] is not None:
            decisions[key] = {
                "split": split,
                "stem": item["stem"],
                "decision": "ACCEPT_AUTO",
                "note": (
                    "Human-reviewed automatic label. "
                    "Accepted without pixel-level redrawing."
                ),
                "timestamp_utc": now_utc,
                "bulk_confirmed": True,
                "weak_label": True,
            }
            decision_counts["ACCEPT_AUTO"] += 1

        else:
            decisions[key] = {
                "split": split,
                "stem": item["stem"],
                "decision": "EXCLUDED_NO_HUMAN_LABEL",
                "note": (
                    "Original annotation-pool item visually inspected, "
                    "but no independent human mask was created. "
                    "Excluded from the 170 verified release."
                ),
                "timestamp_utc": now_utc,
                "bulk_confirmed": False,
                "weak_label": False,
            }
            decision_counts["EXCLUDED_NO_HUMAN_LABEL"] += 1

DECISION_FILE.write_text(
    json.dumps(
        decisions,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

REVIEW_FILE = (
    VERIFY_ROOT /
    "all_220_visual_review_confirmation.json"
)

REVIEW_FILE.write_text(
    json.dumps(
        review_record,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

print("=" * 80)
print("BULK REVIEW FINALIZATION")
print("=" * 80)

for key, value in decision_counts.items():
    print(
        f"{key:34s}: {value}"
    )

if decision_counts["ACCEPT_AUTO"] != 170:
    raise RuntimeError(
        "Release requires exactly 170 ACCEPT_AUTO labels."
    )

if decision_counts["EXCLUDED_NO_HUMAN_LABEL"] != 50:
    raise RuntimeError(
        "Release requires exactly 50 excluded blind/monsoon patches."
    )

print("\n✓ 170 auto-label patches accepted")
print("✓ 50 manual-only patches explicitly excluded")
print("✓ No fabricated manual masks")


# ============================================================
# CELL 4 — LABEL VALIDATION + BINARY DERIVATION
# ============================================================

CLASS_NAMES = {
    0: "no_change",
    1: "water_gain",
    2: "water_loss",
    3: "construction",
    4: "veg_loss",
    5: "veg_gain",
    6: "other",
    255: "uncertain",
}

def validate_mask(
    mask,
    expected_shape=(PATCH_SIZE, PATCH_SIZE)
):
    arr = np.asarray(
        mask,
        dtype=np.uint8
    )

    if arr.shape != expected_shape:
        raise ValueError(
            f"Mask shape {arr.shape} != {expected_shape}"
        )

    values = set(
        np.unique(arr).tolist()
    )

    illegal = values - ALLOWED_LABELS

    if illegal:
        raise ValueError(
            f"Illegal class IDs: {sorted(illegal)}"
        )

    return arr

def load_auto_mask(item):
    if item["autolabel"] is None:
        raise RuntimeError(
            f"No auto-label exists for {patch_key(item)}"
        )

    with rasterio.open(
        item["autolabel"]
    ) as src:

        if src.count != 1:
            raise ValueError(
                f"{patch_key(item)}: "
                f"expected one label band, got {src.count}"
            )

        mask = src.read(1)

    return validate_mask(mask)

def make_binary_target(mask):
    """
    Stored binary target:

    0   = no change
    1   = any semantic change class 1..6
    255 = uncertain / ignore

    255 is deliberately preserved.
    """
    mask = validate_mask(mask)

    binary = np.full(
        mask.shape,
        255,
        dtype=np.uint8
    )

    binary[mask == 0] = 0
    binary[
        (mask >= 1) &
        (mask <= 6)
    ] = 1

    return binary

def validate_binary_target(binary):
    arr = np.asarray(
        binary,
        dtype=np.uint8
    )

    if arr.shape != (
        PATCH_SIZE,
        PATCH_SIZE
    ):
        raise ValueError(
            f"Binary shape {arr.shape} "
            f"!= {(PATCH_SIZE, PATCH_SIZE)}"
        )

    illegal = (
        set(np.unique(arr).tolist())
        - {0, 1, 255}
    )

    if illegal:
        raise ValueError(
            f"Illegal binary IDs: {sorted(illegal)}"
        )

    return arr

print("✓ Mask validation functions ready.")


# ============================================================
# CELL 5 — BUILD 170 RELEASE ARRAYS
# ============================================================

RELEASE_ROOT = VERIFIED_ROOT
RELEASE_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

def build_release_split(
    split_order,
    multi_filename,
    binary_filename,
):
    expected = sum(
        RELEASE_SPLITS[s]
        for s in split_order
    )

    masks = []
    binary_masks = []
    manifest = []
    errors = []

    for split in split_order:
        for item in PATCHES[split]:

            key = patch_key(item)

            # This release includes only patches with an authoritative auto-label.
            if item["autolabel"] is None:
                continue

            record = decisions.get(key)

            if record is None:
                errors.append({
                    "patch": key,
                    "reason": "missing decision"
                })
                continue

            if record.get("decision") != "ACCEPT_AUTO":
                errors.append({
                    "patch": key,
                    "reason": (
                        "release expects ACCEPT_AUTO, got "
                        f"{record.get('decision')!r}"
                    )
                })
                continue

            try:
                mask = load_auto_mask(item)
                binary = validate_binary_target(
                    make_binary_target(mask)
                )
            except Exception as exc:
                errors.append({
                    "patch": key,
                    "reason": str(exc)
                })
                continue

            masks.append(mask)
            binary_masks.append(binary)

            manifest.append({
                "split": split,
                "stem": item["stem"],
                "decision": "ACCEPT_AUTO",
                "source": "authoritative_autolabel",
                "weak_label": True,
                "mask_sha256": hashlib.sha256(
                    mask.tobytes()
                ).hexdigest(),
                "binary_mask_sha256": hashlib.sha256(
                    binary.tobytes()
                ).hexdigest(),
            })

    if errors:
        return {
            "ok": False,
            "expected": expected,
            "received": len(masks),
            "errors": errors,
        }

    multi = np.stack(masks).astype(
        np.uint8
    )

    binary = np.stack(binary_masks).astype(
        np.uint8
    )

    expected_shape = (
        expected,
        PATCH_SIZE,
        PATCH_SIZE
    )

    if multi.shape != expected_shape:
        return {
            "ok": False,
            "expected": expected,
            "received": int(multi.shape[0]),
            "errors": [{
                "reason": (
                    f"multi-class shape "
                    f"{multi.shape} != "
                    f"{expected_shape}"
                )
            }],
        }

    if binary.shape != expected_shape:
        return {
            "ok": False,
            "expected": expected,
            "received": int(binary.shape[0]),
            "errors": [{
                "reason": (
                    f"binary shape "
                    f"{binary.shape} != "
                    f"{expected_shape}"
                )
            }],
        }

    multi_path = RELEASE_ROOT / multi_filename
    binary_path = RELEASE_ROOT / binary_filename

    np.save(
        multi_path,
        multi
    )

    np.save(
        binary_path,
        binary
    )

    manifest_stem = Path(multi_filename).stem.replace("_labels", "")
    manifest_path = RELEASE_ROOT / f"{manifest_stem}_manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    return {
        "ok": True,
        "expected": expected,
        "received": expected,
        "multi_path": str(multi_path),
        "binary_path": str(binary_path),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }

results = {}

results["mh_val"] = build_release_split(
    ["MH_VAL"],
    "mh_val_labels.npy",
    "mh_val_binary_labels.npy",
)

results["mh_adapt"] = build_release_split(
    ["MH_ADAPT"],
    "mh_adapt_labels.npy",
    "mh_adapt_binary_labels.npy",
)

results["mh_test_partial"] = build_release_split(
    [
        "MH_TEST_DRY_A",
        "MH_TEST_DRY_B",
        "MH_TEST_VIDARBHA",
    ],
    "mh_test_partial_labels.npy",
    "mh_test_partial_binary_labels.npy",
)

for name, result in results.items():
    print(
        f"{name:18s} "
        f"expected={result['expected']:3d} "
        f"received={result['received']:3d} "
        f"status={'PASS' if result['ok'] else 'BLOCKED'}"
    )

    if not result["ok"]:
        for error in result["errors"][:10]:
            print("  -", error)

if not all(
    result["ok"]
    for result in results.values()
):
    raise RuntimeError(
        "170-patch release gate failed."
    )

total_released = sum(
    result["received"]
    for result in results.values()
)

if total_released != 170:
    raise RuntimeError(
        f"Expected 170 released masks, got {total_released}"
    )

print("\n✓ 170 multi-class masks PASS")
print("✓ 170 binary masks PASS")


# ============================================================
# CELL 6 — QC REPORT + RELEASE MANIFESTS
# ============================================================

def write_json(path, payload):
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

release_scope = {
    "project": "Geo-Nexus v3.2",
    "release_id": "170-reviewed-weak-labels",
    "original_annotation_pool": 220,
    "released_verified_patches": 170,
    "excluded_manual_only_patches": 50,
    "released_counts": RELEASE_SPLITS,
    "excluded_counts": {
        "MH_TEST_BLIND": 20,
        "MH_TEST_MONSOON": 30,
    },
    "label_status": (
        "Human-reviewed automatic labels; "
        "not independently redrawn pixel-level ground truth."
    ),
    "binary_target": {
        "0": "no_change",
        "1": "change_any_of_classes_1_to_6",
        "255": "uncertain_ignore",
        "preserve_255": True,
    },
    "excluded_reason": (
        "Blind and monsoon patches were exported without "
        "automatic masks and no independent human masks were created."
    ),
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
}

write_json(
    RELEASE_ROOT / "release_scope.json",
    release_scope
)

artifact_status = {
    name: {
        "ok": result["ok"],
        "expected": result["expected"],
        "received": result["received"],
        "multi_path": result["multi_path"],
        "binary_path": result["binary_path"],
        "manifest_path": result["manifest_path"],
    }
    for name, result in results.items()
}

def summarize_labels(path):
    array = np.load(
        path,
        mmap_mode="r"
    )

    summary = {}

    for cls in [
        0, 1, 2, 3, 4, 5, 6, 255
    ]:
        count = int(
            np.sum(array == cls)
        )
        summary[str(cls)] = {
            "name": CLASS_NAMES[cls],
            "pixels": count,
            "fraction": float(
                count / array.size
            ),
        }

    return summary

def summarize_binary(path):
    array = np.load(
        path,
        mmap_mode="r"
    )

    return {
        str(cls): {
            "name": name,
            "pixels": int(
                np.sum(array == cls)
            ),
            "fraction": float(
                np.sum(array == cls) /
                array.size
            ),
        }
        for cls, name in [
            (0, "no_change"),
            (1, "change"),
            (255, "uncertain_ignore"),
        ]
    }

for name, result in results.items():
    print(f"\n{name}")

    print(
        "multi-class:",
        summarize_labels(
            result["multi_path"]
        )
    )

    print(
        "binary:",
        summarize_binary(
            result["binary_path"]
        )
    )

qc_report = {
    "project": "Geo-Nexus v3.2",
    "release_id": "170-reviewed-weak-labels",
    "patch_size": PATCH_SIZE,
    "channel_count_per_timestamp": 17,
    "times_per_patch": 2,
    "class_ids": CLASS_NAMES,
    "binary_ids": {
        "0": "no_change",
        "1": "change_any_of_classes_1_to_6",
        "255": "uncertain_ignore",
    },
    "expected_release_counts": RELEASE_SPLITS,
    "artifact_status": artifact_status,
    "visual_review_record": str(
        VERIFY_ROOT /
        "all_220_visual_review_confirmation.json"
    ),
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
}

write_json(
    RELEASE_ROOT /
    "label_qc_report.json",
    qc_report
)

write_json(
    RELEASE_ROOT /
    "verification_decisions_snapshot.json",
    decisions
)

print("\n✓ QC report written.")
print("✓ Release scope written.")
print("✓ Decision snapshot written.")


# ============================================================
# CELL 7 — WRITE PUBLIC DATASET DOCUMENTATION
# ============================================================

DATASET_DOCS = {
    "README.md": """# Geo-Nexus Maharashtra Change Detection v3.2 — 170 Reviewed Release

## 1. Overview

This Kaggle dataset is the Maharashtra component of the **Geo-Nexus v3.2**
remote-sensing change-detection project.

It contains bi-temporal Sentinel-2/Sentinel-1 derived model tensors,
weak multi-class change labels generated by the project's evidence-fusion
pipeline, binary change targets derived from those labels, confidence/metadata
files from the broader preprocessing release, and a **170-patch human-reviewed
verification release**.

### Important release status

This is **not** the original 220-patch human-ground-truth release.

The original annotation pool contains 220 review targets:

| Split | Original count | Included in this release |
|---|---:|---:|
| MH-VAL | 30 | 30 |
| MH-ADAPT | 30 | 30 |
| MH-TEST_DRY_A | 40 | 40 |
| MH-TEST_DRY_B | 40 | 40 |
| MH-TEST_BLIND | 20 | 0 |
| MH-TEST_MONSOON | 30 | 0 |
| MH-TEST_VIDARBHA | 30 | 30 |
| **Total** | **220** | **170** |

The 170 included patches have an authoritative automatic label that was
visually inspected by the project user and accepted without pixel-level
redrawing. They are therefore **human-reviewed weak labels**, not
independently drawn ground truth.

The 50 excluded patches are intentionally omitted from the verified-label
release because they were exported without automatic labels and would require
independent human masks:

- 20 MH-TEST-BLIND
- 30 MH-TEST-MONSOON

Do not interpret the absence of those 50 masks as missing pixels in the source
dataset. They are an intentional release-scope decision.

---

## 2. Intended use

The dataset can be used for:

- weakly supervised Maharashtra change-detection experiments;
- training/evaluating the Geo-Nexus multi-class and binary heads;
- testing data-loading and preprocessing pipelines;
- reproducibility of the Geo-Nexus v3.2 Maharashtra preprocessing stage;
- educational remote-sensing experiments.

The 170 verified labels should be described as **human-reviewed automatic
labels** or **weak labels**. They should not be described as fully independent
manual ground truth.

The 30 MH-ADAPT labels in this release are also weak/human-reviewed labels.
Consequently, using them for adaptation should be reported as adaptation on
human-reviewed weak labels, not as the original protocol's clean few-shot
human-label experiment.

---

## 3. Model-input tensor

The Maharashtra model input is:

```text
X.shape = [N, 2, 17, 128, 128]
```

where:

```text
N   = number of patches
2   = T1 and T2
17  = channels per timestamp
128 = patch height
128 = patch width
```

### 17-channel order

```text
0   B2
1   B3
2   B4
3   B5
4   B6
5   B7
6   B8
7   B8A
8   B9
9   B11
10  B12
11  NDVI
12  NDBI
13  VV
14  VH
15  CR = VH - VV
16  q  = observation-quality channel
```

The first 11 channels are Sentinel-2 optical bands. NDVI and NDBI are derived
optical channels. VV, VH and the cross-ratio channel come from Sentinel-1.
The final channel is the quality value.

The source preprocessing stores model arrays as `int16`. The training loader
uses the project's normalization/statistics files to convert them back to
the expected numeric representation.

---

## 4. Label definitions

### Multi-class target

The authoritative semantic label array is:

```text
0   no_change
1   water_gain
2   water_loss
3   construction
4   veg_loss
5   veg_gain
6   other
255 uncertain / ignore
```

### Binary target

A second target is derived directly from the multi-class label:

```text
0   no_change
1   change = any class 1..6
255 uncertain / ignore
```

**`255` is deliberately preserved in the stored binary target.**

It is not converted to `0`.

During supervised loss computation, `255` is the ignore/uncertain value and
must be excluded through the training loss validity mask.

### Viewer products versus training data

| Product | Training role | Explanation |
|---|---|---|
| Auto-label mask | Yes, for Maharashtra weak supervision | 7-class weak target |
| Binary mask | Yes, derived target | Binary target derived from the final class mask |
| T2 + mask | No | Visualization/quality-control overlay |
| AlphaEarth change | No | Evidence, QA and baseline, not the supervised target |
| T1 / T2 tensors | Yes | Actual model input |

---

## 5. How the automatic labels were produced

The project's automatic-label pipeline fuses several evidence sources:

- Open Buildings Temporal;
- Hansen Global Forest Change;
- Dynamic World;
- NDVI;
- NDBI;
- MNDWI;
- JRC Global Surface Water;
- AlphaEarth change evidence used as the arbitration signal.

The upstream GEE evidence export (`03_export_autolabels.js`) writes a
10-band Int16 evidence raster without applying the final class thresholds.
Final evidence fusion is performed in the project's Python/Colab
preprocessing stage.

The label vocabulary and the `255 = UNCERTAIN` convention are defined by
the project's `autolabel.py`.

---

## 6. Temporal configuration

The dry-season comparison uses:

```text
T1 = 2020-01-01 to 2020-04-01
T2 = 2024-01-01 to 2024-04-01
```

This is a January–March dry-season window represented using Earth Engine's
half-open interval convention.

### Open Buildings limitation

Open Buildings Temporal V1 provides annual snapshots through **2023**.
Therefore the label fusion does not contain a 2024 Open Buildings snapshot.
Construction/change between 2023 and 2024 is partly represented through other
spectral/evidence sources.

This limitation is intentional and should be disclosed in research papers
and downstream analyses.

---

## 7. Geographic zones and CRS

| Zone | Region | CRS |
|---|---|---|
| A | Pune | EPSG:32643 |
| B | Satara | EPSG:32643 |
| C | Vidarbha | EPSG:32644 |

Zone C uses UTM 44N; Zones A/B use UTM 43N.

---

## 8. Verified-label release files

The following files are added under `verified/`:

```text
verified/
├── mh_val_labels.npy
├── mh_val_binary_labels.npy
├── mh_adapt_labels.npy
├── mh_adapt_binary_labels.npy
├── mh_test_partial_labels.npy
├── mh_test_partial_binary_labels.npy
├── mh_val_manifest.json
├── mh_adapt_manifest.json
├── mh_test_partial_manifest.json
├── label_qc_report.json
├── verification_decisions_snapshot.json
└── release_scope.json
```

### Shapes

```text
mh_val_labels.npy                 (30, 128, 128)
mh_val_binary_labels.npy          (30, 128, 128)

mh_adapt_labels.npy               (30, 128, 128)
mh_adapt_binary_labels.npy        (30, 128, 128)

mh_test_partial_labels.npy        (110, 128, 128)
mh_test_partial_binary_labels.npy (110, 128, 128)
```

The 110 test patches are:

```text
40 Pune dry
40 Satara dry
30 Vidarbha
```

There are no blind or monsoon human masks in this release.

---

## 9. File-loading example

```python
import numpy as np

root = "/kaggle/input/geonexus-mh-v3"

mh_val = np.load(
    f"{root}/verified/mh_val_labels.npy",
    mmap_mode="r",
)

mh_val_binary = np.load(
    f"{root}/verified/mh_val_binary_labels.npy",
    mmap_mode="r",
)

print(mh_val.shape)
print(mh_val_binary.shape)
```

Expected:

```text
(30, 128, 128)
(30, 128, 128)
```

For large source arrays, use `mmap_mode="r"` to avoid unnecessarily loading
the complete array into RAM.

---

## 10. Recommended handling of 255

When training a semantic/multi-class loss:

```python
valid = (y_mc != 255)
```

When training the binary target:

```python
valid = (y_bin != 255)
```

Do not reinterpret `255` as a real change class.

Do not convert `255` to `0` during dataset preparation.

---

## 11. Quality and provenance

The label manifests contain:

- patch split;
- patch identifier;
- decision/source;
- multi-class mask SHA-256;
- binary mask SHA-256.

The release also contains a QC report and a snapshot of the verification
decision log.

For the 170 included weak labels, the decision source is:

```text
authoritative_autolabel
```

with a release-level record that the user visually inspected the pairs.

This is an **audit record**, not a claim that every pixel was independently
redrawn.

---

## 12. Source datasets and attribution

This project combines and derives information from multiple public
Earth-observation sources. The Kaggle dataset therefore uses the Kaggle
license option **Other** rather than incorrectly applying a single upstream
license to every derived artifact.

Upstream source terms remain applicable to the corresponding source-derived
content.

See `DATA_SOURCES.md` in this release for the source-by-source attribution
and licensing notes.

---

## 13. Important scientific limitations

1. The 170 verified labels are weak labels that were visually reviewed, not
   independently digitized masks.
2. The 20 blind patches are absent from the verified release.
3. The 30 monsoon patches are absent from the verified release.
4. Therefore the original blind-label auto-label-quality experiment cannot be
   reproduced from this release.
5. The original monsoon/SAR-under-cloud hypothesis evaluation cannot be
   reproduced from this release.
6. Do not report this 110-patch test subset as the full original 160-patch
   MH-TEST protocol.
7. Do not call the MH-ADAPT labels "clean manual labels" in experiments using
   this release.
8. Upstream evidence products have their own uncertainties and geographic/
   temporal limitations.

---

## 14. Citation

When using this release, cite the Geo-Nexus project and the upstream datasets
that materially contribute to your experiment.

### Geo-Nexus

```text
Geo-Nexus v3.2 — Maharashtra multimodal change-detection dataset,
17-channel Sentinel-1/Sentinel-2 representation with evidence-fused
weak labels and a 170-patch human-reviewed verification release.
```

### Dynamic World

Brown, C. F., Brumby, S. P., Guzder-Williams, B., et al. (2022).
Dynamic World, Near real-time global 10 m land use land cover mapping.
Scientific Data 9, 251.
DOI: 10.1038/s41597-022-01307-4

### Hansen Global Forest Change

Hansen, M. C., Potapov, P. V., Moore, R., et al. (2013).
High-Resolution Global Maps of 21st-Century Forest Cover Change.
Science 342, 850–853.

### Global Surface Water

Pekel, J.-F., Cottam, A., Gorelick, N., & Belward, A. S. (2016).
High-resolution mapping of global surface water and its long-term changes.
Nature 540, 418–422.

### Open Buildings

Sirko, W., Brempong, E. A., Marcos, J. T. C., et al. (2023).
High-Resolution Building and Road Detection from Sentinel-2.

---

## 15. License and redistribution notice

Kaggle metadata for this release uses:

```text
other
```

because the release combines derived products from multiple providers with
different upstream terms.

**Do not assume that every byte in this dataset is covered by CC-BY-4.0 or
another single license.** Users should follow the terms and attribution
requirements of the relevant upstream source.

The `DATA_SOURCES.md` file is part of the release so downstream users can
identify the relevant providers and citations.

---

## 16. Version

```text
Geo-Nexus v3.2
Release: 170-reviewed-weak-labels
Patch size: 128 × 128
Input channels: 17 per timestamp
Times per patch: 2
Verified labels: 170
Excluded original annotation slots: 50
```
""",
    "DATA_SOURCES.md": """# Geo-Nexus Maharashtra Public Dataset — Source and Licensing Notes

This file records the provenance and licensing information used when preparing
the public Kaggle release.

## 1. Copernicus Sentinel data

### Sentinel-2

The project uses `COPERNICUS/S2_SR_HARMONIZED` surface-reflectance imagery.

Copernicus states that Sentinel data are available on a free, full and open
basis subject to the Sentinel Data Legal Notice.

Official sources:

- Copernicus Data Space Ecosystem terms:
  https://dataspace.copernicus.eu/terms-and-conditions
- Sentinel Data Legal Notice:
  https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice

### Sentinel-1

The project uses Sentinel-1 GRD VV/VH measurements for the multimodal input.

Sentinel data are likewise provided under the Copernicus open-data policy,
subject to the applicable legal notice.

---

## 2. Open Buildings Temporal V1

The Geo-Nexus auto-label evidence export uses:

```text
GOOGLE/Research/open-buildings-temporal/v1
```

Relevant bands:

```text
building_presence
building_height
```

The Earth Engine catalog states that Open Buildings Temporal V1 covers annual
snapshots from 2016 through 2023 and is shared under CC-BY 4.0 and ODbL 1.0,
with the user able to choose the applicable license.

Official source:

https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings-temporal_v1

Citation:

Sirko, W., Brempong, E. A., Marcos, J. T. C., et al. (2023).
High-Resolution Building and Road Detection from Sentinel-2.

---

## 3. Dynamic World V1

The evidence fusion uses Dynamic World built and tree probabilities.

Official source:

https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1

The catalog states that Dynamic World V1 is licensed under CC-BY 4.0 and gives
the required attribution to the Dynamic World project, Google, National
Geographic Society, and World Resources Institute.

Citation:

Brown, C. F., Brumby, S. P., Guzder-Williams, B., et al. (2022).
Dynamic World, Near real-time global 10 m land use land cover mapping.
Scientific Data 9, 251.
DOI: 10.1038/s41597-022-01307-4

---

## 4. Hansen Global Forest Change

The evidence fusion uses Hansen Global Forest Change v1.12:

```text
UMD/hansen/global_forest_change_2024_v1_12
```

Relevant layers:

```text
lossyear
treecover2000
```

The Hansen Global Forest Change source documentation states that the data are
licensed under CC-BY 4.0 and provides the attribution:

```text
Hansen/UMD/Google/USGS/NASA
```

Official source:

https://earthenginepartners.appspot.com/science-2013-global-forest/

Citation:

Hansen, M. C., Potapov, P. V., Moore, R., et al. (2013).
High-Resolution Global Maps of 21st-Century Forest Cover Change.
Science 342, 850–853.

---

## 5. JRC Global Surface Water

The evidence fusion uses the JRC Global Surface Water occurrence layer.

Official data-access page:

https://global-surface-water.appspot.com/download

The current data-access page states that the data are produced under the
Copernicus Programme and are provided free of charge without restriction of
use, with required acknowledgment and citation.

Required map attribution:

```text
Source: EC JRC/Google
```

Citation:

Pekel, J.-F., Cottam, A., Gorelick, N., & Belward, A. S. (2016).
High-resolution mapping of global surface water and its long-term changes.
Nature 540, 418–422.
DOI: 10.1038/nature20584

---

## 6. AlphaEarth

AlphaEarth-derived change evidence is used by the project's preprocessing
pipeline as an evidence/QA component and baseline.

This release should not be interpreted as granting a new independent license
to any upstream AlphaEarth product. Users should consult the applicable
AlphaEarth/data-provider terms for redistribution or separate reuse of any
AlphaEarth-derived artifacts.

---

## 7. Geo-Nexus processing layer

The following are project-generated processing artifacts:

- 17-channel tensors;
- derived NDVI/NDBI/SAR cross-ratio/quality channels;
- evidence-fused class labels;
- binary targets;
- patch manifests;
- SHA-256 QC reports;
- verification audit records.

The label source is the project's evidence-fusion implementation, not a single
upstream map.

---

## 8. Auto-label evidence bands

`03_export_autolabels.js` defines the following evidence bands:

```text
0  ob_p2020
1  ob_p2023
2  ob_h2023
3  hansen_ly
4  hansen_tc00
5  dw_built_t1
6  dw_built_t2
7  dw_trees_t1
8  dw_trees_t2
9  jrc_occ
```

The script exports these evidence bands without applying the final change-class
thresholds; class fusion happens later in Python/Colab.

---

## 9. Public-release license decision

Because the dataset combines multiple upstream sources with different terms,
the Kaggle metadata uses:

```text
other
```

and this document provides the source-by-source attribution.

This is intentional. It prevents the public release from implying a blanket
third-party license that the project does not control.
""",
    "CHANGELOG.md": """# Geo-Nexus Maharashtra v3.2 — Changelog

## 170-reviewed-weak-labels

### Added

- `verified/mh_val_labels.npy`
- `verified/mh_val_binary_labels.npy`
- `verified/mh_adapt_labels.npy`
- `verified/mh_adapt_binary_labels.npy`
- `verified/mh_test_partial_labels.npy`
- `verified/mh_test_partial_binary_labels.npy`
- per-split manifests
- `label_qc_report.json`
- `verification_decisions_snapshot.json`
- `release_scope.json`
- `README.md`
- `DATA_SOURCES.md`
- `CHANGELOG.md`
- `dataset-metadata.json`

### Label semantics

Multi-class:

`0..6` are semantic classes and `255` is uncertain/ignore.

Binary:

`0` = no change, `1` = any real change class, `255` = uncertain/ignore.

`255` is preserved in the stored binary target.

### Scope

170 patches are released after visual review of the authoritative automatic
labels.

50 original annotation slots are intentionally excluded from verified labels:

- 20 blind
- 30 monsoon

No human-only mask was fabricated for those 50.
""",
}

RELEASE_ROOT.parent.mkdir(
    parents=True,
    exist_ok=True
)

DOC_ROOT = LOCAL_ROOT / "public_dataset_docs"

if DOC_ROOT.exists():
    shutil.rmtree(DOC_ROOT)

DOC_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

for filename, content in DATASET_DOCS.items():
    (DOC_ROOT / filename).write_text(
        content,
        encoding="utf-8"
    )

kaggle_metadata = {
    "title": "Geo-Nexus Maharashtra Change Detection v3.2",
    "subtitle": "17-channel Sentinel-1/2 tensors with 170 human-reviewed weak labels",
    "description": (
        "Geo-Nexus v3.2 Maharashtra multimodal change-detection release. "
        "Contains the original processed dataset plus a 170-patch human-reviewed "
        "weak-label verification release. Multi-class masks use 0..6 and 255 "
        "uncertain/ignore; binary masks use 0/1/255 with 255 preserved. "
        "20 blind and 30 monsoon annotation slots are intentionally excluded "
        "because no independent human masks were created. See README.md and "
        "DATA_SOURCES.md for complete provenance, usage, limitations and licensing."
    ),
    "id": KAGGLE_HANDLE,
    "licenses": [
        {"name": "other"}
    ],
    "keywords": [
        "remote-sensing",
        "change-detection",
        "Sentinel-1",
        "Sentinel-2",
        "Maharashtra",
        "India",
        "geospatial",
        "PyTorch",
        "weak-supervision",
        "land-change",
    ],
    "expectedUpdateFrequency": "never",
    "userSpecifiedSources": (
        "Copernicus Sentinel-1/2; Google Open Buildings Temporal V1; "
        "Dynamic World V1; Hansen Global Forest Change; JRC Global Surface Water; "
        "project-generated evidence-fusion and verification artifacts. "
        "See DATA_SOURCES.md."
    ),
    "resources": [
        {
            "path": "verified/mh_val_labels.npy",
            "description": "30 human-reviewed weak multi-class MH-VAL masks."
        },
        {
            "path": "verified/mh_val_binary_labels.npy",
            "description": "Binary MH-VAL targets with 0/1/255."
        },
        {
            "path": "verified/mh_adapt_labels.npy",
            "description": "30 human-reviewed weak multi-class MH-ADAPT masks."
        },
        {
            "path": "verified/mh_adapt_binary_labels.npy",
            "description": "Binary MH-ADAPT targets with 0/1/255."
        },
        {
            "path": "verified/mh_test_partial_labels.npy",
            "description": "110-patch partial test labels: 40 Pune dry, 40 Satara dry, 30 Vidarbha."
        },
        {
            "path": "verified/mh_test_partial_binary_labels.npy",
            "description": "Binary 110-patch partial test targets with 0/1/255."
        },
        {
            "path": "README.md",
            "description": "Complete dataset card and usage documentation."
        },
        {
            "path": "DATA_SOURCES.md",
            "description": "Source attribution and licensing notes."
        },
    ],
}

(DOC_ROOT / "dataset-metadata.json").write_text(
    json.dumps(
        kaggle_metadata,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

print("Documentation directory:")
for path in sorted(
    DOC_ROOT.iterdir()
):
    print(
        " ",
        path.name,
        path.stat().st_size,
        "bytes"
    )

print("\n✓ Public dataset documentation ready.")


# ============================================================
# CELL 8 — FINAL RELEASE SELF-TEST
# ============================================================

# Expected shapes.
expected_shapes = {
    "mh_val": (30, 128, 128),
    "mh_adapt": (30, 128, 128),
    "mh_test_partial": (110, 128, 128),
}

for name, expected_shape in expected_shapes.items():

    result = results[name]

    assert result["ok"] is True

    multi = np.load(
        result["multi_path"],
        mmap_mode="r"
    )

    binary = np.load(
        result["binary_path"],
        mmap_mode="r"
    )

    assert multi.shape == expected_shape
    assert binary.shape == expected_shape

    multi_values = set(
        np.unique(multi).tolist()
    )
    binary_values = set(
        np.unique(binary).tolist()
    )

    assert multi_values <= ALLOWED_LABELS
    assert binary_values <= {0, 1, 255}

    # The uncertain mask must be preserved exactly.
    assert int(
        np.sum(multi == 255)
    ) == int(
        np.sum(binary == 255)
    )

    # Class 0 maps to binary 0.
    assert np.all(
        binary[multi == 0] == 0
    )

    # Classes 1..6 map to binary 1.
    for cls in range(1, 7):
        assert np.all(
            binary[multi == cls] == 1
        )

print("✓ VAL shape/labels PASS")
print("✓ ADAPT shape/labels PASS")
print("✓ TEST_PARTIAL shape/labels PASS")
print("✓ Binary values restricted to {0,1,255}")
print("✓ Exact 255 preservation PASS")
print("✓ Binary derived from multi-class PASS")

assert sum(
    result["received"]
    for result in results.values()
) == 170

print("✓ Total released patches = 170")
print("✓ FINAL RELEASE SELF-TEST PASS")


# ============================================================
# CELL 9 — ASSEMBLE AND UPLOAD NEW KAGGLE DATASET VERSION
# ============================================================

# The default is the existing project dataset.
PUBLISH = True

if not PUBLISH:
    print("Kaggle publication disabled.")
else:

    import kagglehub

    kagglehub.login()

    FULL_ROOT = Path(
        "/content/geonexus_mh_v3_full"
    )

    if FULL_ROOT.exists():
        shutil.rmtree(FULL_ROOT)

    print(
        "Downloading existing Kaggle dataset:",
        KAGGLE_HANDLE
    )

    downloaded = kagglehub.dataset_download(
        KAGGLE_HANDLE,
        output_dir=str(FULL_ROOT),
        force_download=False,
    )

    downloaded = Path(downloaded)

    candidates = [
        downloaded,
        FULL_ROOT,
        FULL_ROOT / "geonexus-mh-v3",
    ]

    dataset_root = None

    for candidate in candidates:
        if (
            candidate.exists()
            and any(candidate.iterdir())
        ):
            dataset_root = candidate
            break

    if dataset_root is None:
        raise RuntimeError(
            "Could not locate the downloaded Kaggle dataset."
        )

    # --------------------------------------------------------
    # Replace verified/ with exactly the 170-release artifacts.
    # --------------------------------------------------------
    target_verified = (
        dataset_root /
        "verified"
    )

    if target_verified.exists():
        shutil.rmtree(target_verified)

    shutil.copytree(
        VERIFIED_ROOT,
        target_verified
    )

    # --------------------------------------------------------
    # Add public-release documentation.
    # --------------------------------------------------------
    for file in DOC_ROOT.iterdir():

        destination = (
            dataset_root /
            file.name
        )

        shutil.copy2(
            file,
            destination
        )

    # --------------------------------------------------------
    # Hard file gate.
    # --------------------------------------------------------
    required = [
        "verified/mh_val_labels.npy",
        "verified/mh_val_binary_labels.npy",
        "verified/mh_adapt_labels.npy",
        "verified/mh_adapt_binary_labels.npy",
        "verified/mh_test_partial_labels.npy",
        "verified/mh_test_partial_binary_labels.npy",
        "verified/release_scope.json",
        "verified/label_qc_report.json",
        "README.md",
        "DATA_SOURCES.md",
        "CHANGELOG.md",
        "dataset-metadata.json",
    ]

    missing = [
        item
        for item in required
        if not (
            dataset_root / item
        ).exists()
    ]

    if missing:
        raise RuntimeError(
            "Final Kaggle package missing:\n"
            + "\n".join(
                f"  - {item}"
                for item in missing
            )
        )

    # --------------------------------------------------------
    # Version upload.
    # --------------------------------------------------------
    version_notes = (
        "Geo-Nexus v3.2 public 170-reviewed weak-label release. "
        "Adds 30 MH-VAL + 30 MH-ADAPT + 110 MH-TEST-PARTIAL "
        "(40 Pune dry + 40 Satara dry + 30 Vidarbha), "
        "plus multi-class/binary targets, manifests, QC, "
        "source/licensing documentation and release metadata. "
        "20 blind + 30 monsoon masks remain intentionally excluded."
    )

    print("\nUploading new dataset version...")

    kagglehub.dataset_upload(
        KAGGLE_HANDLE,
        str(dataset_root),
        version_notes=version_notes
    )

    print("\n✓ Kaggle version upload completed.")
    print(
        "Dataset:",
        f"https://www.kaggle.com/datasets/{KAGGLE_HANDLE}"
    )


# ============================================================
# CELL 10 — KAGGLE VISIBILITY / PUBLIC RELEASE CHECK
# ============================================================

# Kaggle's dataset API currently treats visibility separately from versioning.
# We can check the current dataset status after the upload.

import json
import subprocess

command = [
    "kaggle",
    "datasets",
    "status",
    KAGGLE_HANDLE,
    "--format",
    "json",
]

try:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    print(
        "Kaggle status return code:",
        result.returncode
    )

    if result.stdout.strip():
        print("\nKaggle status:")
        print(result.stdout)

        try:
            status_json = json.loads(
                result.stdout
            )

            is_private = status_json.get(
                "isPrivate",
                status_json.get("is_private")
            )

            if is_private is True:
                print("\n⚠ Dataset is PRIVATE.")
                print(
                    "Open the dataset on Kaggle and change:"
                )
                print(
                    "Dataset → Settings → Sharing → Public"
                )
            elif is_private is False:
                print("\n✓ Dataset is PUBLIC.")
            else:
                print(
                    "\nVisibility was not returned by this status response."
                )
                print(
                    "Check Dataset → Settings → Sharing manually."
                )

        except json.JSONDecodeError:
            print(
                "\nCould not parse status JSON."
            )
            print(
                "Check Dataset → Settings → Sharing manually."
            )

    if result.stderr.strip():
        print("\nKaggle status stderr:")
        print(result.stderr)

except FileNotFoundError:
    print(
        "Kaggle CLI is not available in this runtime."
    )
    print(
        "The dataset upload may still have completed through kagglehub."
    )
    print(
        "Check Dataset → Settings → Sharing → Public on Kaggle."
    )
