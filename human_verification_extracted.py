# ============================================================
# CELL 1 — ENVIRONMENT, DRIVE, ARCHIVE DISCOVERY
# ============================================================

!pip -q install rasterio ipywidgets matplotlib pandas

from google.colab import drive
drive.mount("/content/drive")

import os
import json
import tarfile
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from IPython.display import display, clear_output
import ipywidgets as widgets


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------
DRIVE_PROC = Path(
    "/content/drive/MyDrive/geonexus_v3_processed"
)

ARCHIVE = DRIVE_PROC / "geonexus_v3_processed.tar"

VERIFY_ROOT = DRIVE_PROC / "verification"
DECISION_FILE = VERIFY_ROOT / "verification_decisions.json"

LOCAL_ROOT = Path("/content/geonexus_verify")
LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

LOCAL_QGIS = LOCAL_ROOT / "qgis"
LOCAL_POOL = LOCAL_ROOT / "annotation_pool.json"
LOCAL_REPORT = LOCAL_ROOT / "autolabel_report.json"

CORRECTED_ROOT = VERIFY_ROOT / "corrected_masks"

VERIFY_ROOT.mkdir(parents=True, exist_ok=True)
CORRECTED_ROOT.mkdir(parents=True, exist_ok=True)

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

PATCH_SIZE = 128

print("=" * 76)
print("Geo-Nexus v3.2 — Verification Setup")
print("=" * 76)

print("Processed Drive directory:", DRIVE_PROC)
print("Archive:", ARCHIVE)
print("Decision file:", DECISION_FILE)

if not DRIVE_PROC.exists():
    raise FileNotFoundError(
        f"Processed Drive directory not found:\n{DRIVE_PROC}"
    )

LOOSE_QGIS = DRIVE_PROC / "qgis"

if not ARCHIVE.exists() and not LOOSE_QGIS.exists():
    raise FileNotFoundError(
        "Neither the processed archive nor a loose qgis/ directory exists.\n"
        f"Expected archive:\n{ARCHIVE}\n"
        f"Or loose directory:\n{LOOSE_QGIS}"
    )

if LOCAL_QGIS.exists():
    shutil.rmtree(LOCAL_QGIS)
LOCAL_QGIS.mkdir(parents=True, exist_ok=True)



# ------------------------------------------------------------
# Prefer a loose qgis folder when it exists.
# Otherwise extract only qgis/ + annotation_pool.json +
# autolabel_report.json from the existing archive.
# ------------------------------------------------------------
if LOOSE_QGIS.exists():

    print("✓ Using existing loose qgis/ directory.")

    shutil.copytree(
        LOOSE_QGIS,
        LOCAL_QGIS,
        dirs_exist_ok=True
    )

    for filename in [
        "annotation_pool.json",
        "autolabel_report.json",
    ]:
        source = DRIVE_PROC / filename
        if source.exists():
            shutil.copy2(
                source,
                LOCAL_ROOT / filename
            )

else:

    print("✓ Loose qgis/ not found.")
    print("✓ Existing archive found.")
    print("→ Extracting only verification files...")

    with tarfile.open(ARCHIVE, "r") as tar:

        selected = []

        for member in tar.getmembers():

            name = member.name.replace("\\", "/")

            if "/qgis/" in f"/{name}":
                selected.append(member)

            elif name.endswith("/annotation_pool.json"):
                selected.append(member)

            elif name.endswith("/autolabel_report.json"):
                selected.append(member)

        if not selected:
            raise RuntimeError(
                "The archive contains no qgis/ or annotation_pool.json entries."
            )

        print(
            "Selected archive members:",
            len(selected)
        )

        for member in selected:

            name = member.name.replace("\\", "/")

            if "/qgis/" in f"/{name}":

                qidx = name.find("qgis/")
                rel = name[qidx + len("qgis/"):]

                if not rel:
                    continue

                rel_path = Path(rel)

                if rel_path.is_absolute() or ".." in rel_path.parts:
                    raise RuntimeError(
                        f"Unsafe qgis archive path: {name}"
                    )

                output = (LOCAL_QGIS / rel_path).resolve()
                output.relative_to(
                    LOCAL_QGIS.resolve()
                )

                if member.isdir():

                    output.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                elif member.isfile():

                    output.parent.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    source = tar.extractfile(member)

                    if source is None:
                        raise RuntimeError(
                            f"Could not read archive member: {name}"
                        )

                    with open(
                        output,
                        "wb"
                    ) as out_file:
                        shutil.copyfileobj(
                            source,
                            out_file
                        )

            elif (
                name.endswith("annotation_pool.json")
                or name.endswith("autolabel_report.json")
            ):

                output = (
                    LOCAL_ROOT /
                    Path(name).name
                )

                source = tar.extractfile(member)

                if source is not None:

                    with open(
                        output,
                        "wb"
                    ) as out_file:
                        shutil.copyfileobj(
                            source,
                            out_file
                        )

# ------------------------------------------------------------
# Persist a browsable snapshot on Drive once.
# ------------------------------------------------------------
PERSIST_QGIS_SNAPSHOT = True
DRIVE_QGIS_SNAPSHOT = VERIFY_ROOT / "qgis_snapshot"

if PERSIST_QGIS_SNAPSHOT:

    if not DRIVE_QGIS_SNAPSHOT.exists():

        print(
            "\nSaving QGIS package snapshot to Drive..."
        )

        shutil.copytree(
            LOCAL_QGIS,
            DRIVE_QGIS_SNAPSHOT,
            dirs_exist_ok=True
        )

        print(
            "✓ Saved:",
            DRIVE_QGIS_SNAPSHOT
        )

    else:

        print(
            "\n✓ Drive QGIS snapshot already exists:",
            DRIVE_QGIS_SNAPSHOT
        )

for split in SPLITS:

    (CORRECTED_ROOT / split).mkdir(
        parents=True,
        exist_ok=True
    )

missing_folders = []

for split in SPLITS:

    folder = LOCAL_QGIS / split

    if folder.exists():

        print(
            f"✓ {split:22s}"
            f"{len(list(folder.iterdir())):4d} files"
        )

    else:
        missing_folders.append(split)

if missing_folders:

    print("\nActual extracted folders:")

    for path in sorted(LOCAL_QGIS.iterdir()):
        print(" ", path.name)

    raise RuntimeError(
        "Missing required split folders:\n"
        + "\n".join(
            f" - {x}"
            for x in missing_folders
        )
    )

print("\n✓ Cell 1 completed successfully.")
print("Local QGIS root:", LOCAL_QGIS)

# ============================================================
# CELL 2 — AUTHORITATIVE POOL + PATCH INVENTORY
# ============================================================

def load_json_if_exists(path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

ANNOTATION_POOL = load_json_if_exists(LOCAL_POOL)

def index_split(split):
    folder = LOCAL_QGIS / split
    items = []

    for path in sorted(folder.glob("*.tif")):

        if path.name.endswith("_ae.tif"):
            continue

        if path.name.endswith("_autolabel.tif"):
            continue

        try:
            with rasterio.open(path) as src:
                if src.count != 8:
                    continue

                if (
                    src.width != PATCH_SIZE
                    or src.height != PATCH_SIZE
                ):
                    continue

        except Exception as exc:

            print(
                f"[WARNING] Could not inspect {path.name}: {exc}"
            )
            continue

        stem = path.stem

        auto_path = folder / f"{stem}_autolabel.tif"
        ae_path = folder / f"{stem}_ae.tif"

        items.append({
            "split": split,
            "stem": stem,
            "image": path,
            "autolabel": auto_path if auto_path.exists() else None,
            "ae": ae_path if ae_path.exists() else None,
            "blind": (
                split == "MH_TEST_BLIND"
                or split == "MH_TEST_MONSOON"
            ),
        })

    return items

PATCHES = {
    split: index_split(split)
    for split in SPLITS
}

# ------------------------------------------------------------
# Use annotation_pool.json order when it has the expected
# zone/row/col fields.
# ------------------------------------------------------------
if isinstance(ANNOTATION_POOL, dict):

    reordered = {}

    for split in SPLITS:

        by_stem = {
            item["stem"]: item
            for item in PATCHES[split]
        }

        ordered = []

        for entry in ANNOTATION_POOL.get(split, []):

            zone = entry.get("zone")
            row = entry.get("row")
            col = entry.get("col")

            if (
                zone is None
                or row is None
                or col is None
            ):
                continue

            stem = f"{zone}_{int(row)}_{int(col)}"

            if stem in by_stem:

                record = dict(
                    by_stem[stem]
                )

                record["pool"] = dict(entry)
                ordered.append(record)

        reordered[split] = (
            ordered
            if len(ordered) == len(PATCHES[split])
            else PATCHES[split]
        )

    PATCHES = reordered

print("\n" + "=" * 76)
print("PATCH INVENTORY")
print("=" * 76)

for split in SPLITS:

    actual = len(PATCHES[split])
    expected = EXPECTED_COUNTS[split]

    auto_count = sum(
        item["autolabel"] is not None
        for item in PATCHES[split]
    )

    ae_count = sum(
        item["ae"] is not None
        for item in PATCHES[split]
    )

    print(
        f"{split:22s}"
        f"actual={actual:3d} "
        f"expected={expected:3d} "
        f"auto={auto_count:3d} "
        f"AE={ae_count:3d}"
    )

    if actual != expected:
        raise RuntimeError(
            f"{split}: expected {expected} patches, found {actual}."
        )

AUTO_REQUIRED_SPLITS = [
    "MH_VAL",
    "MH_ADAPT",
    "MH_TEST_DRY_A",
    "MH_TEST_DRY_B",
    "MH_TEST_VIDARBHA",
]

for split in AUTO_REQUIRED_SPLITS:

    missing_auto = [
        item["stem"]
        for item in PATCHES[split]
        if item["autolabel"] is None
    ]

    if missing_auto:

        raise RuntimeError(
            f"{split}: missing {len(missing_auto)} auto-label files."
        )

for split in [
    "MH_TEST_BLIND",
    "MH_TEST_MONSOON",
]:

    unexpected_auto = [
        item["stem"]
        for item in PATCHES[split]
        if item["autolabel"] is not None
    ]

    if unexpected_auto:
        raise RuntimeError(
            f"{split}: automatic masks found in a manual-only set."
        )

print("\n✓ Exact 220-patch inventory verified.")

# ============================================================
# CELL 3 — VISUALIZATION + MULTI-CLASS / BINARY QA
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

CLASS_LUT = np.array(
    [
        [0, 0, 0],
        [30, 100, 220],
        [80, 200, 220],
        [255, 200, 0],
        [220, 50, 50],
        [50, 180, 50],
        [180, 180, 180],
    ],
    dtype=np.uint8,
)

ALLOWED_LABELS = set(list(range(7)) + [255])

def stretch_band(band):
    band = np.asarray(band, dtype=np.float32)
    finite = np.isfinite(band)

    if not np.any(finite):
        return np.zeros_like(band)

    vals = band[finite]
    low = float(np.percentile(vals, 2))
    high = float(np.percentile(vals, 98))

    if high <= low:
        low = float(vals.min())
        high = float(vals.max())

    if high <= low:
        return np.zeros_like(band)

    return np.clip((band - low) / (high - low), 0, 1)

def make_rgb(arr4):
    out = np.zeros(
        (arr4.shape[1], arr4.shape[2], 3),
        dtype=np.float32,
    )
    for c in range(3):
        out[:, :, c] = stretch_band(arr4[c])
    return out

def colourise_mask(mask):
    mask = np.asarray(mask, dtype=np.uint8)

    rgb = CLASS_LUT[np.clip(mask, 0, 6)]
    rgb = np.array(rgb, copy=True)

    # 255 = UNCERTAIN -> magenta for QA visualization.
    rgb[mask == 255] = [255, 0, 255]
    return rgb

def make_binary_change_target(mask):
    """
    Stored binary target for Geo-Nexus.

    0   -> no change
    1   -> change (any semantic class 1..6)
    255 -> uncertain / ignored

    IMPORTANT:
    255 is deliberately preserved here so the stored binary artifact
    retains the same ignore information as the multi-class target.
    """
    mask = np.asarray(mask, dtype=np.uint8)

    binary = np.full(
        mask.shape,
        255,
        dtype=np.uint8,
    )

    binary[mask == 0] = 0
    binary[(mask >= 1) & (mask <= 6)] = 1

    return binary

def binary_change_rgb(mask):
    """
    Binary target visualization:

    BLACK = 0 = no change
    WHITE = 1 = change
    GRAY  = 255 = uncertain / ignored
    """
    binary = make_binary_change_target(mask)

    rgb = np.zeros(
        (*binary.shape, 3),
        dtype=np.uint8,
    )

    rgb[binary == 1] = [255, 255, 255]
    rgb[binary == 255] = [160, 160, 160]

    return rgb

def validate_mask(mask, expected_shape=(128, 128)):
    mask = np.asarray(mask, dtype=np.uint8)

    if mask.shape != expected_shape:
        raise ValueError(
            f"Mask shape {mask.shape} != {expected_shape}"
        )

    values = set(np.unique(mask).tolist())
    illegal = values - ALLOWED_LABELS

    if illegal:
        raise ValueError(
            "Illegal class IDs: "
            f"{sorted(illegal)}"
        )

    return mask

def validate_binary_target(binary, expected_shape=(128, 128)):
    binary = np.asarray(binary, dtype=np.uint8)

    if binary.shape != expected_shape:
        raise ValueError(
            f"Binary target shape {binary.shape} != {expected_shape}"
        )

    values = set(np.unique(binary).tolist())
    allowed = {0, 1, 255}
    illegal = values - allowed

    if illegal:
        raise ValueError(
            "Illegal binary IDs: "
            f"{sorted(illegal)}"
        )

    return binary

def read_patch(item):
    with rasterio.open(item["image"]) as src:
        if src.count != 8:
            raise ValueError(
                f"{item['image'].name}: expected 8 bands, got {src.count}"
            )
        arr = src.read(out_dtype="float32")
        descriptions = src.descriptions

    if arr.shape != (8, PATCH_SIZE, PATCH_SIZE):
        raise ValueError(
            f"{item['stem']}: unexpected image shape {arr.shape}"
        )

    t1 = arr[:4]
    t2 = arr[4:8]

    result = {
        "t1_rgb": make_rgb(t1),
        "t2_rgb": make_rgb(t2),
        "t1_nir": stretch_band(t1[3]),
        "t2_nir": stretch_band(t2[3]),
        "mask": None,
        "ae": None,
        "descriptions": descriptions,
    }

    if item["autolabel"] is not None:
        with rasterio.open(item["autolabel"]) as src:
            result["mask"] = validate_mask(src.read(1))

    if item["ae"] is not None:
        with rasterio.open(item["ae"]) as src:
            ae = src.read(1).astype(np.float32)

        if ae.shape != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(
                f"AlphaEarth shape mismatch: {ae.shape}"
            )

        result["ae"] = ae

    return result

def show_item(item):
    data = read_patch(item)

    has_mask = data["mask"] is not None
    has_ae = data["ae"] is not None

    ncols = 6 if (has_mask and has_ae) else 5 if has_mask else 3

    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(4 * ncols, 4.8),
    )
    axes = np.atleast_1d(axes)

    axes[0].imshow(data["t1_rgb"])
    axes[0].set_title("T1 — Earlier")
    axes[0].axis("off")

    axes[1].imshow(data["t2_rgb"])
    axes[1].set_title("T2 — Current")
    axes[1].axis("off")

    if has_mask:
        mask_rgb = colourise_mask(data["mask"])

        axes[2].imshow(mask_rgb)
        axes[2].set_title("Auto-label Mask")
        axes[2].axis("off")

        binary_rgb = binary_change_rgb(data["mask"])

        axes[3].imshow(binary_rgb)
        axes[3].set_title("Binary Change Mask")
        axes[3].axis("off")

        axes[4].imshow(data["t2_rgb"])
        axes[4].imshow(mask_rgb, alpha=0.45)
        axes[4].set_title("T2 + Mask")
        axes[4].axis("off")

        if has_ae:
            image = axes[5].imshow(
                data["ae"],
                cmap="magma",
            )
            axes[5].set_title("AlphaEarth Change")
            axes[5].axis("off")
            fig.colorbar(
                image,
                ax=axes[5],
                fraction=0.046,
                pad=0.04,
            )
    else:
        axes[2].imshow(data["t2_rgb"])
        axes[2].set_title("Manual Label Required")
        axes[2].axis("off")

    fig.suptitle(
        f"{item['split']} | {item['stem']}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()

    if has_mask:
        unique, counts = np.unique(
            data["mask"],
            return_counts=True,
        )

        print("Existing multi-class mask:")

        for cls, count in zip(unique, counts):
            print(
                f"  {int(cls):3d} "
                f"{CLASS_NAMES.get(int(cls), 'UNKNOWN'):16s} "
                f"{count:7d} px "
                f"{100*count/data['mask'].size:6.2f}%"
            )

        binary = make_binary_change_target(data["mask"])

        print("\nBinary target summary:")
        print(
            f"  change=1        {int((binary == 1).sum()):7d} px "
            f"{100*(binary == 1).mean():6.2f}%"
        )
        print(
            f"  no_change=0     {int((binary == 0).sum()):7d} px "
            f"{100*(binary == 0).mean():6.2f}%"
        )
        print(
            f"  uncertain=255   {int((binary == 255).sum()):7d} px "
            f"{100*(binary == 255).mean():6.2f}% "
            "(ignored by binary loss)"
        )

        print(
            "\nBinary convention: BLACK=0 no change, "
            "WHITE=1 change, GRAY=255 ignored."
        )
    else:
        print(
            "\nMANUAL-ONLY PATCH: "
            "No automatic result mask exists for this patch."
        )


# ============================================================
# CELL 4 — PERSISTENT INTERACTIVE REVIEWER
# ============================================================

if DECISION_FILE.exists():

    try:
        with open(
            DECISION_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            DECISIONS = json.load(f)

    except Exception as exc:

        print(
            "Existing decision file could not be loaded:"
        )
        print(exc)
        DECISIONS = {}

else:

    DECISIONS = {}

def decision_key(item):
    return (
        f"{item['split']}/"
        f"{item['stem']}"
    )

def save_decisions():

    VERIFY_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = DECISION_FILE.with_suffix(
        ".tmp"
    )

    tmp.write_text(
        json.dumps(
            DECISIONS,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tmp.replace(
        DECISION_FILE
    )

def corrected_mask_path(item):
    return (
        CORRECTED_ROOT
        / item["split"]
        / f"{item['stem']}.tif"
    )

def validate_corrected_mask(item):

    path = corrected_mask_path(
        item
    )

    if not path.exists():
        raise FileNotFoundError(
            "Mask not found:\n"
            f"{path}"
        )

    with rasterio.open(path) as src:

        if src.count != 1:
            raise ValueError(
                f"{path.name}: expected one band, "
                f"found {src.count}"
            )

        mask = src.read(1)

        if (
            src.width != PATCH_SIZE
            or src.height != PATCH_SIZE
        ):
            raise ValueError(
                f"{path.name}: expected 128x128, "
                f"found {src.height}x{src.width}"
            )

    return validate_mask(
        mask
    )

CURRENT_SPLIT = SPLITS[0]
CURRENT_INDEX = 0

split_selector = widgets.Dropdown(
    options=SPLITS,
    value=SPLITS[0],
    description="Split:",
    layout=widgets.Layout(width="380px"),
)

index_slider = widgets.IntSlider(
    value=0,
    min=0,
    max=EXPECTED_COUNTS[SPLITS[0]] - 1,
    step=1,
    continuous_update=False,
    description="Patch:",
    layout=widgets.Layout(width="570px"),
)

decision_selector = widgets.ToggleButtons(
    options=[
        "UNREVIEWED",
        "ACCEPT_AUTO",
        "CORRECTED",
        "MANUAL_VERIFIED",
        "REJECT",
    ],
    value="UNREVIEWED",
    description="Decision:",
)

note_box = widgets.Text(
    value="",
    description="Note:",
    placeholder="Optional audit note",
    layout=widgets.Layout(width="820px"),
)

previous_button = widgets.Button(
    description="← Previous",
)

next_button = widgets.Button(
    description="Next →",
)

next_unreviewed_button = widgets.Button(
    description="Next Unreviewed",
)

save_button = widgets.Button(
    description="Save",
    button_style="success",
)

save_next_button = widgets.Button(
    description="Save + Next",
    button_style="primary",
)

status_output = widgets.Output()
image_output = widgets.Output()

def current_item():
    return PATCHES[
        CURRENT_SPLIT
    ][CURRENT_INDEX]

def get_current_record():

    item = current_item()

    return DECISIONS.get(
        decision_key(item)
    )

def load_current_decision():

    record = get_current_record()

    if record is None:

        decision_selector.value = "UNREVIEWED"
        note_box.value = ""

    else:

        decision_selector.value = record.get(
            "decision",
            "UNREVIEWED",
        )

        note_box.value = record.get(
            "note",
            "",
        )

def split_counts(split):

    counts = {
        "UNREVIEWED": 0,
        "ACCEPT_AUTO": 0,
        "CORRECTED": 0,
        "MANUAL_VERIFIED": 0,
        "REJECT": 0,
    }

    for item in PATCHES[split]:

        record = DECISIONS.get(
            decision_key(item)
        )

        decision = (
            record.get("decision", "UNREVIEWED")
            if record
            else "UNREVIEWED"
        )

        if decision not in counts:
            counts[decision] = 0

        counts[decision] += 1

    return counts

def render_status():

    with status_output:

        clear_output(
            wait=True
        )

        counts = split_counts(
            CURRENT_SPLIT
        )

        total = len(
            PATCHES[CURRENT_SPLIT]
        )

        reviewed = (
            total
            - counts["UNREVIEWED"]
        )

        item = current_item()

        print(
            f"{CURRENT_SPLIT}: "
            f"{reviewed}/{total} reviewed | "
            f"accept={counts['ACCEPT_AUTO']} | "
            f"corrected={counts['CORRECTED']} | "
            f"manual={counts['MANUAL_VERIFIED']} | "
            f"reject={counts['REJECT']}"
        )

        print(
            f"\nCurrent: "
            f"{CURRENT_INDEX + 1}/{total}"
        )

        print(
            "Patch:",
            item["stem"]
        )

        print(
            "Auto mask:",
            "YES" if item["autolabel"] is not None else "NO"
        )

        if item["blind"]:

            print(
                "\n⚠ MANUAL-ONLY SPLIT"
            )

            print(
                "ACCEPT_AUTO is disabled."
            )

            print(
                "MANUAL_VERIFIED requires "
                "a human-created mask."
            )

def render():

    load_current_decision()

    with image_output:

        clear_output(
            wait=True
        )

        show_item(
            current_item()
        )

    render_status()

def save_current():

    item = current_item()
    decision = decision_selector.value

    if (
        decision == "ACCEPT_AUTO"
        and item["autolabel"] is None
    ):

        raise ValueError(
            "ACCEPT_AUTO is not permitted for "
            "manual-only patches."
        )

    if decision in [
        "CORRECTED",
        "MANUAL_VERIFIED",
    ]:

        validate_corrected_mask(
            item
        )

    DECISIONS[
        decision_key(item)
    ] = {
        "split": item["split"],
        "stem": item["stem"],
        "decision": decision,
        "note": note_box.value,
        "blind": bool(item["blind"]),
        "source_image_sha256": hashlib.sha256(
            item["image"].read_bytes()
        ).hexdigest(),
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    save_decisions()

    render_status()

    print(
        f"✓ Saved {decision_key(item)}"
    )

def safe_save_then_next(_):

    try:
        save_current()
    except Exception as exc:

        with status_output:
            print(
                "\nNOT SAVED:",
                exc,
            )
        return

    move_next_unreviewed()

def save_only(_):

    try:
        save_current()
    except Exception as exc:

        with status_output:
            print(
                "\nNOT SAVED:",
                exc,
            )

def move_previous(_):

    global CURRENT_INDEX

    if CURRENT_INDEX > 0:

        CURRENT_INDEX -= 1
        index_slider.value = (
            CURRENT_INDEX
        )

def move_next(_):

    global CURRENT_INDEX

    if CURRENT_INDEX < (
        len(PATCHES[CURRENT_SPLIT]) - 1
    ):

        CURRENT_INDEX += 1
        index_slider.value = (
            CURRENT_INDEX
        )

def move_next_unreviewed():

    global CURRENT_INDEX

    items = PATCHES[
        CURRENT_SPLIT
    ]

    start = CURRENT_INDEX + 1

    search_order = (
        list(
            range(
                start,
                len(items)
            )
        )
        +
        list(
            range(
                0,
                start
            )
        )
    )

    for idx in search_order:

        record = DECISIONS.get(
            decision_key(items[idx])
        )

        decision = (
            record.get("decision", "UNREVIEWED")
            if record
            else "UNREVIEWED"
        )

        if decision == "UNREVIEWED":

            CURRENT_INDEX = idx
            index_slider.value = idx
            return

    with status_output:
        print(
            "\n✓ No unreviewed patch remains in this split."
        )

def move_next_unreviewed_button(_):
    move_next_unreviewed()

def split_changed(change):

    global CURRENT_SPLIT
    global CURRENT_INDEX

    if change["name"] != "value":
        return

    CURRENT_SPLIT = change["new"]
    CURRENT_INDEX = 0

    index_slider.max = (
        len(PATCHES[CURRENT_SPLIT]) - 1
    )

    index_slider.value = 0

def slider_changed(change):

    global CURRENT_INDEX

    if change["name"] != "value":
        return

    CURRENT_INDEX = int(
        change["new"]
    )

    render()

split_selector.observe(
    split_changed,
    names="value",
)

index_slider.observe(
    slider_changed,
    names="value",
)

previous_button.on_click(
    move_previous
)

next_button.on_click(
    move_next
)

next_unreviewed_button.on_click(
    move_next_unreviewed_button
)

save_button.on_click(
    save_only
)

save_next_button.on_click(
    safe_save_then_next
)

display(
    widgets.VBox(
        [
            split_selector,
            index_slider,
            widgets.HBox(
                [
                    previous_button,
                    next_button,
                    next_unreviewed_button,
                ]
            ),
            decision_selector,
            note_box,
            widgets.HBox(
                [
                    save_button,
                    save_next_button,
                ]
            ),
            status_output,
            image_output,
        ]
    )
)

render()

# ============================================================
# CELL 5 — VERIFICATION DASHBOARD
# ============================================================

def verification_dashboard():

    rows = []

    for split in SPLITS:

        counts = split_counts(
            split
        )

        rows.append({
            "split": split,
            "total": len(PATCHES[split]),
            **counts,
        })

    import pandas as pd

    df = pd.DataFrame(rows)

    display(df)

    totals = {
        col: int(
            df[col].sum()
        )
        for col in [
            "total",
            "UNREVIEWED",
            "ACCEPT_AUTO",
            "CORRECTED",
            "MANUAL_VERIFIED",
            "REJECT",
        ]
    }

    print("\nTOTAL")
    for key, value in totals.items():
        print(
            f"{key:18s}: {value}"
        )

    print(
        "\nDecision file:",
        DECISION_FILE
    )

verification_dashboard()

# ============================================================
# CELL 6 — BUILD FINAL VERIFIED MULTI-CLASS + BINARY ARRAYS
# ============================================================

VERIFIED_ROOT = VERIFY_ROOT / "verified"
VERIFIED_ROOT.mkdir(parents=True, exist_ok=True)

def get_final_mask(item):
    record = DECISIONS.get(decision_key(item))

    if record is None:
        raise RuntimeError(
            f"UNREVIEWED: {decision_key(item)}"
        )

    decision = record.get("decision", "UNREVIEWED")

    if decision == "ACCEPT_AUTO":
        if item["autolabel"] is None:
            raise RuntimeError(
                f"{decision_key(item)}: "
                "ACCEPT_AUTO but no auto mask exists."
            )

        with rasterio.open(item["autolabel"]) as src:
            mask = src.read(1)

        source = "authoritative_autolabel"

    elif decision in ["CORRECTED", "MANUAL_VERIFIED"]:
        mask = validate_corrected_mask(item)
        source = "human_mask"

    elif decision == "REJECT":
        raise RuntimeError(
            f"REJECTED: {decision_key(item)}"
        )

    else:
        raise RuntimeError(
            f"Unresolved decision {decision!r}: "
            f"{decision_key(item)}"
        )

    return validate_mask(mask), source

def build_verified_arrays(
    split_order,
    multiclass_output_name,
    binary_output_name,
):
    expected = sum(
        EXPECTED_COUNTS[s]
        for s in split_order
    )

    masks = []
    binary_masks = []
    manifest = []
    unresolved = []

    for split in split_order:
        for item in PATCHES[split]:

            try:
                mask, source = get_final_mask(item)
            except Exception as exc:
                unresolved.append({
                    "split": split,
                    "stem": item["stem"],
                    "reason": str(exc),
                })
                continue

            binary = make_binary_change_target(mask)
            binary = validate_binary_target(binary)

            masks.append(mask)
            binary_masks.append(binary)

            record = DECISIONS[decision_key(item)]

            manifest.append({
                "split": split,
                "stem": item["stem"],
                "decision": record["decision"],
                "source": source,
                "mask_sha256": hashlib.sha256(
                    mask.tobytes()
                ).hexdigest(),
                "binary_mask_sha256": hashlib.sha256(
                    binary.tobytes()
                ).hexdigest(),
            })

    if unresolved:
        return {
            "ok": False,
            "expected": expected,
            "received": len(masks),
            "multiclass_output": None,
            "binary_output": None,
            "manifest": manifest,
            "unresolved": unresolved,
        }

    multiclass_arr = np.stack(masks).astype(np.uint8)
    binary_arr = np.stack(binary_masks).astype(np.uint8)

    expected_shape = (
        expected,
        PATCH_SIZE,
        PATCH_SIZE,
    )

    if multiclass_arr.shape != expected_shape:
        return {
            "ok": False,
            "expected": expected,
            "received": int(multiclass_arr.shape[0]),
            "multiclass_output": None,
            "binary_output": None,
            "manifest": manifest,
            "unresolved": [{
                "reason": (
                    f"multi-class shape {multiclass_arr.shape} "
                    f"!= {expected_shape}"
                )
            }],
        }

    if binary_arr.shape != expected_shape:
        return {
            "ok": False,
            "expected": expected,
            "received": int(binary_arr.shape[0]),
            "multiclass_output": None,
            "binary_output": None,
            "manifest": manifest,
            "unresolved": [{
                "reason": (
                    f"binary shape {binary_arr.shape} "
                    f"!= {expected_shape}"
                )
            }],
        }

    multiclass_path = VERIFIED_ROOT / multiclass_output_name
    binary_path = VERIFIED_ROOT / binary_output_name

    np.save(multiclass_path, multiclass_arr)
    np.save(binary_path, binary_arr)

    return {
        "ok": True,
        "expected": expected,
        "received": int(multiclass_arr.shape[0]),
        "multiclass_output": str(multiclass_path),
        "binary_output": str(binary_path),
        "manifest": manifest,
        "unresolved": [],
    }

results = {}

results["mh_val"] = build_verified_arrays(
    ["MH_VAL"],
    "mh_val_labels.npy",
    "mh_val_binary_labels.npy",
)

results["mh_adapt"] = build_verified_arrays(
    ["MH_ADAPT"],
    "mh_adapt_labels.npy",
    "mh_adapt_binary_labels.npy",
)

results["mh_test"] = build_verified_arrays(
    [
        "MH_TEST_DRY_A",
        "MH_TEST_DRY_B",
        "MH_TEST_BLIND",
        "MH_TEST_MONSOON",
        "MH_TEST_VIDARBHA",
    ],
    "mh_test_labels.npy",
    "mh_test_binary_labels.npy",
)

for name, result in results.items():

    print(f"\n{name}")

    if result["ok"]:
        print("  ✓ multi-class PASS")
        print(
            "  multi-class shape:",
            np.load(
                result["multiclass_output"],
                mmap_mode="r",
            ).shape,
        )

        print("  ✓ binary PASS")
        print(
            "  binary shape:",
            np.load(
                result["binary_output"],
                mmap_mode="r",
            ).shape,
        )
    else:
        print("  ⛔ BLOCKED")
        print("  expected:", result["expected"])
        print("  received:", result["received"])
        print(
            "  unresolved:",
            len(result["unresolved"]),
        )
        for row in result["unresolved"][:10]:
            print("   -", row)

def class_balance(
    array_path,
    title,
    labels=(0, 1, 2, 3, 4, 5, 6, 255),
):
    arr = np.load(
        array_path,
        mmap_mode="r",
    )

    print(f"\n{title}")

    for cls in labels:
        count = int(np.sum(arr == cls))
        print(
            f"{cls:3d} "
            f"{CLASS_NAMES[cls]:16s} "
            f"{count:10,d} "
            f"{100.0*count/arr.size:8.3f}%"
        )

def binary_balance(
    array_path,
    title,
):
    arr = np.load(
        array_path,
        mmap_mode="r",
    )

    print(f"\n{title}")

    for cls, name in [
        (0, "no_change"),
        (1, "change"),
        (255, "uncertain"),
    ]:
        count = int(np.sum(arr == cls))
        print(
            f"{cls:3d} "
            f"{name:12s} "
            f"{count:10,d} "
            f"{100.0*count/arr.size:8.3f}%"
        )

for name, result in results.items():
    if result["ok"]:
        class_balance(
            result["multiclass_output"],
            f"{name} multi-class",
        )
        binary_balance(
            result["binary_output"],
            f"{name} binary",
        )


# ============================================================
# CELL 7 — MANIFESTS + QC REPORT + SHA256
# ============================================================

def write_json(path, obj):
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

artifact_status = {}

for key, result in results.items():

    artifact_status[key] = {
        "ok": bool(result["ok"]),
        "expected": result["expected"],
        "received": result["received"],
        "multiclass_output": result["multiclass_output"],
        "binary_output": result["binary_output"],
        "unresolved": result["unresolved"],
    }

    if result["ok"]:

        manifest_name = {
            "mh_val": "mh_val_manifest.json",
            "mh_adapt": "mh_adapt_manifest.json",
            "mh_test": "mh_test_manifest.json",
        }[key]

        write_json(
            VERIFIED_ROOT / manifest_name,
            result["manifest"],
        )

decision_summary = {
    split: split_counts(split)
    for split in SPLITS
}

qc_report = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "project": "Geo-Nexus v3.2",
    "patch_size": PATCH_SIZE,
    "expected_counts": EXPECTED_COUNTS,
    "class_ids": CLASS_NAMES,
    "binary_ids": {
        "0": "no_change",
        "1": "change_any_of_classes_1_to_6",
        "255": "uncertain_ignore",
    },
    "decision_counts": decision_summary,
    "artifact_status": artifact_status,
    "decision_file": str(DECISION_FILE),
    "corrected_mask_root": str(CORRECTED_ROOT),
    "source_archive": str(ARCHIVE),
}

write_json(
    VERIFIED_ROOT / "label_qc_report.json",
    qc_report,
)

write_json(
    VERIFIED_ROOT / "verification_decisions_snapshot.json",
    DECISIONS,
)

print(
    "✓ QC report written:",
    VERIFIED_ROOT / "label_qc_report.json",
)

print("\nVerified artifacts currently present:")

for path in sorted(VERIFIED_ROOT.iterdir()):
    if path.is_file():
        print(" ", path.name)


# ============================================================
# CELL 8 — FINAL PACKAGE SNAPSHOT + HASH MANIFEST
# ============================================================

PACKAGE_ROOT = (
    DRIVE_PROC /
    "verification_package"
)

if PACKAGE_ROOT.exists():
    shutil.rmtree(
        PACKAGE_ROOT
    )

PACKAGE_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

PACKAGE_VERIFIED = (
    PACKAGE_ROOT /
    "verified"
)

PACKAGE_VERIFIED.mkdir(
    parents=True,
    exist_ok=True
)

for path in VERIFIED_ROOT.iterdir():

    if path.is_file():

        shutil.copy2(
            path,
            PACKAGE_VERIFIED /
            path.name
        )

file_manifest = []

for path in sorted(
    PACKAGE_VERIFIED.iterdir()
):

    if not path.is_file():
        continue

    digest = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    file_manifest.append({
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest,
    })

write_json(
    PACKAGE_ROOT /
    "package_manifest.json",
    {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "files": file_manifest,
        "artifact_status": artifact_status,
    },
)

print(
    "Package root:",
    PACKAGE_ROOT
)

for row in file_manifest:

    print(
        f"{row['file']:42s}"
        f"{row['bytes']:12,d} bytes  "
        f"{row['sha256'][:16]}..."
    )


# ============================================================
# CELL 9 — OPTIONAL KAGGLE NEW-VERSION PUBLICATION
# ============================================================

RUN_KAGGLE_UPLOAD = False

KAGGLE_HANDLE = (
    "sumit07125/geonexus-mh-v3"
)

if not RUN_KAGGLE_UPLOAD:

    print(
        "Kaggle publication is DISABLED."
    )

    print(
        "Set RUN_KAGGLE_UPLOAD = True after all "
        "220 patches are finalized."
    )

else:

    # --------------------------------------------------------
    # Exact final artifact gate
    # --------------------------------------------------------
    if not all(
        result["ok"]
        for result in results.values()
    ):

        raise RuntimeError(
            "Publication blocked: verified artifacts are incomplete."
        )

    if (
        results["mh_val"]["received"] != 30
        or results["mh_adapt"]["received"] != 30
        or results["mh_test"]["received"] != 160
    ):

        raise RuntimeError(
            "Publication blocked: expected "
            "30 MH-VAL + 30 MH-ADAPT + 160 MH-TEST."
        )

    !pip -q install kagglehub

    import kagglehub

    kagglehub.login()

    FULL_DATASET_ROOT = Path(
        "/content/geonexus_mh_v3_full"
    )

    if FULL_DATASET_ROOT.exists():
        shutil.rmtree(
            FULL_DATASET_ROOT
        )

    print(
        "Downloading the current Kaggle dataset..."
    )

    downloaded = kagglehub.dataset_download(
        KAGGLE_HANDLE,
        output_dir=str(
            FULL_DATASET_ROOT
        ),
        force_download=False,
    )

    print(
        "Downloaded to:",
        downloaded
    )

    target_verified = (
        FULL_DATASET_ROOT /
        "verified"
    )

    if target_verified.exists():
        shutil.rmtree(
            target_verified
        )

    shutil.copytree(
        VERIFIED_ROOT,
        target_verified
    )

    print(
        "\nCreating new Kaggle dataset version..."
    )

    kagglehub.dataset_upload(
        KAGGLE_HANDLE,
        str(
            FULL_DATASET_ROOT
        ),
        version_notes=(
            "Geo-Nexus v3.2 human-verified "
            "Maharashtra labels: "
            "MH-VAL 30, MH-ADAPT 30, "
            "MH-TEST 160, plus audit manifests."
        ),
    )

    print(
        "\n✓ Kaggle publication completed."
    )

# ============================================================
# CELL 10 — LOCAL SELF-TEST
# ============================================================

dummy = np.zeros(
    (128, 128),
    dtype=np.uint8,
)

dummy[10:20, 10:20] = 3
dummy[30:40, 30:40] = 255
dummy[50:60, 50:65] = 5

validated = validate_mask(dummy)

assert validated.shape == (128, 128)

rgb = colourise_mask(validated)
assert rgb.shape == (128, 128, 3)
assert rgb.dtype == np.uint8

binary = make_binary_change_target(validated)

assert binary.shape == (128, 128)
assert binary.dtype == np.uint8

# 3 + 5 = change, 0 = no change, 255 = preserved ignore.
assert int((binary == 1).sum()) == 250
assert int((binary == 0).sum()) == (128*128 - 250 - 100)
assert int((binary == 255).sum()) == 100

binary = validate_binary_target(binary)

assert set(np.unique(binary).tolist()) <= {0, 1, 255}

binary_rgb = binary_change_rgb(validated)

assert binary_rgb.shape == (128, 128, 3)
assert binary_rgb.dtype == np.uint8

# Visual convention check:
assert tuple(binary_rgb[0, 0]) == (0, 0, 0)          # no change
assert tuple(binary_rgb[10, 10]) == (255, 255, 255)  # change
assert tuple(binary_rgb[30, 30]) == (160, 160, 160)  # uncertain

assert (
    decision_key(PATCHES["MH_VAL"][0])
    == f"MH_VAL/{PATCHES['MH_VAL'][0]['stem']}"
)

print("✓ Multi-class mask validation PASS")
print("✓ Multi-class colourization PASS")
print("✓ Binary 0/1/255 conversion PASS")
print("✓ Binary uncertain=255 preservation PASS")
print("✓ Binary visualization PASS")
print("✓ Decision key PASS")
print("✓ Local self-test PASS")
