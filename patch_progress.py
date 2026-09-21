import re
from pathlib import Path

path = Path(r"c:\Users\sumit_mali\Desktop\Geo_Watch\progress.md")
content = path.read_text(encoding="utf-8")

# Update 10.2 DAPT Training
new_dapt = """### 10.2 DAPT Training (READY 🚀)
With the dataset fully processed, patched, and staged, the project has advanced to **Phase P3: Multi-modal Domain-Adaptive Pretraining (DAPT)** using the Decoupled Common & Unique Representations (DeCUR) architecture.
* **Execution Notebook:** [`notebooks/geonexus-dapt-final-6664-persistent-artifacts.ipynb`](file:///c:/Users/sumit_mali/Desktop/Geo_Watch/notebooks/geonexus-dapt-final-6664-persistent-artifacts.ipynb)
* **Architecture Compliance:** Fully conforms to Phase P3 specifications (5-epoch linear warmup + cosine decay, stem cosine similarity > 0.98, backbone LR `3e-5`, head LR `3e-4`).
* **Persistent Checkpoint & Artifact Workflow:** The DAPT notebook now features an advanced Kaggle artifact publisher:
  - Overwrites the latest training state after **every single completed epoch** (removing the old 10-epoch boundary).
  - Automatically publishes a private Kaggle Dataset (`sumit07125/geonexus-v3-2-dapt-6664-artifacts`) at the end of each session.
  - Automatically resumes from the highest compatible epoch by attaching the published artifact dataset.
  - Generates comprehensive live timing, remaining budget guardrails, diagnostic loss curves, and post-DAPT t-SNE clustering graphs.
* **Next Steps:** Mount `geonexus-mh-v3` and `ssl4eo-weights` datasets into Kaggle, run the stem-surgery and initialization verification, and execute the 100-epoch training schedule across sessions."""

content = re.sub(
    r"### 10\.2 DAPT Training \(READY 🚀\).*?(?=\n\n---|\Z)",
    new_dapt,
    content,
    flags=re.DOTALL
)

# Add Phase P4
phase_4 = """

---

## 12. Phase P4: Human Verification & Public Dataset Release

The Geo-Nexus project's public Maharashtra dataset verification and packaging pipeline is officially complete, comprising the exact 220-patch QGIS target pool defined in the architecture.

### 12.1 Interactive Human Verification Interface (COMPLETE ✅)
* **Execution Notebook:** [`notebooks/GeoNexus_MH_Human_Verification_v3_2_FINAL_WITH_BINARY_STORAGE.ipynb`](file:///c:/Users/sumit_mali/Desktop/Geo_Watch/notebooks/GeoNexus_MH_Human_Verification_v3_2_FINAL_WITH_BINARY_STORAGE.ipynb)
* **Patch Inventory Gate:** Strictly asserts the expected 220 patches across 7 predefined splits (MH_VAL: 30, MH_ADAPT: 30, MH_TEST_DRY_A: 40, MH_TEST_DRY_B: 40, MH_TEST_BLIND: 20, MH_TEST_MONSOON: 30, MH_TEST_VIDARBHA: 30).
* **Binary Target Logic:** Validated the derivation of binary labels from multi-class (1-6 $\to$ 1, 0 $\to$ 0). Crucially, the uncertain/ignore class (`255`) is **preserved exactly as `255`** in the binary masks to prevent unfair penalties during supervised training.
* **Blind-Set Protection:** Hard-coded security gates disable the `ACCEPT_AUTO` button for blind and monsoon sets, enforcing independent verification and preventing data leakage.

### 12.2 Public 170-Reviewed Weak Labels Release (COMPLETE ✅)
* **Execution Notebook:** [`GeoNexus_MH_Public_170_Verified_Final.ipynb`](file:///c:/Users/sumit_mali/Desktop/Geo_Watch/GeoNexus_MH_Public_170_Verified_Final.ipynb)
* **Scientific Transparency:** The release explicitly packages exactly **170 human-reviewed automatic labels**. The 50 patches belonging to the blind and monsoon test sets were actively excluded from this artifact since they were not independently redrawn by a human. This honest separation prevents downstream users from claiming false superiority on the full test sets.
* **Documentation Engine:** Dynamically generates robust Kaggle dataset metadata (`README.md` and `DATA_SOURCES.md`), documenting the 17-channel input composition, Open Buildings temporal limits (stopping at 2023), and the correct "other" Kaggle license type reflecting mixed Copernicus/Google/JRC attribution.
* **Kaggle Synchronization:** Performed shape/content self-tests and successfully pushed the newly verified Numpy arrays (`mh_val`, `mh_adapt`, and `mh_test_partial`) to Kaggle (`sumit07125/geonexus-mh-v3`).
"""

if "## 12. Phase P4: Human Verification & Public Dataset Release" not in content:
    content += phase_4

path.write_text(content, encoding="utf-8")
