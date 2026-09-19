# Geo-Nexus (MH-DAPT-CD v3.1) — Project Progress & Engineering Log

**Document Purpose:** Comprehensive record of all architecture decisions, data acquisition, radiometric verification, bug resolutions, and dataset specifications completed to date.  
**Reference Architecture:** [`FINAL_ARCH_3.md`](file:///c:/Users/sumit_mali/Desktop/Geo_Watch/FINAL_ARCH_3.md) (v3.1 Frozen Architecture)  
**Last Updated:** 2026-09-16 (Late Night Session)  
**Status:** Phase P0 (Radiometric Verification) & Phase P1 (GEE Export Queue) **Completed / In Flight**.

---

## 1. Project Overview & Architectural Foundation

The **Geo-Nexus / MH-DAPT-CD** project (*Maharashtra Domain-Adaptive Pretraining for Change Detection*) investigates multi-modal (Optical Sentinel-2 + SAR Sentinel-1) bi-temporal change detection over the state of Maharashtra, India (2020 vs 2024).

The core scientific pipeline consists of:
1. **Pretrained Foundation Initialization:** ResNet-18 encoders initialized from **SSL4EO-S12** (MoCo-v2 pretraining on 1M global Sentinel-1/2 pairs).
2. **Domain-Adaptive Pretraining (DAPT):** Unsupervised continued pretraining on multi-modal Maharashtra imagery using **DeCUR** (Decoupled Common and Unique Representations), separating common optical/radar features from unique SAR cloud-penetration signals.
3. **Supervised Transfer Learning:** Supervised change decoder (Large Kernel Attention - LKA) trained exclusively on the **OSCD** (Onera Satellite Change Detection) dataset in optical-only mode ($g = 1.0$).
4. **Zero-Shot & Few-Shot Evaluation:** Cross-domain transfer of the OSCD-trained change decoder to Maharashtra (P-ZS zero-shot and P-FS few-shot with $k \in \{5, 10, 20, 30\}$ labels), with rigorous evaluation on clear and cloudy (monsoon) benchmarks.

---

## 2. Satellite Data Acquisition Specifications (Google Earth Engine)

### 2.1 Spatial Study Zones & Geographic Coordinate Reference Systems

To ensure robust geographic generalizability without spatial autocorrelation leakage, three distinct zones were established with strict CRS definitions:

| Zone Identifier | Location / Purpose | Bounding Box (WGS84 [minLon, minLat, maxLon, maxLat]) | Projected CRS | Relative SAR Orbit | Pass Direction |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Zone A** | Pune (Urbanization / Suburban Sprawl) | `[73.70, 18.30, 74.20, 18.80]` (~55 × 55 km) | **EPSG:32643** (UTM 43N) | **136** | Descending |
| **Zone B** | Satara (Agricultural / Reservoir Dynamics) | `[73.50, 17.50, 74.00, 18.00]` (~55 × 55 km) | **EPSG:32643** (UTM 43N) | **136** | Descending |
| **Zone C** | Vidarbha (Never-trained Geographic Holdout) | `[78.90, 20.90, 79.30, 21.30]` (~44 × 44 km) | **EPSG:32644** (UTM 44N) | **165** | Descending |

> **Critical Coordinate Engineering:** Zone C is situated at $79^\circ\text{ E}$, placing it in **UTM Zone 44N (`EPSG:32644`)**, whereas Zones A and B ($73.5^\circ - 74.2^\circ\text{ E}$) are in **UTM Zone 43N (`EPSG:32643`)**. Reprojecting Zone C to UTM 44N eliminates grid distortion and scale errors.

---

### 2.2 Temporal Filtering & The Earth Engine Half-Open Interval Fix

The target dry-season quarters were defined as:
* **T1 Baseline:** Early 2020 Dry Season
* **T2 Comparison:** Early 2024 Dry Season

#### The Date Boundary Fix
* **Initial Filter:** `filterDate('2020-01-01', '2020-03-31')` and `filterDate('2024-01-01', '2024-03-31')`
* **Defect Identified:** Google Earth Engine evaluates dates as half-open intervals:
  $$\text{filterDate}(t_{\text{start}}, t_{\text{end}}) \equiv [t_{\text{start}}, t_{\text{end}})$$
  Because Sentinel-2 overpasses India at ~05:30 UTC, setting $t_{\text{end}}$ to `03-31` (which resolves to `03-31 00:00:00 UTC`) strictly excluded all acquisitions on March 31.
* **Resolved Filter:** Updated to **`'2020-01-01'` $\to$ `'2020-04-01'`** and **`'2024-01-01'` $\to$ `'2024-04-01'`**.
* **Direct Empirical Result:** In Zone C, S1 acquisitions increased from $11 \to 13$, and S2 acquisitions increased from $92 \to 94$, successfully capturing end-of-quarter passes.

#### Verified Scene Counts

| Zone | T1 S2 Optical (Jan–Mar 2020) | T2 S2 Optical (Jan–Mar 2024) | T1 S1 SAR (Orbit locked) | T2 S1 SAR (Orbit locked) |
| :---: | :---: | :---: | :---: | :---: |
| **Zone A** (Pune) | 38 scenes | 36 scenes | 10 passes (Orbit 136) | 5 passes (Orbit 136) |
| **Zone B** (Satara) | 36 scenes | 36 scenes | 14 passes (Orbit 136) | 10 passes (Orbit 136) |
| **Zone C** (Vidarbha) | 102 scenes | 94 scenes | 13 passes (Orbit 165) | 12 passes (Orbit 165) |

---

### 2.3 Radiometric & Optical Band Composition

1. **Collection Specification:** `COPERNICUS/S2_SR_HARMONIZED` (Bottom-of-Atmosphere Level-2A surface reflectance with ESA Processing Baseline $\ge 04.00$ $+1000\text{ DN}$ offset removed).
2. **11 Ingested Spectral Bands:**  
   `['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']`  
   *(B1 60m coastal and B10 cirrus are excluded; B9 940nm water vapour is preserved).*
3. **Cloud & Shadow Masking:** Google Cloud Score+ (`GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`), applying a strict quality gate threshold:
   $$\text{Mask} = \text{cs\_cdf} \ge 0.60$$
4. **Observation Density Quality Band ($n\_clear$):**  
   Rather than collapsing composites into binary cloud-free masks (which deletes uncertainty information), the number of valid clear observations per pixel is recorded as band 12 (`n_clear`).
5. **Compositing & Scaling:** Temporal median over unmasked pixels, scaled by $\times 10000$ to signed 16-bit integers (`Int16`).

---

### 2.4 SAR Radar Processing Pipeline (Defect D5 Fix)

1. **Collection:** `COPERNICUS/S1_GRD` (Interferometric Wide swath, 10m pixel spacing).
2. **Polarization:** Dual polarization (`VV`, `VH`).
3. **Geometry Locking:** Fixed single relative orbit per zone (136 for A & B, 165 for C, all descending pass) to guarantee identical radar incidence angles and eliminate false change caused by topographic slope variation.
4. **Linear-Power Averaging:** Averaging SAR scenes in logarithmic dB space introduces negative bias. The correct physics pipeline was implemented:
   $$\text{SAR}_{\text{composite}} = 10 \cdot \log_{10}\left( \text{median}\left(10^{\text{dB} / 10}\right) \right)$$
5. **Adaptive Speckle Mitigation:** Spatial focal median filtering ($3 \times 3$, 30m) is applied **only if** the temporal stack contains $< 5$ looks ($N < 5$), preserving narrow 1–3 pixel road structures in high-look temporal medians.
6. **SAR Scaling:** $\text{dB} \times 100 \to \text{Int16}$ (maps $[-35, +5]\text{ dB}$ safely to $[-3500, +500]$, avoiding integer overflow).

---

### 2.5 Auxiliary Export Layers

* **AlphaEarth Annual Embedding Change Score:** Google Satellite Embedding annual mosaics (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`) for 2020 and 2024. Because embeddings are unit-length normalized, the cosine change metric is:
  $$\text{change\_score} = 1 - \sum_{i=1}^{64} (e_{2020, i} \cdot e_{2024, i}) \in [0, 2]$$
  Exported as $\text{Int16}$ scaled by $\times 10000$. Used for baseline benchmarking and active stratified sampling of annotations.
* **JRC Global Surface Water Mask:** Seasonality band from `JRC/GSW1_4/GlobalSurfaceWater` ($\ge 1 \implies \text{permanent or seasonal water}$) exported as `Byte` (`uint8`) to prevent false-positive change detection along oscillating reservoir margins in Zone B.

---

### 2.6 Real Monsoon Cloudy Benchmark (Zone A)

Designed specifically to stress-test **Hypothesis H2** (SAR rescuing predictions when optical is cloud-obscured):
* **Single-Date Optical:** S2 scene with $30\% - 70\%$ real cloud cover (NOT a composite):
  * **T1:** `20200911T052651_20200911T053945_T43QCA` (41.2% scene cloud)
  * **T2:** `20240920T052651_20240920T053347_T43QCA` (38.0% scene cloud)
* **Quality Signal ($q255$):** Unthresholded continuous Cloud Score+ cumulative distribution function:
  $$q255 = \text{cs\_cdf} \times 255 \to \text{Int16}$$
* **Nearest Single-Date SAR Acquisition:** Selected strictly as the single nearest acquisition on locked Orbit 136 within $\pm 14$ days:
  * **T1 SAR:** `S1A_IW_GRDH_1SDV_20200913T005513...` (2020-09-13 00:55:13 UTC, $\Delta t = 1.8$ days)
  * **T2 SAR:** `S1A_IW_GRDH_1SDV_20240916T005528...` (2024-09-16 00:55:28 UTC, $\Delta t = 4.2$ days)

---

## 3. Phase P0 Radiometric Verification (Pass/Fail Results)

Executed via [`data/gee/00_verify_harmonization.js`](file:///c:/Users/sumit_mali/Desktop/Geo_Watch/data/gee/00_verify_harmonization.js):

### Test 0: Same-Scene Baseline Harmonization Proof (Defect D1 Verification)
* **Method:** Retreived the exact same ESA Baseline $\ge 04.00$ scene asset (`20240104T053221_20240104T053220_T43QCA`) from `COPERNICUS/S2_SR` and `COPERNICUS/S2_SR_HARMONIZED`, computing the direct subtraction across Pseudo-Invariant Features (PIFs).
* **Observed Offset across all 11 bands:**  
  $$\Delta \rho = 0.100000000000 \pm 0.000000$$
* **Verdict:** **PASS ✅**. Mathematically proves that the $+1000\text{ DN}$ shift is completely eliminated by the harmonized collection without conflating natural 4-year weathering.

### Test 1: 4-Year Multi-Date Temporal Stability Diagnostic
* **Method:** Measured absolute reflectance drift $|\rho_{2024} - \rho_{2020}|$ over verified invariant bare basalt plateau rocks (Sinhagad) and stable warehouse roofs (Chakan).
* **Results:** Maximum observed difference is $\le 0.025$ in visible bands (B3/B4), and $\le 0.008$ in NIR/SWIR bands.
* **Verdict:** **DIAGNOSTIC PASSED ✅**. Natural environmental drift is minimal and stable.

### Test 2: Stable Evergreen Forest Phenology ($\Delta\text{NDVI}$)
* **Method:** Evaluated mean NDVI difference over protected evergreen forest ridge in Sinhagad ($500\text{ m}$ buffer).
* **Result:** $\Delta\text{NDVI} = -0.02220038$
* **Criterion:** $|\Delta\text{NDVI}| < 0.05$
* **Verdict:** **PASS ✅**. Confirms no false phenological degradation between dry seasons.

### Test 3: Quality Band Spatial Heterogeneity (Defect D3 Verification)
* **Method:** Evaluated spatial distribution of $n\_clear$ across Zone A.
* **Results:** Mean $= 19.79$, Min $= 12$, Max $= 38$, $\sigma = \mathbf{6.6898}$.
* **Criterion:** $\sigma > 0.5$
* **Verdict:** **PASS ✅**. Clear looks vary significantly across the landscape, proving the quality signal is non-degenerate.

---

## 4. Engineering Bug Resolution: GeoTIFF Export Data Type Inconsistency

### The Bug (Error Code 3)
When the initial export tasks for `pune_T1_optical` and `pune_T2_optical` were submitted, they crashed within 1–2 seconds:
```text
Error: Exported bands must have compatible data types; found inconsistent types: Int16 and Byte. (Error code: 3)
```

### Root Cause
Under the TIFF 6.0 specification and GDAL `GTiff` driver used by Earth Engine, all bands in a single multi-band raster container must share the exact same `GDALDataType`. In `01_export_zone.js`, the 11 optical bands were cast to `Int16` while `n_clear` was cast to `Byte` (`toUint8()`).

### The Fix
In both [`data/gee/01_export_zone.js`](file:///c:/Users/sumit_mali/Desktop/Geo_Watch/data/gee/01_export_zone.js) and [`data/gee/02_export_monsoon.js`](file:///c:/Users/sumit_mali/Desktop/Geo_Watch/data/gee/02_export_monsoon.js):
* `n_clear.toUint8()` was changed to `n_clear.toInt16()`.
* `cs.select('cs_cdf').multiply(255).toUint8()` was changed to `...toInt16()`.

Because observation counts ($12 \dots 38$) and $q255$ ($0 \dots 255$) are small positive integers, casting them to `Int16` ($[-32768, +32767]$) incurs **zero loss of precision, zero truncation risk, and negligible compressed disk overhead ($< 2\text{ MB}$)**, while creating a 100% homogeneous 12-band GeoTIFF container.

---

## 5. Current Execution Status: 22 of 22 Tasks Completed (100% Main GEE Export Complete)

All **22 submitted tasks are 100% Completed** and saved to the Google Drive folder `geonexus_v3_raw/`.

### Verified Task Execution Table:

| Task Name | Phase | Runtime | Batch Compute Usage (EECU-s) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **`pune_T1_optical`** | Completed | 5m | 3,929.23 EECU-s | ✅ In Drive |
| **`pune_T2_optical`** | Completed | 11m | 5,898.36 EECU-s | ✅ In Drive |
| **`pune_T1_sar`** | Completed | 18m | 595.92 EECU-s | ✅ In Drive (ID: 3V4NTZ4NO3SPSZKCJS4I4BUQ) |
| **`pune_T2_sar`** | Completed | 5m | 307.98 EECU-s | ✅ In Drive |
| **`pune_watermask`** | Completed | 8m | 34.11 EECU-s | ✅ In Drive |
| **`pune_alphaearth_change`** | Completed | 6m | 2,086.47 EECU-s | ✅ In Drive |
| **`satara_T1_sar`** | Completed | 6m | 816.11 EECU-s | ✅ In Drive |
| **`satara_T1_optical`** | Completed | 21m | 5,136.24 EECU-s | ✅ In Drive |
| **`satara_T2_optical`** | Completed | 23m | 5,566.09 EECU-s | ✅ In Drive |
| **`satara_alphaearth_change`** | Completed | 4m | 2,361.90 EECU-s | ✅ In Drive |
| **`satara_watermask`** | Completed | 8m | 60.97 EECU-s | ✅ In Drive |
| **`satara_T2_sar`** | Completed | 6m | 455.52 EECU-s | ✅ In Drive |
| **`vidarbha_T1_optical`** | Completed | 9m | 6,183.00 EECU-s | ✅ In Drive |
| **`vidarbha_T1_sar`** | Completed | 6m | 456.77 EECU-s | ✅ In Drive |
| **`vidarbha_T2_optical`** | Completed | 11m | 5,767.06 EECU-s | ✅ In Drive |
| **`vidarbha_alphaearth_change`** | Completed | 10m | 2,907.72 EECU-s | ✅ In Drive |
| **`vidarbha_T2_sar`** | Completed | 6m | 293.79 EECU-s | ✅ In Drive |
| **`vidarbha_watermask`** | Completed | 51s | 30.84 EECU-s | ✅ In Drive |
| **`pune_monsoon_T1_optical`** | Completed | 21m | 239.23 EECU-s | ✅ In Drive |
| **`pune_monsoon_T1_sar`** | Completed | 5m | 87.37 EECU-s | ✅ In Drive |
| **`pune_monsoon_T2_optical`** | Completed | 7m | 251.17 EECU-s | ✅ In Drive |
| **`pune_monsoon_T2_sar`** | Completed | 4m | 85.31 EECU-s | ✅ In Drive |

* **Total Compute Consumed Across All 22 Tasks:** **~37,870 EECU-seconds**
* **Total Quota Allowance:** **540,000 EECU-seconds**
* **Total Quota Used:** **~7.01%** *(Over 92% of your allowance is untouched!)*

---

## 6. External Benchmark & Pretrained Model Architecture


### 6.2 Pretrained Foundation Weights (v3.2 Weight-Source Amendment)
* **Script:** [`data/download_pretrained_weights_v3_2_amended.py`](file:///c:/Users/sumit_mali/Desktop/Geo_Watch/data/download_pretrained_weights_v3_2_amended.py)
* **Optical Backbone:** `resnet18_s2c_moco.pth` (13 input channels, 44.91 MB) from SSL4EO-S12 MoCo-v2 (`torchgeo/resnet18_sentinel2_all_moco`).
* **SAR Backbone (Explicit Amendment):** `resnet18_s1_bigearthnet.pth` (2 input channels, 44.81 MB) from BigEarthNet v2.0 Sentinel-1 ResNet-18 (`BIFOLD-BigEarthNetv2-0/resnet18-s1-v0.2.0`).
  - *Rationale:* The official SSL4EO-S12 release trained S1 MoCo only for ResNet-50 and ViT. BigEarthNet v2.0 provides the official, peer-reviewed 2-band ResNet-18 classifier trunk for SAR.
  - The downloader verifies the exact ResNet-18 trunk structure (`conv1` shape `(64, 2, 7, 7)` and all 4 layer shapes) and strips the classifier head.
* **Stem Surgery Gate:** [`models/stem.py`](file:///c:/Users/sumit_mali/Desktop/Geo_Watch/models/stem.py) verifies cosine similarity $> 0.98$ on the optical S2 conv1 weight before proceeding.
* **Execution Status: 100% COMPLETE & VERIFIED ✅:**
  - `python data/download_pretrained_weights_v3_2_amended.py --out data/weights`:
    - `resnet18_s2c_moco.pth`: shape `(64, 13, 7, 7)`, 44.91 MB, SHA256 `68cd481c23012963`
    - `resnet18_s1_bigearthnet.pth`: shape `(64, 2, 7, 7)`, 44.81 MB, SHA256 `8d89fb450c408352`
    - `pretrained_weights_manifest.json`: saved and documented
    - Stem surgery gate: **Cosine Similarity = 0.9957** ($> 0.98$ threshold) -> **PASS ✅**.
* **Kaggle Staging & Notebooks Prepared:**
  - [`data/stage_kaggle.py`](file:///c:/Users/sumit_mali/Desktop/Geo_Watch/data/stage_kaggle.py): per-dataset legal licenses mapped (`other` for mixed weights, `CC-BY-NC-SA-4.0` for OSCD, `CC-BY-4.0` for Maharashtra).
  - [`notebooks/02_dapt.ipynb`](file:///c:/Users/sumit_mali/Desktop/Geo_Watch/notebooks/02_dapt.ipynb): updated to load `resnet18_s1_bigearthnet.pth`.

### 6.3 Phase P1b: GEE Auto-Label Evidence Verification (Zones A, B, C)
* **Script:** [`data/gee/03_export_autolabels.js`](file:///c:/Users/sumit_mali/Desktop/Geo_Watch/data/gee/03_export_autolabels.js)
* **10-Band Int16 Container Design:**
  - `ob_p2020`, `ob_p2023`: Open Buildings temporal presence ($\times 10000$, range $[0, 10000]$).
  - `ob_h2023`: Open Buildings building height in meters ($\times 100$, range $[0, 10000]$).
  - `hansen_ly`: Hansen Global Forest Change v1.12 lossyear ($0 \dots 24$, unmasked to 0).
  - `hansen_tc00`: Hansen year-2000 tree canopy cover percentage ($0 \dots 100$, unmasked to 0).
  - `dw_built_t1`, `dw_built_t2`: Dynamic World built probability ($\times 10000$, range $[0, 10000]$).
  - `dw_trees_t1`, `dw_trees_t2`: Dynamic World tree probability ($\times 10000$, range $[0, 10000]$).
  - `jrc_occ`: JRC Global Surface Water occurrence percentage ($0 \dots 100$, unmasked to 0).
* **Computational Engineering Resolution (The $2^{31}$ Pixel Boundary Fix):**
  - Evaluating `reduceResolution` on a whole-AOI $55 \times 55\text{ km}$ mosaic at $0.5\text{ m}$ forces Earth Engine to instantiate $1.21 \times 10^{10}\text{ pixels} > 2^{31}-1$, triggering fatal user memory limit errors.
  - Resolved by applying `reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 4096})` and `.reproject()` at the individual $2048 \times 2048$ tile level prior to `.mosaic()`. Tile-level mean reduction followed by mosaicking serves as a computationally bounded implementation of the intended 10 m mean aggregation.
* **Empirical 5 km Sample Statistics (Centroid Windows):**
  - **Zone A (Pune Urban Center Centroid):**
    - Dynamic World built: $0.4333 \to 0.4991$; trees: $0.0702 \to 0.0599$.
    - Open Buildings presence: $0.1324 \to 0.1446$ (max $0.9957$).
    - Open Buildings height: mean $2.46\text{ m}$, max $97.8\text{ m}$ ($9780 / 100$).
    - Hansen lossyear: 0 valid loss pixels in the 5 km sample. Low sampled 2000 tree cover ($0.15\%$) is consistent with predominantly non-forest urban land in this sampled window, but does not characterize the peripheral hill tracts of the full AOI.
  - **Zone B (Satara Centroid):**
    - Dynamic World built: $0.0541 \to 0.0670$; trees: $0.2703 \to 0.2743$.
    - Open Buildings presence: $0.0058 \to 0.0063$ (max $0.9639$).
    - Open Buildings height: mean $0.04\text{ m}$, max $9.55\text{ m}$.
    - Hansen lossyear: 4 valid pixels in sample with $\text{lossyear} = 14$ (2014, pre-dating the 2020–2024 monitoring window).
  - **Zone C (Nagpur Centroid):**
    - Dynamic World built: $0.6549 \to 0.6331$; trees: $0.0444 \to 0.0455$.
    - Open Buildings presence: $0.2596 \to 0.2611$ (max $0.9911$).
    - Open Buildings height: mean $2.45\text{ m}$, max $67.14\text{ m}$.
    - Hansen lossyear: 0 valid loss pixels in sample ($tc00 = 0.03\%$).
* **Sanity Check & Cloud Export Execution:**
  - `satara_autolabel_evidence`: **Completed** in 10m (11,261.75 EECU-s, ID: `ITWS7FTD4T7FNGKNQ5QJU6V6`) ✅ In Drive
  - `vidarbha_autolabel_evidence`: **Completed** in 25m (10,872.63 EECU-s, ID: `KBEOF7LSDY36MW3W5EHW4K4Q`) ✅ In Drive
  - `pune_autolabel_evidence`: **Completed** and verified in Drive ✅
* **Preprocessing Notebook & Engine Assembled & Audited:**
  - [`data/autolabel.py`](file:///c:/Users/sumit_mali/Desktop/Geo_Watch/data/autolabel.py): Multi-modal evidence fusion engine (Part 27.3), with active MMU 4-pixel connected component filtering (`remove_small_blobs`).
  - [`data/export_qgis.py`](file:///c:/Users/sumit_mali/Desktop/Geo_Watch/data/export_qgis.py): QGIS patch triplet & QML styling generator (Part 27.5). Anchoring bias strictly controlled: `blind=True` returns immediately after exporting the 8-band optical imagery, omitting both auto-label and AlphaEarth rasters.
  - [`notebooks/01_preprocess.ipynb`](file:///c:/Users/sumit_mali/Desktop/Geo_Watch/notebooks/01_preprocess.ipynb): Complete Colab notebook for Phase P2, updated to:
    1. Import `data.autolabel` as single source of truth (with verified fallback).
    2. Enforce strict `RuntimeError` failure on Zone A AlphaEarth percentile sweep (no silent fallback).
    3. Implement exact 220-patch allocation matrix (Part 26.2: 30 val, 30 adapt, 40 dry A, 40 dry B, 20 blind, 30 monsoon, 30 Vidarbha).
    4. **Independent Monsoon Sampling & Hard 10/10/10 Gate:** `build_monsoon_pool_sample` builds candidate grid patches directly from the `pune_monsoon_T2_optical` raster across the Pune Test AOI grid, completely independent of dry `pune_test_meta.json` and dry auto-labels. Enforces hard scientific assertions: exactly 10 patches per stratum ($q \le 0.40$ heavy, $0.40 < q \le 0.80$ moderate, $q > 0.80$ clear) per Line 366. The generic `fill` fallback has been completely removed in favor of strict candidate assertions (`assert len(idx) >= 10` and `assert counts == {'heavy_cloud': 10, 'moderate_cloud': 10, 'clear_sky': 10}`).
    5. **Optimized I/O & Memory Cache:** Memoized `get_ae_chg(zone)` eliminates redundant AlphaEarth whole-scene reads within Cell 9. Full-scene exports in Cell 9 are streamed zone-by-zone with explicit memory clearing (`gc.collect()`), keeping single-zone memory footprint well within available Colab RAM.
    6. **Production-Grade Streaming Memory Architecture:** 
       - **Cell 7:** Sequential T1/T2 loading with `quantize_i16_inplace()` ensures only one float32 scene is in RAM at a time. Replaced in-memory array allocation with `np.lib.format.open_memmap()`, streaming patches directly to disk `.npy` files. Each zone is flushed and freed before the next begins.
       - **Cell 8:** In-place chunked reduction (`CHUNK_SIZE = 128` $\approx 272\text{ MiB}$ working buffer) directly reduces along axes `(0, 1, 3, 4)` without `.transpose()` or `.reshape()` memory copies, reusing the buffer in-place with `np.square(out=chunk)`. **Critical Scaling Resolution:** Restores SAR dB scale (`chunk[:, :, ch] *= 100.0` for channels 13–15) prior to computing $\mu$ and $\sigma$, ensuring normalization statistics match the true dB model input scale restored by the `MHPatches` loader (eliminating a 100x variance mismatch).
       - **Live RAM Telemetry:** Added `show_ram()` tracking process RSS and system available memory across all major pipeline checkpoints.
    7. Staging preparation with compliant `other` license metadata documenting mixed upstream open access sources.

---

## 7. Current Project Phase Status

```text
 [P0] Radiometric Verification        ████████████████████  DONE   PASS (Offset = 0.1000, Forest dNDVI = -0.022)
 [P1] GEE Satellite Export (Dry/Mon)  ████████████████████  DONE   22/22 in Drive geonexus_v3_raw/ (100%)
 [P1b] GEE Auto-Label Evidence Export ████████████████████  DONE   A, B, C TIFFs in Drive (100%)
 [P1c] OSCD Verification & Patches    ████████████████████  DONE   PASS (860 train / 100 val / 146 test)
 [P1d] Foundation Weights Verified    ████████████████████  DONE   PASS (Cosine = 0.9957)
 [Staging] ssl4eo-weights to Kaggle   ████████████████████  DONE   sumit07125/ssl4eo-weights (Mounted)
 [P2] Zone A Sweep & Gate Check       ████████████████████  DONE   PASS (ae_pct_lo = 65: uncertain 26.4%, change 8.6%)
 [P2] Multi-Zone Fusion & Tiling      ░░░░░░░░░░░░░░░░░░░░  READY TO RUN (Cells 6–10 in Colab)
```

---

## 8. Empirical Zone A Percentile Sweep & Gate Results (Colab Verified)

Executing [`notebooks/01_preprocess.ipynb`](file:///c:/Users/sumit_mali/Desktop/Geo_Watch/notebooks/01_preprocess.ipynb) in Google Colab yielded real empirical results:

### 8.1 P0 In-Flight Assertions (Cell 3)
* **Pune:** median band difference $|\Delta \rho| = 0.0030 \ll 0.05$; $q$ mean $0.989$ (std $0.106$); VV $-10.2\text{ dB}$, VH $-17.9\text{ dB}$ $\implies$ **ALL PASS** ✅
* **Satara:** median band difference $|\Delta \rho| = 0.0020 \ll 0.05$; $q$ mean $0.987$ (std $0.114$); VV $-10.6\text{ dB}$, VH $-17.5\text{ dB}$ $\implies$ **ALL PASS** ✅
* **Vidarbha:** median band difference $|\Delta \rho| = 0.0029 \ll 0.05$; $q$ mean $0.976$ (std $0.152$); VV $-9.9\text{ dB}$, VH $-17.1\text{ dB}$ $\implies$ **ALL PASS** ✅

### 8.2 Zone A AlphaEarth Sweep Gate (Cell 5)
Scientific Target Window: `uncertain` in $[15\%, 30\%]$, `change` in $[8\%, 20\%]$.

```text
ae_pct_lo=45: uncertain= 46.2%  change=  8.7%  [OUT OF BOUNDS]
ae_pct_lo=50: uncertain= 41.2%  change=  8.7%  [OUT OF BOUNDS]
ae_pct_lo=55: uncertain= 36.2%  change=  8.6%  [OUT OF BOUNDS]
ae_pct_lo=60: uncertain= 31.4%  change=  8.6%  [OUT OF BOUNDS]
ae_pct_lo=65: uncertain= 26.4%  change=  8.6%  [VALID (PASS)]
```

* **Outcome:** Exactly one candidate satisfies the predefined scientific gate: **`ae_pct_lo = 65`**.
* **Gate Status:** **PASSED**. `SELECTED_LO = 65` is permanently frozen and will be applied identically across Pune, Satara, and Vidarbha.
* **Mechanism:** As `ae_pct_lo` rises from 45 to 65, the ambiguous gap $[\tau_{\text{lo}}, \tau_{\text{hi}}]$ narrows from $[45\%, 92\%] \to [65\%, 92\%]$, reducing uncertain pixels from $46.2\% \to 26.4\%$, while change remains rock-solid at $8.6\%$.

---

## 9. Next Step: Execute Cells 6 through 10 in Google Colab
With `SELECTED_LO = 65` active in Colab memory:
1. Run **Cell 6**: Multi-Zone Auto-Label Generation (creates `*_autolabel_full.npz` and `autolabel_report.json`).
2. Run **Cell 7**: Spatial AOI Splitting & 128x128 Tiling with 128px hard buffer (creates `*_train.npy`, `*_test.npy`, `_label.npy`).
3. Run **Cell 8**: Channel Normalization Statistics on **TRAIN AOI ONLY** (creates `norm_stats_trainonly.json`).
4. Run **Cell 9**: Independent 220-Patch Annotation Pool & High-Performance QGIS Export (preloads scenes into RAM, exports packages to `/content/proc/qgis/`).
5. Run **Cell 10**: Create `geonexus_v3_processed.tar` archive to Google Drive and configure `dataset-metadata.json`.


