# ============ CELL 4: FUSION ENGINE + RAM-BOUNDED STREAMING HELPERS ============
import gc
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening
from scipy.ndimage import label as nd_label
from data.autolabel import (
    load_evidence, build_evidence_masks, fuse, report,
    CLASS_NAMES, IGNORE, CFG, EV, remove_small_blobs
)

# Runtime knobs: deliberately conservative so a 12.7 GiB Colab instance does not need
# large full-scene float32 intermediates.
FUSION_STRIPE = 128
FUSION_HALO = 20  # JRC buffer is 20 px; this also covers 3x3 morphology safely.

print('Successfully imported data.autolabel (Single Source of Truth):')
print('  Classes:', CLASS_NAMES)
print('  Ignore Index:', IGNORE)
print('  Fusion Parameters:', CFG)
print('  Evidence band map:', EV)
print(f'  Streaming fusion stripe={FUSION_STRIPE}px halo={FUSION_HALO}px')


def common_scene_shape(zone, period):
    ho, wo = mosaic_shape(f'{zone}_{period}_optical', DRIVE_RAW)
    hs, ws = mosaic_shape(f'{zone}_{period}_sar', DRIVE_RAW)
    return min(ho, hs), min(wo, ws)


def _normdiff_f32(a, b):
    out = np.empty(a.shape, dtype=np.float32)
    den = np.empty(a.shape, dtype=np.float32)
    np.subtract(a, b, out=out, dtype=np.float32)
    np.add(a, b, out=den, dtype=np.float32)
    den += np.float32(1e-6)
    np.divide(out, den, out=out)
    del den
    return out


def _read_fusion_inputs(zone, row0, row1):
    """Read one stripe plus halo for all inputs consumed by data.autolabel."""
    H, W = mosaic_shape(f'{zone}_T1_optical', DRIVE_RAW)
    rr0 = max(0, row0 - FUSION_HALO)
    rr1 = min(H, row1 + FUSION_HALO)
    h = rr1 - rr0
    bands_s2 = [IDX['B3'] + 1, IDX['B4'] + 1, IDX['B8'] + 1, IDX['B11'] + 1]
    a1 = read_mosaic_window(f'{zone}_T1_optical', bands_s2, rr0, 0, h, W, DRIVE_RAW).astype(np.float32)
    a2 = read_mosaic_window(f'{zone}_T2_optical', bands_s2, rr0, 0, h, W, DRIVE_RAW).astype(np.float32)
    a1 /= np.float32(S2_SCALE); a2 /= np.float32(S2_SCALE)
    n1 = _normdiff_f32(a1[2], a1[1])
    b1 = _normdiff_f32(a1[3], a1[2])
    m1 = _normdiff_f32(a1[0], a1[3])
    n2 = _normdiff_f32(a2[2], a2[1])
    b2 = _normdiff_f32(a2[3], a2[2])
    m2 = _normdiff_f32(a2[0], a2[3])
    del a1, a2

    ev_band_nums = [EV['ob_p2020']+1, EV['ob_p2023']+1, EV['hansen_ly']+1,
                    EV['dw_built_t1']+1, EV['dw_built_t2']+1,
                    EV['dw_trees_t1']+1, EV['dw_trees_t2']+1, EV['jrc_occ']+1]
    ev_raw = read_mosaic_window(f'{zone}_autolabel_evidence', ev_band_nums,
                                rr0, 0, h, W, DRIVE_RAW)
    ev = {
        'ob_p2020': ev_raw[0].astype(np.float32) / np.float32(10000.0),
        'ob_p2023': ev_raw[1].astype(np.float32) / np.float32(10000.0),
        'hansen_ly': ev_raw[2].astype(np.float32),
        'dw_built_t1': ev_raw[3].astype(np.float32) / np.float32(10000.0),
        'dw_built_t2': ev_raw[4].astype(np.float32) / np.float32(10000.0),
        'dw_trees_t1': ev_raw[5].astype(np.float32) / np.float32(10000.0),
        'dw_trees_t2': ev_raw[6].astype(np.float32) / np.float32(10000.0),
        'jrc_occ': ev_raw[7].astype(np.float32),
    }
    del ev_raw
    return rr0, rr1, (n1, n2, b1, b2, m1, m2, ev)


def _build_raw_class_masks(zone, work_dir, cfg):
    """Build raw evidence masks to disk in stripes; equations match data.autolabel exactly."""
    H, W = mosaic_shape(f'{zone}_T1_optical', DRIVE_RAW)
    work_dir = Path(work_dir); work_dir.mkdir(parents=True, exist_ok=True)
    mask_paths = {}
    for name in ['water_gain','water_loss','construction','veg_loss','veg_gain']:
        p = work_dir / f'{zone}_{name}.npy'
        mm = np.lib.format.open_memmap(p, mode='w+', dtype=np.bool_, shape=(H,W))
        mm[:] = False
        mm.flush(); del mm
        mask_paths[name] = p

    mms = {name: np.lib.format.open_memmap(path, mode='r+') for name, path in mask_paths.items()}
    try:
        for row0 in range(0, H, FUSION_STRIPE):
            row1 = min(H, row0 + FUSION_STRIPE)
            rr0, rr1, (n1,n2,b1,b2,m1,m2,ev) = _read_fusion_inputs(zone, row0, row1)
            d_ndvi = n2 - n1
            d_ndbi = b2 - b1
            e_ob = ((ev['ob_p2023'] >= cfg['ob_hi']) &
                    (ev['ob_p2020'] <= cfg['ob_lo']) &
                    ((ev['ob_p2023'] - ev['ob_p2020']) >= cfg['ob_jump']))
            e_ndbi = (d_ndbi >= cfg['tau_ndbi']) & (d_ndvi <= -cfg['ndvi_drop_min'])
            e_dw_b = (ev['dw_built_t2'] - ev['dw_built_t1']) >= cfg['dw_margin']
            construction = e_ob | (e_ndbi & e_dw_b)

            e_hansen = (ev['hansen_ly'] >= 20) & (ev['hansen_ly'] <= 23)
            e_hansen = binary_erosion(e_hansen, np.ones((3,3), dtype=bool))
            e_ndvi_l = (d_ndvi <= -cfg['tau_ndvi_loss']) & (n1 >= cfg['ndvi_t1_min'])
            e_dw_t = (ev['dw_trees_t1'] - ev['dw_trees_t2']) >= cfg['dw_margin']
            veg_loss = e_hansen | (e_ndvi_l & e_dw_t)
            veg_gain = ((d_ndvi >= cfg['tau_ndvi_gain']) &
                        (n1 <= cfg['ndvi_t1_max']) &
                        (n2 >= cfg['ndvi_t2_min']))
            w1 = m1 >= cfg['mndwi_thresh']; w2 = m2 >= cfg['mndwi_thresh']
            prior = binary_dilation(
                ev['jrc_occ'] >= cfg['jrc_occ_min'],
                np.ones((cfg['jrc_buffer_px']*2+1,)*2, dtype=bool)
            )
            water_gain = (~w1) & w2 & prior
            water_loss = w1 & (~w2) & prior

            c0, c1 = row0-rr0, row1-rr0
            mms['water_gain'][row0:row1] = water_gain[c0:c1]
            mms['water_loss'][row0:row1] = water_loss[c0:c1]
            mms['construction'][row0:row1] = construction[c0:c1]
            mms['veg_loss'][row0:row1] = veg_loss[c0:c1]
            mms['veg_gain'][row0:row1] = veg_gain[c0:c1]
            del n1,n2,b1,b2,m1,m2,ev,d_ndvi,d_ndbi,e_ob,e_ndbi,e_dw_b,construction
            del e_hansen,e_ndvi_l,e_dw_t,veg_loss,veg_gain,w1,w2,prior,water_gain,water_loss
            gc.collect()
    finally:
        for mm in mms.values():
            mm.flush(); del mm
    return mask_paths, (H,W)


def _exclusive_morphology(mask_paths, shape):
    """Apply the exact fuse() class precedence, opening, and min-blob filter on disk-backed masks."""
    H,W = shape
    claimed = np.zeros((H,W), dtype=np.bool_)
    order = [('water_gain',1), ('water_loss',2), ('construction',3), ('veg_loss',4), ('veg_gain',5)]
    structure = np.ones((3,3), dtype=bool)
    for name, _cid in order:
        mm = np.lib.format.open_memmap(mask_paths[name], mode='r+')
        opened = binary_opening(mm, structure=structure, border_value=0)
        # This calls the project's own implementation and therefore preserves its
        # default connected-component semantics (ndimage.label, min_size=4).
        cleaned = remove_small_blobs(opened, 4)
        exclusive = cleaned & ~claimed
        mm[:] = exclusive
        mm.flush()
        claimed |= exclusive
        del mm, opened, cleaned, exclusive
        gc.collect()
    return claimed


def _load_ae_score(zone):
    H,W = mosaic_shape(f'{zone}_alphaearth_change', DRIVE_RAW)
    ae = read_mosaic_window(f'{zone}_alphaearth_change', [1], 0, 0, H, W, DRIVE_RAW)[0]
    return ae.astype(np.float32) / np.float32(10000.0)


def _stream_counts_for_thresholds(claimed, ae, lo_list, hi_pct):
    tau_hi = float(np.percentile(ae, hi_pct))
    out = []
    H,W = ae.shape
    for lo in lo_list:
        tau_lo = float(np.percentile(ae, lo))
        uncertain = 0
        change = 0
        for r0 in range(0,H,FUSION_STRIPE):
            r1=min(H,r0+FUSION_STRIPE)
            a=ae[r0:r1]
            chg=a >= tau_hi
            stab=a <= tau_lo
            uncertainty = (claimed[r0:r1] & stab) | (~claimed[r0:r1] & ~chg & ~stab)
            uncertain += int(uncertainty.sum())
            change += int(chg.sum())
        denom = H*W
        out.append((lo, 100.0*uncertain/denom, 100.0*change/denom))
    return out


def streaming_fuse_zone(zone, selected_lo, save=True):
    """RAM-bounded equivalent of data.autolabel.fuse over one complete zone."""
    FINAL_CFG = {**CFG, 'ae_pct_lo': int(selected_lo)}
    work_dir = LOCAL / f'_fusion_work_{zone}'
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f'\n===== Fusing evidence layers for [{zone}] =====')
    show_ram(f'{zone} start')

    mask_paths, shape = _build_raw_class_masks(zone, work_dir, FINAL_CFG)
    H,W = shape
    ae = _load_ae_score(zone)
    show_ram(f'{zone} after disk masks + AE')
    tau_hi = float(np.percentile(ae, FINAL_CFG['ae_pct_hi']))
    tau_lo = float(np.percentile(ae, FINAL_CFG['ae_pct_lo']))
    print(f'[{zone}] global AlphaEarth thresholds: hi={tau_hi:.8f} (p{FINAL_CFG["ae_pct_hi"]}), '
          f'lo={tau_lo:.8f} (p{FINAL_CFG["ae_pct_lo"]})')

    claimed = _exclusive_morphology(mask_paths, shape)
    show_ram(f'{zone} after morphology')

    label = np.zeros((H,W), dtype=np.uint8)
    conf = np.zeros((H,W), dtype=np.float32)
    order = [('water_gain',1), ('water_loss',2), ('construction',3), ('veg_loss',4), ('veg_gain',5)]
    # Apply the exact fuse() confidence/ignore rules, but row-wise.
    for row0 in range(0,H,FUSION_STRIPE):
        row1=min(H,row0+FUSION_STRIPE)
        a=ae[row0:row1]
        ae_chg=a >= tau_hi
        ae_stab=a <= tau_lo
        for name,cid in order:
            m=np.lib.format.open_memmap(mask_paths[name], mode='r')[row0:row1]
            take=m & ae_chg
            label[row0:row1][take] = cid
            conf[row0:row1][take] = 1.00
            take=m & ~ae_chg & ~ae_stab
            conf[row0:row1][take] = 0.60
            take=m & ae_stab
            label[row0:row1][take] = IGNORE
            del m, take
        unclaimed=~claimed[row0:row1]
        take=unclaimed & ae_chg
        label[row0:row1][take]=6; conf[row0:row1][take]=0.50
        take=unclaimed & ae_stab
        label[row0:row1][take]=0; conf[row0:row1][take]=1.00
        take=unclaimed & ~ae_chg & ~ae_stab
        label[row0:row1][take]=IGNORE
        conf[row0:row1][take]=0.00
        del a, ae_chg, ae_stab, unclaimed, take

    rep = report(label, zone)
    assert 8.0 <= rep['uncertain'] <= 40.0, f'[{zone}] uncertain {rep["uncertain"]}% outside [8,40].'
    if zone == 'pune':
        assert rep['construction'] >= 1.0, f'Zone A construction ({rep["construction"]}%) < 1.0%.'

    if save:
        np.savez_compressed(LOCAL / f'{zone}_autolabel_full.npz',
                            label=label, conf=(conf*100).astype(np.uint8))
        print(f'[{zone}] saved successfully.')

    # Keep only final outputs; work masks are temporary implementation artifacts.
    del ae, claimed, label, conf
    shutil.rmtree(work_dir, ignore_errors=True)
    gc.collect()
    show_ram(f'{zone} freed')
    return rep
