#!/usr/bin/env python3
"""Freeze the 11/3/10 split with hard disjointness assertions.
Per FINAL_ARCH_3.md v3.2 (Part 28.3).
"""
import json
from pathlib import Path

# v3.2 CORRECTION: montpellier is an OFFICIAL TEST city. v3.1 wrongly put it
# in validation. beirut replaces it -- see Part 21.1 for the selection logic.
OSCD_VAL = ['rennes', 'saclay_e', 'beirut']

TRAIN_14 = ['abudhabi','aguasclaras','beihai','beirut','bercy','bordeaux',
            'cupertino','hongkong','mumbai','nantes','paris','pisa','rennes','saclay_e']
TEST_10  = ['brasilia','chongqing','dubai','lasvegas','milano','montpellier',
            'norcia','rio','saclay_w','valencia']


def build(out='data/OSCD/splits/oscd_splits.json'):
    train = [c for c in TRAIN_14 if c not in OSCD_VAL]

    assert len(TRAIN_14) == 14 and len(TEST_10) == 10
    assert set(OSCD_VAL) <= set(TRAIN_14), \
        'a validation city is not in the official TRAIN 14 -- this is the ' \
        'montpellier bug. Validation may only draw from official train cities.'
    assert not (set(OSCD_VAL) & set(TEST_10)), 'val/test overlap'
    assert not (set(train)   & set(TEST_10)), 'train/test overlap'
    assert not (set(train)   & set(OSCD_VAL)), 'train/val overlap'
    assert len(train) == 11, f'expected 11 train cities, got {len(train)}'

    m = {
        'train': sorted(train), 'val': sorted(OSCD_VAL), 'test': sorted(TEST_10),
        'n': {'train': len(train), 'val': len(OSCD_VAL), 'test': len(TEST_10)},
        'frozen': '2026-09-17',
        'caveats': [
            'saclay_e (val) and saclay_w (test) are ADJACENT halves of the same '
            'area. This spatial-autocorrelation leak is inherent to the official '
            'OSCD split. saclay_e is placed in VAL rather than TRAIN so it never '
            'produces a training gradient. Report OSCD test F1 over all 10 cities '
            'AND over 9 cities excluding saclay_w.',
            'OSCD is L1C top-of-atmosphere; Maharashtra is L2A surface '
            'reflectance. This is a SECOND domain shift on top of geography and '
            'must be stated in the paper.',
            'Original label TIFFs encode 1 = no-change, 2 = change. Subtract 1.',
        ],
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(m, open(out, 'w'), indent=2)
    print(json.dumps(m, indent=2))
    return m


if __name__ == '__main__':
    build()
