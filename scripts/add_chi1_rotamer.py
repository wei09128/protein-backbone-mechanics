#!/usr/bin/env python3
"""add_chi1_rotamer.py — derive g-/t/g+ rotamer bins from chi1_rad
and write an augmented CSV for export_mechlib_tables.py --chi1_csv.
"""
import numpy as np
import pandas as pd

IN_CSV = 'features_lj_v3_FINAL_CLEAN.csv'
OUT_CSV = 'features_lj_v3_chi1.csv'

ROTAMER_CENTERS = {'g-': -60.0, 't': 180.0, 'g+': 60.0}

def circ_diff(a, b):
    d = (a - b + 180) % 360 - 180
    return np.abs(d)

def classify(chi1_deg):
    diffs = {k: circ_diff(chi1_deg, c) for k, c in ROTAMER_CENTERS.items()}
    return min(diffs, key=diffs.get)

df = pd.read_csv(IN_CSV, low_memory=False)

# only residues that actually have a chi1
df = df[df['has_chi1'].astype(bool)].copy()

df['chi1_deg'] = np.degrees(df['chi1_rad'])
df['chi1_deg'] = (df['chi1_deg'] + 180) % 360 - 180  # wrap to (-180, 180]
df['chi1_rotamer'] = df['chi1_deg'].apply(classify)

df.to_csv(OUT_CSV, index=False)
print(f"{len(df)} chi1-bearing rows -> {OUT_CSV}")
print(df['chi1_rotamer'].value_counts())