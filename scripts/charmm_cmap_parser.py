#!/usr/bin/env python3
"""
charmm_cmap_parser.py — Parse CHARMM36 CMAP grid from par_all36m_prot.prm
============================================================================
CHARMM's CMAP term is a 2D bicubic-spline-interpolated energy correction
E_cmap(phi, psi) applied on top of the standard torsional terms. It is NOT
a harmonic bond/angle parameter like AMBER's eq/k — it's a full energy
surface over the Ramachandran plot, typically on a 24x24 grid (15 degree
spacing, -180 to 180).

This script parses the real CHARMM36m CMAP block so we work with actual
published values, not reconstructed approximations.

WHERE TO GET THE FILE:
  The CHARMM36m parameter file is distributed by the MacKerell lab and
  via CHARMM-GUI's toppar archive:
    http://mackerell.umaryland.edu/charmm_ff.shtml
  or
    https://www.charmm-gui.org/?doc=toppar
  Download `toppar_c36_jul21.tgz` (or latest), extract, and locate:
    toppar/par_all36m_prot.prm
  The file contains a "CMAP" section near the end with the format:

    CMAP
    X   CT1  C    NH1  X   NH1  CT1  C    X     24   24
    <24 lines of 24 values each — the correction grid in kcal/mol>
    ...

  There is typically ONE CMAP grid for the general protein backbone
  (the "backbone CMAP") plus occasionally residue-specific ones (e.g.
  glycine, proline) depending on the CHARMM version.

Usage:
  python charmm_cmap_parser.py --prm par_all36m_prot.prm --out cmap_grid.npz
"""

import argparse
import re
import numpy as np
from pathlib import Path


def parse_cmap_section(prm_text):
    """
    Parse all CMAP grids from a CHARMM .prm file.

    Returns a list of dicts:
      {
        'atom_types': (8 atom type strings defining the two dihedrals),
        'n_phi': int, 'n_psi': int,
        'grid': np.ndarray shape (n_phi, n_psi),  # kcal/mol
      }
    """
    lines = prm_text.splitlines()

    # Find the CMAP section
    cmap_start = None
    for i, line in enumerate(lines):
        if re.match(r'^\s*CMAP\b', line, re.IGNORECASE):
            cmap_start = i
            break
    if cmap_start is None:
        raise ValueError("No CMAP section found in this parameter file")

    grids = []
    i = cmap_start + 1
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        i += 1
        if not line or line.startswith('!'):
            continue
        if re.match(r'^(END|BONDS|ANGLES|DIHEDRALS|IMPROPER|NONBONDED)\b',
                     line, re.IGNORECASE):
            break

        # Header line: 8 atom types + grid dimension.
        # CHARMM36m format uses a SINGLE dimension number for a square
        # grid, e.g.:
        #   C  NH1  CT1  C  NH1  CT1  C  NH1   24
        # (8 atom types, then one integer = n_phi = n_psi)
        # Some older CHARMM files use two numbers (n_phi n_psi) instead;
        # support both.
        parts = line.split('!')[0].split()
        if len(parts) < 9:
            continue  # not a valid header line, skip
        atom_types = tuple(parts[:8])
        try:
            if len(parts) >= 10:
                n_phi, n_psi = int(parts[8]), int(parts[9])
            else:
                n_phi = n_psi = int(parts[8])
        except ValueError:
            continue

        # Read n_phi * n_psi values, which may be spread across many lines
        values = []
        while len(values) < n_phi * n_psi and i < n:
            row = lines[i].strip()
            i += 1
            if not row or row.startswith('!'):
                continue
            row_vals = row.split('!')[0].split()
            for v in row_vals:
                try:
                    values.append(float(v))
                except ValueError:
                    pass

        if len(values) != n_phi * n_psi:
            print(f"  WARNING: grid for {atom_types} expected "
                  f"{n_phi*n_psi} values, got {len(values)} — skipping")
            continue

        grid = np.array(values, dtype=float).reshape(n_phi, n_psi)
        grids.append({
            'atom_types': atom_types,
            'n_phi': n_phi,
            'n_psi': n_psi,
            'grid': grid,
        })

    return grids


def grid_to_lookup(grid_dict):
    """
    Convert a CMAP grid dict into a lookup function f(phi_deg, psi_deg) ->
    energy (kcal/mol), using bilinear interpolation on the periodic grid.
    CHARMM CMAP grids are defined on [-180, 180) with uniform spacing.
    """
    grid = grid_dict['grid']
    n_phi, n_psi = grid_dict['n_phi'], grid_dict['n_psi']
    d_phi = 360.0 / n_phi
    d_psi = 360.0 / n_psi

    def lookup(phi_deg, psi_deg):
        p = ((phi_deg + 180.0) % 360.0) / d_phi
        q = ((psi_deg + 180.0) % 360.0) / d_psi
        i0, j0 = int(np.floor(p)) % n_phi, int(np.floor(q)) % n_psi
        i1, j1 = (i0 + 1) % n_phi, (j0 + 1) % n_psi
        fp, fq = p - np.floor(p), q - np.floor(q)

        v00 = grid[i0, j0]; v10 = grid[i1, j0]
        v01 = grid[i0, j1]; v11 = grid[i1, j1]
        v0 = v00 * (1 - fp) + v10 * fp
        v1 = v01 * (1 - fp) + v11 * fp
        return float(v0 * (1 - fq) + v1 * fq)

    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prm', required=True,
                    help='Path to par_all36m_prot.prm (or similar CHARMM36 prm file)')
    ap.add_argument('--out', default='cmap_grid.npz',
                    help='Output .npz file with parsed grids')
    args = ap.parse_args()

    print(f"Reading {args.prm} ...")
    text = Path(args.prm).read_text(errors='ignore')

    print("Parsing CMAP section...")
    grids = parse_cmap_section(text)

    print(f"Found {len(grids)} CMAP grid(s):")
    for g in grids:
        print(f"  {g['atom_types']}  ->  {g['n_phi']}x{g['n_psi']} grid  "
              f"range=[{g['grid'].min():.3f}, {g['grid'].max():.3f}] kcal/mol")

    if not grids:
        print("ERROR: no valid CMAP grids parsed. Check the file format.")
        return

    # Save all grids to a single npz for downstream use
    save_dict = {}
    for idx, g in enumerate(grids):
        save_dict[f'grid_{idx}'] = g['grid']
        save_dict[f'types_{idx}'] = np.array(g['atom_types'])
    np.savez(args.out, n_grids=len(grids), **save_dict)
    print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
