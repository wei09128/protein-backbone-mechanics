"""
geom_utils.py — Pure geometric primitives for protein backbone analysis
========================================================================
No dependencies beyond numpy. Every function is a pure function of its
inputs. No global state, no file IO.

Contents:
  Vectors:      unit, safe_norm
  Rotations:    rodrigues
  Torques:      torque, local_frame
  Measurements: bond_length, bond_angle, dihedral, improper
  Helpers:      cg_position, ss_bin

Run this file directly to execute Layer 1 tests:
    python geom_utils.py

All tests use synthetic inputs with analytically-known answers.
If any test fails, there is a bug in the primitives and nothing
downstream can be trusted.
"""

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# VECTORS
# ══════════════════════════════════════════════════════════════════════════════

def unit(v):
    """Unit vector. Returns zero vector for near-zero input (no NaN)."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else np.zeros(3)


def safe_norm(v):
    """Euclidean norm as a plain float. Zero for invalid input."""
    v = np.asarray(v, dtype=float)
    if not np.all(np.isfinite(v)):
        return 0.0
    return float(np.linalg.norm(v))


# ══════════════════════════════════════════════════════════════════════════════
# ROTATIONS
# ══════════════════════════════════════════════════════════════════════════════

def rodrigues(points, pivot, axis, angle):
    """
    Rotate `points` (shape (N,3) or (3,)) about `axis` through `pivot` by `angle` (radians).
    Axis must be a unit vector. Returns rotated points with same shape as input.
    """
    points = np.asarray(points, dtype=float)
    single = (points.ndim == 1)
    if single:
        points = points[np.newaxis, :]
    if len(points) == 0:
        return points[0] if single else points
    pivot = np.asarray(pivot, dtype=float)
    axis = np.asarray(axis, dtype=float)
    r = points - pivot
    ca, sa = np.cos(angle), np.sin(angle)
    rotated = (pivot
               + r * ca
               + np.cross(axis, r) * sa
               + axis * (r @ axis)[:, np.newaxis] * (1 - ca))
    return rotated[0] if single else rotated


# ══════════════════════════════════════════════════════════════════════════════
# TORQUES & FRAMES
# ══════════════════════════════════════════════════════════════════════════════

def torque(force, position, axis_point, axis_hat):
    """
    Scalar torque of `force` applied at `position`, about the axis
    passing through `axis_point` with unit direction `axis_hat`.

    τ = ((r − r_axis) × F) · axis_hat
    """
    force = np.asarray(force, dtype=float)
    position = np.asarray(position, dtype=float)
    axis_point = np.asarray(axis_point, dtype=float)
    axis_hat = np.asarray(axis_hat, dtype=float)
    return float(np.dot(np.cross(position - axis_point, force), axis_hat))


def local_frame(N, CA, C):
    """
    Right-handed orthonormal frame at a residue.
      x = unit(CA − N)       ← along N→Cα
      z = unit(x × (C − CA)) ← normal to N–Cα–C plane
      y = z × x              ← in-plane, perpendicular to x

    Returns 3×3 matrix whose rows are [x, y, z].
    Degenerate cases return a safe default frame.
    """
    x = unit(np.asarray(CA) - np.asarray(N))
    c = unit(np.asarray(C) - np.asarray(CA))
    z = np.cross(x, c)
    if np.linalg.norm(z) < 1e-10:
        z = np.array([0., 0., 1.])
    else:
        z = unit(z)
    y = np.cross(z, x)
    return np.stack([x, y, z])


# ══════════════════════════════════════════════════════════════════════════════
# MEASUREMENTS — bond length, bond angle, dihedral, improper
# ══════════════════════════════════════════════════════════════════════════════

def bond_length(a, b):
    """Distance |a − b| in Å."""
    return safe_norm(np.asarray(a) - np.asarray(b))


def bond_angle(a, b, c):
    """
    Angle at vertex b formed by a–b–c, in degrees.
    Returns NaN for degenerate input.
    """
    a, b, c = map(np.asarray, (a, b, c))
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(c))):
        return float('nan')
    v1 = unit(a - b)
    v2 = unit(c - b)
    if np.linalg.norm(v1) < 1e-10 or np.linalg.norm(v2) < 1e-10:
        return float('nan')
    cos_ang = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def dihedral(a, b, c, d):
    """
    Proper dihedral angle for four points a-b-c-d, in degrees.
    Convention: IUPAC / Ramachandran (right-handed, sign from sin term).
    Returns NaN for degenerate input.
    """
    a, b, c, d = map(np.asarray, (a, b, c, d))
    if not all(np.all(np.isfinite(p)) for p in (a, b, c, d)):
        return float('nan')
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n2 = np.linalg.norm(b2)
    if n2 < 1e-10:
        return float('nan')
    b2u = b2 / n2
    n1 = np.cross(b1, b2)
    n3 = np.cross(b2, b3)
    m1 = np.cross(n1, b2u)
    x = np.dot(n1, n3)
    y = np.dot(m1, n3)
    return float(np.degrees(np.arctan2(y, x)))


def improper(center, a, b, c):
    """
    Improper dihedral at `center` with the three neighbours a, b, c.
    Computed as the dihedral a-center-b-c.
    Useful for measuring chirality and out-of-plane distortions at Cα.
    """
    return dihedral(a, center, b, c)


# ══════════════════════════════════════════════════════════════════════════════
# RESIDUE-SPECIFIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def cg_position(CA, CB, chi1_rad):
    """
    Virtual Cγ placement given Cα, Cβ, and χ₁ (radians).
    Uses ideal bond length 1.52 Å and angle 111°.
    Returns None for invalid input.
    """
    if chi1_rad is None or CB is None:
        return None
    CA = np.asarray(CA, dtype=float)
    CB = np.asarray(CB, dtype=float)
    if not (np.all(np.isfinite(CA)) and np.all(np.isfinite(CB))):
        return None
    b1 = unit(CB - CA)
    perp = np.array([1., 0., 0.]) if abs(b1[0]) < 0.9 else np.array([0., 1., 0.])
    n_vec = unit(np.cross(b1, perp))
    m_vec = np.cross(b1, n_vec)
    ang = np.radians(111.0)
    d = np.array([-np.cos(ang),
                   np.sin(ang) * np.cos(-chi1_rad),
                   np.sin(ang) * np.sin(-chi1_rad)])
    return CB + 1.52 * (d[0] * b1 + d[1] * m_vec + d[2] * n_vec)


def ss_bin(phi_deg, psi_deg):
    """
    Coarse Ramachandran region bin (same as v5).
      0 = αR   1 = β    2 = PPII   3 = 3₁₀   4 = loop   5 = αL
    """
    p, q = phi_deg, psi_deg
    if p > 0 and -20 <= q <= 80:
        return 5
    if -100 <= p <= -40 and -60 <= q <= 20:
        return 0
    if p <= -90 and q >= 90:
        return 1
    if -90 <= p <= -50 and q >= 120:
        return 2
    if -80 <= p <= -30 and -40 <= q <= 0:
        return 3
    return 4


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 TESTS — synthetic inputs with analytically-known answers
# ══════════════════════════════════════════════════════════════════════════════

def _check(name, passed, detail=""):
    marker = "PASS" if passed else "FAIL"
    print(f"  [{marker}] {name}" + (f"  ({detail})" if detail else ""))
    return passed


def _validate():
    print("=" * 70)
    print("geom_utils.py — Layer 1 physics sanity tests")
    print("=" * 70)
    all_pass = True

    # ── unit ──────────────────────────────────────────────────────────────────
    print("\n[unit]")
    all_pass &= _check("unit([3,0,0]) → [1,0,0]",
                       np.allclose(unit([3, 0, 0]), [1, 0, 0]))
    all_pass &= _check("unit([0,0,0]) → [0,0,0] (no NaN)",
                       np.allclose(unit([0, 0, 0]), [0, 0, 0]))
    all_pass &= _check("unit has length 1 for random vector",
                       abs(np.linalg.norm(unit([1.7, -2.3, 0.4])) - 1.0) < 1e-10)

    # ── safe_norm ─────────────────────────────────────────────────────────────
    print("\n[safe_norm]")
    all_pass &= _check("safe_norm([3,4,0]) = 5",
                       abs(safe_norm([3, 4, 0]) - 5.0) < 1e-10)
    all_pass &= _check("safe_norm with NaN → 0",
                       safe_norm([np.nan, 1, 0]) == 0.0)

    # ── bond_length ───────────────────────────────────────────────────────────
    print("\n[bond_length]")
    all_pass &= _check("bond_length([0,0,0],[3,4,0]) = 5",
                       abs(bond_length([0, 0, 0], [3, 4, 0]) - 5.0) < 1e-10)

    # ── bond_angle ────────────────────────────────────────────────────────────
    print("\n[bond_angle]")
    # Right angle: points at (1,0,0), (0,0,0), (0,1,0) → 90°
    all_pass &= _check("right angle → 90°",
                       abs(bond_angle([1, 0, 0], [0, 0, 0], [0, 1, 0]) - 90.0) < 1e-8)
    # Straight line: (1,0,0), (0,0,0), (-1,0,0) → 180°
    all_pass &= _check("straight line → 180°",
                       abs(bond_angle([1, 0, 0], [0, 0, 0], [-1, 0, 0]) - 180.0) < 1e-8)
    # 60° equilateral
    p1 = [1, 0, 0]
    p2 = [0, 0, 0]
    p3 = [0.5, np.sqrt(3) / 2, 0]
    all_pass &= _check("equilateral → 60°",
                       abs(bond_angle(p1, p2, p3) - 60.0) < 1e-8)
    # Ideal tetrahedral angle = 109.4712°
    tet = [1, 1, 1]
    tet2 = [1, -1, -1]
    all_pass &= _check("tetrahedral → 109.47°",
                       abs(bond_angle(tet, [0, 0, 0], tet2) - 109.4712206) < 1e-5)

    # ── dihedral ──────────────────────────────────────────────────────────────
    print("\n[dihedral]")
    # Four points: a=(1,0,0), b=(0,0,0), c=(0,1,0), d=(0,1,1)
    # Dihedral a-b-c-d should be 90°
    d90 = dihedral([1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 1])
    all_pass &= _check("90° test dihedral",
                       abs(abs(d90) - 90.0) < 1e-8,
                       f"got {d90:.4f}")
    # Cis (coplanar, same side): a=(1,0,0), b=(0,0,0), c=(0,1,0), d=(1,1,0) → 0°
    d0 = dihedral([1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0])
    all_pass &= _check("cis coplanar → 0°",
                       abs(d0) < 1e-8,
                       f"got {d0:.4f}")
    # Trans (coplanar, opposite): d=(-1,1,0) → 180°
    d180 = dihedral([1, 0, 0], [0, 0, 0], [0, 1, 0], [-1, 1, 0])
    all_pass &= _check("trans coplanar → ±180°",
                       abs(abs(d180) - 180.0) < 1e-8,
                       f"got {d180:.4f}")
    # IUPAC sign convention: check both signs with a symmetric pair.
    # From cis (d=(1,1,0)), moving d into +z vs −z gives opposite signs.
    d_pos = dihedral([1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 1])
    d_neg = dihedral([1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, -1])
    all_pass &= _check("sign convention: opposite sides give opposite signs",
                       abs(d_pos + d_neg) < 1e-8,
                       f"got {d_pos:.4f} and {d_neg:.4f}")
    all_pass &= _check("sign convention: magnitudes both 90°",
                       abs(abs(d_pos) - 90.0) < 1e-8 and abs(abs(d_neg) - 90.0) < 1e-8)

    # ── improper ──────────────────────────────────────────────────────────────
    print("\n[improper]")
    # Chiral center: L-amino acid Cα improper N-Cα-C-Cβ should be negative
    # Place Cα at origin, N, C, Cβ at tetrahedral positions for L-config
    # (this is a geometric sanity check, not a chemistry test)
    imp = improper([0, 0, 0], [1, 1, 1], [1, -1, -1], [-1, 1, -1])
    all_pass &= _check("improper returns finite value",
                       np.isfinite(imp),
                       f"got {imp:.4f}")

    # ── rodrigues ─────────────────────────────────────────────────────────────
    print("\n[rodrigues]")
    # Rotate (1,0,0) by 90° about z at origin → (0,1,0)
    r = rodrigues([1, 0, 0], [0, 0, 0], [0, 0, 1], np.pi / 2)
    all_pass &= _check("rotate (1,0,0) 90° about z → (0,1,0)",
                       np.allclose(r, [0, 1, 0], atol=1e-10),
                       f"got {r}")
    # Rotate (1,0,0) by 180° about z → (-1,0,0)
    r = rodrigues([1, 0, 0], [0, 0, 0], [0, 0, 1], np.pi)
    all_pass &= _check("rotate (1,0,0) 180° about z → (-1,0,0)",
                       np.allclose(r, [-1, 0, 0], atol=1e-10))
    # Identity: 0° rotation
    r = rodrigues([[1, 2, 3], [4, 5, 6]], [0, 0, 0], [0, 0, 1], 0.0)
    all_pass &= _check("0° rotation is identity",
                       np.allclose(r, [[1, 2, 3], [4, 5, 6]], atol=1e-10))
    # Rotation preserves distance from pivot
    pt = np.array([1.5, 2.7, -0.3])
    pivot = np.array([0.1, 0.2, 0.3])
    axis = unit([0.5, 0.5, 0.5])
    rotated = rodrigues(pt, pivot, axis, 1.234)
    d_before = np.linalg.norm(pt - pivot)
    d_after = np.linalg.norm(rotated - pivot)
    all_pass &= _check("rotation preserves distance from pivot",
                       abs(d_before - d_after) < 1e-10,
                       f"Δd = {abs(d_before - d_after):.2e}")

    # ── torque ────────────────────────────────────────────────────────────────
    print("\n[torque]")
    # Force at (1,0,0), direction (0,1,0), axis through origin along z
    # τ = (r × F) · ẑ = ((1,0,0) × (0,1,0)) · (0,0,1) = (0,0,1) · (0,0,1) = 1
    t = torque([0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1])
    all_pass &= _check("unit force at unit lever arm → τ = 1",
                       abs(t - 1.0) < 1e-10,
                       f"got {t}")
    # Force parallel to lever arm → τ = 0
    t = torque([1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1])
    all_pass &= _check("parallel force → τ = 0",
                       abs(t) < 1e-10)
    # Force passing through axis point → τ = 0
    t = torque([0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1])
    all_pass &= _check("force at axis point → τ = 0",
                       abs(t) < 1e-10)
    # Sign flip: reverse force → τ negated
    t1 = torque([0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1])
    t2 = torque([0, -1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1])
    all_pass &= _check("sign flip on force",
                       abs(t1 + t2) < 1e-10)

    # ── local_frame ───────────────────────────────────────────────────────────
    print("\n[local_frame]")
    # Simple case: N=(0,0,0), CA=(1,0,0), C=(1,1,0)
    # x = (1,0,0), z = cross(x, (0,1,0)) = (0,0,1), y = cross(z,x) = (0,1,0)
    R = local_frame([0, 0, 0], [1, 0, 0], [1, 1, 0])
    all_pass &= _check("local_frame x = N→Cα direction",
                       np.allclose(R[0], [1, 0, 0]))
    all_pass &= _check("local_frame z = normal to peptide plane",
                       np.allclose(R[2], [0, 0, 1]))
    all_pass &= _check("local_frame y = z × x",
                       np.allclose(R[1], [0, 1, 0]))
    all_pass &= _check("local_frame is orthonormal",
                       np.allclose(R @ R.T, np.eye(3), atol=1e-10))

    # ── cg_position ───────────────────────────────────────────────────────────
    print("\n[cg_position]")
    cg = cg_position([0, 0, 0], [1.52, 0, 0], 0.0)
    all_pass &= _check("cg_position returns finite point",
                       cg is not None and np.all(np.isfinite(cg)))
    # Distance CB → CG should be 1.52 Å
    if cg is not None:
        d_cg = np.linalg.norm(cg - np.array([1.52, 0, 0]))
        all_pass &= _check("|CB−CG| = 1.52 Å",
                           abs(d_cg - 1.52) < 1e-6,
                           f"got {d_cg:.6f}")
    all_pass &= _check("cg_position handles None chi1",
                       cg_position([0, 0, 0], [1.52, 0, 0], None) is None)

    # ── ss_bin ────────────────────────────────────────────────────────────────
    print("\n[ss_bin]")
    all_pass &= _check("αR region (−60, −45) → 0",
                       ss_bin(-60, -45) == 0)
    all_pass &= _check("β region (−120, 120) → 1",
                       ss_bin(-120, 120) == 1)
    all_pass &= _check("αL region (60, 40) → 5",
                       ss_bin(60, 40) == 5)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if all_pass:
        print("  ALL LAYER 1 TESTS PASSED")
    else:
        print("  *** SOME TESTS FAILED — FIX BEFORE PROCEEDING ***")
    print("=" * 70)
    return all_pass


if __name__ == '__main__':
    import sys
    sys.exit(0 if _validate() else 1)