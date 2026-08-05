#!/usr/bin/env python3
"""
apply_library_corrections.py
=============================
DEPRECATED SHIM. This file used to contain its own, independent
implementation of the OpenMM integration -- one that never actually
computed phi/psi from the structure's coordinates, and instead
hardcoded every residue in every structure to the alpha-helix bin
("phi_key, psi_key = '-65', '-45'"), regardless of that residue's
real conformation. That implementation has been removed rather than
patched, to avoid maintaining two divergent copies of the same logic
under the same function name.

The real, correct implementation -- which computes phi/psi per
residue from the actual atomic coordinates via NeRF-style dihedral
geometry, and supports AMBER/CHARMM/OPLS defaults -- lives in
backbone_geometry_library.apply_corrections(). This module now just
re-exports it under the original name so existing imports of
`apply_library_corrections.apply_corrections` keep working without
silently getting the broken behavior.

If your code does:
    from apply_library_corrections import apply_corrections

it will now receive the real implementation. Prefer importing
directly from backbone_geometry_library going forward.
"""

from backbone_geometry_library import apply_corrections  # noqa: F401

__all__ = ["apply_corrections"]
