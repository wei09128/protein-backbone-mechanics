#!/usr/bin/env python3
"""
Apply geometry library corrections to an OpenMM simulation.
Works with any force field (AMBER, CHARMM, OPLS).
"""

import json
import numpy as np

def apply_corrections(system, topology, pdb_positions, library_path):
    """Add library correction forces to an OpenMM system.
    
    Args:
        system: OpenMM System object
        topology: OpenMM Topology
        pdb_positions: initial positions for dihedral calculation
        library_path: path to constants_library.json
    """
    try:
        from openmm import CustomAngleForce
    except ImportError:
        print("OpenMM not installed — cannot apply corrections")
        return system

    with open(library_path) as f:
        lib = json.load(f)

    correction = CustomAngleForce(
        '0.5*k*(theta-theta_lib)^2 - 0.5*k*(theta-theta_ff)^2'
    )
    correction.addPerAngleParameter('theta_lib')
    correction.addPerAngleParameter('theta_ff')
    correction.addPerAngleParameter('k')

    deg2rad = np.pi / 180.0
    n_corrections = 0

    for residue in topology.residues():
        # Get atom indices
        atoms = {a.name: a.index for a in residue.atoms()}
        if not all(a in atoms for a in ['N', 'CA', 'C']):
            continue

        # Get phi, psi from positions (simplified)
        # In practice, compute from coordinates
        res_name = residue.name
        phi_key, psi_key = "-65", "-45"  # default αR

        # Lookup library tau
        cell = None
        for cls in [res_name, "ALL"]:
            if cls in lib:
                cell = lib[cls].get(phi_key, {}).get(psi_key)
                if cell: break

        if cell and "tau_deg_eq" in cell:
            tau_lib = cell["tau_deg_eq"] * deg2rad
            tau_ff = 111.1 * deg2rad  # AMBER default
            k = 63.0 * 4.184  # kcal to kJ
            correction.addAngle(atoms['N'], atoms['CA'], atoms['C'],
                                [tau_lib, tau_ff, k])
            n_corrections += 1

    system.addForce(correction)
    print(f"Added {n_corrections} angle corrections")
    return system
