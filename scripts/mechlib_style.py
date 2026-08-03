"""
mechlib_style.py — Shared plotting style for all MechLib paper figures
=========================================================================
Import and call setup_style() at the top of each figure script, and use
the shared COLORS dict so AMBER/OPLS/CHARMM/MechLib are represented
consistently across every figure in the paper.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {
    'amber':    '#B2182B',   # red
    'opls':     '#5AAE61',   # green
    'charmm':   '#762A83',   # purple
    'mechlib':  '#2166AC',   # blue
    'neutral':  '#4D4D4D',   # dark gray, for reference lines/text
}

FONT_SIZES = {
    'title': 13,
    'label': 11,
    'tick': 9.5,
    'legend': 10,
    'annotation': 9,
}


def setup_style():
    """Call once at the start of each figure script."""
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.titlesize': FONT_SIZES['title'],
        'axes.titleweight': 'bold',
        'axes.labelsize': FONT_SIZES['label'],
        'xtick.labelsize': FONT_SIZES['tick'],
        'ytick.labelsize': FONT_SIZES['tick'],
        'legend.fontsize': FONT_SIZES['legend'],
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.8,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def panel_label(ax, letter, x=-0.08, y=1.05):
    """Add a bold panel letter (A, B, C...) above-left of an axis."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=15,
            fontweight='bold', va='bottom', ha='right')


def clean_observable_name(col):
    """Turn a raw column name like 'angle_N_CA_CB' into a readable label."""
    name = col.replace('bond_', '').replace('angle_', '')
    replacements = {
        'N_CA': 'N-Cα', 'CA_C': 'Cα-C', 'C_O': 'C=O', 'C_N_next': 'C-N(i+1)',
        'CA_CB': 'Cα-Cβ', 'N_CA_CB': 'N-Cα-Cβ', 'C_CA_CB': 'C-Cα-Cβ',
        'CaCN': 'Cα-C-N(i+1)', 'CNCa': 'C(i-1)-N-Cα', 'CA_C_O': 'Cα-C=O',
    }
    return replacements.get(name, name)
