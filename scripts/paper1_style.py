"""
paper1_style.py — Shared plotting style for Paper 1 (Figures 2-7)
=====================================================================
Formalizes the color palette already established in paper1_2_backbone_analysis.py
(Figures 2, 3, 4, 6) as the canonical scheme for this paper, and corrects
Figure 7 (which was rebuilt separately, using Paper 4/MechLib's palette by
mistake) to match it -- addressing Minor Comment 5 (consistent color
scales, axis labels, resolution).

Usage: import and call setup_style() once near the top of each script's
main plotting entry point.
"""

import matplotlib

# Paper 1's ESTABLISHED palette (already in use in Figs 2/3/4/6 before this
# module existed) -- adopted here as canonical rather than replaced.
COLORS = {
    'green':  '#1D9E75',   # Group A / steric / "current states" / restoring
    'orange': '#D85A30',   # "new states" / driving / secondary series
    'blue':   '#378ADD',   # Group B / forces / CV markers
    'amber':  '#BA7517',   # Group C / context
    'dark_green':  '#085041',
    'dark_orange': '#712B13',
    'dark_blue':   '#0C447C',
    'neutral': '#4D4D4D',
}

FIG_DPI = 300  # publication resolution (was 150 -- fine for screen, not print)


def setup_style():
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.8,
        'savefig.dpi': FIG_DPI,
        'savefig.bbox': 'tight',
    })
