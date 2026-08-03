"""
paper1_fig3_plot.py — Render Figure 3 from the saved RF importance CSVs
================================================================================
paper1_table1_rf_importance.py computed and saved the importances but never
plotted them. This reads those CSVs directly (no refitting) and renders the
actual bar chart.

Usage:
  python paper1_fig3_plot.py --out_dir ./figures/paper1
"""

import argparse
import pandas as pd
from pathlib import Path
from paper1_style import COLORS, setup_style

GROUP_A = {
    'steric_N_3A', 'steric_N_4A', 'steric_N_5A',
    'steric_CA_3A', 'steric_CA_4A', 'steric_CA_5A',
    'steric_C_3A', 'steric_C_4A', 'steric_C_5A',
    'steric_O_3A', 'steric_O_4A', 'steric_O_5A',
    'steric_asym_x', 'steric_asym_y', 'steric_asym_z', 'improper_ca',
    'steric_clash_phi_plus', 'steric_clash_phi_minus',
    'steric_clash_psi_plus', 'steric_clash_psi_minus',
    'steric_mindist_phi', 'steric_mindist_psi',
    'sc_contact_nm1_to_bb', 'sc_contact_np1_to_bb',
}
GROUP_B = {
    'tau_phi_correct', 'tau_psi_correct', 'tau_phi_bb_donor', 'tau_psi_bb_donor',
    'tau_phi_bb_acc', 'tau_psi_bb_acc', 'tau_phi_sc_hb', 'tau_psi_sc_hb',
    'tau_phi_steric', 'tau_psi_steric', 'tau_phi_elec_corr', 'tau_psi_elec_corr',
    'chi1_rad', 'has_chi1',
}


def color_for(feat):
    if feat in GROUP_A: return COLORS['green']
    if feat in GROUP_B: return COLORS['blue']
    return COLORS['amber']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phi_csv', default='rf_importance_phi.csv')
    ap.add_argument('--psi_csv', default='rf_importance_psi.csv')
    ap.add_argument('--out_dir', default='.')
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup_style()
    import matplotlib.pyplot as plt

    imp_phi = pd.read_csv(args.phi_csv, index_col=0)['importance']
    imp_psi = pd.read_csv(args.psi_csv, index_col=0)['importance']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, imp, label in [(axes[0], imp_phi, 'phi'), (axes[1], imp_psi, 'psi')]:
        top = imp.head(10)
        colors = [color_for(f) for f in top.index]
        ax.barh(range(len(top)), top.values[::-1], color=colors[::-1],
                edgecolor='#333333', linewidth=0.5)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index[::-1], fontsize=9)
        ax.set_xlabel('Importance')
        ax.set_title(f'{label} prediction (Random Forest, 58 features)')

    handles = [
        plt.Rectangle((0,0),1,1, color=COLORS['green'], label='Group A (steric)'),
        plt.Rectangle((0,0),1,1, color=COLORS['blue'], label='Group B (forces)'),
        plt.Rectangle((0,0),1,1, color=COLORS['amber'], label='Group C (context)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    out_path = out / 'feature_importance.png'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")
    print(f"\nTop phi feature: {imp_phi.index[0]} ({imp_phi.iloc[0]:.4f})")
    print(f"Top psi feature: {imp_psi.index[0]} ({imp_psi.iloc[0]:.4f})")


if __name__ == '__main__':
    main()
