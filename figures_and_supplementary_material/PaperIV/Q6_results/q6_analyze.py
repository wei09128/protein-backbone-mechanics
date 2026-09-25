#!/usr/bin/env python3
"""Evaluate the frozen MechLib-QM-Q1-v1 primary endpoint."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path("work_q5")
OUT = Path("work_q6")
SEED = 20260915
N_BOOT = 10_000


def read_tsv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows, fields=None):
    if fields is None:
        fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_xyz(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    n = int(lines[0])
    rows = lines[2:2+n]
    if len(rows) != n:
        raise ValueError(f"Malformed XYZ: {path}")
    return np.asarray([[float(v) for v in row.split()[1:4]] for row in rows])


def distance(xyz, a, b):
    return float(np.linalg.norm(xyz[a-1] - xyz[b-1]))


def angle_deg(xyz, a, b, c):
    u = xyz[a-1] - xyz[b-1]
    v = xyz[c-1] - xyz[b-1]
    cosv = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))


def atom_index(atom_rows, residue_index, atom_name):
    found = [int(r["atom_index_1based"]) for r in atom_rows
             if int(r["residue_index_1based"]) == residue_index and r["atom_name"] == atom_name]
    if len(found) != 1:
        raise ValueError(f"Expected one atom {residue_index}:{atom_name}, got {found}")
    return found[0]


def geometry_terms(case_dir: Path):
    atoms = read_tsv(case_dir / "atom_map.tsv")
    xyz = read_xyz(case_dir / "qm" / "qm_final_geometry.xyz")
    pC = atom_index(atoms, 1, "C")
    N = atom_index(atoms, 2, "N")
    CA = atom_index(atoms, 2, "CA")
    C = atom_index(atoms, 2, "C")
    O = atom_index(atoms, 2, "O")
    nN = atom_index(atoms, 3, "N")
    values = {
        "tau_deg": angle_deg(xyz, N, CA, C),
        "angle_CaCN": angle_deg(xyz, CA, C, nN),
        "angle_CNCa": angle_deg(xyz, pC, N, CA),
        "angle_CA_C_O": angle_deg(xyz, CA, C, O),
        "bond_N_CA": distance(xyz, N, CA),
        "bond_CA_C": distance(xyz, CA, C),
        "bond_C_O": distance(xyz, C, O),
        "bond_C_N_next": distance(xyz, C, nN),
    }
    cb = [r for r in atoms if int(r["residue_index_1based"]) == 2 and r["atom_name"] == "CB"]
    if cb:
        CB = int(cb[0]["atom_index_1based"])
        values.update({
            "angle_N_CA_CB": angle_deg(xyz, N, CA, CB),
            "angle_C_CA_CB": angle_deg(xyz, C, CA, CB),
            "bond_CA_CB": distance(xyz, CA, CB),
        })
    return values


def ci(values, rng, n_boot=N_BOOT):
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = values[idx].mean(axis=1)
    return float(np.quantile(boots, .025)), float(np.quantile(boots, .975)), boots


def cluster_ci(case_rows, rng):
    groups = defaultdict(list)
    for row in case_rows:
        groups[row["residue"]].append(float(row["delta"]))
    residues = sorted(groups)
    means = np.asarray([np.mean(groups[r]) for r in residues])
    return (*ci(means, rng)[:2], means)


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, idx in enumerate(order):
        running = max(running, (m-rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    predictions = read_tsv(ROOT / "frozen_model_predictions.tsv")
    cases = read_tsv(ROOT / "cases_confirmatory.tsv")
    case_meta = {r["case_id"]: r for r in cases}
    pred_by_case = defaultdict(list)
    for row in predictions:
        pred_by_case[row["case_id"]].append(row)

    detail = []
    case_scores = []
    for case_id in sorted(case_meta):
        case_dir = ROOT / "confirmatory_qm_out_v103" / case_id
        status = json.loads((case_dir / "case_status.json").read_text())["status"]
        if status != "PASS":
            continue
        qm = geometry_terms(case_dir)
        rows = []
        for p in pred_by_case[case_id]:
            if p["mechlib_value"] == "NA":
                continue
            term = p["term"]
            kind = p["kind"]
            q = qm[term]
            amber = float(p["amber_value"])
            mech = float(p["mechlib_value"])
            chi = float(p["chi1_value"])
            shuffled = float(p["shuffled_value"])
            k = float(p["force_constant"])
            scale = math.pi / 180.0 if kind == "angle" else 1.0
            amber_abs = abs(amber-q)
            mech_abs = abs(mech-q)
            chi_abs = abs(chi-q)
            shuffled_abs = abs(shuffled-q)
            row = {
                "case_id": case_id, "residue": p["residue"], "region": p["region"],
                "term": term, "kind": kind, "force_constant": k,
                "qm_value": q, "amber_value": amber, "mechlib_value": mech,
                "chi1_value": chi, "shuffled_value": shuffled,
                "amber_abs_error": amber_abs, "mechlib_abs_error": mech_abs,
                "chi1_abs_error": chi_abs, "shuffled_abs_error": shuffled_abs,
                "amber_weighted_error": math.sqrt(k)*amber_abs*scale,
                "mechlib_weighted_error": math.sqrt(k)*mech_abs*scale,
                "chi1_weighted_error": math.sqrt(k)*chi_abs*scale,
                "shuffled_weighted_error": math.sqrt(k)*shuffled_abs*scale,
            }
            row["delta"] = row["amber_weighted_error"] - row["mechlib_weighted_error"]
            rows.append(row)
            detail.append(row)
        case_scores.append({
            "case_id": case_id, "residue": case_meta[case_id]["residue"],
            "region": case_meta[case_id]["region"], "terms_available": len(rows),
            "amber_score": np.mean([r["amber_weighted_error"] for r in rows]),
            "mechlib_score": np.mean([r["mechlib_weighted_error"] for r in rows]),
            "chi1_score": np.mean([r["chi1_weighted_error"] for r in rows]),
            "shuffled_score": np.mean([r["shuffled_weighted_error"] for r in rows]),
            "delta": np.mean([r["delta"] for r in rows]),
        })

    rng = np.random.default_rng(SEED)
    cluster_low, cluster_high, residue_means = cluster_ci(case_scores, rng)
    case_low, case_high, _ = ci([r["delta"] for r in case_scores], rng)
    primary_mean = float(np.mean([r["delta"] for r in case_scores]))

    residue_rows = []
    for residue in sorted({r["residue"] for r in case_scores}):
        vals = [float(r["delta"]) for r in case_scores if r["residue"] == residue]
        residue_rows.append({"residue": residue, "mean_delta": np.mean(vals),
                             "positive": np.mean(vals) > 0, "cases": len(vals)})
    positive_residues = int(sum(r["positive"] for r in residue_rows))

    region_rows = []
    for region in ["alpha", "beta", "ppii", "left"]:
        vals = [float(r["delta"]) for r in case_scores if r["region"] == region]
        low, high, _ = ci(vals, rng)
        region_rows.append({"region": region, "mean_delta": np.mean(vals),
                            "ci95_low": low, "ci95_high": high,
                            "positive_cases": sum(v > 0 for v in vals), "cases": len(vals)})

    term_rows = []
    term_groups = defaultdict(list)
    for row in detail:
        term_groups[row["term"]].append(row)
    one_sided_worse = []
    for term in sorted(term_groups):
        rows = term_groups[term]
        deltas = np.asarray([r["delta"] for r in rows])
        # One-sided paired test: negative delta means MechLib is worse.
        test = stats.ttest_1samp(deltas, popmean=0.0, alternative="less")
        one_sided_worse.append(float(test.pvalue))
        kind = rows[0]["kind"]
        term_rows.append({
            "term": term, "kind": kind, "n": len(rows),
            "amber_mae": np.mean([r["amber_abs_error"] for r in rows]),
            "mechlib_mae": np.mean([r["mechlib_abs_error"] for r in rows]),
            "amber_rmse": np.sqrt(np.mean([r["amber_abs_error"]**2 for r in rows])),
            "mechlib_rmse": np.sqrt(np.mean([r["mechlib_abs_error"]**2 for r in rows])),
            "amber_signed_bias": np.mean([r["amber_value"]-r["qm_value"] for r in rows]),
            "mechlib_signed_bias": np.mean([r["mechlib_value"]-r["qm_value"] for r in rows]),
            "mean_weighted_delta": np.mean(deltas),
            "positive_cases": int(np.sum(deltas > 0)),
            "one_sided_p_mechlib_worse": float(test.pvalue),
        })
    adjusted = holm(one_sided_worse)
    for row, value in zip(term_rows, adjusted):
        row["holm_p_mechlib_worse"] = float(value)
        row["holm_significant_worse"] = bool(value < 0.05 and row["mean_weighted_delta"] < 0)

    qc_pass = len(case_scores)
    no_term_worse = not any(r["holm_significant_worse"] for r in term_rows)
    criteria = {
        "clustered_ci_lower_gt_zero": bool(cluster_low > 0),
        "minimum_12_positive_residues": bool(positive_residues >= 12),
        "no_holm_significant_worse_term": bool(no_term_worse),
        "minimum_68_qc_pass_cases": bool(qc_pass >= 68),
    }
    status = "PASS" if all(criteria.values()) else ("INCONCLUSIVE" if qc_pass < 68 else "FAIL")
    result = {
        "contract_id": "MechLib-QM-Q1-v1",
        "status": status,
        "primary_endpoint_evaluated": True,
        "registered_cases": 80,
        "qc_pass_cases": qc_pass,
        "primary_mean_delta": primary_mean,
        "cluster_bootstrap_ci95": [cluster_low, cluster_high],
        "case_bootstrap_ci95_sensitivity": [case_low, case_high],
        "bootstrap_replicates": N_BOOT,
        "bootstrap_seed": SEED,
        "positive_residues": positive_residues,
        "total_residues": 20,
        "criteria": criteria,
        "interpretation_boundary": "Constrained local equilibrium geometry only; not dynamics, populations, folding thermodynamics, or NMR/RDC observables.",
    }

    write_tsv(OUT / "Q6_TERM_DETAIL.tsv", detail)
    write_tsv(OUT / "Q6_CASE_SCORES.tsv", case_scores)
    write_tsv(OUT / "Q6_RESIDUE_RESULTS.tsv", residue_rows)
    write_tsv(OUT / "Q6_REGION_RESULTS.tsv", region_rows)
    write_tsv(OUT / "Q6_TERM_RESULTS.tsv", term_rows)
    (OUT / "Q6_PRIMARY_ENDPOINT.json").write_text(json.dumps(result, indent=2) + "\n")

    # Compact manuscript figure.
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.8), constrained_layout=True)
    ax = axes[0, 0]
    x = np.asarray([r["amber_score"] for r in case_scores])
    y = np.asarray([r["mechlib_score"] for r in case_scores])
    colors = {"alpha":"#2878B5", "beta":"#E07B39", "ppii":"#59A14F", "left":"#B07AA1"}
    for region in colors:
        mask = np.asarray([r["region"] == region for r in case_scores])
        ax.scatter(x[mask], y[mask], s=25, alpha=.8, label=region, color=colors[region])
    lim = max(x.max(), y.max()) * 1.04
    ax.plot([0, lim], [0, lim], "--", color="0.35", lw=1)
    ax.set(xlabel="AMBER-fixed weighted error", ylabel="MechLib weighted error", title="A  Paired confirmatory cases")
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax = axes[0, 1]
    ordered = sorted(residue_rows, key=lambda r: r["mean_delta"])
    vals = [r["mean_delta"] for r in ordered]
    ax.barh([r["residue"] for r in ordered], vals,
            color=["#2A9D8F" if v > 0 else "#D55E00" for v in vals])
    ax.axvline(0, color="0.3", lw=.8)
    ax.set(xlabel="Mean Δ (positive favors MechLib)", title=f"B  Residue effects ({positive_residues}/20 positive)")

    ax = axes[1, 0]
    ordered_t = sorted(term_rows, key=lambda r: r["mean_weighted_delta"])
    vals_t = [r["mean_weighted_delta"] for r in ordered_t]
    labels = [r["term"].replace("angle_", "∠ ").replace("bond_", "") for r in ordered_t]
    ax.barh(labels, vals_t, color=["#2A9D8F" if v > 0 else "#D55E00" for v in vals_t])
    ax.axvline(0, color="0.3", lw=.8)
    ax.set(xlabel="Mean weighted Δ", title="C  Registered geometry terms")

    ax = axes[1, 1]
    means = np.asarray([r["mean_delta"] for r in region_rows])
    lows = means - np.asarray([r["ci95_low"] for r in region_rows])
    highs = np.asarray([r["ci95_high"] for r in region_rows]) - means
    ax.errorbar(range(4), means, yerr=[lows, highs], fmt="o", capsize=4,
                color="#3B5B92", markersize=6)
    ax.axhline(0, color="0.3", lw=.8)
    ax.set_xticks(range(4), [r["region"] for r in region_rows])
    ax.set(ylabel="Mean Δ (95% case-bootstrap CI)", title="D  Ramachandran regions")
    fig.suptitle(
        "Independent constrained-QM benchmark of MechLib equilibrium geometry\n"
        "Overall mean favors MechLib; strict registered endpoint fails on three bond terms",
        fontsize=12, weight="bold"
    )
    fig.savefig(OUT / "Figure_QM_validation.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "Figure_QM_validation.pdf", bbox_inches="tight")
    plt.close(fig)

    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    (OUT / "SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
