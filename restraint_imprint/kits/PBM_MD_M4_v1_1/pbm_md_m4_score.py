#!/usr/bin/env python3
"""
pbm_md_m4_score.py (v1.0.0, 2026-11-13) -- prespecified 3J(HN,Ha) scoring for the M4 MD referee.
Its SHA-256 is recorded in M4_LOCK.json before production; do not edit afterwards.
Refuses to score anything until MECHLIB_V3 has been added and every lineage of every arm is complete, so no NMR
result can inform the construction of v3.

Delta MSE = MSE(arm A) - MSE(arm B); positive favours B.
Primary contrast (M4_PROTOCOL.md): AMBER_FIXED - MECHLIB_V3. Secondary: AMBER_FIXED - MECHLIB_V2; MECHLIB_V2 - MECHLIB_V3.
  frames: production only; discard the first 20% of every lineage
  per lineage: mean predicted 3J per residue (J = 6.7cos^2(phi-60) - 1.3cos(phi-60) + 1.5 Hz) -> MSE vs 50 GB3 / 63 ubiquitin
  per protein: mean of the 8 paired seed contrasts, SD, 95% t interval (df = 7); equal-protein mean
  outcome per contrast: SUPPORTED (both means > 0, both intervals exclude 0) / DIRECTIONAL / MIXED / NOT_SUPPORTED
Sensitivities (reported; cannot change the primary outcome): S1 automated equilibration on Ca RMSD, S2 no discard,
S3 second half; 5-block convergence.
"""
import argparse, csv, hashlib, json, math, sys
from pathlib import Path

CASES = {"M2-GB3": ("M2E_GB3_TableS1_transcription_v1.tsv", 50),
         "M2-UBQ": ("M2E_UBQ_TableS3_HNHA_transcription_v1.tsv", 63)}
ARMS = ("AMBER_FIXED", "MECHLIB_V2", "MECHLIB_V3")
CONTRASTS = (("AMBER_FIXED", "MECHLIB_V3"), ("AMBER_FIXED", "MECHLIB_V2"), ("MECHLIB_V2", "MECHLIB_V3"))
T95 = {2: 4.302652729911275, 3: 3.182446305284263, 4: 2.776445105197799, 5: 2.570581835636314, 6: 2.446911848791681, 7: 2.364624251592785}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stat_ineff(x):
    import numpy as np
    x = np.asarray(x, float); n = len(x); x = x - x.mean(); v = (x * x).mean()
    if n < 10 or v == 0:
        return 1.0
    g = 1.0
    for t in range(1, n - 1):
        c = (x[:-t] * x[t:]).mean() / v
        if c <= 0:
            break
        g += 2 * c * (1 - t / n)
    return max(1.0, g)


def detect_equilibration(x, step_frac=0.01):
    n = len(x); best = (0, -1.0)
    for t0 in range(0, int(0.5 * n), max(1, int(step_frac * n))):
        g = stat_ineff(x[t0:]); neff = (n - t0) / g
        if neff > best[1]:
            best = (t0, neff)
    return best[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="~/pbm_md_m4")
    ap.add_argument("--nmr-dir", type=Path, default=Path("/mnt/c/Wei/PBM"))
    ap.add_argument("--preflight", action="store_true", help="check mapping and inputs only; no frames scored")
    a = ap.parse_args()
    import numpy as np, mdtraj as md
    out = Path(a.output).expanduser()
    lock = json.loads((out / "M4_LOCK.json").read_text())
    if not a.preflight and list(lock["arms"]) != list(ARMS):
        sys.exit("MECHLIB_V3 not yet added; scoring is refused until every arm is locked and complete")
    if sha256(Path(__file__).resolve()) != lock["scorer_sha256"]:
        sys.exit("scorer changed since lock; refusing to score")
    n_chunks = int(round(lock["production_ns"] / lock["protocol"]["production_chunk_ns"]))

    obs, phi_sel, tops = {}, {}, {}
    for case, (fn, expected) in CASES.items():
        top = md.load_pdb(str(out / "build" / case / "protein.pdb")); tops[case] = top
        phi_idx, _ = md.compute_phi(top)
        mapping = {top.topology.atom(int(q[2])).residue.resSeq: i for i, q in enumerate(phi_idx)}
        with open(a.nmr_dir / fn, newline="") as f:
            rows = [r for r in csv.DictReader(f, delimiter="\t") if r["primary_non_gly"] == "YES"]
        if len(rows) != expected:
            sys.exit(f"{case}: expected {expected} primary rows, found {len(rows)}")
        sel = []
        for r in rows:
            n = int(r["residue_number"])
            if n not in mapping:
                sys.exit(f"{case}: residue {n} has no phi in topology")
            res = top.topology.atom(int(phi_idx[mapping[n], 2])).residue
            if res.code != r["residue"]:
                sys.exit(f"{case}: residue {n} is {res.code}, table says {r['residue']}")
            sel.append(mapping[n])
        obs[case] = np.array([float(r["J_HN_Halpha_Hz"]) for r in rows])
        phi_sel[case] = (phi_idx, sel, [r["residue_number"] for r in rows])
    if a.preflight:
        print("SCORE PREFLIGHT PASS: 113 non-Gly observations mapped (50 GB3, 63 ubiquitin); no frames scored")
        return 0

    seeds = lock["seeds"]; lineage = {}; residue_rows = []
    for case in CASES:
        phi_idx, sel, resnums = phi_sel[case]; ca = tops[case].topology.select("name CA")
        for arm in ARMS:
            for seed in seeds:
                d = out / "runs" / case / arm / f"seed_{seed}"
                files = [d / f"prod_{c:03d}.dcd" for c in range(n_chunks)]
                if not all(f.exists() for f in files):
                    sys.exit(f"incomplete lineage: {d}")
                tr = md.load([str(f) for f in files], top=tops[case])
                phi = md.compute_dihedrals(tr, phi_idx)[:, sel]
                c = np.cos(phi - np.pi / 3); J = 6.7 * c * c - 1.3 * c + 1.5
                rmsd = md.rmsd(tr, tr, 0, atom_indices=ca)
                n = tr.n_frames
                windows = {"primary_drop20": int(0.2 * n), "S1_auto_equil": detect_equilibration(rmsd),
                           "S2_no_discard": 0, "S3_second_half": n // 2}
                res = {}
                for w, t0 in windows.items():
                    m = J[t0:].mean(0); res[w] = {"t0_frame": int(t0), "mse": float(((m - obs[case]) ** 2).mean())}
                t0 = windows["primary_drop20"]; blocks = np.array_split(np.arange(t0, n), 5)
                res["block_mse"] = [float(((J[b].mean(0) - obs[case]) ** 2).mean()) for b in blocks]
                res["frames"] = n; res["rmsd_ca_nm_mean_last_half"] = float(rmsd[n // 2:].mean())
                lineage[(case, arm, seed)] = res
                m = J[t0:].mean(0)
                for i, rn in enumerate(resnums):
                    residue_rows.append({"case": case, "arm": arm, "seed": seed, "residue_number": rn,
                                         "observed_hz": float(obs[case][i]), "predicted_hz": float(m[i])})
                print(f"  {case} {arm} {seed}: frames {n}, primary MSE {res['primary_drop20']['mse']:.4f} Hz²", flush=True)

    summary = {"lock_sha256": sha256(out / "M4_LOCK.json"), "positive_favours": "second arm of each contrast", "contrasts": {}}
    for A, Bm in CONTRASTS:
        cw = {}
        for w in ("primary_drop20", "S1_auto_equil", "S2_no_discard", "S3_second_half"):
            per = {}
            for case in CASES:
                d = [lineage[(case, A, s_)][w]["mse"] - lineage[(case, Bm, s_)][w]["mse"] for s_ in seeds]
                mu = float(np.mean(d)); sd = float(np.std(d, ddof=1)); half = T95[len(d) - 1] * sd / math.sqrt(len(d))
                per[case] = {"paired_deltas_hz2": d, "mean": mu, "sd": sd, "ci95": [mu - half, mu + half]}
            means = [per[c]["mean"] for c in CASES]
            if all(m > 0 for m in means):
                status = "SUPPORTED" if all(per[c]["ci95"][0] > 0 for c in CASES) else "DIRECTIONAL"
            elif all(m <= 0 for m in means):
                status = "NOT_SUPPORTED"
            else:
                status = "MIXED"
            cw[w] = {"per_protein": per, "equal_protein_mean_hz2": float(np.mean(means)), "outcome": status}
        blk = {}
        for case in CASES:
            bd = np.array([[x - y for x, y in zip(lineage[(case, A, s_)]["block_mse"], lineage[(case, Bm, s_)]["block_mse"])] for s_ in seeds])
            blk[case] = {"block_mean_delta_hz2": bd.mean(0).tolist(), "block_sd_across_blocks": float(bd.mean(0).std(ddof=1))}
        summary["contrasts"][f"{A}-{Bm}"] = {"windows": cw, "block_convergence": blk}
    summary["primary_outcome"] = summary["contrasts"]["AMBER_FIXED-MECHLIB_V3"]["windows"]["primary_drop20"]["outcome"]
    summary["lineages"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in lineage.items()}
    (out / "M4_SCORE_SUMMARY.json").write_text(json.dumps(summary, indent=1))
    with open(out / "M4_SCORE_RESIDUES.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(residue_rows[0]), delimiter="\t"); w.writeheader(); w.writerows(residue_rows)
    for k, v in summary["contrasts"].items():
        p = v["windows"]["primary_drop20"]
        print(k, json.dumps({c: {"mean": round(x["mean"], 4), "ci95": [round(y, 4) for y in x["ci95"]]} for c, x in p["per_protein"].items()}), p["outcome"])
    print("PRIMARY OUTCOME (AMBER_FIXED - MECHLIB_V3):", summary["primary_outcome"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
