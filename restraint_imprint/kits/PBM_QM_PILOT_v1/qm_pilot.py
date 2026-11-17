#!/usr/bin/env python3
"""QM noise-floor pilot (spec QM_PILOT_SPEC.md v1.0).

Commands:
  selftest                     offline checks of geometry, inputs and gates (no QM)
  preflight [--q5-root DIR]    find the frozen Q5 geometries, verify them against Q6, test each arm on water
  run [--workers 2 --threads 4] run the 12 cases x 4 arms (resumable; single writer)
  status                       progress
  analyse                      shifts from the frozen reference, per arm and term
No library is read anywhere in this kit.
"""
import argparse, csv, fcntl, hashlib, json, math, os, re, shutil, subprocess, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

KIT = Path(__file__).resolve().parent
OUT_DEFAULT = Path.home() / "pbm_qm_pilot"
SHA = {"cases": "160e367216a7c1205d5d069c98a0addf8cbcb4cf74ed6fd771c7efe0491552c1",
       "q6": "3af8c2fee6113422004b1a2a9f36d9d8c0887e7c41077bbd8dda0b2e7c3170b1"}
RESIDUES = ("ALA", "GLY", "SER")
REGIONS = ("alpha", "beta", "ppii", "left")
CASES = [f"C-{r}-{g}" for r in RESIDUES for g in REGIONS]
ARMS = {  # order = run order
    "R0": dict(method="wb97x-d", basis="def2-svp", medium="gas"),
    "S1": dict(method="wb97x-d", basis="def2-svp", medium="water"),
    "B1": dict(method="wb97x-d", basis="def2-tzvp", medium="gas"),
    "F1": dict(method="wb97m-v", basis="def2-tzvp", medium="gas"),
}
F1_CANDIDATES = ("wb97m-v", "m06-2x", "pbe0")  # first that passes the water test is used (spec amendment 1)
SUBSET_ARMS = ("B1", "F1")                          # amendment 3: def2-TZVP arms on ALA only (runtime)
SUBSET_CASES = tuple(f"C-ALA-{g}" for g in ("alpha", "beta", "ppii", "left"))
Q6_TERMS = ["tau_deg", "angle_N_CA_CB", "angle_C_CA_CB", "angle_CaCN", "angle_CNCa", "angle_CA_C_O",
            "bond_N_CA", "bond_CA_C", "bond_C_O", "bond_C_N_next", "bond_CA_CB"]
TERMS = Q6_TERMS + ["angle_O_C_N"]
SOLVENT_BLOCKS = {
    "ddx": ("set {\n    ddx true\n    ddx_model pcm\n    ddx_solvent water\n    ddx_radii_set uff\n}\n", ""),
    "pcmsolver": ("set {\n    pcm true\n    pcm_scf_type total\n}\n",
                  "pcm = {\n   Units = Angstrom\n   Medium {\n   SolverType = IEFPCM\n   Solvent = Water\n   }\n"
                  "   Cavity {\n   RadiiSet = UFF\n   Type = GePol\n   Scaling = False\n   Area = 0.3\n   Mode = Implicit\n   }\n}\n"),
}
GATE_TOL = {"deg": 1e-4, "A": 1e-6}
R0_FLAG = {"deg": 0.05, "A": 0.0005}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def read_tsv(p):
    with open(p, newline="") as f: return list(csv.DictReader(f, delimiter="\t"))


def read_xyz(p):
    L = Path(p).read_text().splitlines(); n = int(L[0]); rows = L[2:2 + n]
    if len(rows) != n: raise ValueError(f"malformed xyz {p}")
    return [r.split()[0] for r in rows], np.array([[float(v) for v in r.split()[1:4]] for r in rows]), L[1]


def write_xyz(p, sym, xyz, comment=""):
    Path(p).write_text(f"{len(sym)}\n{comment}\n" + "".join(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n" for s, (x, y, z) in zip(sym, xyz)))


def last_frame(traj, out):
    L = Path(traj).read_text().splitlines(); n = int(L[0]); size = n + 2
    if len(L) % size: raise ValueError("malformed trajectory")
    Path(out).write_text("\n".join(L[-size:]) + "\n"); return len(L) // size


def ang(x, a, b, c):
    u, v = x[a - 1] - x[b - 1], x[c - 1] - x[b - 1]
    return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))


def dist(x, a, b): return float(np.linalg.norm(x[a - 1] - x[b - 1]))


def dihedral(x, ids):
    p0, p1, p2, p3 = (x[i - 1] for i in ids)
    b0, b1, b2 = -(p1 - p0), p2 - p1, p3 - p2; b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - (b0 @ b1) * b1, b2 - (b2 @ b1) * b1
    return math.degrees(math.atan2(np.cross(b1, v) @ w, v @ w))


def wrap(d): return (d + 180.0) % 360.0 - 180.0


def idx(atoms, res, name):
    f = [int(r["atom_index_1based"]) for r in atoms if int(r["residue_index_1based"]) == res and r["atom_name"] == name]
    if len(f) != 1: raise ValueError(f"expected one {res}:{name}, got {f}")
    return f[0]


def terms(atoms, x):
    """Q6 definitions (q6_analyze.py) plus O-C-N."""
    pC, N, CA, C, O, nN = idx(atoms, 1, "C"), idx(atoms, 2, "N"), idx(atoms, 2, "CA"), idx(atoms, 2, "C"), idx(atoms, 2, "O"), idx(atoms, 3, "N")
    v = {"tau_deg": ang(x, N, CA, C), "angle_CaCN": ang(x, CA, C, nN), "angle_CNCa": ang(x, pC, N, CA),
         "angle_CA_C_O": ang(x, CA, C, O), "angle_O_C_N": ang(x, O, C, nN),
         "bond_N_CA": dist(x, N, CA), "bond_CA_C": dist(x, CA, C), "bond_C_O": dist(x, C, O), "bond_C_N_next": dist(x, C, nN)}
    cb = [r for r in atoms if int(r["residue_index_1based"]) == 2 and r["atom_name"] == "CB"]
    if cb:
        CB = int(cb[0]["atom_index_1based"])
        v.update({"angle_N_CA_CB": ang(x, N, CA, CB), "angle_C_CA_CB": ang(x, C, CA, CB), "bond_CA_CB": dist(x, CA, CB)})
    return v


def unit(t): return "A" if t.startswith("bond") else "deg"


def load_inputs():
    for k, f in (("cases", "cases_confirmatory.tsv"), ("q6", "Q6_TERM_DETAIL.tsv")):
        if sha256(KIT / "inputs" / f) != SHA[k]: sys.exit(f"REFUSING: inputs/{f} hash differs from the frozen file")
    cases = {r["case_id"]: r for r in read_tsv(KIT / "inputs" / "cases_confirmatory.tsv")}
    q6 = {}
    for r in read_tsv(KIT / "inputs" / "Q6_TERM_DETAIL.tsv"): q6.setdefault(r["case_id"], {})[r["term"]] = float(r["qm_value"])
    return cases, q6


def reference_gate(q5case, q6vals):
    atoms = read_tsv(q5case / "atom_map.tsv"); _, x, _ = read_xyz(q5case / "qm" / "qm_final_geometry.xyz")
    v = terms(atoms, x); worst = {}
    for t, ref in q6vals.items():
        worst[t] = abs(v[t] - ref)
    bad = {t: d for t, d in worst.items() if d > GATE_TOL[unit(t)]}
    return not bad, bad, v


def find_q5_roots(roots):
    hits = []
    for r in roots:
        r = Path(r).expanduser()
        if not r.exists(): continue
        try:
            out = subprocess.run(["find", str(r), "-maxdepth", "6", "-name", "GATE_Q5_EXECUTION_SUMMARY.json",
                                  "-not", "-path", "*pbm_md*"], capture_output=True, text=True, timeout=600).stdout
        except subprocess.TimeoutExpired:
            continue
        hits += [Path(p).parent for p in out.split()]
    return sorted(set(hits))


# ---------------------------------------------------------------- QM input
def psi4_input(xyz_path, charge, mult, arm, solvent_kind, f1_method=None):
    sym, x, _ = read_xyz(xyz_path); a = dict(ARMS[arm])
    if arm == "F1" and f1_method: a["method"] = f1_method
    geom = "\n".join(f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}" for s, p in zip(sym, x))
    extra_set, extra_block = ("", "")
    # Amendment 2: the F1 functional (meta-GGA fallbacks are grid-sensitive) uses a (99, 590) grid
    grid = "set {\n    dft_radial_points 99\n    dft_spherical_points 590\n}\n" if arm == "F1" else ""
    if a["medium"] == "water":
        if solvent_kind not in SOLVENT_BLOCKS: raise RuntimeError("SOLVENT_UNAVAILABLE")
        extra_set, extra_block = SOLVENT_BLOCKS[solvent_kind]
    return (f"memory 4 GB\n\nmolecule peptide {{\n{charge} {mult}\n{geom}\nsymmetry c1\nno_com\nno_reorient\n}}\n\n"
            f"set {{\n    basis {a['basis']}\n    scf_type df\n    reference rhf\n    e_convergence 1.0e-8\n"
            f"    d_convergence 1.0e-8\n    maxiter 200\n}}\n{grid}{extra_set}{extra_block}\ngradient('{a['method']}')\n")


def constraints_text(build, row):
    return ("$set\n" + "dihedral " + " ".join(map(str, build["phi_indices_1based"])) + f" {float(row['phi_deg']):.8f}\n"
            + "dihedral " + " ".join(map(str, build["psi_indices_1based"])) + f" {float(row['psi_deg']):.8f}\n")


def run_cmd(cmd, cwd, log, env, timeout=None):
    t = time.time()
    with open(log, "w") as h:
        try:
            r = subprocess.run(cmd, cwd=cwd, stdout=h, stderr=subprocess.STDOUT, text=True, env=env, timeout=timeout)
            code = r.returncode
        except subprocess.TimeoutExpired:
            code = -9
    return code, time.time() - t


def geometric_opt(d, env, threads, restarts=2):
    for att in range(restarts + 1):
        code, sec = run_cmd(["geometric-optimize", "--engine", "psi4", "--nt", str(threads), "--enforce", "0.01",
                             "--conmethod", "1", "psi4_opt.in", "constraints.txt"], d, d / f"geometric_attempt_{att}.log", env)
        log = (d / f"geometric_attempt_{att}.log").read_text(errors="replace"); traj = d / "psi4_opt_optim.xyz"
        if code == 0 and "Converged!" in log and traj.is_file(): return att, True
        if traj.is_file() and att < restarts:
            last_frame(traj, d / "start.xyz")
            traj.rename(d / f"attempt_{att}_psi4_opt_optim.xyz")
            tmp = d / "psi4_opt.tmp"
            if tmp.exists(): tmp.rename(d / f"attempt_{att}_psi4_opt.tmp")
            # rebuild input from the last frame
            txt = (d / "psi4_opt.in").read_text()
            sym, x, _ = read_xyz(d / "start.xyz")
            new_geom = "\n".join(f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}" for s, p in zip(sym, x))
            txt = re.sub(r"(molecule peptide \{\n-?\d+ \d+\n)(.*?)(\nsymmetry c1)", lambda m: m.group(1) + new_geom + m.group(3), txt, flags=re.S)
            (d / "psi4_opt.in").write_text(txt)
        else:
            return att, False
    return restarts, False


def run_one(arm, case_id, row, q5root, out, threads, solvent_kind, f1_method=None):
    d = out / arm / case_id; d.mkdir(parents=True, exist_ok=True)
    st = {"arm": arm, "case_id": case_id, "status": "PENDING", **ARMS[arm]}
    if arm == "F1" and f1_method: st["method"] = f1_method
    env = os.environ.copy(); env.update({k: str(threads) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    try:
        src = q5root / case_id
        build = json.loads((src / "build_audit.json").read_text())
        shutil.copy2(src / "qm" / "qm_final_geometry.xyz", d / "start.xyz")
        (d / "constraints.txt").write_text(constraints_text(build, row))
        (d / "psi4_opt.in").write_text(psi4_input(d / "start.xyz", row["charge"], row["multiplicity"], arm, solvent_kind, f1_method))
        t0 = time.time(); att, ok = geometric_opt(d, env, threads); st["seconds"] = round(time.time() - t0, 1); st["restarts_used"] = att
        if not ok: raise RuntimeError("OPT_FAILED")
        frames = last_frame(d / "psi4_opt_optim.xyz", d / "final.xyz")
        sym, x, comment = read_xyz(d / "final.xyz"); sym0, x0, _ = read_xyz(src / "qm" / "qm_final_geometry.xyz")
        if sym != sym0: raise RuntimeError("ATOM_ORDER_CHANGED")
        phi_err = abs(wrap(dihedral(x, build["phi_indices_1based"]) - float(row["phi_deg"])))
        psi_err = abs(wrap(dihedral(x, build["psi_indices_1based"]) - float(row["psi_deg"])))
        if max(phi_err, psi_err) > 0.5: raise RuntimeError("CONSTRAINT_FAILED")
        bonds = [(int(r["atom_i_1based"]), int(r["atom_j_1based"])) for r in read_tsv(src / "connectivity.tsv")]
        excess = max(dist(x, i, j) - (2.15 if "S" in (sym[i - 1], sym[j - 1]) else 1.80) for i, j in bonds)
        change = max(abs(dist(x, i, j) - dist(x0, i, j)) for i, j in bonds)
        if excess > 1e-6 or change > 0.25: raise RuntimeError("CONNECTIVITY_CHANGED")
        chi = None
        if build.get("chi1_indices_1based"): chi = [dihedral(x0, build["chi1_indices_1based"]), dihedral(x, build["chi1_indices_1based"])]
        st.update(status="PASS", frames=frames, phi_error_deg=phi_err, psi_error_deg=psi_err, chi1_start_final_deg=chi,
                  max_bond_change_A=change, energy_comment=comment,
                  terms=terms(read_tsv(src / "atom_map.tsv"), x))
    except Exception as e:
        st.update(status="FAIL", failure=str(e)); (d / "traceback.txt").write_text(traceback.format_exc())
    (d / "status.json").write_text(json.dumps(st, indent=2) + "\n")
    return st


# ---------------------------------------------------------------- commands
def cmd_preflight(a):
    cases, q6 = load_inputs(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if any((out / arm).exists() for arm in ARMS): sys.exit("This output already holds pilot runs; preflight would change the record. Use a new OUT.")
    if (out / "RUN_RECORD.json").exists(): (out / "RUN_RECORD.json").rename(out / f"RUN_RECORD.void_{int(time.time())}.json")
    rep = {"kit_sha256": {str(f.relative_to(KIT)): sha256(f) for f in sorted(KIT.rglob("*")) if f.is_file() and "__pycache__" not in str(f)}, "nproc": os.cpu_count()}
    for exe in ("psi4", "geometric-optimize"):
        if not shutil.which(exe): sys.exit(f"Missing executable {exe}: activate mechlib_qm_clean")
    rep["psi4_version"] = subprocess.run(["psi4", "--version"], capture_output=True, text=True).stdout.strip()
    try:
        import geometric; rep["geometric_version"] = geometric.__version__
    except Exception as e: rep["geometric_version"] = f"import failed: {e}"
    roots = [a.q5_root] if a.q5_root else ["/mnt/c/Wei/PBM", "~"]
    cands = find_q5_roots(roots) if not a.q5_root else [Path(a.q5_root)]
    rep["q5_candidates"] = []; chosen = None
    for c in cands:
        info = {"path": str(c)}
        try:
            info["summary"] = json.loads((c / "GATE_Q5_EXECUTION_SUMMARY.json").read_text())
            res = {cid: reference_gate(c / cid, q6[cid])[:2] for cid in CASES}
            info["gate_pass"] = all(ok for ok, _ in res.values())
            info["failures"] = {cid: bad for cid, (ok, bad) in res.items() if not ok}
        except Exception as e:
            info["gate_pass"] = False; info["error"] = str(e)
        rep["q5_candidates"].append(info)
        if info["gate_pass"] and chosen is None: chosen = c
    if chosen is None:
        (out / "PREFLIGHT.json").write_text(json.dumps(rep, indent=2, default=str))
        sys.exit("No Q5 output reproduces the frozen Q6 reference values for the 12 pilot cases; see PREFLIGHT.json, "
                 "or pass --q5-root /path/to/work_q5")
    rep["q5_root"] = str(chosen)
    print(f"Reference gate PASS: {chosen} reproduces Q6 qm_value for all 12 pilot cases")
    # water tests through the same geomeTRIC route
    wt = out / "_preflight_water"; shutil.rmtree(wt, ignore_errors=True); wt.mkdir(parents=True)
    env = os.environ.copy(); env.update({k: str(a.threads) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    sym = ["O", "H", "H"]; xyz = np.array([[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]])
    rep["water_tests"] = {}; rep["solvent_kind"] = None; rep["f1_method"] = None

    def wtest(label, arm, sk=None, f1=None):
        d = wt / label; d.mkdir(); write_xyz(d / "w.xyz", sym, xyz)
        (d / "psi4_opt.in").write_text(psi4_input(d / "w.xyz", 0, 1, arm, sk, f1))
        code, sec = run_cmd(["geometric-optimize", "--engine", "psi4", "--nt", str(a.threads), "psi4_opt.in"], d, d / "geometric.log", env, timeout=3600)
        log = (d / "geometric.log").read_text(errors="replace"); ok = code == 0 and "Converged!" in log
        tails = {}
        if not ok:
            for f in [d / "geometric.log"] + sorted(d.rglob("*.out")) + sorted(d.rglob("output.dat")) + sorted(d.rglob("*.err")):
                L = [l for l in f.read_text(errors="replace").splitlines() if l.strip()]
                err = [l for l in L if re.search(r"error|Error|ERROR|not |Unknown|unknown|No module|Traceback|Exception", l)]
                tails[str(f.relative_to(d))] = (err[-6:] or L[-6:])
        rep["water_tests"][label] = {"ok": ok, "seconds": round(sec, 1), "returncode": code, "errors": tails}
        print(f"water {label}: {'ok' if ok else 'FAILED'} ({sec:.0f} s)")
        for f, ls in tails.items():
            for l in ls: print(f"    [{f}] {l[:200]}")
        return ok

    avail = []
    if wtest("R0_gas", "R0"): avail.append("R0")
    for sk in ("ddx", "pcmsolver"):
        if wtest(f"S1_{sk}", "S1", sk=sk): rep["solvent_kind"] = sk; avail.append("S1"); break
    if wtest("B1_gas", "B1"): avail.append("B1")
    for f1 in F1_CANDIDATES:
        if wtest(f"F1_{f1}", "F1", f1=f1): rep["f1_method"] = f1; avail.append("F1"); break
    gas_ok = "R0" in avail
    rep["arms_available"] = [k for k in ARMS if k in avail]
    (out / "PREFLIGHT.json").write_text(json.dumps(rep, indent=2, default=str))
    print("Arms available:", rep["arms_available"], "| solvent:", rep["solvent_kind"] or "NONE (S1 skipped)",
          "| F1 functional:", rep["f1_method"] or "NONE (F1 skipped)")
    print("PREFLIGHT", "PASS" if gas_ok else "FAIL")


def cmd_run(a):
    cases, q6 = load_inputs(); out = Path(a.out)
    pf = json.loads((out / "PREFLIGHT.json").read_text()) if (out / "PREFLIGHT.json").exists() else sys.exit("run preflight first")
    if "q5_root" not in pf: sys.exit("preflight did not find the Q5 reference")
    lock = open(out / "RUN.lock", "a+")
    try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: sys.exit("Another runner is using this output directory")
    if "R0" not in pf.get("arms_available", []): sys.exit("Preflight did not pass R0; fix preflight first (nothing run)")
    rec = {"q5_root": pf["q5_root"], "solvent_kind": pf["solvent_kind"], "f1_method": pf.get("f1_method"), "arms": pf["arms_available"],
           "kit": pf["kit_sha256"], "threads": a.threads, "workers": a.workers}
    rp = out / "RUN_RECORD.json"
    if rp.exists():
        old = json.loads(rp.read_text())
        if {k: old[k] for k in ("q5_root", "solvent_kind", "f1_method", "arms", "kit") if k in old} != {k: rec[k] for k in ("q5_root", "solvent_kind", "f1_method", "arms", "kit") if k in old}:
            sys.exit("RUN_RECORD differs (reference, solvent choice, arms or kit changed): refusing to mix runs")
    rp.write_text(json.dumps(rec, indent=2))
    q5 = Path(pf["q5_root"])
    for cid in CASES:  # re-verify reference before every run
        ok, bad, _ = reference_gate(q5 / cid, q6[cid])
        if not ok: sys.exit(f"Reference changed for {cid}: {bad}")
    todo = []
    for arm in rec["arms"]:
        for cid in CASES:
            if arm in SUBSET_ARMS and cid not in SUBSET_CASES:   # amendment 3 (runtime)
                continue
            sp = out / arm / cid / "status.json"
            if sp.exists() and json.loads(sp.read_text())["status"] in ("PASS", "FAIL"): continue
            if (out / arm / cid).exists(): (out / arm / cid).rename(out / arm / f"{cid}__INTERRUPTED_{int(time.time())}")
            todo.append((arm, cid))
    print(f"{len(todo)} jobs to run ({len(rec['arms'])} arms x {len(CASES)} cases; done jobs kept)", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:  # submission order = arm order
        fut = {pool.submit(run_one, arm, cid, cases[cid], q5, out, a.threads, rec["solvent_kind"], rec["f1_method"]): (arm, cid) for arm, cid in todo}
        for f in as_completed(fut):
            s = f.result(); print(time.strftime("%H:%M"), s["arm"], s["case_id"], s["status"], s.get("failure", ""), f"{s.get('seconds', 0):.0f}s", flush=True)
    cmd_analyse(a)


def collect(out):
    rows = []
    for sp in sorted(out.glob("*/C-*/status.json")):
        s = json.loads(sp.read_text())
        if s["status"] == "PASS": rows.append(s)
    return rows


def cmd_status(a):
    out = Path(a.out)
    for arm in ARMS:
        st = [json.loads(p.read_text())["status"] for p in (out / arm).glob("C-*/status.json")] if (out / arm).exists() else []
        secs = [json.loads(p.read_text()).get("seconds", 0) for p in (out / arm).glob("C-*/status.json")] if st else []
        print(f"{arm}: {st.count('PASS')} pass, {st.count('FAIL')} fail, of 12" + (f"; mean {np.mean(secs)/60:.0f} min/case" if secs else ""))


def summarise(rows, q6):
    """rows: status dicts with terms; returns per-row shifts and per arm x term summary."""
    det = []
    for s in rows:
        for t in TERMS:
            if t not in s["terms"]: continue
            ref = q6[s["case_id"]].get(t)
            det.append({"arm": s["arm"], "case_id": s["case_id"], "residue": s["case_id"].split("-")[1], "term": t,
                        "value": s["terms"][t], "q6_reference": ref, "shift": None if ref is None else s["terms"][t] - ref})
    return det


def cmd_analyse(a):
    cases, q6 = load_inputs(); out = Path(a.out); rows = collect(out)
    # O-C-N reference = R0-independent: recompute from the frozen geometry
    pf = json.loads((out / "PREFLIGHT.json").read_text()); q5 = Path(pf["q5_root"])
    for cid in CASES:
        _, _, v = reference_gate(q5 / cid, q6[cid]); q6[cid]["angle_O_C_N"] = v["angle_O_C_N"]
    det = summarise(rows, q6)
    with open(out / "PILOT_TERM_SHIFTS.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(det[0]) if det else ["arm"], delimiter="\t"); w.writeheader(); w.writerows(det)
    summ = {}
    for arm in ARMS:
        for t in TERMS:
            d = [r for r in det if r["arm"] == arm and r["term"] == t]
            if not d: continue
            sh = np.array([r["shift"] for r in d])
            res = {}
            for r in d: res.setdefault(r["residue"], []).append(r["shift"])
            shape = np.concatenate([np.array(v) - np.mean(v) for v in res.values() if len(v) == 4]) if any(len(v) == 4 for v in res.values()) else np.array([])
            summ.setdefault(arm, {})[t] = {"unit": unit(t), "n": len(sh), "mean_shift": float(sh.mean()),
                                           "sd_shift": float(sh.std(ddof=1)) if len(sh) > 1 else None,
                                           "mean_abs_shift": float(np.abs(sh).mean()), "max_abs_shift": float(np.abs(sh).max()),
                                           "shape_sd": float(np.sqrt((shape ** 2).sum() / max(len(shape) - len([v for v in res.values() if len(v) == 4]), 1))) if len(shape) else None}
    flags = []
    for t, v in summ.get("R0", {}).items():
        if v["max_abs_shift"] > R0_FLAG[v["unit"]]: flags.append(f"R0 {t} max |shift| {v['max_abs_shift']:.5f} {v['unit']}")
    result = {"spec": "QM_PILOT_SPEC.md v1.0", "cases_done": {arm: len({r['case_id'] for r in det if r['arm'] == arm}) for arm in ARMS},
              "solvent_kind": pf.get("solvent_kind"), "f1_method": pf.get("f1_method"), "R0_reproduction_flags": flags, "summary": summ}
    (out / "PILOT_SUMMARY.json").write_text(json.dumps(result, indent=2))
    print(f"{'arm':4} {'term':15} {'n':>3} {'mean':>9} {'sd':>9} {'shapeSD':>9}")
    for arm, d in summ.items():
        for t, v in d.items():
            f = (lambda z: "" if z is None else f"{z:9.4f}" if v["unit"] == "deg" else f"{z:9.5f}")
            print(f"{arm:4} {t:15} {v['n']:3d} {f(v['mean_shift'])} {f(v['sd_shift'])} {f(v['shape_sd'])}")
    print("R0 reproduction:", "OK" if not flags else "; ".join(flags))
    hashes = {p.name: sha256(p) for p in (out / "PILOT_SUMMARY.json", out / "PILOT_TERM_SHIFTS.tsv")}
    (out / "RESULT_HASHES.json").write_text(json.dumps({"written": time.strftime("%Y-%m-%dT%H:%M:%S"), **hashes}, indent=2))


# ---------------------------------------------------------------- selftest
def cmd_selftest(a):
    import tempfile
    cases, q6 = load_inputs()
    assert len(cases) == 80 and all(c in cases and c in q6 for c in CASES)
    assert all(len(q6[c]) == (8 if "GLY" in c else 11) for c in CASES)
    # geometry functions on a constructed ACE-X-NME skeleton with known values
    x = np.array([[0, 0, 0], [1.33, 0, 0], [1.33 + 1.46 * math.cos(math.radians(70)), 1.46 * math.sin(math.radians(70)), 0],
                  [0, 1, 0], [-1, 0, 0], [3, 3, 0], [3, 3, 1.5], [-0.6, -1.0, 0]], float)
    atoms = [dict(atom_index_1based=str(i + 1), residue_index_1based=str(r), atom_name=n)
             for i, (r, n) in enumerate([(2, "N"), (1, "C"), (2, "CA"), (2, "C"), (3, "N"), (2, "O"), (2, "CB"), (9, "X")])]
    v = terms(atoms, x)
    assert abs(v["angle_CNCa"] - ang(x, 2, 1, 3)) < 1e-12 and abs(v["bond_C_N_next"] - dist(x, 4, 5)) < 1e-12
    assert abs(ang(np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]], float), 1, 2, 3) - 90) < 1e-9
    assert abs(dihedral(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]], float), [1, 2, 3, 4]) - 90) < 1e-9 or \
           abs(abs(dihedral(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]], float), [1, 2, 3, 4])) - 90) < 1e-9
    # input rendering
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "w.xyz"; write_xyz(p, ["O", "H", "H"], np.eye(3))
        for arm in ARMS:
            for sk in ([None] if arm != "S1" else list(SOLVENT_BLOCKS)):
                t = psi4_input(p, 0, 1, arm, sk)
                assert f"gradient('{ARMS[arm]['method']}')" in t and ARMS[arm]["basis"] in t
                assert ("ddx true" in t) == (sk == "ddx") and ("pcm true" in t) == (sk == "pcmsolver")
                assert ("dft_spherical_points 590" in t) == (arm == "F1")
        try: psi4_input(p, 0, 1, "S1", None); raise AssertionError("S1 without solvent must refuse")
        except RuntimeError: pass
        # restart geometry substitution regex
        txt = psi4_input(p, 0, 1, "R0", None); new = "O 9 9 9\nH 1 1 1\nH 2 2 2"
        txt2 = re.sub(r"(molecule peptide \{\n-?\d+ \d+\n)(.*?)(\nsymmetry c1)", lambda m: m.group(1) + new + m.group(3), txt, flags=re.S)
        assert "O 9 9 9" in txt2 and "symmetry c1" in txt2 and txt2.count("molecule peptide") == 1
        # reference gate on a synthetic Q5 case
        q5 = Path(td) / "q5" / "C-ALA-alpha"; (q5 / "qm").mkdir(parents=True)
        write_xyz(q5 / "qm" / "qm_final_geometry.xyz", ["N", "C", "C", "C", "N", "O", "C", "H"], x)
        with open(q5 / "atom_map.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["atom_index_1based", "residue_index_1based", "atom_name"], delimiter="\t"); w.writeheader(); w.writerows(atoms)
        ref = {t: v[t] for t in Q6_TERMS}
        assert reference_gate(q5, ref)[0]
        ref2 = dict(ref); ref2["bond_C_O"] += 2e-6
        assert not reference_gate(q5, ref2)[0]
        # analysis arithmetic
        rows = [{"arm": "B1", "case_id": f"C-ALA-{g}", "terms": {"tau_deg": 110 + k}} for k, g in enumerate(REGIONS)]
        qq = {f"C-ALA-{g}": {"tau_deg": 110.0} for g in REGIONS}
        det = summarise(rows, qq); assert [r["shift"] for r in det] == [0, 1, 2, 3]
    print("SELFTEST PASS: inputs hash-checked, term definitions, input rendering, restart edit, reference gate, shift arithmetic")


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    for n in ("selftest", "preflight", "run", "status", "analyse"):
        s = sub.add_parser(n); s.add_argument("--out", default=str(OUT_DEFAULT))
        s.add_argument("--threads", type=int, default=4); s.add_argument("--workers", type=int, default=2)
        s.add_argument("--q5-root", default=None)
    a = p.parse_args(); globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
