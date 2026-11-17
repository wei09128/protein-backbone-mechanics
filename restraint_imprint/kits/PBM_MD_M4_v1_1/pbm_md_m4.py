#!/usr/bin/env python3
"""
pbm_md_m4.py (v1.0.0, 2026-11-13) -- MD/NMR referee for MechLib v2 and v3 (M4), built on the M3 runner.

Reuses the M3 AMBER_FIXED systems and prepared structures unchanged (hash-checked against M3_LOCK.json) and adds
MechLib arms with merged per-term forces and the next-Pro split (build_m4_systems.py). Design in M4_PROTOCOL.md:
8 paired seeds x 150 ns per protein and arm, 4 fs + HMR; arms AMBER_FIXED and MECHLIB_V2 locked now, MECHLIB_V3
registered now and added with `add-v3` (its system hash recorded) before any V3 frame exists.

Subcommands:
  preflight  check inputs and OpenMM
  build      copy the M3 AMBER_FIXED builds and build MECHLIB_V2 for GB3 and ubiquitin
  bench      speed test (4 fs + HMR) of every built arm
  lock       fix the design and hashes (M4_LOCK.json); required by run
  add-v3     build MECHLIB_V3 from a frozen v3 target file and record it in M4_LOCK.json (refused once any V3 frame exists)
  run        equilibrate + produce all locked lineages; resumable (safe to rerun; completed lineages are skipped)
  status     progress of every lineage
Outputs go to a Linux-side folder (default ~/pbm_md_m4), never /mnt/c.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, platform as _pf, shutil, subprocess, sys, time
from pathlib import Path

PBM = Path("/mnt/c/Wei/PBM")
M2B = PBM / "md_validation/MechLib_MD_M2B_matched_system_gate_v1_0_0"
KIT = Path(__file__).resolve().parent
BUILDER = KIT / "build_m4_systems.py"
M3 = Path("~/pbm_md_m3").expanduser()
M3_AMBER_SHA = {"M2-GB3": "ce21730a6fdde565f0ce32db31c8d57675f0aa1621c79a60393946edc343d409",
                "M2-UBQ": "eb99b46865bda56371935a3a18c74806c3924ca5c201db4ab8ed86c975246e00"}
V2_TARGETS = PBM / "PBM_CLEAN_REBUILD_v1/results/training_targets.npz"
V2_SHA = "ef7b86b359c84492a0be791de52a1e64112a0153142b9f9f5a0417448d2731b1"
CASES = ("M2-GB3", "M2-UBQ")
ARMS = ("AMBER_FIXED", "MECHLIB_V2", "MECHLIB_V3")
SEEDS = tuple(2026111301 + i for i in range(8))     # new seeds; M3 used 20261103xx
PROTEIN = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
           "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "ACE", "NME", "HIE", "HID",
           "HIP", "CYX", "ASH", "GLH", "LYN"}

PROTOCOL = {  # fixed design; production length and timestep are fixed later by `lock`
    "temperature_K": 298.0, "pressure_bar": 1.0, "friction_per_ps": 1.0,
    "barostat_interval_steps": 25,
    "minimize_tolerance_kJ_mol_nm": 10.0, "minimize_max_iterations": 5000,
    "nvt_ps": 100.0, "nvt_timestep_fs": 1.0, "nvt_restraint_kJ_mol_nm2": 1000.0,
    "npt_restrained_stages": [[1000.0, 250.0], [100.0, 250.0], [10.0, 250.0]],   # [k, ps], 2 fs
    "npt_unrestrained_ps": 1000.0,
    "production_chunk_ns": 10.0, "frame_interval_ps": 10.0, "state_interval_ps": 10.0,
    "hmr_hydrogen_mass_amu": 3.024,
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def omm():
    import openmm
    from openmm import app, unit
    return openmm, app, unit


def choose_platform(openmm, requested="auto"):
    names = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
    order = [requested] if requested != "auto" else ["CUDA", "OpenCL", "CPU"]
    for n in order:
        if n in names:
            props = {"Precision": "mixed"} if n in ("CUDA", "OpenCL") else {}
            return openmm.Platform.getPlatformByName(n), props
    raise SystemExit(f"no usable platform among {names} (requested {requested})")


# ------------------------------------------------------------------ helpers
def protein_atom_indices(topology, heavy_only=False):
    out = []
    for a in topology.atoms():
        if a.residue.name in PROTEIN and (not heavy_only or (a.element is not None and a.element.symbol != "H")):
            out.append(a.index)
    return out


def apply_hmr(system, topology, unit, h_mass_amu):
    """Repartition mass from heavy atoms to bonded hydrogens (protein only; water stays rigid)."""
    n = 0
    for a1, a2 in topology.bonds():
        if a1.residue.name not in PROTEIN:
            continue
        for h, heavy in ((a1, a2), (a2, a1)):
            if h.element is not None and h.element.symbol == "H" and heavy.element.symbol != "H":
                mh = system.getParticleMass(h.index).value_in_unit(unit.dalton)
                mx = system.getParticleMass(heavy.index).value_in_unit(unit.dalton)
                d = h_mass_amu - mh
                if d > 0:
                    system.setParticleMass(h.index, h_mass_amu * unit.dalton)
                    system.setParticleMass(heavy.index, (mx - d) * unit.dalton)
                    n += 1
    return n


def add_restraint(system, positions, indices, openmm, unit):
    f = openmm.CustomExternalForce("0.5*kr*periodicdistance(x,y,z,x0,y0,z0)^2")
    f.addGlobalParameter("kr", 0.0)
    for p in ("x0", "y0", "z0"):
        f.addPerParticleParameter(p)
    for i in indices:
        x = positions[i].value_in_unit(unit.nanometer)
        f.addParticle(i, [x[0], x[1], x[2]])
    f.setName("M4 positional restraint")
    system.addForce(f)
    return f


def load_system(case_dir: Path, arm: str, hmr: bool, openmm, app, unit):
    system = openmm.XmlSerializer.deserialize((case_dir / f"{arm}_system.xml").read_text())
    pdb = app.PDBFile(str(case_dir / "prepared.pdb"))
    n_hmr = apply_hmr(system, pdb.topology, unit, PROTOCOL["hmr_hydrogen_mass_amu"]) if hmr else 0
    return system, pdb, n_hmr


# ------------------------------------------------------------------ preflight
def cmd_preflight(a):
    ok = True
    for p in [BUILDER, V2_TARGETS, M3 / "M3_LOCK.json", M3 / "build/M2-GB3/AMBER_FIXED_system.xml", M3 / "build/M2-UBQ/AMBER_FIXED_system.xml",
              a.nmr_dir / "M2E_GB3_TableS1_transcription_v1.tsv",
              a.nmr_dir / "M2E_UBQ_TableS3_HNHA_transcription_v1.tsv"]:
        e = p.exists(); ok &= e
        print(f"  {'ok ' if e else 'MISSING'}  {p}")
    try:
        openmm, app, unit = omm()
        names = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
        print(f"  OpenMM {openmm.__version__}; platforms: {names}")
        if "CUDA" not in names:
            print("  WARNING: no CUDA platform in this environment -- GPU runs need an env with CUDA OpenMM")
        import pdbfixer  # noqa
        print("  pdbfixer ok")
        from openmm.app import DCDReporter
        import inspect
        if "atomSubset" not in inspect.signature(DCDReporter.__init__).parameters:
            print("  WARNING: DCDReporter lacks atomSubset (OpenMM < 8.1); full-system frames would be large"); ok = False
    except Exception as e:
        print("  OpenMM/pdbfixer import failed:", e); ok = False
    out = Path(a.output).expanduser()
    if out.resolve().as_posix().startswith("/mnt/"):
        print(f"  WARNING: output {out} is on a Windows drive; use a Linux path such as ~/pbm_md_m4")
    free = shutil.disk_usage(out.parent if not out.exists() else out).free / 1e9
    print(f"  free disk at output: {free:.0f} GB (production needs about 25 GB)")
    print("PREFLIGHT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ------------------------------------------------------------------ build
def build_arm(b, case, arm, targets):
    d = b / case
    cmd = [sys.executable, str(BUILDER), "--case-dir", str(d), "--targets", str(targets), "--arm", arm]
    print(" ", " ".join(cmd))
    with open(b / f"build_{case}_{arm}.log", "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    audit = json.loads((d / f"{arm}_AUDIT.json").read_text()) if (d / f"{arm}_AUDIT.json").exists() else {}
    print(f"  {case} {arm}: rc={rc} status={audit.get('status')} terms={audit.get('registered')}")
    return {"rc": rc, "status": audit.get("status"), "registered": audit.get("registered"), "targets_sha256": audit.get("targets_sha256")}


def cmd_build(a):
    out = Path(a.output).expanduser(); b = out / "build"
    if b.exists() and any(b.iterdir()):
        raise SystemExit(f"{b} exists; builds are not overwritten")
    if sha256(V2_TARGETS) != V2_SHA:
        raise SystemExit("MechLib v2 target file differs from the frozen ef7b86b3...")
    b.mkdir(parents=True)
    rec = {"v2_targets": str(V2_TARGETS), "v2_sha256": V2_SHA, "builder_sha256": sha256(BUILDER), "cases": {}}
    for case in CASES:
        src = M3 / "build" / case; d = b / case; d.mkdir()
        if sha256(src / "AMBER_FIXED_system.xml") != M3_AMBER_SHA[case]:
            raise SystemExit(f"{case}: M3 AMBER_FIXED system differs from the M3 lock")
        for fn in ("AMBER_FIXED_system.xml", "prepared.pdb", "protein.pdb"):
            shutil.copy2(src / fn, d / fn)
        rec["cases"][case] = {"MECHLIB_V2": build_arm(b, case, "MECHLIB_V2", V2_TARGETS),
                              "prepared_sha256": sha256(d / "prepared.pdb")}
    status = "PASS" if all(v["MECHLIB_V2"]["rc"] == 0 and v["MECHLIB_V2"]["status"] == "PASS" for v in rec["cases"].values()) else "FAIL"
    rec["status"] = status
    (b / "M4_BUILD_RECORD.json").write_text(json.dumps(rec, indent=1))
    print("BUILD", status)
    return 0 if status == "PASS" else 1


# ------------------------------------------------------------------ bench
def cmd_bench(a):
    openmm, app, unit = omm()
    out = Path(a.output).expanduser(); b = out / "build"
    plat, props = choose_platform(openmm, a.platform)
    res = []
    for case in CASES:
        for arm in [x for x in ARMS if (b / case / f"{x}_system.xml").exists()]:
            for ts, hmr in ((4.0, True),):
                system, pdb, _ = load_system(b / case, arm, hmr, openmm, app, unit)
                system.addForce(openmm.MonteCarloBarostat(1 * unit.bar, 298 * unit.kelvin, 25))
                integ = openmm.LangevinMiddleIntegrator(298 * unit.kelvin, 1 / unit.picosecond, ts * unit.femtoseconds)
                sim = app.Simulation(pdb.topology, system, integ, plat, props)
                sim.context.setPositions(pdb.positions)
                sim.minimizeEnergy(maxIterations=500)
                sim.context.setVelocitiesToTemperature(298 * unit.kelvin, 1)
                sim.step(2000)
                t0 = time.time(); n = a.steps; sim.step(n)
                st = sim.context.getState(getEnergy=True)
                dt = time.time() - t0
                nsday = n * ts * 1e-6 / (dt / 86400)
                e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                res.append({"case": case, "arm": arm, "timestep_fs": ts, "hmr": hmr, "ns_per_day": round(nsday, 1),
                            "finite_energy": math.isfinite(e)})
                print(f"  {case} {arm} {ts:.0f} fs{' +HMR' if hmr else ''}: {nsday:8.1f} ns/day  energy finite={math.isfinite(e)}")
                del sim, integ
    proj = {}
    rates = {(r["case"], r["arm"]): r["ns_per_day"] for r in res}
    for L in (100, 150, 200):
        proj[f"{L}ns_x{len(SEEDS)}_per_arm_days"] = {arm: round(sum(len(SEEDS) * (L + 1.85) / rates[(c, arm)] for c in CASES if (c, arm) in rates), 1)
                                                     for arm in sorted({k[1] for k in rates})}
    print("  projected wall-clock days (one worker at a time; parallel workers share the GPU):", json.dumps(proj))
    (out / "M4_BENCH.json").write_text(json.dumps({"platform": plat.getName(), "results": res, "projected_days": proj}, indent=1))
    return 0


# ------------------------------------------------------------------ lock
def cmd_lock(a):
    out = Path(a.output).expanduser()
    if (out / "M4_LOCK.json").exists():
        raise SystemExit("M4_LOCK.json exists; the design is already fixed")
    rec = json.loads((out / "build/M4_BUILD_RECORD.json").read_text())
    if rec["status"] != "PASS":
        raise SystemExit("build did not pass")
    here = Path(__file__).resolve()
    scorer = here.with_name("pbm_md_m4_score.py"); proto = here.with_name("M4_PROTOCOL.md")
    arms = ["AMBER_FIXED", "MECHLIB_V2"]
    lock = {"locked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "production_ns": a.production_ns,
            "timestep_fs": a.timestep_fs, "hmr": a.timestep_fs > 2.0, "seeds": SEEDS, "cases": CASES, "arms": arms,
            "registered_pending_arms": ["MECHLIB_V3"],
            "protocol": PROTOCOL, "runner_sha256": sha256(here), "scorer_sha256": sha256(scorer),
            "builder_sha256": sha256(BUILDER), "protocol_doc_sha256": sha256(proto),
            "systems_sha256": {f"{c}/{arm}": sha256(out / "build" / c / f"{arm}_system.xml") for c in CASES for arm in arms},
            "library_sha256": {"MECHLIB_V2": V2_SHA}, "nmr_scored_before_lock": False, "amendments": []}
    (out / "M4_LOCK.json").write_text(json.dumps(lock, indent=1))
    print(json.dumps(lock, indent=1))
    return 0


def cmd_add_v3(a):
    out = Path(a.output).expanduser(); lock_p = out / "M4_LOCK.json"
    lock = json.loads(lock_p.read_text())
    if "MECHLIB_V3" in lock["arms"]:
        raise SystemExit("MECHLIB_V3 already added")
    if any((out / "runs").glob("*/MECHLIB_V3/seed_*/prod_*.dcd")):
        raise SystemExit("V3 frames exist; refusing")
    if a.v3_targets is None or a.v3_sha256 is None:
        raise SystemExit("add-v3 needs --v3-targets and --v3-sha256 (the frozen v3 hash)")
    if sha256(a.v3_targets) != a.v3_sha256:
        raise SystemExit("v3 target file does not match the given frozen hash")
    b = out / "build"; res = {}
    for case in CASES:
        res[case] = build_arm(b, case, "MECHLIB_V3", a.v3_targets)
        if res[case]["rc"] != 0 or res[case]["status"] != "PASS":
            raise SystemExit(f"{case}: V3 build failed; lock unchanged")
    lock["arms"].append("MECHLIB_V3"); lock["registered_pending_arms"] = []
    for case in CASES:
        lock["systems_sha256"][f"{case}/MECHLIB_V3"] = sha256(b / case / "MECHLIB_V3_system.xml")
    lock["library_sha256"]["MECHLIB_V3"] = a.v3_sha256
    lock["amendments"].append({"id": len(lock["amendments"]) + 1, "applied_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "change": "MECHLIB_V3 arm added as registered in M4_PROTOCOL.md (same seeds, velocities, protocol)",
                               "v3_frames_before": 0, "v3_targets": str(a.v3_targets)})
    lock_p.write_text(json.dumps(lock, indent=1))
    print("MECHLIB_V3 added; rerun the workers to produce its lineages")
    return 0


# ------------------------------------------------------------------ run
def lineage_dir(out, case, arm, seed):
    return out / "runs" / case / arm / f"seed_{seed}"


def paired_velocities(out, case, seed, sim, unit, openmm):
    """One velocity draw per case/seed, reused in both arms (same atoms, same masses)."""
    import numpy as np
    p = out / "runs" / case / f"velocities_seed_{seed}.npy"
    if p.exists():
        return np.load(p)
    sim.context.setVelocitiesToTemperature(PROTOCOL["temperature_K"] * unit.kelvin, seed)
    v = sim.context.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    v = np.asarray(v, float); p.parent.mkdir(parents=True, exist_ok=True); np.save(p, v)
    return v


def run_lineage(out, lock, case, arm, seed, plat, props, openmm, app, unit):
    import numpy as np
    d = lineage_dir(out, case, arm, seed); d.mkdir(parents=True, exist_ok=True)
    prog_p = d / "PROGRESS.json"
    prog = json.loads(prog_p.read_text()) if prog_p.exists() else {"stage": "new", "chunks_done": 0}
    if prog.get("stage") == "complete":
        return prog
    n_target = int(round(lock["production_ns"] / lock["protocol"]["production_chunk_ns"]))
    if prog.get("stage") == "production" and prog.get("chunks_done", 0) >= n_target:
        prog["stage"] = "complete"; prog["completed_by"] = "amendment A2 (production length)"
        prog_p.write_text(json.dumps(prog, indent=1))   # chunks beyond the locked length are kept but never scored
        return prog
    P = lock["protocol"]; ts = lock["timestep_fs"]; T = P["temperature_K"] * unit.kelvin
    system, pdb, n_hmr = load_system(out / "build" / case, arm, lock["hmr"], openmm, app, unit)
    heavy = protein_atom_indices(pdb.topology, heavy_only=True)
    restr = add_restraint(system, pdb.positions, heavy, openmm, unit)
    baro = openmm.MonteCarloBarostat(P["pressure_bar"] * unit.bar, T, P["barostat_interval_steps"])
    baro.setRandomNumberSeed(seed); system.addForce(baro)
    integ = openmm.LangevinMiddleIntegrator(T, P["friction_per_ps"] / unit.picosecond, ts * unit.femtoseconds)
    integ.setRandomNumberSeed(seed)
    sim = app.Simulation(pdb.topology, system, integ, plat, props)
    chk = d / "resume.chk"

    if prog["stage"] != "production":
        # ---- equilibration (never scored); restarted from scratch if interrupted
        sim.context.setPositions(pdb.positions)
        sim.context.setParameter("kr", P["nvt_restraint_kJ_mol_nm2"])
        sim.minimizeEnergy(tolerance=P["minimize_tolerance_kJ_mol_nm"] * unit.kilojoule_per_mole / unit.nanometer,
                           maxIterations=P["minimize_max_iterations"])
        if arm == ARMS[0] or not (out / "runs" / case / f"velocities_seed_{seed}.npy").exists():
            v = paired_velocities(out, case, seed, sim, unit, openmm)
        else:
            v = np.load(out / "runs" / case / f"velocities_seed_{seed}.npy")
        sim.context.setVelocities(v * unit.nanometer / unit.picosecond)
        # NVT: barostat off via frequency 0
        baro.setFrequency(0); sim.context.reinitialize(preserveState=True)
        integ.setStepSize(P["nvt_timestep_fs"] * unit.femtoseconds)
        sim.step(int(round(P["nvt_ps"] * 1000 / P["nvt_timestep_fs"])))
        baro.setFrequency(P["barostat_interval_steps"]); sim.context.reinitialize(preserveState=True)
        integ.setStepSize(2.0 * unit.femtoseconds)
        log = open(d / "equilibration.tsv", "w")
        sim.reporters = [app.StateDataReporter(log, 5000, step=True, time=True, potentialEnergy=True,
                                               temperature=True, volume=True, density=True, separator="\t")]
        for k, ps in P["npt_restrained_stages"]:
            sim.context.setParameter("kr", k); sim.step(int(round(ps * 1000 / 2.0)))
        sim.context.setParameter("kr", 0.0)
        integ.setStepSize(ts * unit.femtoseconds)
        sim.step(int(round(P["npt_unrestrained_ps"] * 1000 / ts)))
        sim.reporters = []; log.close()
        st = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        with open(d / "equilibrated.pdb", "w") as fh:
            app.PDBFile.writeFile(pdb.topology, st.getPositions(), fh)
        sim.context.setTime(0.0); sim.currentStep = 0
        sim.saveCheckpoint(str(chk) + ".tmp"); os.replace(str(chk) + ".tmp", chk)
        prog = {"stage": "production", "chunks_done": 0, "hmr_hydrogens": n_hmr, "platform": plat.getName()}
        prog_p.write_text(json.dumps(prog, indent=1))
    else:
        sim.loadCheckpoint(str(chk))
        sim.context.setParameter("kr", 0.0)

    steps_chunk = int(round(P["production_chunk_ns"] * 1e6 / ts))
    n_chunks = int(round(lock["production_ns"] / P["production_chunk_ns"]))
    frame = int(round(P["frame_interval_ps"] * 1000 / ts)); sint = int(round(P["state_interval_ps"] * 1000 / ts))
    prot = protein_atom_indices(pdb.topology)
    for c in range(prog["chunks_done"], n_chunks):
        t0 = time.time()
        sim.reporters = [app.DCDReporter(str(d / f"prod_{c:03d}.dcd"), frame, atomSubset=prot),
                         app.StateDataReporter(str(d / f"prod_{c:03d}.tsv"), sint, step=True, time=True,
                                               potentialEnergy=True, temperature=True, volume=True, density=True,
                                               speed=True, separator="\t")]
        sim.step(steps_chunk); sim.reporters = []
        sim.saveCheckpoint(str(chk) + ".tmp"); os.replace(str(chk) + ".tmp", chk)
        prog["chunks_done"] = c + 1; prog["last_chunk_minutes"] = round((time.time() - t0) / 60, 1)
        prog["stage"] = "production" if c + 1 < n_chunks else "complete"
        prog_p.write_text(json.dumps(prog, indent=1))
        print(f"    {case} {arm} seed {seed}: chunk {c+1}/{n_chunks} ({prog['last_chunk_minutes']} min)", flush=True)
    st = sim.context.getState(getEnergy=True)
    prog["final_potential_kJ_mol"] = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    prog["finite"] = math.isfinite(prog["final_potential_kJ_mol"])
    prog_p.write_text(json.dumps(prog, indent=1))
    del sim, integ
    return prog


def cmd_amend_length(a):
    """Amendment A2: shorten production (registered before any MECHLIB_V2/V3 production frame and before any scoring)."""
    out = Path(a.output).expanduser(); lock_p = out / "M4_LOCK.json"
    lock = json.loads(lock_p.read_text())
    here = Path(__file__).resolve(); proto = here.with_name("M4_PROTOCOL.md")
    if a.production_ns is None or not a.production_ns < lock["production_ns"]:
        raise SystemExit("amend-length only shortens production: give --production-ns below the locked value")
    if a.production_ns % lock["protocol"]["production_chunk_ns"]:
        raise SystemExit("length must be a whole number of chunks")
    if any((out / "runs").glob("*/MECHLIB_V*/seed_*/prod_*.dcd")):
        raise SystemExit("MechLib production frames exist; refusing (amendment must precede them)")
    if any(out.glob("*SCORE*")) or lock.get("nmr_scored_before_lock"):
        raise SystemExit("scoring output exists; refusing")
    if not a.confirm_workers_stopped:
        raise SystemExit("stop all workers first (pkill -f pbm_md_m4.py), then rerun with --confirm-workers-stopped")
    old = {k: lock[k] for k in ("production_ns", "runner_sha256", "protocol_doc_sha256")}
    lock["production_ns"] = a.production_ns
    lock["runner_sha256"] = sha256(here); lock["protocol_doc_sha256"] = sha256(proto)
    lock["amendments"].append({"id": len(lock["amendments"]) + 1, "name": "A2", "applied_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "change": f"production per lineage {old['production_ns']} -> {a.production_ns} ns (seeds, arms, protocol unchanged); "
                                         "AMBER chunks beyond the new length are kept and never scored; recommended 8 workers, seed-major order",
                               "reason": "throughput (16 workers ~540 ns/day); decided before any MechLib production frame and before any scoring",
                               "previous": old, "mechlib_frames_before": 0})
    lock_p.write_text(json.dumps(lock, indent=1))
    print(f"A2 applied: production {old['production_ns']} -> {a.production_ns} ns; runner {old['runner_sha256'][:8]} -> {lock['runner_sha256'][:8]}")
    return 0


def cmd_run(a):
    openmm, app, unit = omm()
    out = Path(a.output).expanduser()
    lock = json.loads((out / "M4_LOCK.json").read_text())
    here = Path(__file__).resolve()
    if sha256(here) != lock["runner_sha256"]:
        raise SystemExit("runner changed since lock; refusing to run")
    for k, h in lock["systems_sha256"].items():
        c, arm = k.split("/")
        if sha256(out / "build" / c / f"{arm}_system.xml") != h:
            raise SystemExit(f"system changed since lock: {k}")
    plat, props = choose_platform(openmm, a.platform)
    if plat.getName() not in ("CUDA", "OpenCL") and a.platform != "CPU":
        raise SystemExit(f"no GPU platform (found {plat.getName()}); production on CPU would take months. "
                         "Use an OpenMM build with CUDA, or pass --platform CPU deliberately.")
    if not 0 <= a.shard < a.nshards:
        raise SystemExit("--shard must be in [0, --nshards)")
    # replicate-major order; a worker owns whole (seed, protein) pairs so both conditions share velocities
    pairs = [(seed, case) for seed in lock["seeds"] for case in lock["cases"]]
    mine = pairs[a.shard::a.nshards]
    print(f"platform {plat.getName()}; worker {a.shard + 1}/{a.nshards}: {len(mine)} of {len(pairs)} pairs; "
          f"{lock['production_ns']} ns per lineage at {lock['timestep_fs']} fs{' + HMR' if lock['hmr'] else ''}", flush=True)
    for seed, case in mine:
        for arm in lock["arms"]:
            pr = run_lineage(out, lock, case, arm, seed, plat, props, openmm, app, unit)
            print(f"  {case} {arm} seed {seed}: {pr['stage']}", flush=True)
    print(f"WORKER {a.shard + 1}/{a.nshards} COMPLETE")
    return 0


def cmd_status(a):
    out = Path(a.output).expanduser()
    lock = json.loads((out / "M4_LOCK.json").read_text()) if (out / "M4_LOCK.json").exists() else None
    n = int(round(lock["production_ns"] / PROTOCOL["production_chunk_ns"])) if lock else None
    for seed in SEEDS:
        for case in CASES:
            for arm in (lock["arms"] if lock else ARMS):
                p = lineage_dir(out, case, arm, seed) / "PROGRESS.json"
                s = json.loads(p.read_text()) if p.exists() else {"stage": "not started", "chunks_done": 0}
                print(f"  {case:7s} {arm:17s} {seed}: {s['stage']:12s} {s['chunks_done']}/{n}"
                      f"  {s.get('last_chunk_minutes', '')}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["preflight", "build", "bench", "lock", "add-v3", "amend-length", "run", "status"])
    ap.add_argument("--output", default="~/pbm_md_m4")
    ap.add_argument("--v3-targets", type=Path, help="add-v3: frozen MechLib v3 target file (npz, same layout as v2)")
    ap.add_argument("--v3-sha256", help="add-v3: its frozen SHA-256")
    ap.add_argument("--platform", default="auto")
    ap.add_argument("--nmr-dir", type=Path, default=PBM)
    ap.add_argument("--steps", type=int, default=25000, help="bench: timed steps per system")
    ap.add_argument("--shard", type=int, default=0, help="run: this worker's index (0-based)")
    ap.add_argument("--nshards", type=int, default=1, help="run: number of concurrent workers on the GPU (4-6 suggested)")
    ap.add_argument("--production-ns", type=float, help="lock: production length per lineage")
    ap.add_argument("--timestep-fs", type=float, choices=[2.0, 4.0], help="lock: 2 (no HMR) or 4 (HMR)")
    ap.add_argument("--confirm-workers-stopped", action="store_true", help="amend-length: all workers are stopped")
    a = ap.parse_args()
    Path(a.output).expanduser().mkdir(parents=True, exist_ok=True)
    if a.command == "lock" and (a.production_ns is None or a.timestep_fs is None):
        ap.error("lock needs --production-ns and --timestep-fs")
    return {"preflight": cmd_preflight, "build": cmd_build, "bench": cmd_bench, "lock": cmd_lock, "add-v3": cmd_add_v3,
            "amend-length": cmd_amend_length, "run": cmd_run, "status": cmd_status}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
