"""Run one PHENIX arm (stock or swap) with a traced live CDL lookup. Run with phenix.python.

swap: for each live CDL lookup that returns a row (trans, standard residues), mACN, mACO and
mOCN are replaced by MechLib v2 at the lookup node; sigmas and all other fields unchanged.
"""
import argparse, atexit, hashlib, importlib, inspect, json, os, runpy, sys
from pathlib import Path

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
import swapcore as S  # noqa: E402


def atom_id(atom):
    l = atom.fetch_labels()
    return tuple(str(getattr(l, k) or '').strip() for k in ['name', 'altloc', 'resname', 'chain_id', 'resseq', 'icode'])


def main(a):
    lib = importlib.import_module('mmtbx.conformation_dependent_library')
    db = importlib.import_module('mmtbx.conformation_dependent_library.cdl_database')
    from mmtbx.conformation_dependent_library.cdl_setup import columns
    from mmtbx.conformation_dependent_library import cdl_utils
    assert hashlib.sha256(Path(db.__file__).read_bytes()).hexdigest() == S.CDL_SHA, 'CDL source hash differs'
    assert lib.cdl_database is db.cdl_database and len(db.cdl_database) == 8
    assert all(columns.index(k) == i for k, i in S.COLS.items()), 'CDL column layout differs'
    v2 = S.load_v2(a.targets)
    trace = open('LOOKUPS.jsonl', 'w')
    old = lib.get_restraint_values
    sig = inspect.signature(old)
    calls, edited = [0], [0]

    def lookup(*args, **kwargs):
        b = sig.bind(*args, **kwargs); b.apply_defaults()
        assert not b.arguments['interpolate'] and not b.arguments['cdl_svl'], 'Expected discrete stock CDL lookup'
        th = b.arguments['threes']
        out = old(*args, **kwargs)
        calls[0] += 1
        if out is None:
            return out
        key = th.get_cdl_key()
        if key is None or bool(th.cis_group()):
            return out
        r1, r2 = th[1].resname.strip().upper(), th[2].resname.strip().upper()
        group = cdl_utils.get_res_type_group(th[1].resname, th[2].resname)
        before = [float(out[S.COLS[t]]) for t in S.TERMS]
        if a.arm == 'swap' and r1 in S.RES20 and r2 in S.RES20:
            new = list(out)
            for t in S.TERMS:
                new[S.COLS[t]] = S.v2_node(v2, t, r1, r2, key)
            out = new
            edited[0] += 1
        cs = [atom_id(x) for x in th[1].atoms() if x.name.strip() == 'C']
        trace.write(json.dumps({'call': calls[0], 'group': group, 'key': list(key), 'res': [r1, r2],
                                'carbonyl_ids': cs, 'before': before,
                                'after': [float(out[S.COLS[t]]) for t in S.TERMS]}) + '\n')
        return out

    lib.get_restraint_values = lookup
    meta = {'arm': a.arm, 'model_sha256': S.sha(a.model), 'mtz_sha256': S.sha(a.mtz), 'phil_sha256': S.sha(a.phil),
            'targets_sha256': S.sha(a.targets), 'swapcore_sha256': S.sha(KIT / 'swapcore.py')}

    def finish():
        trace.close()
        meta.update(live_calls=calls[0], edited_lookups=edited[0])
        Path('ARM.json').write_text(json.dumps(meta, indent=2) + '\n')
        print(f'PBM discrete CDL calls={calls[0]} edited={edited[0]}', flush=True)
    atexit.register(finish)
    os.environ['LIBTBX_DISPATCHER_NAME'] = 'phenix.refine'
    sys.argv = [str(a.entry), str(a.model), str(a.mtz), str(a.phil), 'refinement.main.nproc=1']
    try:
        runpy.run_path(str(a.entry), run_name='__main__')
    except SystemExit as e:
        if e.code not in (None, 0):
            raise
    prefix = a.model.stem + '_refine_001'
    for src, dst in [(prefix + '.geo', 'initial.geo'), (prefix + '_final.geo', 'final.geo'), (prefix + '.pdb', 'final.pdb'),
                     (prefix + '.mtz', 'final.mtz'), (prefix + '.log', 'final.log')]:
        Path(dst).write_bytes(Path(src).read_bytes())
    assert calls[0] > 0, 'No live CDL lookup'
    if a.arm == 'swap':
        assert edited[0] > 0, 'No lookup was edited'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['stock', 'swap'], required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--mtz', type=Path, required=True)
    p.add_argument('--phil', type=Path, required=True)
    p.add_argument('--entry', type=Path, required=True)
    p.add_argument('--targets', type=Path, required=True)
    main(p.parse_args())
