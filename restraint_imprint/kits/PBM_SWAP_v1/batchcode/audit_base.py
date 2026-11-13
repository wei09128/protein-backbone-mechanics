"""Audit the one-cycle synthetic angle pairs and 4YCA stock repeat."""
import csv
import math
import re
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from statistics import mean, median


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atom_id(label):
    return tuple(label[a:b].strip() for a, b in
                 [(0, 4), (4, 5), (5, 8), (9, 10), (10, 14), (14, 15)])


def geometry(path):
    content = path.read_text()
    pattern = re.compile(
        r'^(bond|angle) (pdb="[^"\n]+"(?:\n +pdb="[^"\n]+"){1,2})\n'
        r' +ideal +model +delta +sigma +weight +residual\n +([^\n]+)', re.M
    )
    restraints = {}
    counts = Counter()
    for kind, labels, values in pattern.findall(content):
        atoms = tuple(map(atom_id, re.findall(r'pdb="([^"]+)"', labels)))
        require(len(atoms) == (2 if kind == 'bond' else 3), (path, atoms))
        key = (kind, min(atoms, atoms[::-1]))
        require(key not in restraints, (path, 'duplicate restraint', key))
        ideal, model, _, sigma, weight, _ = map(Decimal, values.split())
        restraints[key] = (ideal, model, sigma, weight)
        counts[kind] += 1
    for heading, kind in [('Bond', 'bond'), ('Bond angle', 'angle')]:
        expected = sum(map(int, re.findall(
            rf'^{heading} \| .*?restraints: (\d+)', content, re.M)))
        require(expected > 0 and expected == counts[kind],
                (path, kind, expected, counts[kind]))
    return restraints


def check_geometry(stock, shift, repeat=False):
    require(stock.keys() == shift.keys(), 'Initial restraint identities differ')
    targets = []
    for key, (ideal, model, sigma, weight) in stock.items():
        new_ideal, new_model, new_sigma, new_weight = shift[key]
        require((model, sigma, weight) == (new_model, new_sigma, new_weight),
                ('Initial model, printed sigma or weight differs', key))
        delta = new_ideal - ideal
        if delta:
            kind, atoms = key
            require(not repeat and kind == 'angle', ('Unexpected ideal change', key, delta))
            a, b, c = atoms if atoms[0][0] == 'CA' else atoms[::-1]
            require((a[0], b[0], c[0]) == ('CA', 'C', 'N')
                    and a[2:] == b[2:] and b[3] == c[3]
                    and b[4:] != c[4:] and delta == Decimal('1.00'),
                    ('Unexpected ideal change', key, delta))
            targets.append(key)
    require(repeat or targets, 'No +1.0 degree Cα–C–N targets found')
    print(f'Initial GEO PASS: {len(stock)} bonds and angles, {len(targets)} changed targets')
    return targets


def coordinates(path):
    out = {}
    for line in path.read_text().splitlines():
        if line.startswith(('ATOM  ', 'HETATM')):
            key = atom_id(line[12:27])
            require(key not in out, (path, 'duplicate atom', key))
            out[key] = tuple(float(line[i:i+8]) for i in (30, 38, 46))
    return out


def array_map(mtz, label):
    column = mtz.get_column(label)
    values = column.extract_values()
    valid = column.selection_valid()
    hkls = list(mtz.extract_miller_indices())
    require(len(set(hkls)) == len(hkls), ('Duplicate MTZ rows', label))
    return {h: value for h, value, keep in zip(hkls, values, valid) if keep}


def check_reflections(common_path, paths, pdb_id):
    from iotbx import mtz

    common = mtz.object(file_name=str(common_path))
    flags = array_map(common, 'R-free-flags')
    observed = array_map(common, 'FOBS')
    expected = {'4aqo': {0: 1623, 1: 30949}, '4yca': {0: 2086, 1: 57758}}
    require(observed.keys() == flags.keys() and Counter(flags.values()) == expected[pdb_id],
            'Common input observations or flags differ from archive')
    parity = []
    for path in paths:
        output = mtz.object(file_name=str(path))
        raw = array_map(output, 'F-obs')
        filtered = array_map(output, 'F-obs-filtered')
        out_flags = array_map(output, 'R-free-flags')
        require(filtered and filtered.keys() <= raw.keys() <= observed.keys(),
                (path, 'Unexpected observed HKLs'))
        require(raw == observed, (path, 'Raw observations differ from input'))
        anchor = max(filtered, key=lambda h: abs(raw[h]))
        scale = filtered[anchor] / raw[anchor]
        require(math.isfinite(scale) and scale > 0 and
                all(math.isclose(v, scale * raw[h], rel_tol=1e-6, abs_tol=1e-7)
                    for h, v in filtered.items()), (path, 'Nonuniform filtered amplitude scaling'))
        print(f'{path.parent.name}: filtered amplitude scale {scale:.8f}')
        require(out_flags == flags,
                (path, 'Free flags differ from common input'))
        map_only = set(output.extract_miller_indices()) - observed.keys()
        for label in output.column_labels()[3:]:
            if label not in ('2FOFCWT', 'PH2FOFCWT'):
                require(not (map_only & array_map(output, label).keys()),
                        (path, 'Extra HKLs outside filled map columns', label))
        require(map_only <= array_map(output, '2FOFCWT').keys()
                and map_only <= array_map(output, 'PH2FOFCWT').keys(),
                (path, 'Extra HKLs missing filled map coefficients'))
        omitted = sorted((h, int(flags[h])) for h in observed.keys() - filtered.keys())
        print(f'{path.parent.name}: {len(raw)} F-obs, {len(filtered)} filtered, '
              f'{len(out_flags)} flags, {sum(out_flags[h] == 0 for h in filtered)} filtered free, '
              f'{len(map_only)} extra filled-map HKLs')
        print('Input HKLs absent from filtered F-obs (HKL, input flag):', omitted)
        parity.append((set(filtered), out_flags, map_only, scale))
    require(all(p[:3] == parity[0][:3] and math.isclose(p[3], parity[0][3], rel_tol=1e-6)
                for p in parity), 'Filtered HKLs, flags, map coverage or scale differ')
    print('MTZ parity PASS')


def final_r(path):
    log = path.read_text()
    flags = re.findall(r'(?m)^\s*test_flag_value\s*=\s*(\d+)\s*$', log)
    require(flags and set(flags) == {'0'}, (path, 'Missing or inconsistent test flag'))
    match = re.search(r'Final R-work\s*=\s*([\d.]+),\s*R-free\s*=\s*([\d.]+)', log)
    require(match is not None, (path, 'Missing final R summary'))
    return tuple(map(Decimal, match.groups()))


def angle(coords, atoms):
    a, b, c = (coords[k] for k in atoms)
    u = [x - y for x, y in zip(a, b)]
    v = [x - y for x, y in zip(c, b)]
    cosine = sum(x * y for x, y in zip(u, v)) / (math.hypot(*u) * math.hypot(*v))
    return math.degrees(math.acos(max(-1, min(1, cosine))))


