# PBM_COD_PROBE_v1_1: does COD have enough restraint-free peptide junctions?
Counts only. Finds C(=O)-N-CA-C(=O)-N junctions (both amides trans, R1 <= 0.05, no disorder at the junction) in your local COD copy, by residue class and phi/psi region, and applies the rule in COD_ROLE_RULE.json (fixed before any carbonyl angle is computed): enough junctions -> COD is an adequately powered referee; otherwise it is reported as an underpowered referee. COD never trains or calibrates a library (amendment v1.1, 16 Nov 2026, before the probe ran).

    cd /mnt/c/Wei/PBM && unzip -o /mnt/c/Users/*/Downloads/PBM_COD_PROBE_v1_1.zip
    ~/ccp4/ccp4-9/bin/ccp4-python PBM_COD_PROBE_v1_1/cod_probe.py --selftest
    nice -n 19 ~/ccp4/ccp4-9/bin/ccp4-python PBM_COD_PROBE_v1_1/cod_probe.py --cod ~/cod/cif --out ~/cod_probe --workers 4

(ccp4-python already has gemmi and numpy.) Run after the COD download finishes. Paste the printed summary.
