# PBM controlled response in REFMAC (spec v1.0), 13 Nov 2026
Repeats the Figure 1 signed edits (isolated Ca-C-N, isolated O-C-N, angle-sum-preserving) in REFMAC on 6 of the 19 structures (2 per resolution band, alphabetical rule), via a user dictionary that shifts the TRANS/PTRANS link targets. Tests coupling, angle-sum conservation and REFMAC vs PHENIX transmission.

Needs CCP4 (refmac5, CLIBD_MON, ccp4-python with gemmi). The script finds ccp4.setup-sh under ~, /opt or /usr/local, or set CCP4_SETUP.
```bash
cd /mnt/c/Wei/PBM
bash PBM_REFMAC_RESPONSE_v1/run.sh
```
A gate runs first on 2WFI (isolated Ca-C-N edit must move coordinates > 0.10 per degree); if the dictionary override is not active it stops before the main runs. Upload the printed pbm_refmac_response.*.tar.gz (mtz files excluded). Expect roughly 42-50 REFMAC runs.
