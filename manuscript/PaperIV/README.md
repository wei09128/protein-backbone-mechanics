# Paper IV manuscript working copies

`paper4_JCIM_MAIN_v14_full_authors.docx` and `paper4_JCIM_SI_v8.docx` are the current clean manuscript and Supporting Information working copies. The full OpenMM 8 and Psi4 1.4 author lists are present in the main text. The additional mean-target comparator arms and minimization subgroup bootstrap intervals are reported explicitly.

The 80-case prespecified Q6 benchmark passed QC in all cases and had a positive aggregate mean; its **strict registered endpoint was FAIL** because N–Cα, C–O, and C–N(next) worsened after Holm correction. The two-protein NMR comparison is an exploratory short MD pilot; a post hoc last-800-ps sensitivity had mixed protein-specific effects. CDL slightly outperformed MechLib on the post-2020 ten-term crystal-coordinate comparison overall, while the order reversed among REFMAC-labeled entries. Software labels do not establish which restraint targets were used.

Supporting scripts for the post hoc baselines, matched CDL comparison, refinement-program diagnostic, QM offset sensitivity, MD/NMR last-800-ps sensitivity, and minimization subgroup bootstrap are under `scripts/PaperIV/secondary/`. Scripts depend on the frozen source files named in their own usage/help text; the 1.7-million-row feature CSV, QM wavefunctions, and MD trajectories are not stored in GitHub. Published percentages cannot be regenerated from the repository alone without those inputs.

The marked manuscript against the actual originally submitted version, the editor letter, and the point-by-point response belong in the journal submission package and are deliberately excluded from this public repository. These files are working copies pending that marked-version task.
