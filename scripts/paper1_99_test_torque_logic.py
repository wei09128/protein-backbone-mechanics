import numpy as np

# --- UPDATED DICTIONARIES ---
EXCL_PREV = {'N': {'C', 'O', 'CA'}, 'CA': {'C'}}
EXCL_NEXT = {
    'C':  {'N', 'CA', 'H', 'HN'}, 
    'O':  {'N'}, 
    'CA': {'N'}
}

SCALE_14_PREV = {
    ('N', 'N'), ('N', 'CB'), ('N', 'HA'), ('N', 'HA2'), ('N', 'HA3'), 
    ('CA', 'O'), ('CA', 'CA'), ('C', 'C')
}

SCALE_14_NEXT = {
    ('C', 'CB'), ('C', 'C'), ('C', 'HA'), ('C', 'CD'), # Added CD for Proline
    ('CA', 'CA'), ('CA', 'H'), ('CA', 'HN'), 
    ('O', 'CA'), ('O', 'H'), ('O', 'HN'), ('N', 'N')
}

def test_production_logic():
    print(f"{'Interaction Pair':<25} | {'Expected':<8} | {'Result':<10}")
    print("-" * 50)

    test_cases = [
        # --- GAP 1: HA / GLY Naming Coverage ---
        ('N',  'HA',  -1, 0.5),  # Standard 1-4
        ('N',  'HA2', -1, 0.5),  # Glycine 1-4
        
        # --- GAP 2: Proline i+1 Check ---
        ('C',  'CD',   1, 0.5),  # C(i)-N(i+1)-CD(i+1) is 1-4 in Proline
        
        # --- GAP 3: Long-Range Guards (|delta| > 1) ---
        ('CA', 'O',   -2, 1.0),  # Residue i-2: Must be full LJ
        ('C',  'CB',   2, 1.0),  # Residue i+2: Must be full LJ
        ('N',  'N',  -10, 1.0),  # Far away: Must be full LJ
        
        # --- Sanity Checks (Existing) ---
        ('N',  'C',   -1, 0.0),  # Exclusion
        ('CA', 'O',   -1, 0.5),  # 1-4 Scale
    ]

    passed = 0
    for bb_name, env_name, delta, expected in test_cases:
        scale_lj = 1.0
        is_excluded = False
        
        # 1. EXCLUSION LOGIC
        if delta == -1 and env_name in EXCL_PREV.get(bb_name, set()):
            is_excluded = True
        elif delta == 1 and env_name in EXCL_NEXT.get(bb_name, set()):
            is_excluded = True
            
        if is_excluded:
            scale_lj = 0.0
        else:
            # 2. SCALING LOGIC (Only applies to immediate neighbors)
            if delta == -1:
                if (bb_name, env_name) in SCALE_14_PREV:
                    scale_lj = 0.5
            elif delta == 1:
                if (bb_name, env_name) in SCALE_14_NEXT:
                    scale_lj = 0.5
            # 3. |delta| > 1 implicitly stays at 1.0

        label = f"{bb_name}(i)-{env_name}(i{delta:+})"
        status = "✅ PASS" if scale_lj == expected else "❌ FAIL"
        if scale_lj == expected: passed += 1
        print(f"{label:<25} | {expected:<8} | {status} (Got {scale_lj})")

    print("-" * 50)
    print(f"Final Coverage: {passed}/{len(test_cases)} cases passed.")

if __name__ == "__main__":
    test_production_logic()