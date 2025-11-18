import numpy as np
import os
import glob

# Directory con gli array salvati
ATTN_DIR = "./outputs/attention_arrays"

# Trova tutti i file per un layer specifico
layer = 30
mode = 'viral'

files = glob.glob(f"{ATTN_DIR}/layer{layer:02d}_{mode}_*.npy")

if len(files) < 2:
    print(f"[INFO] Found only {len(files)} file(s). Need at least 2 for comparison.")
else:
    print(f"[INFO] Found {len(files)} attention arrays for layer {layer}, mode {mode}")
    
    # Carica tutti gli array
    arrays = {}
    for f in files:
        filename = os.path.basename(f)
        arr = np.load(f)
        arrays[filename] = arr
        print(f"\n  File: {filename}")
        print(f"    Shape: {arr.shape}")
        print(f"    Range: [{arr.min():.6f}, {arr.max():.6f}]")
    
    # Confronta tutti contro il primo
    filenames = list(arrays.keys())
    reference_name = filenames[0]
    reference = arrays[reference_name]
    
    print(f"\n[COMPARISON] Using '{reference_name}' as reference")
    print("="*70)
    
    for name in filenames[1:]:
        arr = arrays[name]
        
        # Confronti
        identical = np.array_equal(reference, arr)
        close = np.allclose(reference, arr, rtol=1e-6, atol=1e-8)
        max_diff = np.abs(reference - arr).max()
        mean_diff = np.abs(reference - arr).mean()
        
        print(f"\n  vs '{name}':")
        print(f"    Exactly identical: {identical}")
        print(f"    Close (rtol=1e-6): {close}")
        print(f"    Max difference: {max_diff:.10f}")
        print(f"    Mean difference: {mean_diff:.10f}")
        
        if close:
            print(f"    ✅ Arrays are effectively identical")
        else:
            print(f"    ❌ Arrays differ significantly!")