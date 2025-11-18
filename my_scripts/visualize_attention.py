import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init

# ^ ----------------------------
# ^ CONFIG
# ^ ----------------------------
# MODEL_PATH = "/cluster/project/cvg/students/fbondi/sem-project/VIRAL/checkpoints/viral_checkpoints/llava-v1.5-7b-instruct-repa-dino-single-16"
MODEL_PATH = "/cluster/project/cvg/students/fbondi/checkpoints/llava-viral-merged"
# MODEL_BASE = "/cluster/project/cvg/students/fbondi/checkpoints/llava-v1.5-7b"
MODEL_BASE = None
# MODEL_NAME = "llava-v1.5-7b-lora"
MODEL_NAME = "llava-viral"
IMAGE_PATH = "./images/llava_logo.png"
PROMPT = "How many legs does the animal have?"
LAYER_INDICES = [16, 30]  # Layer da visualizzare
ATTENTION_MODE = 'viral' # or 'grounding'
SAVE_ATTENTION_ARRAYS = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ^ ----------------------------
# ^ LOAD MODEL
# ^ ----------------------------
disable_torch_init()
print(f"[INFO] Loading model from {MODEL_PATH}")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL_PATH, model_base=MODEL_BASE, model_name=MODEL_NAME
)
if False:
    print("[INFO] Salvo il modello post merge")
    model.save_pretrained(
        "/cluster/project/cvg/students/fbondi/checkpoints/llava-viral-merged"
    )
    tokenizer.save_pretrained(
        "/cluster/project/cvg/students/fbondi/checkpoints/llava-viral-merged"
    )

model.to(DEVICE, dtype=DTYPE).eval()
print(f"[INFO] Model loaded successfully on {DEVICE}")

# ^ ----------------------------
# ^ PREPARE IMAGE + PROMPT
# ^ ----------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
print(f"[INFO] Loaded image: {image.size}")

# Process image
if hasattr(image_processor, "preprocess"):
    pixel_values = image_processor.preprocess(image, return_tensors="pt")[
        "pixel_values"
    ]
else:
    pixel_values = image_processor(image, return_tensors="pt")["pixel_values"]
pixel_values = pixel_values.to(DEVICE, dtype=DTYPE)

# Prepare conversation
conv = conv_templates["llava_v1"].copy()
conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + PROMPT)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

# Tokenize
input_ids = (
    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    .unsqueeze(0)
    .to(DEVICE)
)

print(f"[INFO] Input shape: {input_ids.shape}")

# ^ ----------------------------
# ^ RUN INFERENCE WITH ATTENTION
# ^ ----------------------------
print("[INFO] Running inference with attention extraction...")
with torch.inference_mode():
    outputs = model(
        input_ids=input_ids,
        images=pixel_values,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )

# Gli hidden states hanno la sequenza completa (625)
hidden_states = outputs.hidden_states[-1]  # [1, 625, hidden_dim]
print(f"Hidden states shape: {hidden_states.shape}")

# Verifica che abbiamo le attention
if not hasattr(outputs, "attentions") or outputs.attentions is None:
    raise ValueError(
        "Model did not return attention weights. Check model configuration."
    )

# Ottieni le attention weights
attn_list = outputs.attentions
print(f"[INFO] Got attention from {len(attn_list)} layers")

# ^ ----------------------------
# ^ GENERATE MODEL RESPONSE (SECONDA CHIAMATA)
# ^ ----------------------------
print("\n[INFO] Generating model response...")
with torch.inference_mode():
    outputs_generation = model.generate(  # ← Seconda chiamata separata
        inputs=input_ids,
        images=pixel_values,
        do_sample=False,
        temperature=0.2,
        top_p=None,
        max_new_tokens=512,
        use_cache=True,
    )

# Estrai la risposta
output_sequences = getattr(outputs_generation, "sequences", outputs_generation)
output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

# Parse la risposta dal template
if conv.sep_style == SeparatorStyle.TWO:
    response = output_text.split(conv.sep2)[-1].strip()
elif conv.sep_style == SeparatorStyle.LLAMA_2:
    response = output_text.split(conv.roles[1] + ":")[-1].strip()
else:
    response = output_text.split(conv.roles[1] + ":")[-1].strip()

print(f"[INFO] Prompt: {PROMPT}")
print(f"[INFO] Response: {response}")
generated_text = response

# ^ ----------------------------
# ^ TROVA LE POSIZIONI DEI TOKEN VISIVI
# ^ ----------------------------
def find_vision_token_positions_correct(input_ids, num_vis_tokens, attn_seq_len):
    """
    Trova le posizioni dei vision tokens nella sequenza finale
    basandosi sul mismatch tra input_ids e attention sequence length
    """
    text_token_count = input_ids.shape[1]  # 50
    total_seq_len = attn_seq_len  # 625
    
    # Calcola quanti token sono stati aggiunti
    added_tokens = total_seq_len - text_token_count  # 575
    
    # LLaVA sostituisce 1 token <image> con 576 vision embeddings
    # Quindi: added_tokens = num_vis_tokens - 1
    # (perché 1 token viene sostituito, non aggiunto)
    
    expected_added = num_vis_tokens - 1  # 576 - 1 = 575
    
    if added_tokens != expected_added:
        print(f"[WARNING] Unexpected token count: {added_tokens} vs {expected_added}")
    
    # Trova dove era il token <image> negli input_ids originali
    image_token_id = -200  # IMAGE_TOKEN_INDEX
    token_ids = input_ids[0].tolist()
    
    try:
        # Posizione del token <image> negli input_ids originali
        image_pos_in_input = token_ids.index(image_token_id)
        print(f"[INFO] Found <image> token at position {image_pos_in_input} in input_ids")
        
        # Nella sequenza finale, i vision tokens iniziano dove era <image>
        vis_start = image_pos_in_input
        vis_end = vis_start + num_vis_tokens
        
    except ValueError:
        # Se non trova <image>, usa euristica
        print("[WARNING] <image> token not found, using heuristic")
        
        # Solitamente in LLaVA v1.5, il system prompt è ~35 token
        # Poi viene l'immagine
        vis_start = 35
        vis_end = vis_start + num_vis_tokens
    
    # Verifica che le posizioni siano valide
    if vis_end > total_seq_len:
        print(f"[ERROR] vis_end ({vis_end}) > seq_len ({total_seq_len})")
        vis_end = total_seq_len
        vis_start = vis_end - num_vis_tokens
    
    return vis_start, vis_end

# Usa questa funzione invece del calcolo attuale
num_vis_tokens = model.get_vision_tower().num_patches
attn_seq_len = attn_list[0].shape[-1]
vis_start, vis_end = find_vision_token_positions_correct(input_ids, num_vis_tokens, attn_seq_len)

# print(f"[INFO] Sequence length (attn): {num_vis_tokens}")
print(f"[INFO] Text tokens: 0-{vis_start-1}  |  Vision tokens: {vis_start}-{vis_end-1}")

# ^ ==============================================
# ^ ANALISI SEQUENZA E DECODIFICA SICURA
# ^ ==============================================
print("\n" + "="*60)
print("DETAILED SEQUENCE ANALYSIS")
print("="*60)

# 1. Analizza input_ids
token_ids = input_ids[0].tolist()
print(f"\nInput IDs length: {len(token_ids)}")
print(f"Contains -200 (IMAGE_TOKEN): {-200 in token_ids}")

if -200 in token_ids:
    image_pos = token_ids.index(-200)
    print(f"Position of <image> in input_ids: {image_pos}")
    print(f"Tokens before <image>: {image_pos}")
    print(f"Tokens after <image>: {len(token_ids) - image_pos - 1}")
    
    # Decodifica separando prima e dopo <image>
    tokens_before = token_ids[:image_pos]
    tokens_after = token_ids[image_pos + 1:]  # Salta il token -200
    
    decoded_before = tokenizer.decode(tokens_before) if tokens_before else ""
    decoded_after = tokenizer.decode(tokens_after) if tokens_after else ""
    
    print(f"\n--- DECODED PROMPT ---")
    print(f"Before <image>: {decoded_before}")
    print(f"<IMAGE TOKEN HERE>")
    print(f"After <image>: {decoded_after}")
    print(f"\nFull prompt: {decoded_before} [IMAGE] {decoded_after}")
else:
    print("\n[WARNING] Cannot find <image> token!")
    # Fallback: prova a decodificare sostituendo -200 con UNK
    safe_ids = [tid if tid != -200 else tokenizer.unk_token_id for tid in token_ids]
    print(f"Decoded (with UNK): {tokenizer.decode(safe_ids)}")

# 2. Verifica finale delle posizioni
print(f"\n--- POSITION VERIFICATION ---")
print(f"Attention sequence length: {attn_seq_len}")
print(f"Vision patches: {num_vis_tokens}")
print(f"Expected final length: {len(token_ids) - 1 + num_vis_tokens}")
print(f"Actual sequence length: {attn_seq_len}")
print(f"Match: {len(token_ids) - 1 + num_vis_tokens == attn_seq_len}")

print(f"\n--- CORRECT TOKEN POSITIONS ---")
print(f"  Text before image: 0-{vis_start-1} ({vis_start} tokens)")
print(f"  Vision tokens: {vis_start}-{vis_end-1} ({vis_end - vis_start} tokens)")
print(f"  Text after image: {vis_end}-{attn_seq_len-1} ({attn_seq_len - vis_end} tokens)")
print("="*60 + "\n")

# ^ ----------------------------
# ^ PROCESS ATTENTION FOR VISUALIZATION
# ^ ----------------------------
def process_attention_layer(attn_tensor, layer_idx, vis_start, vis_end, input_ids, tokenizer, mode='viral'):
    """
    Estrae attention dal layer specificato dell'LLM.
    
    Args:
        attn_tensor: Tensor di attention [batch, heads, seq_len, seq_len]
        layer_idx: Indice del layer
        vis_start: Posizione iniziale dei vision tokens
        vis_end: Posizione finale dei vision tokens
        input_ids: Token IDs originali
        tokenizer: Tokenizer del modello
        mode: 'grounding' (question→vision, prompt-dependent) 
              'viral' (vision→vision, prompt-independent)
    
    Returns:
        mean_attention: Array numpy [576] con attention mediata
    """
    batch_size, num_heads, seq_len, _ = attn_tensor.shape
    
    print("\n" + "="*70)
    print(f"DETAILED ATTENTION ANALYSIS - Layer {layer_idx} | Mode: {mode.upper()}")
    print("="*70)
    
    # Media sulle heads
    attn = attn_tensor[0].mean(0).cpu().numpy()
    
    if mode == 'viral':
        # ========== QUESTION → VISION ATTENTION (PROMPT-DEPENDENT) ==========
        
        # POSIZIONI nell'ATTENTION SEQUENCE
        question_start_attn = vis_end
        question_end_attn = seq_len - 2
        
        # POSIZIONI negli INPUT_IDS ORIGINALI
        token_ids = input_ids[0].tolist()
        image_token_pos = token_ids.index(-200)
        question_start_input = image_token_pos + 1
        question_end_input = len(token_ids) - 2
        
        print(f"\n[TOKEN MAPPING]:")
        print(f"    INPUT_IDS: Question at positions {question_start_input}-{question_end_input}")
        print(f"    ATTENTION: Question at positions {question_start_attn}-{question_end_attn}")
        
        # Estrai i token IDs DALLA POSIZIONE CORRETTA
        question_token_ids = token_ids[question_start_input:question_end_input]
        
        print(f"\n[QUESTION TOKENS]:")
        print(f"    Token IDs: {question_token_ids}")
        
        if question_token_ids:
            question_text = tokenizer.decode(question_token_ids, skip_special_tokens=False)
            print(f"    Decoded: '{question_text}'")
        else:
            print(f"    [ERROR] Could not extract question tokens!")
        
        # Estrai attention dalle POSIZIONI CORRETTE nell'attention tensor
        question_positions = np.arange(question_start_attn, question_end_attn)
        question2vis = attn[question_positions][:, vis_start:vis_end]
        
        print(f"\n[QUESTION→VISION ATTENTION]:")
        print(f"    Shape: {question2vis.shape}")
        print(f"    Range: [{question2vis.min():.6f}, {question2vis.max():.6f}]")
        
        # CRITICO: Ora possiamo decodificare ogni token correttamente
        print(f"\n[PER-TOKEN ATTENTION ANALYSIS]:")
        for i in range(len(question_token_ids)):
            token_attn = question2vis[i]
            token_id = question_token_ids[i]
            token_text = tokenizer.decode([token_id])
            print(f"    Token {i}: '{token_text}' (ID={token_id}) | "
                  f"max={token_attn.max():.6f}, mean={token_attn.mean():.6f}")
        
        # Media finale
        mean_attention = question2vis.mean(0)
        
        print(f"\n[FINAL MEAN ATTENTION (question→vision)]:")
        print(f"    Shape: {mean_attention.shape}")
        print(f"    Range: [{mean_attention.min():.6f}, {mean_attention.max():.6f}]")
    
    elif mode == 'grounding':
        # ========== VISION → VISION SELF-ATTENTION (PROMPT-INDEPENDENT) ==========
        
        vis2vis = attn[vis_start:vis_end, vis_start:vis_end]
        
        print(f"\n[VISION→VISION SELF-ATTENTION]:")
        print(f"    Shape: {vis2vis.shape}")
        print(f"    Range: [{vis2vis.min():.6f}, {vis2vis.max():.6f}]")
        
        # Media: quanto ogni patch è "guardata" dalle altre
        mean_attention = vis2vis.mean(0)
        
        print(f"\n[FINAL MEAN ATTENTION (vision→vision)]:")
        print(f"    Shape: {mean_attention.shape}")
        print(f"    Range: [{mean_attention.min():.6f}, {mean_attention.max():.6f}]")
        
        print(f"\n[NOTE]: VIRAL-style attention is PROMPT-INDEPENDENT")
        print(f"    This shows internal visual structure, not task relevance")
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'grounding' or 'viral'")
    
    # Fingerprint
    import hashlib
    # Arrotonda a 6 decimali (ignora noise da float16)
    mean_attention_rounded = np.round(mean_attention, decimals=6)
    fingerprint = hashlib.md5(mean_attention_rounded.tobytes()).hexdigest()[:16]
    print(f"\n[ATTENTION FINGERPRINT (6 decimals)]: {fingerprint}")
    print("="*70 + "\n")
    
    return mean_attention

# Processa i layer selezionati
print(f"\n[INFO] Processing {len(LAYER_INDICES)} layers: {LAYER_INDICES}")

attention_maps = {}  # Dizionario per salvare le attention map

for LAYER_IDX in LAYER_INDICES:
    if LAYER_IDX >= len(attn_list):
        print(f"[ERROR] Layer {LAYER_IDX} not found. Model has {len(attn_list)} layers.")
        continue
    
    print(f"\n{'='*70}")
    print(f"PROCESSING LAYER {LAYER_IDX}")
    print(f"{'='*70}")
    
    attn_vector = process_attention_layer(
        attn_list[LAYER_IDX], LAYER_IDX, vis_start, vis_end, input_ids, tokenizer, mode=ATTENTION_MODE
    )
    
    # Salva per processare dopo
    attention_maps[LAYER_IDX] = attn_vector

    if SAVE_ATTENTION_ARRAYS:
        os.makedirs("./outputs/attention_arrays", exist_ok=True)
        
        # Crea un filename descrittivo
        import hashlib
        prompt_hash = hashlib.md5(PROMPT.encode()).hexdigest()[:8]
        
        save_path = f"./outputs/attention_arrays/layer{LAYER_IDX:02d}_{ATTENTION_MODE}_prompt{prompt_hash}.npy"
        np.save(save_path, attn_vector)
        print(f"[INFO] Saved raw attention array to: {save_path}")

# Reshape to 2D attention map
if attn_vector is None or len(attn_vector) == 0:
    raise ValueError(f"[ERROR] Empty attention vector at layer {LAYER_IDX}")

# Processa ogni attention map
normalized_maps = {}

for LAYER_IDX, attn_vector in attention_maps.items():
    print(f"\n[INFO] Processing attention map for layer {LAYER_IDX}...")
    
    # Reshape to 2D attention map
    if attn_vector is None or len(attn_vector) == 0:
        print(f"[ERROR] Empty attention vector at layer {LAYER_IDX}, skipping")
        continue
    
    print(f"[DEBUG] Attention vector stats (Layer {LAYER_IDX}):")
    print(f"  - Length: {len(attn_vector)}")
    print(f"  - Min: {attn_vector.min():.6f}")
    print(f"  - Max: {attn_vector.max():.6f}")
    print(f"  - Mean: {attn_vector.mean():.6f}")
    print(f"  - Has NaN: {np.isnan(attn_vector).any()}")
    print(f"  - Has Inf: {np.isinf(attn_vector).any()}")
    
    # Reshape più robusto
    expected_patches = int(np.sqrt(num_vis_tokens))
    if expected_patches**2 == num_vis_tokens:
        h = w = expected_patches
    else:
        h = w = int(np.sqrt(len(attn_vector)))
        if h * w < len(attn_vector):
            attn_vector = attn_vector[: h * w]
        elif h * w > len(attn_vector):
            attn_vector = np.pad(attn_vector, (0, h * w - len(attn_vector)), 'constant')
    
    print(f"[DEBUG] Reshaping to ({h}, {w})")
    attn_map = attn_vector.reshape(h, w)
    
    # Normalizza l'attention map
    attn_map_clean = attn_map.copy()
    
    if np.isnan(attn_map_clean).any() or np.isinf(attn_map_clean).any():
        print("[WARNING] Found NaN/Inf in attention map, replacing with 0")
        attn_map_clean = np.nan_to_num(attn_map_clean, nan=0.0, posinf=1.0, neginf=0.0)
    
    attn_min = attn_map_clean.min()
    attn_max = attn_map_clean.max()
    if attn_max - attn_min > 1e-8:
        attn_map_normalized = (attn_map_clean - attn_min) / (attn_max - attn_min)
    else:
        print("[WARNING] Attention map is constant, using uniform distribution")
        attn_map_normalized = np.ones_like(attn_map_clean) * 0.5
    
    attn_map_normalized = attn_map_normalized.astype(np.float32)
    
    print(f"[DEBUG] Normalized attention map:")
    print(f"  - Shape: {attn_map_normalized.shape}")
    print(f"  - Dtype: {attn_map_normalized.dtype}")
    print(f"  - Range: [{attn_map_normalized.min():.6f}, {attn_map_normalized.max():.6f}]")
    
    normalized_maps[LAYER_IDX] = attn_map_normalized

# ^ ----------------------------
# ^ VISUALIZATION
# ^ ----------------------------
def create_attention_overlay(img_pil, attn_map, alpha=0.5, colormap="jet"):
    """Crea overlay dell'attention map sull'immagine"""
    img = np.array(img_pil)

    # Resize attention map to image size
    attn_resized = cv2.resize(
        attn_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC
    )

    # Apply Gaussian blur for smoother visualization
    attn_resized = cv2.GaussianBlur(attn_resized, (5, 5), 0)

    # Colormap
    cmap = plt.cm.get_cmap(colormap)
    attn_colored = cmap(attn_resized)[..., :3]

    # Blend
    img_norm = img.astype(np.float32) / 255.0
    overlay = (1 - alpha) * img_norm + alpha * attn_colored
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    return overlay, attn_resized

def create_visualization_figure(image, attn_map_normalized, layer_idx, prompt, response, model_path, image_path, fingerprint=None):
    """
    Crea una figura completa con layout professionale per un singolo layer
    
    Args:
        image: PIL Image
        attn_map_normalized: numpy array [24, 24] normalizzato in [0,1]
        layer_idx: int, numero del layer
        prompt: str, la domanda
        response: str, la risposta del modello
        model_path: str, path del modello
        image_path: str, path dell'immagine
    
    Returns:
        fig: matplotlib figure
        timestamp: str, timestamp della creazione
    """
    import textwrap
    from datetime import datetime
    
    overlay, attn_resized = create_attention_overlay(image, attn_map_normalized, alpha=0.4)
    
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.8], height_ratios=[1, 0.15], 
                          hspace=0.35, wspace=0.25)
    
    # --- IMMAGINI (top row) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=11, pad=8)
    ax1.axis("off")
    
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(attn_map_normalized, cmap="hot", interpolation="nearest")
    ax2.set_title(f"Attention Map (Layer {layer_idx})", fontsize=11, pad=8)
    ax2.axis("off")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(overlay)
    ax3.set_title("Attention Overlay", fontsize=11, pad=8)
    ax3.axis("off")
    
    # --- INFO PANEL (right side) ---
    ax_info = fig.add_subplot(gs[0, 3])
    ax_info.axis('off')
    
    def wrap_text(text, width=55):
        return "\n".join(textwrap.wrap(text, width=width))

    # Costruisci info_text
    info_parts = [
        f"PROMPT:\n{wrap_text(prompt, width=60)}",
        f"\nRESPONSE:\n{wrap_text(response, width=60)}"
    ]

    if fingerprint:
        info_parts.append(f"\n\nFINGERPRINT:\n{fingerprint}")

    info_text = "".join(info_parts)
    
    ax_info.text(0.05, 0.95, info_text,
                ha='left', va='top', fontsize=9, family='monospace',
                transform=ax_info.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # --- FOOTER (bottom) ---
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    ax_footer = fig.add_subplot(gs[1, :])
    ax_footer.axis('off')
    footer = f"Model: {model_path.split('/')[-1]} | Image: {image_path.split('/')[-1]} | Layer {layer_idx} | {timestamp}"
    ax_footer.text(0.5, 0.5, footer, ha='center', va='center', 
                  fontsize=9, style='italic', transform=ax_footer.transAxes)
    
    if ATTENTION_MODE == 'grounding':
        plt.suptitle(f"Vision→Vision Attention Analysis - Layer {layer_idx}", fontsize=14, fontweight='bold', y=0.98)
    else:
        plt.suptitle(f"Question→Vision Attention Analysis - Layer {layer_idx}", fontsize=14, fontweight='bold', y=0.98)
    
    return fig, timestamp

print("\n[INFO] Creating visualizations for all layers...")

# Crea directory output
os.makedirs("./outputs/attn_viz", exist_ok=True)

# Crea una visualizzazione per ogni layer
for LAYER_IDX, attn_map_normalized in normalized_maps.items():
    print(f"\n[INFO] Creating visualization for layer {LAYER_IDX}...")

    import hashlib
    raw_attention = attention_maps[LAYER_IDX]
    raw_rounded = np.round(raw_attention, decimals=6)
    fingerprint = hashlib.md5(raw_rounded.tobytes()).hexdigest()[:16]
    
    fig, timestamp = create_visualization_figure(
        image=image,
        attn_map_normalized=attn_map_normalized,
        layer_idx=LAYER_IDX,
        prompt=PROMPT,
        response=generated_text,
        model_path=MODEL_PATH,
        image_path=IMAGE_PATH,
        fingerprint=fingerprint
    )
    
    out_path = f"./outputs/attn_viz/attention_layer_{LAYER_IDX:02d}_analysis_{timestamp}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)  # Importante: chiudi la figura per liberare memoria
    print(f"[INFO] Saved visualization to {out_path}")

print(f"\n[INFO] All visualizations saved successfully!")
