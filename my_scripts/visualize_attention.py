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
PROMPT = "What is shown in the image?"
LAYER_IDX = 16  # Layer da visualizzare
SAVE_ALL_LAYERS = False  # Se True, salva visualizzazioni per tutti i layer

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

# Dopo il forward pass
""" print(f"\n=== SEQUENCE ANALYSIS ===")
print(f"Input tokens (text only): {input_ids.shape[1]}")
print(f"Attention sequence length: {attn_list[0].shape[-1]}")
print(f"Expected vision tokens: {num_vis_tokens}")
print(f"Missing tokens: {attn_list[0].shape[-1] - input_ids.shape[1]}")
print(
    f"Match vision tokens? {attn_list[0].shape[-1] - input_ids.shape[1] == num_vis_tokens}"
) """

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
def process_attention_layer(attn_tensor, layer_idx, vis_start, vis_end):
    """Processa l'attention di un singolo layer - Focus su question tokens"""
    
    batch_size, num_heads, seq_len, _ = attn_tensor.shape
    
    print(f"[DEBUG] Layer {layer_idx}: attn shape = {attn_tensor.shape}")
    print(f"[DEBUG] vis_start={vis_start}, vis_end={vis_end}")
    
    if vis_end > seq_len:
        print(f"[WARNING] vis_end ({vis_end}) > seq_len ({seq_len})")
        vis_end = seq_len
    
    attn = attn_tensor[0].mean(0).cpu().numpy()  # [seq_len, seq_len]
    
    # NUOVO: Usa solo i token della question (dopo l'immagine)
    # Questi sono i token più rilevanti per capire dove il modello guarda
    question_start = vis_end  # Token dopo l'immagine
    question_end = seq_len    # Fino alla fine
    question_positions = np.arange(question_start, question_end)
    
    print(f"[DEBUG] Using only question tokens: positions {question_start}-{question_end-1} ({len(question_positions)} tokens)")
    
    if len(question_positions) == 0:
        print(f"[WARNING] No question tokens found")
        return None
    
    # Question → Vision attention
    question2vis = attn[question_positions][:, vis_start:vis_end]
    
    print(f"[DEBUG] question2vis shape: {question2vis.shape}")
    print(f"[DEBUG] question2vis range: [{question2vis.min():.6f}, {question2vis.max():.6f}]")
    
    # Media sui token della question
    mean_attention = question2vis.mean(0)
    
    print(f"[DEBUG] mean_attention range: [{mean_attention.min():.6f}, {mean_attention.max():.6f}]")
    
    return mean_attention


# Processa il layer selezionato
if LAYER_IDX >= len(attn_list):
    print(f"[ERROR] Layer {LAYER_IDX} not found. Model has {len(attn_list)} layers.")
    LAYER_IDX = len(attn_list) - 1
    print(f"[INFO] Using layer {LAYER_IDX} instead.")

attn_vector = process_attention_layer(
    attn_list[LAYER_IDX], LAYER_IDX, vis_start, vis_end
)

# Reshape to 2D attention map
if attn_vector is None or len(attn_vector) == 0:
    raise ValueError(f"[ERROR] Empty attention vector at layer {LAYER_IDX}")

# Reshape to 2D attention map
if attn_vector is None or len(attn_vector) == 0:
    raise ValueError(f"[ERROR] Empty attention vector at layer {LAYER_IDX}")

print(f"\n[DEBUG] Attention vector stats:")
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
    # Non perfettamente quadrato (es. con CLS token)
    h = w = int(np.sqrt(len(attn_vector)))
    if h * w < len(attn_vector):
        # Tronca
        attn_vector = attn_vector[: h * w]
    elif h * w > len(attn_vector):
        # Padda con zeri
        attn_vector = np.pad(attn_vector, (0, h * w - len(attn_vector)), 'constant')

print(f"[DEBUG] Reshaping to ({h}, {w})")
attn_map = attn_vector.reshape(h, w)

# CRITICO: Normalizza l'attention map per cv2
attn_map_clean = attn_map.copy()

# Rimuovi NaN e Inf
if np.isnan(attn_map_clean).any() or np.isinf(attn_map_clean).any():
    print("[WARNING] Found NaN/Inf in attention map, replacing with 0")
    attn_map_clean = np.nan_to_num(attn_map_clean, nan=0.0, posinf=1.0, neginf=0.0)

# Normalizza in [0, 1]
attn_min = attn_map_clean.min()
attn_max = attn_map_clean.max()
if attn_max - attn_min > 1e-8:  # Evita divisione per zero
    attn_map_normalized = (attn_map_clean - attn_min) / (attn_max - attn_min)
else:
    print("[WARNING] Attention map is constant, using uniform distribution")
    attn_map_normalized = np.ones_like(attn_map_clean) * 0.5

# Converti in float32 per OpenCV
attn_map_normalized = attn_map_normalized.astype(np.float32)

print(f"[DEBUG] Normalized attention map:")
print(f"  - Shape: {attn_map_normalized.shape}")
print(f"  - Dtype: {attn_map_normalized.dtype}")
print(f"  - Range: [{attn_map_normalized.min():.6f}, {attn_map_normalized.max():.6f}]")

# DIAGNOSTICA

print(f"[DIAGNOSTIC] Total sequence length: {attn_list[0].shape[-1]}")
print(f"[DIAGNOSTIC] Vision patches: {num_vis_tokens}")
print(f"[DIAGNOSTIC] Input IDs shape: {input_ids.shape}")
print(f"[DIAGNOSTIC] Unique token IDs: {len(torch.unique(input_ids))}")
print(f"[DIAGNOSTIC] First 20 token IDs: {input_ids[0][:20].tolist()}")


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


# Crea visualizzazione
overlay, attn_resized = create_attention_overlay(image, attn_map_normalized, alpha=0.4)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

# Attention map
im = axes[1].imshow(attn_map, cmap="hot", interpolation="nearest")
axes[1].set_title(f"Attention Map (Layer {LAYER_IDX})")
axes[1].axis("off")
plt.colorbar(im, ax=axes[1], fraction=0.046)

# Overlay
axes[2].imshow(overlay)
axes[2].set_title(f"Attention Overlay (Layer {LAYER_IDX})")
axes[2].axis("off")

plt.suptitle(
    f"Cross-Modal Attention Analysis - {MODEL_PATH.split('/')[-1]}", fontsize=14
)
plt.tight_layout()

# Save
os.makedirs("./outputs/attn_viz", exist_ok=True)
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
out_path = f"./outputs/attn_viz/attention_layer_{LAYER_IDX}_analysis_{timestamp}.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"[INFO] Saved visualization to {out_path}")

# ^ ----------------------------
# ^ OPTIONAL: SAVE ALL LAYERS
# ^ ----------------------------
if SAVE_ALL_LAYERS:
    print("[INFO] Processing all layers...")
    os.makedirs("../outputs/attn_viz/all_layers", exist_ok=True)

    for layer_idx in range(len(attn_list)):
        attn_vector = process_attention_layer(
            attn_list[layer_idx], layer_idx, vis_start, vis_end
        )

        if attn_vector is not None:
            h = w = int(np.sqrt(len(attn_vector)))
            attn_map = attn_vector.reshape(h, w)
            attn_map = (attn_map - attn_map.min()) / (
                attn_map.max() - attn_map.min() + 1e-8
            )

            overlay, _ = create_attention_overlay(image, attn_map)

            plt.figure(figsize=(6, 6))
            plt.imshow(overlay)
            plt.title(f"Layer {layer_idx}")
            plt.axis("off")
            plt.savefig(
                f"../outputs/attn_viz/all_layers/layer_{layer_idx:02d}.png",
                dpi=100,
                bbox_inches="tight",
            )
            plt.close()

    print(f"[INFO] Saved all {len(attn_list)} layer visualizations")
