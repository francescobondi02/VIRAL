import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "/cluster/project/cvg/students/fbondi/sem-project/VIRAL/checkpoints/viral_checkpoints/llava-v1.5-7b-instruct-repa-dino-single-16"
MODEL_BASE = "liuhaotian/llava-v1.5-7b"
IMAGE_PATH = "../images/llava_logo.png"
PROMPT = "What is shown in the image?"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ----------------------------
# LOAD MODEL
# ----------------------------
disable_torch_init()
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL_PATH, model_base=MODEL_BASE, model_name="llava-v1.5-7b-lora"
)
model.to(DEVICE, dtype=DTYPE).eval()

# ----------------------------
# REGISTER HOOKS
# ----------------------------
attention_maps = {}


def save_attention_hook(name):
    def hook(module, input, output):
        # output = (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) > 1:
            attn = output[1]
            attention_maps[name] = attn.detach().cpu()

    return hook


for i, layer in enumerate(model.model.model.layers):
    layer.self_attn.register_forward_hook(save_attention_hook(f"attn_{i}"))

# ----------------------------
# PREPARE IMAGE + PROMPT
# ----------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
if hasattr(image_processor, "preprocess"):
    pixel_values = image_processor.preprocess(image, return_tensors="pt")[
        "pixel_values"
    ]
else:
    pixel_values = image_processor(image, return_tensors="pt")["pixel_values"]
pixel_values = pixel_values.to(DEVICE, dtype=DTYPE)

conv = conv_templates["llava_v1"].copy()
conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + PROMPT)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = (
    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    .unsqueeze(0)
    .to(DEVICE)
)

# ----------------------------
# RUN INFERENCE (capturing hooks)
# ----------------------------
with torch.inference_mode():
    _ = model.generate(
        inputs=input_ids, images=pixel_values, max_new_tokens=40, do_sample=False
    )

# ----------------------------
# EXTRACT CROSS-MODAL ATTENTION (TEXT→VISION)
# ----------------------------
# pick one layer to visualize
layer_name = "attn_16"
attn = attention_maps[layer_name][0]  # [num_heads, seq, seq]
attn = attn.mean(0)  # average over heads

# figure out token partition
num_vis_tokens = model.get_vision_tower().num_patches  # e.g. 576
num_text_tokens = attn.shape[0] - num_vis_tokens

# text-to-vision attention submatrix
text2vis = attn[:num_text_tokens, num_text_tokens:]  # [text, vision]

# aggregate attention over all text tokens
mean_text2vis = text2vis.mean(0).numpy()  # [vision_tokens]

# reshape to spatial grid
h = w = int(np.sqrt(num_vis_tokens))
attn_map = mean_text2vis.reshape(h, w)
attn_map = attn_map / attn_map.max()


# ----------------------------
# VISUALIZE
# ----------------------------
def overlay_attention(image_pil, attn_map):
    img = np.array(image_pil.resize((attn_map.shape[1], attn_map.shape[0])))
    attn_map = np.clip(attn_map, 0, 1)
    attn_map = plt.cm.jet(attn_map)[..., :3]
    overlay = (0.6 * img / 255 + 0.4 * attn_map) / ((0.6 + 0.4))
    return (overlay * 255).astype(np.uint8)


overlay = overlay_attention(image, attn_map)

os.makedirs("../outputs/attn_viz", exist_ok=True)
out_path = f"../outputs/attn_viz/attn_text2vision_{layer_name}.png"
Image.fromarray(overlay).save(out_path)

print(f"[INFO] Saved overlay attention map → {out_path}")
