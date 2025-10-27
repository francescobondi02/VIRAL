# analyse_feature_coherence_batch.py
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from pycocotools.coco import COCO
from PIL import Image
from llava.model.builder import load_pretrained_model

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------
COCO_IMG_DIR = "/cluster/project/cvg/data/mscoco/mscoco/train2017"
COCO_ANN_PATH = "/cluster/project/cvg/data/mscoco/annotations/instances_train2017.json"
FEATURE_DIR = "./outputs/features"
OUTPUT_PATH = "./outputs/analysis/feature_coherence_summary.json"
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Model parameters
MODEL_PATH = "liuhaotian/llava-v1.5-7b-lora"
MODEL_BASE = "lmsys/vicuna-7b-v1.5"
MODEL_NAME = "liuhaotian/llava-v1.5-7b-lora"

# Number of images to analyze
NUM_IMAGES = 100  # puoi aumentare gradualmente
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32


# ----------------------------------------------------
# UTILS
# ----------------------------------------------------
def cosine_similarity(a, b):
    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return a_norm @ b_norm.T


def downsample_mask(mask, target_size=(24, 24)):
    """Utils to understand which patches correspond to the object"""
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_resized = mask_img.resize(target_size, resample=Image.NEAREST)
    return np.array(mask_resized) > 127


@torch.no_grad()
def extract_features(image_path, model, image_processor):
    """Extract visual features from a single image using the vision tower."""
    image = Image.open(image_path).convert("RGB")
    if hasattr(image_processor, "preprocess"):
        pixel_values = image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ]
    else:
        pixel_values = image_processor(image, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device=DEVICE, dtype=DTYPE)

    vision_tower = model.get_vision_tower()

    # --- robust fallback ---
    try:
        vision_out = vision_tower(pixel_values, output_hidden_states=True)
        # * Obtain the visual features from the vision tower
        if hasattr(vision_out, "last_hidden_state"):
            feats = vision_out.last_hidden_state
        elif isinstance(vision_out, torch.Tensor):
            feats = vision_out
        else:
            feats = vision_out[0]
    except TypeError:
        # fallback for vision towers that don’t accept `output_hidden_states`
        feats = vision_tower(pixel_values)

    return feats.to(device=DEVICE, dtype=DTYPE)


# ----------------------------------------------------
# MAIN ANALYSIS
# ----------------------------------------------------
def analyze_image(coco, img_id, model=None, image_processor=None):
    img_info = coco.loadImgs([img_id])[0]
    img_path = os.path.join(COCO_IMG_DIR, img_info["file_name"])
    feature_path = os.path.join(FEATURE_DIR, f"features_{img_id}.pt")

    # Load or extract features
    if os.path.exists(feature_path):
        features = torch.load(feature_path, map_location=DEVICE)
    else:
        if model is None or image_processor is None:
            raise RuntimeError(
                "Model and image_processor required for feature extraction."
            )
        features = extract_features(img_path, model, image_processor)
        torch.save(features, feature_path)

    features = features.squeeze(0)
    num_patches = int(np.sqrt(features.size(0)))
    features = features.view(num_patches, num_patches, -1)

    # * Load annotations
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)
    results = []

    for ann in anns:
        m = coco.annToMask(ann)
        if m.sum() == 0:
            continue
        label = coco.loadCats([ann["category_id"]])[0]["name"]
        down_mask = downsample_mask(m, target_size=(num_patches, num_patches))

        flat_features = features.view(-1, features.size(-1))
        flat_mask = down_mask.flatten()

        inside = flat_features[flat_mask]
        outside = flat_features[~flat_mask]
        if len(inside) < 5 or len(outside) < 5:
            continue

        sim_in = cosine_similarity(inside, inside)
        intra = sim_in[torch.triu(torch.ones_like(sim_in), 1) == 1].mean().item()
        inter = cosine_similarity(inside, outside).mean().item()

        results.append(
            {
                "image_id": img_id,
                "object": label,
                "intra": intra,
                "inter": inter,
                "contrast": intra - inter,
            }
        )
    return results


def main():
    print("[INFO] Loading COCO dataset...")
    coco = COCO(COCO_ANN_PATH)
    img_ids = coco.getImgIds()[:NUM_IMAGES]

    print(f"[INFO] Preparing model on {DEVICE} ({DTYPE})...")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=MODEL_PATH, model_base=MODEL_BASE, model_name=MODEL_NAME
    )
    model.to(device=DEVICE, dtype=DTYPE).eval()

    all_results = []
    for img_id in tqdm(img_ids, desc="Analysing COCO images"):
        try:
            res = analyze_image(coco, img_id, model, image_processor)
            all_results.extend(res)
        except Exception as e:
            print(f"[WARN] Failed on image {img_id}: {e}")

    if not all_results:
        print("[WARN] No valid data found.")
        return

    # Save all results
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[INFO] Saved summary to {OUTPUT_PATH}")

    # Print quick stats
    contrasts = [r["contrast"] for r in all_results]
    print(f"[INFO] Analysed {len(all_results)} objects across {NUM_IMAGES} images.")
    print(f"[INFO] Mean contrast: {np.mean(contrasts):.3f} ± {np.std(contrasts):.3f}")


if __name__ == "__main__":
    main()
