# analyse_feature_coherence_batch.py
import os
import json
import math
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from llava.model.builder import load_pretrained_model

# -----------------------------
# CONFIG
# -----------------------------
COCO_IMG_DIR = "/cluster/project/cvg/data/mscoco/mscoco/train2017"
COCO_ANN_PATH = "/cluster/project/cvg/data/mscoco/annotations/instances_train2017.json"
FEATURE_DIR = "./outputs/features-simple"
OUTPUT_PATH = "./outputs/analysis/feature_coherence_summary.json"

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

MODEL_PATH = "/cluster/project/cvg/students/fbondi/sem-project/VIRAL/checkpoints/viral_checkpoints/llava-v1.5-7b-instruct-repa-dino-single-16/checkpoint-5000"
MODEL_BASE = "liuhaotian/llava-v1.5-7b"
MODEL_NAME = "llava-v1.5-7b-lora"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32

SAVE_INTERVAL = 200

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# -----------------------------
# ARGPARSE
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser("Analyse feature coherence on COCO")
    parser.add_argument("--start", type=int, default=0, help="Start index of images")
    parser.add_argument(
        "--end", type=int, default=None, help="End index of images (exclusive)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional limit on number of images"
    )
    return parser.parse_args()


# -----------------------------
# UTILS
# -----------------------------
def cosine_similarity(a, b):
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return a @ b.T


def downsample_mask(mask, target_hw):
    h, w = target_hw
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_resized = mask_img.resize((w, h), resample=Image.NEAREST)
    return np.array(mask_resized) > 127


def strip_cls_and_grid(features: torch.Tensor):
    seq_len = features.size(0)
    g = int(round(math.sqrt(seq_len)))
    if g * g == seq_len:
        return features.view(g, g, -1), (g, g)
    seq_len_wo_cls = seq_len - 1
    g2 = int(round(math.sqrt(seq_len_wo_cls)))
    if g2 * g2 == seq_len_wo_cls:
        return features[1:].view(g2, g2, -1), (g2, g2)
    raise ValueError(f"Unexpected vision sequence length: {seq_len}.")


@torch.no_grad()
def extract_features(image_path, model, image_processor):
    image = Image.open(image_path).convert("RGB")
    if hasattr(image_processor, "preprocess"):
        pixel_values = image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ]
    else:
        pixel_values = image_processor(image, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device=DEVICE, dtype=DTYPE)

    vision_tower = model.get_vision_tower()
    try:
        vision_out = vision_tower(pixel_values, output_hidden_states=True)
        if hasattr(vision_out, "last_hidden_state"):
            feats = vision_out.last_hidden_state
        elif isinstance(vision_out, torch.Tensor):
            feats = vision_out
        else:
            feats = vision_out[0]
    except TypeError:
        feats = vision_tower(pixel_values)

    if feats.dim() == 3:
        feats = feats.squeeze(0)
    return feats.to(device=DEVICE, dtype=DTYPE)


def analyze_image(coco, img_id):
    img_info = coco.loadImgs([img_id])[0]
    feature_path = os.path.join(FEATURE_DIR, f"features_{img_id}.pt")
    if not os.path.exists(feature_path):
        return []

    features = torch.load(feature_path, map_location="cpu").squeeze(0)
    if features.dtype == torch.float16:
        features = features.float()
    # print("features shape:", features.shape)
    features_hw, (H, W) = strip_cls_and_grid(features)

    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)
    results = []

    for ann in anns:
        mask = coco.annToMask(ann)
        if mask.sum() == 0:
            continue
        label = coco.loadCats([ann["category_id"]])[0]["name"]
        down_mask = downsample_mask(mask, (H, W))

        features_hw = features_hw.contiguous()  # garantisce layout
        flat_features = features_hw.view(-1, features_hw.size(-1))
        flat_mask = torch.from_numpy(down_mask.reshape(-1)).to(flat_features.device)

        inside = flat_features[flat_mask]
        outside = flat_features[~flat_mask]
        # print(f"DEBUG img {img_id}: inside={inside.size(0)} outside={outside.size(0)}")
        if inside.size(0) < 5 or outside.size(0) < 5:
            continue

        sim_in = cosine_similarity(inside, inside)
        triu_mask = torch.triu(torch.ones_like(sim_in, dtype=torch.bool), 1)
        intra = sim_in[triu_mask].mean().item()
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
    print(f"[DEBUG] Image {img_id}: {len(results)} valid objects")
    return results


# -----------------------------
# MAIN
# -----------------------------
def main():
    args = parse_args()

    print("[INFO] Loading COCO dataset...")
    coco = COCO(COCO_ANN_PATH)
    img_ids = coco.getImgIds()

    # Usa flag start/end/limit
    start = args.start
    end = args.end or len(img_ids)
    img_ids = img_ids[start:end]
    if args.limit:
        img_ids = img_ids[: args.limit]
    print(f"[INFO] Processing images {start}-{end} (total {len(img_ids)})")

    print(f"[INFO] Preparing model on {DEVICE} ({DTYPE})...")
    _, model, image_processor, _ = load_pretrained_model(
        model_path=MODEL_PATH, model_base=MODEL_BASE, model_name=MODEL_NAME
    )
    model.to(device=DEVICE, dtype=DTYPE).eval()

    all_results, failed = [], []

    for i, img_id in enumerate(tqdm(img_ids, desc="Processing images")):
        try:
            feature_path = os.path.join(FEATURE_DIR, f"features_{img_id}.pt")
            if not os.path.exists(feature_path):
                img_info = coco.loadImgs([img_id])[0]
                img_path = os.path.join(COCO_IMG_DIR, img_info["file_name"])
                print(f"[DEBUG] Extracting features for image {img_id}", flush=True)
                feats = extract_features(img_path, model, image_processor)
                torch.save(feats.cpu(), feature_path)
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

            res = analyze_image(coco, img_id)
            if not res:
                print(f"[DEBUG] No valid results for image {img_id}")
            else:
                print(f"[DEBUG] Got {len(res)} results for image {img_id}")
            all_results.extend(res)
        except Exception as e:
            print(f"[ERROR] Image {img_id} failed: {e}", flush=True)
            failed.append((img_id, str(e)))

        if i > 0 and i % SAVE_INTERVAL == 0:
            tmp_path = OUTPUT_PATH.replace(".json", f"_partial_{i}.json")
            with open(tmp_path, "w") as f:
                json.dump(all_results, f)
            print(f"[INFO] Checkpoint saved ({i}/{len(img_ids)}) → {tmp_path}")

    if not all_results:
        print("[WARN] No valid data found.")
        return

    final_path = OUTPUT_PATH.replace(".json", f"_{start}_{end}.json")
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[INFO] Saved summary to {final_path}")

    contrasts = [r["contrast"] for r in all_results]
    print(f"[INFO] Analysed {len(all_results)} objects across {len(img_ids)} images.")
    print(f"[INFO] Mean contrast: {np.mean(contrasts):.3f} ± {np.std(contrasts):.3f}")

    if failed:
        print(f"[WARN] Failed on {len(failed)} images. Example: {failed[:5]}")


if __name__ == "__main__":
    main()
