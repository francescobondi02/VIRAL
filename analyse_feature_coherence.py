import os
import torch
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms.functional as F

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------
COCO_IMG_ID = 9  # corrisponde al file 000000000009.jpg
COCO_IMG_DIR = "/cluster/project/cvg/data/mscoco/mscoco/train2017"
COCO_ANN_PATH = "/cluster/project/cvg/data/mscoco/annotations/instances_train2017.json"
FEATURE_PATH = "./outputs/features/coco_sample.pt"
OUTPUT_DIR = "./outputs/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------
# UTILS
# ----------------------------------------------------
def cosine_similarity(a, b):
    """Compute cosine similarity matrix between two sets of vectors."""
    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return a_norm @ b_norm.T


def downsample_mask(mask, target_size=(24, 24)):
    """Convert full-res segmentation mask to match the 24x24 patch grid."""
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_resized = mask_img.resize(target_size, resample=Image.NEAREST)
    return np.array(mask_resized) > 127


# ----------------------------------------------------
# LOAD FEATURES
# ----------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Loading features from {FEATURE_PATH}")
features = torch.load(FEATURE_PATH, map_location=device)  # stay on GPU if available
features = features.squeeze(0)  # (576, 1024)
num_patches = int(np.sqrt(features.size(0)))  # 24
features = features.view(num_patches, num_patches, -1)
print(
    f"[INFO] Feature map shape: {features.shape}, dtype: {features.dtype}, device: {features.device}"
)

# ----------------------------------------------------
# LOAD COCO ANNOTATION + MASKS
# ----------------------------------------------------
print("[INFO] Loading COCO annotations...")
coco = COCO(COCO_ANN_PATH)
print(f"[DEBUG] Total images in COCO: {len(coco.getImgIds())}")
print(f"[DEBUG] Looking for image id {COCO_IMG_ID}")
ann_ids = coco.getAnnIds(imgIds=[COCO_IMG_ID])
anns = coco.loadAnns(ann_ids)

img_info = coco.loadImgs([COCO_IMG_ID])[0]
img_path = os.path.join(COCO_IMG_DIR, img_info["file_name"])
print(f"[INFO] Loaded image: {img_path}")

# Create binary mask per object category
masks = []
labels = []
for ann in anns:
    m = coco.annToMask(ann)
    if m.sum() == 0:
        continue
    masks.append(m)
    labels.append(coco.loadCats([ann["category_id"]])[0]["name"])

print(f"[INFO] Found {len(masks)} objects: {labels}")

# ----------------------------------------------------
# ANALYZE FEATURE COHERENCE
# ----------------------------------------------------
results = []
grid_size = (num_patches, num_patches)

for mask, label in zip(masks, labels):
    # Downsample segmentation to patch grid
    down_mask = downsample_mask(mask, target_size=grid_size)

    # Select token embeddings inside and outside the mask
    flat_features = features.view(-1, features.size(-1))  # (576, 1024)
    flat_mask = down_mask.flatten()  # (576,)

    inside_tokens = flat_features[flat_mask]
    outside_tokens = flat_features[~flat_mask]

    if len(inside_tokens) < 5 or len(outside_tokens) < 5:
        continue

    # Compute similarities
    sim_matrix = cosine_similarity(inside_tokens, inside_tokens)
    intra_sim = (
        sim_matrix[torch.triu(torch.ones_like(sim_matrix), 1) == 1].mean().item()
    )

    inter_sim_matrix = cosine_similarity(inside_tokens, outside_tokens)
    inter_sim = inter_sim_matrix.mean().item()

    results.append(
        {
            "object": label,
            "intra_sim": intra_sim,
            "inter_sim": inter_sim,
            "contrast": intra_sim - inter_sim,
        }
    )

# ----------------------------------------------------
# OUTPUT
# ----------------------------------------------------
if not results:
    print("[WARN] No valid masks or features to analyze.")
else:
    print("\n[RESULTS]")
    for r in results:
        print(
            f" - {r['object']:<15} | intra: {r['intra_sim']:.3f} | inter: {r['inter_sim']:.3f} | Δ: {r['contrast']:.3f}"
        )

    avg_contrast = np.mean([r["contrast"] for r in results])
    print(f"\n[INFO] Mean object-background contrast: {avg_contrast:.3f}")

    torch.save(results, os.path.join(OUTPUT_DIR, f"feature_coherence_{COCO_IMG_ID}.pt"))
    print(f"[INFO] Saved results to {OUTPUT_DIR}/feature_coherence_{COCO_IMG_ID}.pt")

# ----------------------------------------------------
# BETTER VISUALIZATION
# ----------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

if results:
    print(f"[INFO] Creating clearer visualization for image {COCO_IMG_ID}...")

    # Carica immagine originale
    img = np.array(Image.open(img_path).convert("RGB"))

    cmap = plt.cm.get_cmap("tab10", len(masks))
    overlay = img.copy()

    for i, (mask, label) in enumerate(zip(masks, labels)):
        color = tuple(int(c * 255) for c in cmap(i)[:3])

        # Contorni della maschera
        mask_uint8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, thickness=3)

        # Bounding box
        x, y, w, h = cv2.boundingRect(mask_uint8)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

        # Label testuale
        cv2.putText(
            overlay,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    # Salva output
    save_path = os.path.join(OUTPUT_DIR, f"coco_masks_outline_{COCO_IMG_ID}.png")
    Image.fromarray(overlay).save(save_path)
    print(f"[INFO] Saved improved visualization to {save_path}")
