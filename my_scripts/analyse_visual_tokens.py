import os
import argparse
import torch
from PIL import Image
from llava.model.builder import load_pretrained_model

def parse_args():
    parser = argparse.ArgumentParser(description="Extract raw vision features from VIRAL/LLaVA model.")
    parser.add_argument("--model-path", type=str, default="./checkpoints/viral-7b",
                        help="Path to model checkpoint (e.g., ./checkpoints/viral-7b)")
    parser.add_argument("--image-path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--output-path", type=str, default="./vision_features.pt",
                        help="Where to save the extracted feature tensor")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda or cpu)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use float16 precision for faster extraction")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    print(f"[INFO] Loading model from {args.model_path} on {device} ({dtype})")

    # Identify the base and model name automatically
    is_7b = "7b" in args.model_path.lower()
    model_base = "lmsys/vicuna-7b-v1.5" if is_7b else "lmsys/vicuna-13b-v1.5"
    model_name = "liuhaotian/llava-v1.5-7b-lora" if is_7b else "liuhaotian/llava-v1.5-13b-lora"

    # Load model components
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path, model_base=model_base, model_name=model_name
    )
    model.to(device=device, dtype=dtype).eval()

    # Load image
    image = Image.open(args.image_path).convert("RGB")
    if hasattr(image_processor, "preprocess"):
        pixel_values = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
    else:
        pixel_values = image_processor(image, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device=device, dtype=dtype)

    print("[INFO] Extracting visual tokens from vision tower...")
    vision_tower = model.get_vision_tower()

    with torch.no_grad():
        try:
            vision_out = vision_tower(pixel_values, output_hidden_states=True)
            if hasattr(vision_out, "last_hidden_state"):
                vision_features = vision_out.last_hidden_state
            elif isinstance(vision_out, torch.Tensor):
                vision_features = vision_out
            else:
                vision_features = vision_out[0]
        except TypeError:
            vision_features = vision_tower(pixel_values)

    print(f"[INFO] Feature shape: {tuple(vision_features.shape)}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(vision_features.cpu(), args.output_path)
    print(f"[INFO] Saved features to {args.output_path}")

    # Optional: compute a quick summary for sanity check
    mean_activation = vision_features.mean().item()
    std_activation = vision_features.std().item()
    print(f"[INFO] Mean activation: {mean_activation:.4f}, Std: {std_activation:.4f}")


if __name__ == "__main__":
    main()