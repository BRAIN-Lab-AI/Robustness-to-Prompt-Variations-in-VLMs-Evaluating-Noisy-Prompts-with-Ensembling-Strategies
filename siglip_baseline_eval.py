"""
SigLIP Baseline Evaluation on Multiple Datasets

This script evaluates SigLIP zero-shot performance on vision datasets
without prompt noise. It computes Top-1, Top-5, and per-class accuracy.

Usage:
    python siglip_baseline_eval.py \
        --dataset_name oxford_pets \
        --dataset_dir ./data/oxford_pets \
        --batch_size 64
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
import numpy as np
from sklearn.metrics import confusion_matrix
import json

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- Collate: keep images as list of PILs ----
def collate_pil(batch):
    imgs, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return list(imgs), labels


# ---- Auto-detect the split folder if needed ----
def resolve_imagefolder_root(base_root):
    """
    base_root: e.g. ./data/food101
    This tries to find the actual folder that contains class subfolders.
    It checks common patterns like: base_root/{train,val,test,images}
    If none match, it assumes base_root is already the ImageFolder root.
    """
    if not os.path.isdir(base_root):
        raise FileNotFoundError(f"Base root does not exist: {base_root}")

    # Candidates in order of preference
    candidates = ["test", "val", "validation", "train", "images"]

    for c in candidates:
        p = os.path.join(base_root, c)
        if os.path.isdir(p):
            # Check if it looks like ImageFolder (subdirs = classes)
            subdirs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
            if len(subdirs) > 0:
                print(f"  -> Using split folder: {p}")
                return p

    # Fallback: assume base_root is already an ImageFolder root
    print(f"  -> Using base root directly as ImageFolder: {base_root}")
    return base_root


# ---- SigLIP helpers ----
@torch.no_grad()
def get_text_features(processor, model, classnames, templates, device):
    """
    Computes normalized text embeddings for each class using multiple templates.
    """
    all_embeds = []
    for cname in classnames:
        prompts = [t.format(cname) for t in templates]

        inputs = processor(
            text=prompts,
            return_tensors="pt",
            padding=True
        ).to(device)

        # SigLIP text encoder
        text_embeds = model.get_text_features(**inputs)  # (num_prompts, dim)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Average over templates
        class_embed = text_embeds.mean(dim=0, keepdim=True)
        class_embed = class_embed / class_embed.norm(dim=-1, keepdim=True)
        all_embeds.append(class_embed)

    return torch.cat(all_embeds, dim=0)  # (num_classes, dim)


@torch.no_grad()
def get_image_features(processor, model, dataloader, device):
    """
    Computes normalized image embeddings for all samples in a DataLoader.
    Assumes dataloader yields (list_of_PIL_images, labels).
    """
    all_feats = []
    all_labels = []

    for imgs, labels in tqdm(dataloader, desc="Extracting SigLIP image features"):
        inputs = processor(
            images=imgs,
            return_tensors="pt"
        ).to(device)

        # SigLIP image encoder
        img_embeds = model.get_image_features(**inputs)  # (batch, dim)
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

        all_feats.append(img_embeds.cpu())
        all_labels.append(labels)

    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


def run_siglip_on_imagefolder(
    dataset_name,
    base_root,
    model_id="google/siglip-so400m-patch14-384",
    batch_size=64,
    compute_confusion=True,
    save_json="",
):
    """
    base_root: the dataset root, e.g. "./data/food101"

    This function will:
      - resolve the actual split folder
      - run zero-shot SigLIP
      - compute Top-1, Top-5, per-class accuracy, confusion matrix
    """
    print(f"\n===== Dataset: {dataset_name} =====")
    print(f"Using device: {device}")
    print("Base root:", base_root)

    root = resolve_imagefolder_root(base_root)

    # 1) Dataset & loader (ImageFolder)
    dataset = ImageFolder(root=root)
    classnames = dataset.classes
    print(f"  Classes: {len(classnames)}, Samples: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_pil,
    )

    # 2) Load SigLIP
    print("  Loading SigLIP model:", model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    # 3) Image features
    img_feats, labels = get_image_features(processor, model, dataloader, device)

    # 4) Text features (generic templates)
    templates = [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a close-up photo of a {}.",
    ]
    text_feats = get_text_features(processor, model, classnames, templates, device)

    # 5) Compute logits
    img_feats_dev = img_feats.to(device)
    text_feats_dev = text_feats.to(device)
    logits = img_feats_dev @ text_feats_dev.t()   # (N, C)

    preds = logits.argmax(dim=-1).cpu()
    labels_cpu = labels.cpu()

    # Top-1
    top1 = (preds == labels_cpu).float().mean().item() * 100.0

    # Top-5
    top5_idx = logits.topk(5, dim=-1).indices.cpu()
    correct_top5 = (top5_idx == labels_cpu.unsqueeze(1)).any(dim=1)
    top5 = correct_top5.float().mean().item() * 100.0

    # Per-class accuracy
    per_class = {}
    for idx, cname in enumerate(classnames):
        mask = (labels_cpu == idx)
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == labels_cpu[mask]).float().mean().item() * 100.0
        per_class[cname] = acc

    # Confusion matrix (optional)
    cm = None
    if compute_confusion:
        cm = confusion_matrix(labels_cpu.numpy(), preds.numpy())

    print(f"\n  Top-1 accuracy: {top1:.2f}%")
    print(f"  Top-5 accuracy: {top5:.2f}%")
    print(f"  Per-class accuracy computed for {len(per_class)} classes.")

    results = {
        "dataset": dataset_name,
        "model": model_id,
        "top1": top1,
        "top5": top5,
        "per_class": per_class,
        "num_classes": len(classnames),
        "num_samples": len(dataset),
    }

    # Save results
    if save_json:
        os.makedirs(os.path.dirname(save_json), exist_ok=True)
        # Don't save confusion matrix in JSON (too large)
        with open(save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved results to: {save_json}")

    # Add confusion matrix to return dict
    results["confusion_matrix"] = cm
    results["labels"] = labels_cpu.numpy()
    results["preds"] = preds.numpy()

    return results


def main():
    parser = argparse.ArgumentParser(description="SigLIP Baseline Evaluation")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset name for logging")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--model_id", type=str,
                        default="google/siglip-so400m-patch14-384",
                        help="SigLIP model ID from HuggingFace")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for evaluation")
    parser.add_argument("--save_json", type=str, default="",
                        help="Path to save results JSON")
    parser.add_argument("--no_confusion", action="store_true",
                        help="Skip confusion matrix computation")

    args = parser.parse_args()

    results = run_siglip_on_imagefolder(
        dataset_name=args.dataset_name,
        base_root=args.dataset_dir,
        model_id=args.model_id,
        batch_size=args.batch_size,
        compute_confusion=not args.no_confusion,
        save_json=args.save_json,
    )

    print("\n===== Evaluation Complete =====")


if __name__ == "__main__":
    main()
