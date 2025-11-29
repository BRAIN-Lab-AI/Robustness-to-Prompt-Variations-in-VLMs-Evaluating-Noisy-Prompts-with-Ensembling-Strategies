
import os, json, argparse, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets import (
    OxfordIIITPet,
    Caltech101,
    Food101,
    DTD,
    EuroSAT,
    ImageFolder  # <--- Added ImageFolder here
)
from transformers import AutoProcessor, AutoModel

# ========================
# Tiny linear text adapter (optional)
# ========================
class TextAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=True)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(dim))
            self.fc.bias.zero_()

    def forward(self, x):
        return self.fc(x)

# ========================
# Prompt noise operators
# ========================
def typo(name, p=0.1, rng=None):
    rng = rng or np.random.default_rng(42)
    s = list(name)
    if len(s) < 4:
        return name
    i = int(rng.integers(1, len(s)-1))
    if rng.random() < 0.5:
        s.pop(i)
    else:
        if i < len(s)-1:
            s[i], s[i+1] = s[i+1], s[i]
    return "".join(s)

def random_case(name, p=0.2, rng=None):
    rng = rng or np.random.default_rng(42)
    out = []
    for ch in name:
        if ch.isalpha() and rng.random() < p:
            out.append(ch.upper() if ch.islower() else ch.lower())
        else:
            out.append(ch)
    return "".join(out)

def extra_space(name, rng=None):
    rng = rng or np.random.default_rng(42)
    if len(name) < 3:
        return name
    i = int(rng.integers(1, len(name)))
    return name[:i] + " " + name[i:]

def emoji_tail(name, rng=None):
    rng = rng or np.random.default_rng(42)
    EMOJIS = ["🐾", "✨", "🙂", "🌸"]
    return name + " " + EMOJIS[int(rng.integers(0, len(EMOJIS)))]

PROMPT_NOISES = {
    "typo": typo,
    "case": random_case,
    "space": extra_space,
    "emoji": emoji_tail,
}

def apply_prompt_noise(cls, names, severity, seed=42):
    if severity == 0 or not names:
        return cls
    rng = np.random.default_rng(seed)

    ops = [PROMPT_NOISES[n] for n in names]
    # Pick 'severity' number of operators
    ops = list(rng.choice(ops, size=min(severity, len(ops)), replace=False))

    out = cls
    for f in ops:
        out = f(out, rng=rng)
    return out

# ========================
# Dataset builder (5 datasets)
# ========================
def build_dataset(dataset_name, root, split):
    name = dataset_name.lower()

    if name in ["oxford_pets", "oxfordiiitpet", "pets"]:
        ds = OxfordIIITPet(
            root=root,
            split=split,
            target_types="category",
            transform=None,
            download=False,
        )
    elif name in ["caltech101", "caltech"]:
        ds = Caltech101(
            root=root,
            transform=None,
            download=False,
        )
    elif name in ["food101", "food-101", "food"]:
        ds = Food101(
            root=root,
            split=split,
            transform=None,
            download=False,
        )
    elif name in ["dtd", "textures"]:
        ds = DTD(
            root=root,
            split=split,
            transform=None,
            download=False,
        )
    elif name in ["eurosat", "euro_sat"]:
        # CHANGED: Use ImageFolder instead of EuroSAT class
        # This bypasses strict folder structure checks
        ds = ImageFolder(
            root=root,
            transform=None,
        )
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
    class_names = [idx_to_class[i].replace("_", " ") for i in range(len(idx_to_class))]

    return ds, class_names

# ========================
# Collate function for PIL images
# ========================
def collate_pil(batch):
    imgs, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return list(imgs), labels

# ========================
# Prompt bank with ensembles
# ========================
def build_prompt_bank(class_names, template, prompt_noises, severity, K, include_clean, seed):
    rng = np.random.default_rng(seed)
    banks = []

    for c in class_names:
        variants = []
        if include_clean:
            variants.append(template.replace("{class}", c))  # clean prompt
            need = max(0, K - 1)
        else:
            need = K

        for _ in range(need):
            s_seed = int(rng.integers(0, 1_000_000))
            noisy = apply_prompt_noise(c, prompt_noises, severity, seed=s_seed)
            variants.append(template.replace("{class}", noisy))

        banks.append(variants[:K])

    return banks  # list length C, each is list of K strings

# ========================
# Main evaluation (SigLIP)
# ========================
def evaluate(opts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading SigLIP model: {opts.model_id}")
    processor = AutoProcessor.from_pretrained(opts.model_id)
    model = AutoModel.from_pretrained(opts.model_id).to(device)
    model.eval()

    ds, class_names = build_dataset(
        opts.dataset_name,
        opts.dataset_dir,
        opts.split,
    )
    loader = DataLoader(ds, batch_size=opts.batch_size, shuffle=False, num_workers=2, collate_fn=collate_pil)

    C = len(class_names)
    print(f"Loaded dataset '{opts.dataset_name}' with {C} classes and {len(ds)} images.")

    adapter = None
    if opts.adapter_path:
        print(f" Loading adapter from {opts.adapter_path}")
        ckpt = torch.load(opts.adapter_path, map_location=device)
        # infer text dim if not stored
        dummy_inputs = processor(text=["a"], return_tensors="pt", padding=True).to(device)
        dummy_emb = model.get_text_features(**dummy_inputs)
        text_dim = ckpt.get("text_dim", dummy_emb.shape[-1])

        adapter = TextAdapter(text_dim).to(device).float()
        adapter.load_state_dict(ckpt["state_dict"])
        adapter.eval()
    else:
        print(" Evaluating SigLIP baseline (no adapter).")

    results = []
    severities = [int(s) for s in opts.severity_list.split(",")]

    for sev in severities:
        print(f"Evaluating severity {sev} (K={opts.ensemble_k}, include_clean={opts.include_clean})")

        banks = build_prompt_bank(
            class_names,
            opts.template,
            opts.prompt_noises,
            sev,
            opts.ensemble_k,
            opts.include_clean,
            seed=opts.seed,
        )

        # Flatten prompts => encode => reshape to [C, K, D]
        flat_prompts = [p for bank in banks for p in bank]

        with torch.no_grad():
            text_inputs = processor(
                text=flat_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            text_feats = model.get_text_features(**text_inputs)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            text_feats = text_feats.float()

            if adapter is not None:
                text_feats = adapter(text_feats)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

            text_feats = text_feats.view(C, opts.ensemble_k, -1)

        correct = total = 0
        for imgs, labels in tqdm(loader, desc=f"sev={sev}"):
            labels = labels.to(device)

            with torch.no_grad():
                img_inputs = processor(
                    images=imgs,
                    return_tensors="pt"
                ).to(device)

                img_feats = model.get_image_features(**img_inputs)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                img_feats = img_feats.float()

                # Average probabilities over K prompt variants
                probs_list = []
                for k in range(opts.ensemble_k):
                    tf = text_feats[:, k, :]            # [C, D]
                    logits = img_feats @ tf.t()         # [B, C]
                    probs_list.append(torch.softmax(logits, dim=-1))

                probs = torch.stack(probs_list, dim=0).mean(dim=0)
                preds = probs.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = 100.0 * correct / total
        results.append({"severity": sev, "accuracy": acc})
        print(f"Severity {sev}: {acc:.2f}%")

    if opts.save_json:
        os.makedirs(os.path.dirname(opts.save_json), exist_ok=True)
        with open(opts.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f" Saved results to {opts.save_json}")

# ========================
# CLI
# ========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, default="oxford_pets",
                   help="one of: oxford_pets, caltech101, food101, dtd, eurosat")
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--model_id", type=str, default="google/siglip-so400m-patch14-384",
                   help="HuggingFace model id for SigLIP")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--severity_list", type=str, default="0,1,2,3")
    p.add_argument("--prompt_noises", type=str, default="typo,case,emoji,space")
    p.add_argument("--template", type=str, default="a photo of a {class}")
    p.add_argument("--ensemble_k", type=int, default=1)
    p.add_argument("--include_clean", type=lambda s: s.lower() in ["1","true","yes","y"], default=True)
    p.add_argument("--adapter_path", type=str, default="")
    p.add_argument("--save_json", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    args.prompt_noises = [s for s in args.prompt_noises.split(",") if s.strip() != ""]
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
