
import os, json, math, argparse, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets import OxfordIIITPet
import clip

# ----- prompt noise (same ops you used) -----
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
    return ''.join(s)

def random_case(name, p=0.2, rng=None):
    rng = rng or np.random.default_rng(42)
    out = []
    for ch in name:
        if ch.isalpha() and rng.random() < p:
            out.append(ch.upper() if ch.islower() else ch.lower())
        else:
            out.append(ch)
    return ''.join(out)

def extra_space(name, rng=None):
    rng = rng or np.random.default_rng(42)
    if len(name) < 3:
        return name
    i = int(rng.integers(1, len(name)))
    return name[:i] + " " + name[i:]

def emoji_tail(name, rng=None):
    rng = rng or np.random.default_rng(42)
    EMOJIS = ['🐾','✨','🙂','🌸']
    return name + " " + EMOJIS[int(rng.integers(0, len(EMOJIS)))]

PROMPT_NOISES = {
    "typo": typo,
    "case": random_case,
    "space": extra_space,
    "emoji": emoji_tail
}

def apply_prompt_noise(cls, names, severity, seed=42):
    if severity == 0 or not names:
        return cls
    rng = np.random.default_rng(seed)
    ops = [PROMPT_NOISES[n] for n in names]
    ops = list(rng.choice(ops, size=min(severity, len(ops)), replace=False))
    out = cls
    for f in ops:
        out = f(out, rng=rng)
    return out

# ----- simple identity-initialized adapter W,b on text features -----
class TextAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=True)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(dim))
            self.fc.bias.zero_()
    def forward(self, x):  # x: [..., dim]
        return self.fc(x)

def encode_texts(model, device, prompts):
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

def train(opts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(opts.backbone, device=device)
    model = model.float()
    for p in model.parameters():
        p.requires_grad_(False)  # keep CLIP frozen

    # Data
    split = "trainval" if opts.use_trainval else "train"
    ds = OxfordIIITPet(
        root=opts.dataset_dir,
        split=split,
        target_types="category",
        transform=preprocess,
        download=False
    )
    loader = DataLoader(ds, batch_size=opts.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
    class_names = [idx_to_class[i].replace("_", " ") for i in range(len(idx_to_class))]
    C = len(class_names)

    # Adapter
    text_dim = model.encode_text(clip.tokenize("a").to(device)).shape[-1]
    adapter = TextAdapter(text_dim).to(device)

    # Optim
    opt = torch.optim.AdamW(adapter.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=opts.epochs * len(loader))

    # helper to build banks each epoch (random seeds)
    def build_banks(severity, k, seed):
        rng = np.random.default_rng(seed)
        banks = []
        for c in class_names:
            variants = []
            if opts.include_clean:
                variants.append(opts.template.replace("{class}", c))  # 1 clean
                need = k - 1
            else:
                need = k
            for _ in range(need):
                s_seed = int(rng.integers(0, 1_000_000))
                noisy = apply_prompt_noise(c, opts.prompt_noises, severity, seed=s_seed)
                variants.append(opts.template.replace("{class}", noisy))
            banks.append(variants)

        flat = [p for bank in banks for p in bank]
        # encode with CLIP only (no adapter, no grad)
        base = encode_texts(model, device, flat).view(C, k, -1).detach()  # [C,K,D], no grad

        if opts.include_clean:
            clean_base = base[:, :1, :].squeeze(1)   # [C,D]
            noisy_base = base[:, 1:, :]              # [C,K-1,D]
        else:
            clean_base = None
            noisy_base = base                        # [C,K,D]
        return clean_base, noisy_base

    global_step = 0
    for ep in range(opts.epochs):
        adapter.train()
        ce_meter, kl_meter = 0.0, 0.0
        clean_base, noisy_base = build_banks(opts.train_severity, opts.k, seed=opts.seed + ep)

        for imgs, labels in tqdm(loader, desc=f"epoch {ep+1}/{opts.epochs}"):
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                img_feats = model.encode_image(imgs).float()
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            # Apply the learnable adapter per batch -> fresh graph each iteration
            noisy_feats = adapter(noisy_base)                    # [C,K',D]
            noisy_feats = noisy_feats / noisy_feats.norm(dim=-1, keepdim=True)

            logits_list = []
            for k in range(noisy_feats.shape[1]):
                tf = noisy_feats[:, k, :]                        # [C,D]
                logits = 100.0 * img_feats @ tf.t()
                logits_list.append(logits)
            logits_avg = torch.stack(logits_list, dim=0).mean(dim=0)

            ce = F.cross_entropy(logits_avg, labels)

            if clean_base is not None:
                clean_feats = adapter(clean_base)                # [C,D]
                clean_feats = clean_feats / clean_feats.norm(dim=-1, keepdim=True)
                logits_clean = 100.0 * img_feats @ clean_feats.t()
                p_clean = F.log_softmax(logits_clean, dim=-1)
                p_noisy = F.softmax(logits_avg, dim=-1)
                kl = F.kl_div(p_clean, p_noisy, reduction="batchmean")
            else:
                kl = torch.tensor(0.0, device=img_feats.device)

            loss = ce + opts.lambda_consistency * kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            sched.step()

            ce_meter += ce.item()
            kl_meter += kl.item()
            global_step += 1

        print(f"[epoch {ep+1}] CE={ce_meter/len(loader):.4f}  KL={kl_meter/len(loader):.4f}")

        # save checkpoint each epoch
        os.makedirs(opts.output_dir, exist_ok=True)
        ckpt = {
            "state_dict": adapter.state_dict(),
            "text_dim": text_dim,
            "backbone": opts.backbone,
            "template": opts.template,
            "prompt_noises": opts.prompt_noises,
            "include_clean": opts.include_clean,
            "k": opts.k,
            "train_severity": opts.train_severity,
            "lambda_consistency": opts.lambda_consistency,
            "epochs_done": ep + 1,
        }
        torch.save(ckpt, os.path.join(opts.output_dir, f"adapter_ep{ep+1}.pth"))

    # final
    torch.save(ckpt, os.path.join(opts.output_dir, "adapter_last.pth"))
    print("Saved:", os.path.join(opts.output_dir, "adapter_last.pth"))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--backbone", type=str, default="ViT-B/16")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=5e-2)
    p.add_argument("--template", type=str, default="a photo of a {class}")
    p.add_argument("--prompt_noises", type=str, default="typo,case,emoji,space")
    p.add_argument("--k", type=int, default=5, help="no. of prompts per class during training")
    p.add_argument("--include_clean", type=lambda s: s.lower() in ["1","true","yes","y"], default=True)
    p.add_argument("--train_severity", type=int, default=2)
    p.add_argument("--lambda_consistency", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_trainval", action="store_true")
    p.add_argument("--output_dir", type=str, default="output/robust_train/text_adapter")
    return p.parse_args()

if __name__ == "__main__":
    opts = parse_args()
    opts.prompt_noises = [s for s in opts.prompt_noises.split(",") if s.strip() != ""]
    train(opts)
