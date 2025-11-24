import clip, torch, os, numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import datasets, transforms

# ---------- CONFIG ----------
DATASET = "/content/drive/MyDrive/DL_CoOp_Robustness10/data/oxford_pets"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "ViT-B/16"
SEVERITY = 2  # choose 0,1,2,3 later
# ----------------------------

# Noise functions
def typo(name, p=0.1, rng=None):
    rng = rng or np.random.default_rng(42)
    s = list(name)
    if len(s) < 4: return name
    i = int(rng.integers(1, len(s)-1))
    if rng.random() < 0.5:
        s.pop(i)
    else:
        if i < len(s)-1: s[i], s[i+1] = s[i+1], s[i]
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

def emoji_tail(name, rng=None):
    rng = rng or np.random.default_rng(42)
    EMOJIS = ['🐾','✨','🙂','🌸']
    return name + ' ' + EMOJIS[int(rng.integers(0, len(EMOJIS)))]

NOISES = [typo, random_case, emoji_tail]

def build_noisy_label(name, severity=1, seed=42):
    rng = np.random.default_rng(seed)
    if severity == 0: return name
    funcs = rng.choice(NOISES, size=severity, replace=False)
    out = name
    for f in funcs:
        out = f(out, rng=rng)
    return out

# Load model
print("Loading CLIP model...")
model, preprocess = clip.load(BACKBONE, device=DEVICE)

# Load dataset
from torchvision.datasets import OxfordIIITPet

# Load Oxford-IIIT Pets via Torchvision (uses annotations for labels)
dataset = OxfordIIITPet(
    root=DATASET,            # e.g., ".../data/oxford_pets"
    split="test",            # or "trainval" / "train" / "val" depending on what you want
    target_types="category",
    transform=preprocess,
    download=False           # we already downloaded it
)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

# Class names (sorted by index)
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
class_names = [idx_to_class[i].replace("_", " ") for i in range(len(idx_to_class))]

prompts = [f"a photo of a {build_noisy_label(c, severity=SEVERITY)}" for c in class_names]
text_tokens = clip.tokenize(prompts).to(DEVICE)

# Encode text
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Evaluate
print(f"Evaluating with severity {SEVERITY}")
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in tqdm(loader):
        imgs = imgs.to(DEVICE)
        image_features = model.encode_image(imgs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features.T
        preds = logits.argmax(dim=-1)
        correct += (preds == labels.to(DEVICE)).sum().item()
        total += labels.size(0)

acc = correct / total * 100
print(f"\nAccuracy at severity {SEVERITY}: {acc:.2f}%")
