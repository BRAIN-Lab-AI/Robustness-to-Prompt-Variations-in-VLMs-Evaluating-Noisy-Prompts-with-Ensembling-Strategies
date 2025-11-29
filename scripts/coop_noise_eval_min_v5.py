# ===================== PATH FIX (must be first) =====================
from pathlib import Path
import sys, os

_here = Path(__file__).resolve()         # .../CoOp/robust_eval/coop_noise_eval_v5.py
_repo = _here.parents[1]                 # .../CoOp
_dassl = _repo / "Dassl.pytorch"         # .../CoOp/Dassl.pytorch

for _p in (str(_repo), str(_dassl)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("PYTHONPATH", f"{_repo}:{_dassl}")
# ===================================================================

import json
import argparse
import numpy as np

try:
    from dassl.config import get_cfg_default
    from dassl.engine import build_trainer
except ModuleNotFoundError as e:
    raise RuntimeError(
        "Could not import 'dassl'.\\n"
        f"Checked paths:\\n  repo={_repo}\\n  dassl={_dassl}\\n"
        "First 5 sys.path entries:\\n  " + "\\n  ".join(sys.path[:5])
    ) from e

import trainers.coop  # register CoOp trainer
#  Register CoOp datasets in Dassl
from datasets.oxford_pets import OxfordPets
from datasets.caltech101 import Caltech101
from datasets.food101 import Food101
from datasets.dtd import DTD
from datasets.eurosat import EuroSAT



def typo(name, p=0.1, rng=None):
    rng = rng or np.random.default_rng(42)
    s = list(name)
    if len(s) < 4:
        return name
    i = int(rng.integers(1, len(s) - 1))
    if rng.random() < 0.5:
        s.pop(i)
    else:
        if i < len(s) - 1:
            s[i], s[i + 1] = s[i + 1], s[i]
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


def extra_space(name, severity=1, rng=None):
    rng = rng or np.random.default_rng(42)
    s = name
    k = max(1, int(severity))
    for _ in range(k):
        if len(s) < 3:
            break
        i = int(rng.integers(1, len(s)))
        s = s[:i] + " " + s[i:]
    return s


def emoji_tail(name, severity=1, rng=None):
    rng = rng or np.random.default_rng(42)
    EMOJIS = ["🐾", "✨", "⭐", "🙂", "🔥"]
    k = max(1, int(severity))
    return name + " " + "".join(EMOJIS[int(rng.integers(0, len(EMOJIS)))] for _ in range(k))


NOISES = {
    "typo": typo,
    "case": random_case,
    "space": extra_space,
    "emoji": emoji_tail,
}


def build_cfg(args):
    cfg = get_cfg_default()
    from yacs.config import CfgNode as CN

    if not hasattr(cfg.TRAINER, "COOP"):
        cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16
    cfg.TRAINER.COOP.CTX_INIT = ""
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.COOP.CSC = False
    cfg.TRAINER.COOP.PREC = "fp32"

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.DATASET.NUM_SHOTS = -1

    cfg.SEED = 1
    cfg.OUTPUT_DIR = "output/tmp_eval"

    cfg.DATASET.NAME = args.dataset_name
    cfg.DATASET.ROOT = args.dataset_dir
    cfg.TRAINER.NAME = "CoOp"
    cfg.MODEL.BACKBONE.NAME = args.backbone

    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.batch_size
    cfg.TEST.EVALUATOR = "Classification"
    cfg.INPUT.PROMPT_TEMPLATE = "a photo of a {}"

    return cfg


def make_noised_classnames(names, noise_kind, severity, rng):
    fn = NOISES[noise_kind]
    out = []
    for n in names:
        if noise_kind in ("typo", "case"):
            p = 0.05 * severity
            out.append(fn(n, p=p, rng=rng))
        else:
            out.append(fn(n, severity=severity, rng=rng))
    return out


def run(args):
    import clip
    cfg = build_cfg(args)

    def eval_with_noised_names(noised_classnames):
        import torch, clip

        trainer = build_trainer(cfg)
        trainer.load_model(args.coop_ckpt)

        pl = getattr(trainer.model, "prompt_learner", None)
        if pl is None:
            raise RuntimeError("PromptLearner not found on trainer.model (unexpected for CoOp).")

        device = next(trainer.model.parameters()).device

        if hasattr(pl, "classnames"):
            pl.classnames = list(noised_classnames)

        template = getattr(cfg.INPUT, "PROMPT_TEMPLATE", "a photo of a {}")
        texts = [template.format(c) for c in noised_classnames]
        tok = clip.tokenize(texts).to(device, non_blocking=True)

        if hasattr(pl, "tokenized_prompts"):
            pl.tokenized_prompts = tok
        else:
            setattr(pl, "tokenized_prompts", tok)

        for name, buf in pl.named_buffers(recurse=True):
            if "token" in name and isinstance(buf, torch.Tensor) and buf.device != device:
                try:
                    setattr(pl, name, buf.to(device, non_blocking=True))
                except Exception:
                    pass

        metrics = trainer.test()
        if isinstance(metrics, (int, float)):
            return float(metrics)
        return float(metrics.get("accuracy", metrics.get("top1", metrics.get("acc", 0.0))))

    base = build_trainer(cfg)
    base.load_model(args.coop_ckpt)
    clean_classnames = list(base.dm.dataset.classnames)

    results = {
        "meta": {
            "dataset": args.dataset_name,
            "backbone": args.backbone,
            "k": "1",
            "noises": args.prompt_noises.split(","),
            "severities": args.severity_list,
        },
        "curves": {},
    }

    for nk in args.prompt_noises.split(","):
        nk = nk.strip()
        if not nk:
            continue
        accs = []
        for sev in args.severity_list:
            rng = np.random.default_rng(42 + sev)
            noised = make_noised_classnames(clean_classnames, nk, sev, rng)
            acc = eval_with_noised_names(noised)
            accs.append(acc)
            print(f"[{args.dataset_name}][{nk}] severity {sev}: {acc:.2f}%")
        results["curves"][nk] = accs

    out_dir = Path("output/robust_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"coop_{args.dataset_name}_k1_v5.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", required=True,
                    help="CoOp dataset name, e.g. OxfordPets, Caltech101, Food101, DTD, EuroSAT")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--backbone", default="ViT-B/16")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--severity_list", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--prompt_noises", default="typo,case,emoji,space")
    ap.add_argument("--ensemble_k", type=int, default=1)
    ap.add_argument("--include_clean", action="store_true")
    ap.add_argument("--coop_ckpt", required=True)
    args = ap.parse_args()
    run(args)
