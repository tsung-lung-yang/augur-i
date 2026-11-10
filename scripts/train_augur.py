"""AUGUR-I training entry point (T0 and, later, the foresight conditions).

Runs LeRobot's lerobot-train unchanged except for three hooks:
  1. LoRA on the VLM text layers (attention projections) of SmolVLA. The vision
     encoder stays frozen; the action expert trains as in the stock recipe.
     Checkpoints are saved with LoRA merged into the base weights, so a saved
     model is a plain SmolVLA that lerobot-eval / scripts/eval_mixed.py can load.
  2. Base augmentation from LOG-003 section 5, applied on GPU to every training
     batch after LeRobot's preprocessor: random shift (replicate-pad then crop),
     colour jitter (brightness, contrast, saturation, hue, in that fixed order),
     each camera sampled independently; state dropout sets the normalised state
     vector of a sample to 0 (the dataset mean) with probability p.
  3. Nothing else: data, optimiser, LR schedule and checkpointing are LeRobot's.

Environment variables (defaults are the registered settings):
  AUGUR_AUG          base | strong | none      (LOG-003 sections 5 and 6)
  AUGUR_LORA_R       LoRA rank, default 16
  AUGUR_LORA_ALPHA   LoRA alpha, default 2 * rank
  AUGUR_LORA_TARGETS comma list, default q_proj,k_proj,v_proj,o_proj
All other arguments are lerobot-train arguments, e.g.
  python scripts/train_augur.py --policy.type=smolvla --policy.load_vlm_weights=true \
    --policy.push_to_hub=false --policy.use_amp=true --dataset.repo_id=lerobot/libero \
    --dataset.video_backend=torchcodec --batch_size=32 --steps=60000 --save_freq=5000 \
    --log_freq=100 --num_workers=8 --wandb.enable=false --seed=0 --output_dir=runs/train/t0_curve_s0
Resuming from a checkpoint is not supported (saved weights are merged).
"""
import json
import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF

AUG_PRESETS = {
    "base": dict(pad=12, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, state_p=0.3),
    "strong": dict(pad=24, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, state_p=0.5),
    "none": None,
}
AUG_NAME = os.environ.get("AUGUR_AUG", "base")
AUG = AUG_PRESETS[AUG_NAME]
LORA_R = int(os.environ.get("AUGUR_LORA_R", "16"))
LORA_ALPHA = float(os.environ.get("AUGUR_LORA_ALPHA", str(2 * LORA_R)))
LORA_TARGETS = os.environ.get("AUGUR_LORA_TARGETS", "q_proj,k_proj,v_proj,o_proj").split(",")


# ----------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Linear):
    """nn.Linear that keeps the original `weight`/`bias` names and adds a low-rank update."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight = base.weight
        self.bias = base.bias
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return out + delta * self.scale

    @torch.no_grad()
    def merged_weight(self):
        w = self.weight.float() + self.scale * (self.lora_B @ self.lora_A)
        return w.to(self.weight.dtype).contiguous()


def inject_lora(policy) -> int:
    text = policy.model.vlm_with_expert.get_vlm_model().text_model
    n = 0
    for layer in text.layers:
        attn = layer.self_attn
        for name in LORA_TARGETS:
            base = getattr(attn, name)
            if isinstance(base, LoRALinear):
                continue
            setattr(attn, name, LoRALinear(base, LORA_R, LORA_ALPHA))
            n += 1
    return n


def _save_merged(self, save_directory, state_dict=None):
    """Save a plain SmolVLA state dict: LoRA merged into weight, lora_* tensors dropped."""
    sd = {k: v for k, v in self.state_dict().items() if ".lora_A" not in k and ".lora_B" not in k}
    for name, m in self.named_modules():
        if isinstance(m, LoRALinear):
            sd[f"{name}.weight"] = m.merged_weight()
    sd = {k: v.detach().contiguous() for k, v in sd.items()}
    type(self)._save_pretrained(self, save_directory, state_dict=sd)
    with open(os.path.join(str(save_directory), "augur_train.json"), "w") as f:
        json.dump(
            {"aug": AUG_NAME, "aug_params": AUG, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
             "lora_targets": LORA_TARGETS, "lora_merged": True},
            f, indent=1,
        )


# ----------------------------------------------------------------------------- augmentation
def _shift(x: torch.Tensor, pad: int) -> torch.Tensor:
    n, _, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    oy = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    ox = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    rows = oy[:, None] + torch.arange(h, device=x.device)[None, :]
    cols = ox[:, None] + torch.arange(w, device=x.device)[None, :]
    idx = torch.arange(n, device=x.device)[:, None, None]
    out = xp.permute(0, 2, 3, 1)[idx, rows[:, :, None], cols[:, None, :]]
    return out.permute(0, 3, 1, 2).contiguous()


def _uniform(n, mag, device):
    return 1.0 + (torch.rand(n, device=device) * 2 - 1) * mag


def _gray(x):
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)


def _jitter(x: torch.Tensor, a: dict) -> torch.Tensor:
    n = x.shape[0]
    f = _uniform(n, a["brightness"], x.device).view(n, 1, 1, 1)
    x = (x * f).clamp(0, 1)
    f = _uniform(n, a["contrast"], x.device).view(n, 1, 1, 1)
    m = _gray(x).mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - m) * f + m).clamp(0, 1)
    f = _uniform(n, a["saturation"], x.device).view(n, 1, 1, 1)
    g = _gray(x)
    x = ((x - g) * f + g).clamp(0, 1)
    hs = (torch.rand(n) * 2 - 1) * a["hue"]
    x = torch.stack([TF.adjust_hue(x[i], float(hs[i])) for i in range(n)])
    return x


def augment_batch(batch: dict) -> dict:
    if AUG is None:
        return batch
    for k in list(batch.keys()):
        if not k.startswith("observation.images.") or k.endswith("_padding_mask"):
            continue
        v = batch[k]
        if not torch.is_tensor(v) or v.ndim not in (4, 5):
            continue
        shape = v.shape
        x = v.reshape(-1, *shape[-3:]).float()
        x = _jitter(_shift(x, AUG["pad"]), AUG)
        batch[k] = x.reshape(shape).to(v.dtype)
    s = batch.get("observation.state")
    if torch.is_tensor(s) and AUG["state_p"] > 0:
        drop = torch.rand(s.shape[0], device=s.device) < AUG["state_p"]
        s = s.clone()
        s[drop] = 0
        batch["observation.state"] = s
    return batch


# ----------------------------------------------------------------------------- hooks
import lerobot.scripts.lerobot_train as T  # noqa: E402

_make_policy = T.make_policy
_update_policy = T.update_policy


def make_policy_with_lora(*args, **kwargs):
    policy = _make_policy(*args, **kwargs)
    n = inject_lora(policy)
    policy._save_pretrained = _save_merged.__get__(policy)
    n_lora = sum(p.numel() for name, p in policy.named_parameters() if "lora_" in name)
    logging.info(f"[augur] LoRA r={LORA_R} alpha={LORA_ALPHA} on {n} VLM projections ({n_lora:,} params); aug={AUG_NAME}")
    print(f"[augur] LoRA r={LORA_R} alpha={LORA_ALPHA} on {n} VLM projections ({n_lora:,} params); aug={AUG_NAME}", flush=True)
    return policy


def update_policy_with_aug(train_metrics, policy, batch, *args, **kwargs):
    return _update_policy(train_metrics, policy, augment_batch(batch), *args, **kwargs)


T.make_policy = make_policy_with_lora

# LeRobot's --policy.use_amp uses torch's default CUDA autocast dtype (float16). The registered
# precision is bfloat16 (LOG-002, LOG-004), so set the default before LeRobot reads it.
import torch as _torch  # noqa: E402

_AMP = {"bf16": _torch.bfloat16, "fp16": _torch.float16}[os.environ.get("AUGUR_AMP_DTYPE", "bf16")]
if _torch.cuda.is_available():
    _torch.set_autocast_dtype("cuda", _AMP)

T.update_policy = update_policy_with_aug

if __name__ == "__main__":
    T.main()
