"""LOG-007: the cached TT training path equals the original two-pass path. CPU only.

  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= python scripts/test_tt_cached.py
Builds a randomly initialised SmolVLA with LoRA and a TT foresight module, and computes the
training loss and all gradients twice on the same batch, noise and time: once with the
original implementation (prefix-only pass + full prefix/suffix pass) and once through the KV
cache (one prefix pass). The two must agree up to floating-point rounding; as a reference
for the size of rounding, the original path is also compared with itself after a harmless
change of summation order (a different batch order).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(0)
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

import augur_foresight as AF  # noqa: E402
import train_augur as TA  # noqa: E402

cfg = SmolVLAConfig(load_vlm_weights=False, device="cpu")
cfg.input_features = {
    "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
}
cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
pol = SmolVLAPolicy(cfg)
TA.inject_lora(pol)
fs = AF.install(pol, "t5", "tt", k=4)
with torch.no_grad():  # non-trivial LoRA and foresight weights, as after training
    for n, p in pol.named_parameters():
        if ".lora_B" in n or n.startswith("model.augur."):
            p.add_(torch.randn_like(p) * 0.02)
pol.train()
print("policy built", flush=True)

B = 3
g = torch.Generator().manual_seed(1)
batch = {
    "observation.images.image": torch.rand(B, 3, 256, 256, generator=g),
    "observation.images.image2": torch.rand(B, 3, 256, 256, generator=g),
    "observation.state": torch.randn(B, 8, generator=g),
    "observation.language.tokens": torch.randint(0, 1000, (B, 12), generator=g),
    "observation.language.attention_mask": torch.tensor([[1] * 12, [1] * 9 + [0] * 3, [1] * 7 + [0] * 5], dtype=torch.bool),
    "action": torch.randn(B, 50, 7, generator=g),
    "action_is_pad": torch.zeros(B, 50, dtype=torch.bool),
}
noise = torch.randn(B, 50, 32, generator=g)
tm = torch.rand(B, generator=g) * 0.9 + 0.05
params = [(n, p) for n, p in pol.named_parameters() if p.requires_grad]


def run(cached, perm=None):
    AF.TT_CACHED_TRAINING = cached
    bt, nz, t = batch, noise, tm
    if perm is not None:
        bt = {k: v[perm] for k, v in batch.items()}
        nz, t = noise[perm], tm[perm]
    pol.zero_grad(set_to_none=True)
    loss, _ = pol.forward(bt, noise=nz.clone(), time=t.clone())
    loss.backward()
    grads = {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in params}
    return float(loss), grads, fs.last_pred.detach().clone()


def rel(a, b):
    num = sum(float((a[n] - b[n]).float().pow(2).sum()) for n in a)
    den = sum(float(b[n].float().pow(2).sum()) for n in a) + 1e-30
    return (num / den) ** 0.5


l_old, g_old, z_old = run(False)
l_new, g_new, z_new = run(True)
l_prm, g_prm, _ = run(False, perm=torch.tensor([2, 0, 1]))
ok = True


def check(name, cond, info=""):
    global ok
    print(f"  [{'ok' if cond else 'FAIL'}] {name} {info}", flush=True)
    ok &= bool(cond)


floor = rel(g_prm, g_old)
err = rel(g_new, g_old)
check("loss equal", abs(l_new - l_old) <= 1e-3 * abs(l_old) + 1e-6, f"old {l_old:.6f} new {l_new:.6f}")
check("foresight prediction equal", float((z_new - z_old).abs().max()) < 1e-3, f"max diff {float((z_new - z_old).abs().max()):.2e}")
check("all gradients equal (relative L2)", err <= max(10 * floor, 1e-3), f"{err:.2e} (rounding reference {floor:.2e})")
groups = {"LoRA": ".lora_", "queries/head/cond": "model.augur.", "action expert": "lm_expert"}
for gname, key in groups.items():
    sub_old = {n: g_old[n] for n in g_old if key in n}
    sub_new = {n: g_new[n] for n in g_new if key in n}
    if sub_old:
        e = rel(sub_new, sub_old)
        nz = sum(float(v.abs().sum()) for v in sub_old.values()) > 0
        check(f"{gname} gradients equal and non-zero", nz and e <= max(10 * floor, 1e-3), f"{e:.2e}")
print("\nALL OK" if ok else "\nSOME TESTS FAILED", flush=True)
sys.exit(0 if ok else 1)
