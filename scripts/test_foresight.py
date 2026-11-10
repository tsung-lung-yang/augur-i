"""Tests for scripts/augur_foresight.py on a randomly initialised SmolVLA (CPU, ~2-4 min).

Run in the `augur` env:  HF_HUB_OFFLINE=1 python scripts/test_foresight.py
Checks
  1. TR: with the same weights, action loss and sampled actions equal a model without
     foresight tokens; changing the queries changes the prediction but not the actions.
  2. TT: actions depend on the imagined future (changing the queries changes the action
     loss); gradients reach queries, head and the conditioning projection.
  3. t3 target: shape, EMA swap restores the online LoRA weights, EMA changes the target.
  4. quadrant pooling layout, masked foresight loss, foresight error.
"""
import copy
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
import train_augur as TA  # noqa: E402  (LoRALinear / inject_lora)

cfg = SmolVLAConfig(load_vlm_weights=False, device="cpu")
cfg.input_features = {
    "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
}
cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
base = SmolVLAPolicy(cfg).eval()
n_lora = TA.inject_lora(base)
with torch.no_grad():  # make LoRA non-trivial so the EMA test means something
    for n, p in base.named_parameters():
        if ".lora_B" in n:
            p.normal_(0, 0.01)
print("policy built, LoRA projections:", n_lora, flush=True)

B = 2
g = torch.Generator().manual_seed(1)
images = [torch.rand(B, 3, 512, 512, generator=g) * 2 - 1 for _ in range(2)]
img_masks = [torch.ones(B, dtype=torch.bool) for _ in range(2)]
lang_tokens = torch.randint(0, 1000, (B, 12), generator=g)
lang_masks = torch.ones(B, 12, dtype=torch.bool)
lang_masks[1, 9:] = False
state = torch.randn(B, 32, generator=g)
actions = torch.randn(B, 50, 32, generator=g)
noise = torch.randn(B, 50, 32, generator=g)
time = torch.rand(B, generator=g) * 0.9 + 0.05
args = (images, img_masks, lang_tokens, lang_masks, state)

tt_policy = copy.deepcopy(base)
m = base.model
with torch.no_grad():
    ref_loss = m.forward(*args, actions, noise=noise.clone(), time=time.clone())
    ref_act = m.sample_actions(*args, noise=noise.clone())
    # numerical-noise control: 4 extra *masked* language tokens are a logical no-op that also
    # lengthens the sequence by 4, like the foresight queries do
    pad_tokens = torch.cat([lang_tokens, torch.zeros(B, 4, dtype=lang_tokens.dtype)], 1)
    pad_masks = torch.cat([lang_masks, torch.zeros(B, 4, dtype=torch.bool)], 1)
    pargs = (images, img_masks, pad_tokens, pad_masks, state)
    ctl_loss = m.forward(*pargs, actions, noise=noise.clone(), time=time.clone())
    ctl_act = m.sample_actions(*pargs, noise=noise.clone())
floor_loss = (ctl_loss - ref_loss).abs().max().item()
floor_act = (ctl_act - ref_act).abs().max().item()
print("reference loss", float(ref_loss.mean()), flush=True)
print(f"numerical floor (masked padding, +4 tokens): max|d loss| {floor_loss:.2e}, max|d action| {floor_act:.2e}", flush=True)

# ------------------------------------------------------------------ 1. TR invariance
fs = AF.install(base, "t3", "tr")
with torch.no_grad():
    tr_loss = m.forward(*args, actions, noise=noise.clone(), time=time.clone())
    pred1 = fs.last_pred.clone()
    tr_act = m.sample_actions(*args, noise=noise.clone())
    fs.queries.add_(torch.randn_like(fs.queries))
    tr_loss2 = m.forward(*args, actions, noise=noise.clone(), time=time.clone())
    pred2 = fs.last_pred.clone()
    tr_act2 = m.sample_actions(*args, noise=noise.clone())
d_loss = (tr_loss - ref_loss).abs().max().item()
d_act = (tr_act - ref_act).abs().max().item()
q_loss = (tr_loss2 - tr_loss).abs().max().item()
q_act = (tr_act2 - tr_act).abs().max().item()
print(f"[1] TR vs no-foresight: max|d loss| {d_loss:.2e}, max|d action| {d_act:.2e}")
print(f"[1] TR, queries changed (same length): max|d loss| {q_loss:.2e}, max|d action| {q_act:.2e}")
assert q_loss < 1e-6 and q_act < 1e-6, "queries leak into actions"
assert d_loss <= max(10 * floor_loss, 1e-4) and d_act <= max(10 * floor_act, 1e-4), \
    "TR differs from the no-foresight model by more than the numerical floor"
assert (pred2 - pred1).abs().max() > 1e-3, "queries do not affect the prediction"
assert pred1.shape == (B, 4, fs.hidden), pred1.shape
print("[1] ok: TR actions equal the no-foresight model; queries only change the prediction", flush=True)

# ------------------------------------------------------------------ 2. TT dependence + gradients
ft = AF.install(tt_policy, "t3", "tt")
mt = tt_policy.model
with torch.no_grad():
    ft.cond_proj.weight.normal_(0, 0.05)  # cond_proj is fresh; make the conditioning non-trivial
tt_loss = mt.forward(*args, actions, noise=noise.clone(), time=time.clone())
target = torch.randn(B, 4, ft.hidden)
total = tt_loss.mean() + 0.1 * AF.foresight_loss(ft, target)
total.backward()
for name, p in [("queries", ft.queries), ("head", ft.head[0].weight), ("cond_proj", ft.cond_proj.weight)]:
    gn = 0.0 if p.grad is None else p.grad.abs().sum().item()
    print(f"[2] grad {name}: {gn:.3e}")
    assert gn > 0, f"no gradient to {name}"
lora_grad = sum(p.grad.abs().sum().item() for n, p in tt_policy.named_parameters() if ".lora_" in n and p.grad is not None)
print(f"[2] grad LoRA (sum): {lora_grad:.3e}")
assert lora_grad > 0
with torch.no_grad():
    l_a = mt.forward(*args, actions, noise=noise.clone(), time=time.clone())
    ft.queries.add_(torch.randn_like(ft.queries))
    l_b = mt.forward(*args, actions, noise=noise.clone(), time=time.clone())
    act = mt.sample_actions(*args, noise=noise.clone())
dd = (l_b - l_a).abs().max().item()
print(f"[2] TT: changing the queries changes the action loss by {dd:.2e}")
assert dd > 1e-5, "TT actions do not depend on the imagined future"
assert act.shape == (B, 50, 32)
print("[2] ok", flush=True)

# ------------------------------------------------------------------ 3. t3 target + EMA
fut = torch.randn(B, 64, ft.hidden) * 0.05
online = {n: p.detach().clone() for n, p in ft.lora_params(mt).items()}
ft.ema_init(mt)
t_online = AF.t3_target(tt_policy, fut)
for n in ft.ema:
    ft.ema[n] = ft.ema[n] + torch.randn_like(ft.ema[n]) * 0.05
t_ema = AF.t3_target(tt_policy, fut)
after = {n: p.detach() for n, p in ft.lora_params(mt).items()}
assert all(torch.equal(online[n], after[n]) for n in online), "EMA swap did not restore LoRA"
assert t_ema.shape == (B, 4, ft.hidden)
print(f"[3] t3 target {tuple(t_ema.shape)}, EMA changes target by {(t_ema - t_online).abs().max().item():.3e}")
assert (t_ema - t_online).abs().max() > 1e-4
ft.ema_update(mt)
print("[3] ok", flush=True)

# ------------------------------------------------------------------ 4. helpers
grid = torch.arange(64.0).view(1, 64, 1)
q = AF.quadrant_pool(grid)[0, :, 0].tolist()
assert q == [13.5, 17.5, 45.5, 49.5], q
ft.last_pred = torch.zeros(3, 4, 2)
loss = AF.foresight_loss(ft, torch.ones(3, 4, 2), valid=torch.tensor([1.0, 0.0, 1.0]))
assert abs(loss.item() - 1.0) < 1e-6
err = AF.foresight_error(torch.ones(2, 4, 3), torch.ones(2, 4, 3))
assert err.abs().max() < 1e-6
print("[4] ok")
print("ALL OK")
