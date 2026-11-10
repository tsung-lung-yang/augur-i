"""Tests for the fine-tuning and evaluation integration of foresight (LOG-005), CPU only.

Run in the `augur` env (does not touch the GPU, safe while training runs):
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= python scripts/test_finetune_foresight.py
Uses a randomly initialised SmolVLA and a small fake target cache in a temp directory.
Checks, for T0 (no foresight), t3-TR and t5-TT:
  1. setup: pretraining LoRA / queries / head / EMA are loaded exactly; TT conditioning is new.
  2. loss: the training loss equals action loss + lambda * foresight loss; invalid rows are
     masked; gradients reach LoRA, queries, head (and the conditioning projection for TT).
  3. save: model.safetensors holds no foresight or LoRA tensors; augur.safetensors,
     augur_lora.safetensors and augur_train.json are written (T0 writes target none only).
  4. eval load: a fresh policy from the checkpoint plus load_for_eval reproduces the
     in-memory policy's sampled actions; T0 checkpoints are left untouched.
  5. the preprocessor wrapper keeps the row index; eval_mixed patches make_policy.
"""
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(0)
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

import augur_foresight as AF  # noqa: E402
import train_augur as TA  # noqa: E402

tmp = tempfile.mkdtemp(prefix="augur_ft_test_")
cache = os.path.join(tmp, "cache")
os.makedirs(cache)
N, D = 120, 960
rng = np.random.default_rng(0)
fut = np.minimum(np.arange(N) + 10, N - 1)
valid = np.arange(N) + 10 < N
np.save(f"{cache}/libero_future.npy", fut)
np.save(f"{cache}/libero_valid.npy", valid)
np.save(f"{cache}/libero_emb.npy", rng.normal(0, 1, (N, 64, D)).astype(np.float16))
np.save(f"{cache}/libero_flow.npy", rng.normal(0, 0.02, (N, 4, 32)).astype(np.float16))
json.dump({"libero": {"mean": [0.001] * 32, "std": [0.02] * 32}, "ssv2": {"mean": [0.0] * 32, "std": [0.05] * 32}},
          open(f"{cache}/flow_stats.json", "w"))
TA.FS_CACHE = cache

cfg = SmolVLAConfig(load_vlm_weights=False, device="cpu")
cfg.input_features = {
    "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
}
cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
base = SmolVLAPolicy(cfg)
TA.inject_lora(base)
print("base policy built", flush=True)

B = 3
g = torch.Generator().manual_seed(1)
batch = {
    "observation.images.image": torch.rand(B, 3, 256, 256, generator=g),
    "observation.images.image2": torch.rand(B, 3, 256, 256, generator=g),
    "observation.state": torch.randn(B, 8, generator=g),
    "observation.language.tokens": torch.randint(0, 1000, (B, 12), generator=g),
    "observation.language.attention_mask": torch.ones(B, 12, dtype=torch.bool),
    "action": torch.randn(B, 50, 7, generator=g),
    "action_is_pad": torch.zeros(B, 50, dtype=torch.bool),
    "augur_index": torch.tensor([3, 50, N - 2]),  # last row has no true future -> masked
}
noise = torch.randn(B, 50, 32, generator=g)
tm = torch.rand(B, generator=g) * 0.9 + 0.05
ok = True


def check(name, cond, info=""):
    global ok
    print(f"  [{'ok' if cond else 'FAIL'}] {name} {info}", flush=True)
    ok &= bool(cond)


def fake_pretrain(policy, target):
    """A pretraining checkpoint in pretrain_foresight.py's format with non-trivial values."""
    vwe = policy.model.vlm_with_expert
    fs0 = AF.Foresight(vwe.config.text_config.hidden_size, vwe.expert_hidden_size, target, "tr", 4)
    with torch.no_grad():
        for p in fs0.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        if target == "t5":
            fs0.tgt_mean.fill_(0.001)
            fs0.tgt_std.fill_(0.02)
    lora = {n: torch.randn_like(p) * 0.01 for n, p in policy.named_parameters() if ".lora_" in n}
    pre = "model.vlm_with_expert."
    ema = {n[len(pre):]: t + torch.randn_like(t) * 0.001 for n, t in lora.items()} if target == "t3" else {}
    ck = {"lora": lora, "ema": ema, "augur": fs0.state_dict(), "step": 123,
          "args": {"target": target, "data": "hr", "k": 4}}
    path = os.path.join(tmp, f"pre_{target}.pt")
    torch.save(ck, path)
    return path, ck


def sample(policy):
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    with torch.no_grad():
        return policy.model.sample_actions(images, img_masks, batch["observation.language.tokens"],
                                           batch["observation.language.attention_mask"], state, noise=noise.clone())


def run(target, mode):
    print(f"\n=== {target}-{mode}", flush=True)
    TA._FS.clear()
    TA.FS_TARGET, TA.FS_MODE, TA.FS_K, TA.FS_LAMBDA = target, mode, 4, 0.1
    pol = copy.deepcopy(base)
    pol._save_pretrained = TA._save_merged.__get__(pol)
    ck = None
    if target != "none":
        TA.FS_INIT, ck = fake_pretrain(pol, target)
        fs, msg = TA.setup_foresight(pol)
        TA._FS["policy_ref"] = pol
        print("  ", msg)
        # 1. setup
        named = dict(pol.named_parameters())
        check("LoRA loaded", all(torch.equal(named[n], t) for n, t in ck["lora"].items()))
        check("queries/head loaded", torch.equal(fs.queries, ck["augur"]["queries"]) and
              torch.equal(fs.head[0].weight, ck["augur"]["head.0.weight"]))
        if target == "t3":
            check("EMA loaded", set(fs.ema) == set(ck["ema"]) and all(torch.equal(fs.ema[n], t) for n, t in ck["ema"].items()))
        if mode == "tt":
            check("TT conditioning created", hasattr(fs, "cond_proj"))
        # 2. loss
        pol.train()
        TA.FS_LAMBDA = 0.0
        l0, _ = pol.forward(batch, noise=noise.clone(), time=tm.clone())
        TA.FS_LAMBDA = 0.1
        l1, d = pol.forward(batch, noise=noise.clone(), time=tm.clone())
        tgt, v = TA.foresight_targets(batch["augur_index"])
        check("valid mask", v.tolist() == [True, True, False], str(v.tolist()))
        check("target shape", tuple(tgt.shape) == (B, 4, fs.d), str(tuple(tgt.shape)))
        fl = AF.foresight_loss(fs, tgt, v)
        check("loss = action + 0.1 * foresight", abs(float(l1 - l0) - 0.1 * float(fl)) < 1e-4,
              f"diff {float(l1 - l0):.6f} vs {0.1 * float(fl):.6f}; logged {d.get('foresight_loss'):.4f}")
        pol.zero_grad()
        l1.backward()
        gq, gh = fs.queries.grad, fs.head[0].weight.grad
        gl = sum(float(p.grad.abs().sum()) for n, p in pol.named_parameters() if ".lora_" in n and p.grad is not None)
        check("grads: queries, head, LoRA", gq is not None and gq.abs().sum() > 0 and gh.abs().sum() > 0 and gl > 0)
        if mode == "tt":
            check("grads: conditioning", fs.cond_proj.weight.grad.abs().sum() > 0)
        with torch.no_grad():  # make the online LoRA differ from the pretraining one, as after training
            for n, p in pol.named_parameters():
                if ".lora_B" in n:
                    p.add_(torch.randn_like(p) * 0.005)
        fs.ema_update(pol.model)
    # 3. save
    pol.eval()
    out = os.path.join(tmp, f"ckpt_{target}_{mode}")
    os.makedirs(out)
    pol._save_pretrained(Path(out))
    sd = load_file(os.path.join(out, "model.safetensors"))
    check("no augur / LoRA keys in model.safetensors", not any(k.startswith("model.augur.") or ".lora_" in k for k in sd))
    info = json.load(open(os.path.join(out, "augur_train.json")))
    check("augur_train.json target", info["target"] == target, str({k: info.get(k) for k in ("target", "mode", "k", "lambda")}))
    if target != "none":
        check("augur.safetensors", os.path.exists(os.path.join(out, "augur.safetensors")))
        lz = load_file(os.path.join(out, "augur_lora.safetensors"))
        check("online + EMA LoRA saved", any(k.startswith("online.") for k in lz) and any(k.startswith("ema.") for k in lz))
    # 4. eval load
    ref = sample(pol)
    fresh = SmolVLAPolicy.from_pretrained(out)
    fresh.eval()
    fs2 = AF.load_for_eval(fresh, out)
    check("load_for_eval installs iff foresight", (fs2 is None) == (target == "none"))
    act = sample(fresh)
    err = float((act - ref).abs().max())
    # bf16 VLM: merged vs unmerged LoRA differs by rounding only; a wrong load is far larger (below)
    check("eval actions match in-memory policy", err < 5e-2, f"max diff {err:.2e}")
    if target != "none":
        bad = SmolVLAPolicy.from_pretrained(out)
        bad.eval()
        AF.install(bad, target, mode, 4)  # same architecture, foresight weights NOT loaded
        if target == "t5":
            bad.model.augur.tgt_mean.copy_(fs2.tgt_mean)
            bad.model.augur.tgt_std.copy_(fs2.tgt_std)
        bad_err = float((sample(bad) - ref).abs().max())
        if mode == "tt":
            check("negative control: unloaded TT foresight changes actions", bad_err > 3 * err, f"{bad_err:.2e}")
        else:
            check("TR: actions independent of foresight weights", bad_err < 5e-2, f"{bad_err:.2e}")
    if target != "none" and mode == "tt":
        with torch.no_grad():
            fs2.queries.add_(1.0)
        check("TT eval actions depend on foresight", float((sample(fresh) - act).abs().max()) > 1e-3)
    if target != "none" and mode == "tr":
        with torch.no_grad():
            fs2.queries.add_(1.0)
        check("TR eval actions ignore foresight", float((sample(fresh) - act).abs().max()) < 1e-4)


run("none", "tr")
run("t3", "tr")
run("t5", "tt")

print("\n=== preprocessor wrapper / eval_mixed", flush=True)
w = TA._KeepIndex(lambda b: {k: v for k, v in b.items() if k != "index"})
o = w({"index": torch.tensor([7, 8]), "x": 1})
check("_KeepIndex keeps the row index", torch.equal(o["augur_index"], torch.tensor([7, 8])))
try:
    import eval_mixed as EM  # noqa: F401

    check("eval_mixed patches make_policy", EM.E.make_policy is EM._make_policy_augur)
except ImportError as e:
    print("  [skip] eval_mixed import:", e)

print("\nALL OK" if ok else "\nSOME TESTS FAILED", flush=True)
sys.exit(0 if ok else 1)
