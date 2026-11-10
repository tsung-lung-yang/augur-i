"""Unit test for scripts/train_augur.py (LoRA merge, augmentation, state dropout). Run: python scripts/test_train_hooks.py"""
import sys, types, importlib.util, torch, torch.nn as nn

# stub lerobot so train_augur imports without LeRobot installed
T = types.ModuleType("lerobot.scripts.lerobot_train")
T.make_policy = lambda *a, **k: None
T.update_policy = lambda *a, **k: None
T.main = lambda: None
for name in ["lerobot", "lerobot.scripts"]:
    sys.modules[name] = types.ModuleType(name)
sys.modules["lerobot.scripts.lerobot_train"] = T
sys.modules["lerobot.scripts"].lerobot_train = T

import os
spec = importlib.util.spec_from_file_location("ta", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_augur.py"))
ta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ta)

torch.manual_seed(0)
# LoRA: zero-init B means identical output; merged weight reproduces LoRA forward
base = nn.Linear(16, 8)
x = torch.randn(4, 16)
ref = base(x)
l = ta.LoRALinear(base, 4, 8)
assert torch.allclose(l(x), ref, atol=1e-6), "LoRA with B=0 must equal base"
with torch.no_grad():
    l.lora_B.normal_()
y = l(x)
merged = nn.Linear(16, 8)
merged.weight.data = l.merged_weight()
merged.bias.data = base.bias.data
assert torch.allclose(merged(x), y, atol=1e-5), "merged weight mismatch"
assert l.weight is base.weight
names = [n for n, _ in l.named_parameters()]
assert set(names) == {"weight", "bias", "lora_A", "lora_B"}, names
print("LoRA ok", names)

# augmentation
b = {
    "observation.images.image": torch.rand(6, 1, 3, 256, 256),
    "observation.images.image2": torch.rand(6, 3, 256, 256),
    "observation.images.image_padding_mask": torch.ones(6),
    "observation.state": torch.randn(6, 8) + 5,
}
orig = {k: v.clone() for k, v in b.items()}
torch.manual_seed(1)
out = ta.augment_batch(b)
for k in ["observation.images.image", "observation.images.image2"]:
    assert out[k].shape == orig[k].shape and out[k].min() >= 0 and out[k].max() <= 1
    assert not torch.allclose(out[k], orig[k])
assert torch.equal(out["observation.images.image_padding_mask"], orig["observation.images.image_padding_mask"])
z = (out["observation.state"] == 0).all(dim=1)
print("state rows zeroed:", z.tolist())
# shift only: with jitter disabled the result must be a crop of the replicate-padded image
x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
s = ta._shift(x, 2)
xp = torch.nn.functional.pad(x, (2, 2, 2, 2), mode="replicate")
ok = all(any(torch.equal(s[i], xp[i, :, dy:dy + 8, dx:dx + 8]) for dy in range(5) for dx in range(5)) for i in range(2))
assert ok, "shift is not a crop"
# state-drop rate
torch.manual_seed(2)
big = {"observation.state": torch.ones(20000, 8)}
r = (ta.augment_batch(big)["observation.state"] == 0).all(1).float().mean().item()
print("state drop rate", round(r, 3))
assert abs(r - 0.3) < 0.02
print("ALL OK")
