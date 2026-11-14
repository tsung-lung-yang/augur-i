"""AUGUR-I foresight module for LeRobot SmolVLA (targets x modes; LOG-005).

Installs K learnable foresight query tokens at the end of the VLM prefix (after the
state token). Queries attend to images, language and state; nothing in the prefix
attends to them, and the action expert is masked from them in every layer, so the
action pathway never reads the raw query states.

  mode "tr" (training-time): a head maps the query outputs to z_hat (K vectors in the
      target space); the foresight loss trains LoRA + queries + head. The action suffix is
      masked from the queries and keeps the position ids it would have without them, so
      with the same weights a TR model's actions equal those of a model without
      foresight tokens (verified by test_foresight.py).
  mode "tt" (test-time): z_hat is projected to K conditioning tokens placed before the
      action tokens in the expert suffix; the expert generates actions while attending
      to its own imagined future. Training uses the model's own z_hat (not the true
      future), so training and inference match.

Targets (one 1-second-ahead target per query, the 2x2 quadrants of the main camera):
  "t3"  JEPA latent: the future main-camera image embedding (frozen SigLIP + connector,
        cached) passed through the VLM text layers with EMA LoRA weights, mean-pooled per
        quadrant, then layer-normed. d = VLM hidden size.
  "t5"  low-resolution optical flow: RAFT flow (t -> t+1 s), 8x8 grid, 4x4x2 per quadrant,
        standardised with cached dataset statistics. d = 32.
Loss: mean squared error over valid samples (future inside the episode / clip).
"""
import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_DIMS = {"t5": 32}


class Foresight(nn.Module):
    def __init__(self, hidden: int, expert_hidden: int, target: str, mode: str, k: int = 4, ema_tau: float = 0.996):
        super().__init__()
        assert target in ("t3", "t5") and mode in ("tr", "tt")
        self.target, self.mode, self.k, self.hidden, self.ema_tau = target, mode, k, hidden, ema_tau
        d = hidden if target == "t3" else TARGET_DIMS[target]
        self.d = d
        self.queries = nn.Parameter(torch.randn(k, hidden) * 0.02)
        self.head = nn.Sequential(nn.Linear(hidden, 1024), nn.GELU(), nn.Linear(1024, d))
        if mode == "tt":
            self.cond_proj = nn.Linear(d, expert_hidden)
            self.cond_pos = nn.Parameter(torch.zeros(k, expert_hidden))
        self.register_buffer("tgt_mean", torch.zeros(d), persistent=True)
        self.register_buffer("tgt_std", torch.ones(d), persistent=True)
        # runtime state (not saved)
        self._cond = None
        self._q0 = None
        self.last_pred = None
        self.ema = {}

    # ---- prediction / conditioning
    def predict(self, h_q):
        return self.head(h_q.float())

    def cond_tokens(self, z_hat, dtype):
        return (self.cond_proj(z_hat) + self.cond_pos[None]).to(dtype)

    # ---- EMA of the LoRA weights (target encoder for t3)
    def lora_params(self, model):
        return {n: p for n, p in model.vlm_with_expert.named_parameters() if ".lora_" in n}

    @torch.no_grad()
    def ema_init(self, model):
        self.ema = {n: p.detach().clone() for n, p in self.lora_params(model).items()}

    @torch.no_grad()
    def ema_update(self, model):
        if not self.ema:
            return self.ema_init(model)
        for n, p in self.lora_params(model).items():
            self.ema[n].mul_(self.ema_tau).add_(p.detach(), alpha=1 - self.ema_tau)

    @contextlib.contextmanager
    def ema_swap(self, model):
        if not self.ema:
            self.ema_init(model)
        params = self.lora_params(model)
        saved = {n: p.data for n, p in params.items()}
        try:
            for n, p in params.items():
                p.data = self.ema[n].to(p.device, p.dtype)
            yield
        finally:
            for n, p in params.items():
                p.data = saved[n]


def quadrant_pool(tokens):
    """(B, 64, D) row-major 8x8 grid -> (B, 4, D): TL, TR, BL, BR means."""
    b, n, d = tokens.shape
    s = int(math.isqrt(n))
    g = tokens.view(b, s, s, d)
    h = s // 2
    return torch.stack([g[:, :h, :h].mean((1, 2)), g[:, :h, h:].mean((1, 2)),
                        g[:, h:, :h].mean((1, 2)), g[:, h:, h:].mean((1, 2))], dim=1)


import os as _os

# LOG-007: TT training through the KV cache (one prefix pass). Set AUGUR_TT_CACHED=0 for the
# original two-pass implementation (used for T3-TT seed 0); the two are numerically equivalent
# (scripts/test_tt_cached.py).
TT_CACHED_TRAINING = _os.environ.get("AUGUR_TT_CACHED", "1") == "1"


def install(policy, target: str, mode: str, k: int = 4):
    """Attach Foresight to policy.model and patch its prefix/suffix/forward/inference."""
    import lerobot.policies.smolvla.modeling_smolvla as M

    m = policy.model
    vwe = m.vlm_with_expert
    hidden = vwe.config.text_config.hidden_size
    fs = Foresight(hidden, vwe.expert_hidden_size, target, mode, k).to(next(m.parameters()).device)
    m.augur = fs
    orig_prefix, orig_suffix = m.embed_prefix, m.embed_suffix
    chunk = m.config.chunk_size

    def embed_prefix(images, img_masks, lang_tokens, lang_masks, state=None):
        embs, pad, att = orig_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        b = embs.shape[0]
        q = (fs.queries * math.sqrt(hidden)).to(embs.dtype)[None].expand(b, -1, -1)
        fs._q0 = embs.shape[1]
        qatt = torch.zeros(b, k, dtype=att.dtype, device=att.device)
        qatt[:, 0] = True
        return (torch.cat([embs, q], 1),
                torch.cat([pad, torch.ones(b, k, dtype=pad.dtype, device=pad.device)], 1),
                torch.cat([att, qatt], 1))

    def embed_suffix(noisy_actions, timestep):
        embs, pad, att = orig_suffix(noisy_actions, timestep)
        if fs.mode == "tt" and fs._cond is not None:
            b = embs.shape[0]
            catt = torch.zeros(b, k, dtype=att.dtype, device=att.device)
            catt[:, 0] = 1
            embs = torch.cat([fs._cond.to(embs.dtype), embs], 1)
            pad = torch.cat([torch.ones(b, k, dtype=pad.dtype, device=pad.device), pad], 1)
            att = torch.cat([catt, att], 1)
        return embs, pad, att

    def prefix_only(prefix_embs, prefix_pad, prefix_att):
        att2d = M.make_att_2d_masks(prefix_pad, prefix_att)
        pos = torch.cumsum(prefix_pad, dim=1) - 1
        outs, cache = vwe.forward(attention_mask=att2d, position_ids=pos, past_key_values=None,
                                  inputs_embeds=[prefix_embs, None], use_cache=True)
        return outs[0], cache

    def queries_of(prefix_out):
        return prefix_out[:, fs._q0: fs._q0 + k]

    def forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
        if noise is None:
            noise = m.sample_noise(actions.shape, actions.device)
        if time is None:
            time = m.sample_time(actions.shape[0], actions.device)
        te = time[:, None, None]
        x_t = te * noise + (1 - te) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad, prefix_att = m.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        lp = prefix_embs.shape[1]
        z_hat = None
        if fs.mode == "tt" and TT_CACHED_TRAINING:
            # LOG-007: one prefix pass that fills the KV cache (with gradients), then the suffix
            # attends to the cached prefix -- the same computation as the full pass below
            # (prefix tokens never attend to the suffix), with one prefix pass instead of two.
            p_out, cache = prefix_only(prefix_embs, prefix_pad, prefix_att)
            z_hat = fs.predict(queries_of(p_out))
            fs.last_pred = z_hat
            fs._cond = fs.cond_tokens(z_hat, prefix_embs.dtype)
            try:
                suffix_embs, suffix_pad, suffix_att = m.embed_suffix(x_t, time)
            finally:
                fs._cond = None
            b, sl = suffix_pad.shape
            p2d = prefix_pad[:, None, :].expand(b, sl, lp).clone()
            p2d[:, :, fs._q0:lp] = False
            s2d = M.make_att_2d_masks(suffix_pad, suffix_att)
            full = torch.cat([p2d, s2d], dim=2)
            pos = (torch.sum(prefix_pad, dim=-1)[:, None] - k) + torch.cumsum(suffix_pad, dim=1) - 1
            outs, _ = vwe.forward(attention_mask=full, position_ids=pos, past_key_values=cache,
                                  inputs_embeds=[None, suffix_embs], use_cache=True)
            suffix_out = outs[1][:, -chunk:].to(dtype=torch.float32)
            v_t = m.action_out_proj(suffix_out)
            return F.mse_loss(u_t, v_t, reduction="none")
        if fs.mode == "tt":
            p_out, _ = prefix_only(prefix_embs, prefix_pad, prefix_att)
            z_hat = fs.predict(queries_of(p_out))
            fs._cond = fs.cond_tokens(z_hat, prefix_embs.dtype)
        try:
            suffix_embs, suffix_pad, suffix_att = m.embed_suffix(x_t, time)
        finally:
            fs._cond = None
        pad = torch.cat([prefix_pad, suffix_pad], 1)
        att = torch.cat([prefix_att, suffix_att], 1)
        att2d = M.make_att_2d_masks(pad, att)
        att2d[:, lp:, fs._q0:lp] = False
        pos = torch.cumsum(pad, dim=1) - 1
        pos[:, lp:] -= k  # the suffix is numbered as if the queries were absent
        (p_out2, suffix_out), _ = vwe.forward(attention_mask=att2d, position_ids=pos, past_key_values=None,
                                              inputs_embeds=[prefix_embs, suffix_embs], use_cache=False)
        if z_hat is None:
            z_hat = fs.predict(queries_of(p_out2))
        fs.last_pred = z_hat
        suffix_out = suffix_out[:, -chunk:].to(dtype=torch.float32)
        v_t = m.action_out_proj(suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    def sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        bsize, device = state.shape[0], state.device
        if noise is None:
            noise = m.sample_noise((bsize, m.config.chunk_size, m.config.max_action_dim), device)
        prefix_embs, prefix_pad, prefix_att = m.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        p_out, cache = prefix_only(prefix_embs, prefix_pad, prefix_att)
        z_hat = fs.predict(queries_of(p_out))
        fs.last_pred = z_hat
        fs._cond = fs.cond_tokens(z_hat, prefix_embs.dtype) if fs.mode == "tt" else None
        try:
            return M.euler_integrate(
                lambda x, t: m.denoise_step(x_t=x, prefix_pad_masks=prefix_pad, past_key_values=cache, timestep=t),
                noise, m.config.num_steps, rtc_processor=m.rtc_processor, rtc_enabled=m._rtc_enabled(),
                inference_delay=kwargs.get("inference_delay"), prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
                execution_horizon=kwargs.get("execution_horizon"),
            )
        finally:
            fs._cond = None

    def denoise_step(prefix_pad_masks, past_key_values, x_t, timestep):
        suffix_embs, suffix_pad, suffix_att = m.embed_suffix(x_t, timestep)
        sl, b, pl = suffix_pad.shape[1], prefix_pad_masks.shape[0], prefix_pad_masks.shape[1]
        p2d = prefix_pad_masks[:, None, :].expand(b, sl, pl).clone()
        p2d[:, :, fs._q0:pl] = False
        s2d = M.make_att_2d_masks(suffix_pad, suffix_att)
        full = torch.cat([p2d, s2d], dim=2)
        pos = (torch.sum(prefix_pad_masks, dim=-1)[:, None] - k) + torch.cumsum(suffix_pad, dim=1) - 1
        outs, _ = vwe.forward(attention_mask=full, position_ids=pos, past_key_values=past_key_values,
                              inputs_embeds=[None, suffix_embs], use_cache=m.config.use_cache)
        if past_key_values is not None:
            past_key_values.crop(pl)
        out = outs[1][:, -chunk:].to(dtype=torch.float32)
        return m.action_out_proj(out)

    m.embed_prefix, m.embed_suffix = embed_prefix, embed_suffix
    m.forward, m.sample_actions, m.denoise_step = forward, sample_actions, denoise_step
    m.augur_prefix_only = prefix_only
    return fs


# ---------------------------------------------------------------- targets and losses
@torch.no_grad()
def t3_target(policy, future_emb):
    """future_emb: (B, 64, D) cached main-camera image embedding (before the sqrt(D) scale)."""
    m, fs = policy.model, policy.model.augur
    vwe = m.vlm_with_expert
    x = future_emb.to(next(m.parameters()).dtype) * math.sqrt(fs.hidden)
    b, n, _ = x.shape
    att2d = torch.ones(b, n, n, dtype=torch.bool, device=x.device)
    pos = torch.arange(n, device=x.device)[None].expand(b, -1)
    with fs.ema_swap(m):
        outs, _ = vwe.forward(attention_mask=att2d, position_ids=pos, past_key_values=None,
                              inputs_embeds=[x, None], use_cache=True)
    return F.layer_norm(quadrant_pool(outs[0].float()), (fs.hidden,))


def standardise(fs, target):
    return (target.float() - fs.tgt_mean) / fs.tgt_std


def foresight_loss(fs, target, valid=None):
    """MSE between fs.last_pred (B, K, d) and target (B, K, d), over valid samples."""
    pred = fs.last_pred
    per = ((pred - target.to(pred.dtype)) ** 2).mean(dim=(1, 2))
    if valid is None:
        return per.mean()
    v = valid.to(per.dtype)
    return (per * v).sum() / v.sum().clamp(min=1.0)


def foresight_error(pred, target):
    """Registered foresight error: cosine distance per sample, averaged over the K tokens."""
    return (1 - F.cosine_similarity(pred.float(), target.float(), dim=-1)).mean(dim=-1)


# ---------------------------------------------------------------- checkpoints
def save_augur(fs, model, save_directory, info):
    """Write the foresight module (augur.safetensors) and the online / EMA LoRA tensors
    (augur_lora.safetensors, needed only to recompute t3 targets) next to a merged checkpoint."""
    from safetensors.torch import save_file

    d = str(save_directory)
    save_file({k: v.detach().contiguous().cpu() for k, v in fs.state_dict().items()}, f"{d}/augur.safetensors")
    lora = {f"online.{n}": p.detach().float().contiguous().cpu() for n, p in fs.lora_params(model).items()}
    lora.update({f"ema.{n}": t.detach().float().contiguous().cpu() for n, t in fs.ema.items()})
    if lora:
        save_file(lora, f"{d}/augur_lora.safetensors")
    return {"target": fs.target, "mode": fs.mode, "k": fs.k, **info}


def load_for_eval(policy, path):
    """If a fine-tuned checkpoint directory carries a foresight module, install it on the loaded
    policy (whose LoRA is already merged) and load its weights. Returns the module or None."""
    import json
    from pathlib import Path

    from safetensors.torch import load_file

    p = Path(path)
    j = p / "augur_train.json"
    if not j.exists():
        return None
    info = json.load(open(j))
    if info.get("target", "none") in (None, "none"):
        return None
    fs = install(policy, info["target"], info["mode"], int(info.get("k", 4)))
    fs.load_state_dict(load_file(str(p / "augur.safetensors")), strict=True)
    fs.eval()
    return fs
