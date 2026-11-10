"""lerobot-eval with mixed-variant vector envs (one LIBERO / LIBERO-Plus variant per sub-env).

Stock lerobot-eval builds one vector env per task id, so a plan with one episode
per variant runs with batch size 1. This wrapper replaces create_libero_envs so
each vector env holds n_envs *different* variants; the rest of lerobot-eval
(policy, processors, rollout, metrics) is unchanged.

Environment variables:
  AUGUR_MIXED_SPEC   JSON file: list of [task_id, episode_index]; length must be a
                     multiple of --eval.batch_size. Chunk c of the list becomes
                     task_id c in eval_info.json, sub-env order is preserved.
  AUGUR_EVAL_VIDEOS  "1" to keep LeRobot's videos (default "0": off, saves one
                     extra render per step).
  A chunk that raises (e.g. a broken variant) is logged as "CHUNK FAILED" and its
  episodes get success = NaN; the remaining chunks still run.
  AUGUR_DIAG_LANG    diagnostics only: replace every sub-env's instruction with
                     this string. Never set in registered evaluations.
Foresight checkpoints (augur_train.json names a target, LOG-005): the foresight module is
installed on the loaded policy and its weights loaded from augur.safetensors; checkpoints
without one (T0) are evaluated exactly as before.
Called by scripts/eval_plan.py; all other arguments are lerobot-eval arguments.
"""
import json
import os

import lerobot.envs.libero as L
from lerobot.envs.utils import _LazyAsyncVectorEnv, parse_camera_names


def _with_lang(fn, text):
    def make():
        env = fn()
        env.task_description = text
        return env

    return make


def create_mixed(
    task,
    n_envs,
    gym_kwargs=None,
    camera_name="agentview_image,robot0_eye_in_hand_image",
    init_states=True,
    env_cls=None,
    control_mode="relative",
    episode_length=None,
    camera_name_mapping=None,
    is_libero_plus=False,
):
    spec = json.load(open(os.environ["AUGUR_MIXED_SPEC"]))
    gym_kwargs = dict(gym_kwargs or {})
    gym_kwargs.pop("task_ids", None)
    suite_name = str(task).strip()
    if "," in suite_name:
        raise ValueError("eval_mixed handles one suite per call")
    if len(spec) % n_envs:
        raise ValueError(f"spec length {len(spec)} is not a multiple of batch size {n_envs}")
    suite = L._get_suite(suite_name)
    cams = parse_camera_names(camera_name)
    out = {}
    for c in range(len(spec) // n_envs):
        fns = []
        for tid, ep in spec[c * n_envs : (c + 1) * n_envs]:
            fns.append(
                L._make_env_fns(
                    suite=suite,
                    suite_name=suite_name,
                    task_id=int(tid),
                    n_envs=int(ep) + 1,
                    camera_names=cams,
                    episode_length=episode_length,
                    init_states=init_states,
                    gym_kwargs=gym_kwargs,
                    control_mode=control_mode,
                    camera_name_mapping=camera_name_mapping,
                    is_libero_plus=is_libero_plus,
                )[int(ep)]
            )
        if os.environ.get("AUGUR_DIAG_LANG"):
            fns = [_with_lang(f, os.environ["AUGUR_DIAG_LANG"]) for f in fns]
        out[c] = _LazyAsyncVectorEnv(fns)
    if os.environ.get("AUGUR_DIAG_LANG"):
        print(f"[eval_mixed] DIAGNOSTIC instruction override: {os.environ['AUGUR_DIAG_LANG']!r}")
    print(f"[eval_mixed] {suite_name}: {len(spec)} sub-envs in {len(out)} chunks of {n_envs}")
    return {suite_name: out}


L.create_libero_envs = create_mixed

# LeRobot's --policy.use_amp uses torch's default CUDA autocast dtype (float16). The registered
# precision is bfloat16 (LOG-002, LOG-004), so set the default before LeRobot reads it.
import torch as _torch  # noqa: E402

_AMP = {"bf16": _torch.bfloat16, "fp16": _torch.float16}[os.environ.get("AUGUR_AMP_DTYPE", "bf16")]
if _torch.cuda.is_available():
    _torch.set_autocast_dtype("cuda", _AMP)


import lerobot.scripts.lerobot_eval as E  # noqa: E402

_eval_policy = E.eval_policy
_VIDEOS = os.environ.get("AUGUR_EVAL_VIDEOS", "0") == "1"


def _eval_policy_safe(*args, **kwargs):
    """Run one chunk; if it raises, mark its episodes as errors (NaN) and continue."""
    if not _VIDEOS:
        kwargs["max_episodes_rendered"] = 0
        kwargs["videos_dir"] = None
    try:
        return _eval_policy(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        print(f"[eval_mixed] CHUNK FAILED: {e!r}"[:800], flush=True)
        env = kwargs.get("env")
        inner = getattr(env, "_env", None)
        if inner is not None:
            try:
                inner.close(terminate=True)
            except Exception:  # noqa: BLE001
                pass
            env._env = None
        n = kwargs["n_episodes"]
        nan = float("nan")
        return {
            "per_episode": [
                {"episode_ix": i, "sum_reward": nan, "max_reward": nan, "success": nan, "seed": None}
                for i in range(n)
            ],
            "aggregated": {"avg_sum_reward": nan, "avg_max_reward": nan, "pc_success": nan, "eval_s": 0.0, "eval_ep_s": 0.0},
        }


E.eval_policy = _eval_policy_safe

_make_policy = E.make_policy


def _make_policy_augur(*args, **kwargs):
    policy = _make_policy(*args, **kwargs)
    cfg = kwargs.get("cfg", args[0] if args else None)
    path = getattr(cfg, "pretrained_path", None)
    if path and os.path.isdir(str(path)):
        import sys

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import augur_foresight as AF

        fs = AF.load_for_eval(policy, path)
        if fs is not None:
            print(f"[eval_mixed] foresight module installed: {fs.target}-{fs.mode} K={fs.k}", flush=True)
    return policy


E.make_policy = _make_policy_augur

if __name__ == "__main__":
    E.main()
