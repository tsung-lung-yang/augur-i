#!/bin/bash
# LOG-005: one-time LIBERO-PRO setup (Position axis). CPU and disk only; safe while training runs.
#   1. clone LIBERO-PRO at the registered commit into ~/augur/code/LIBERO-PRO
#   2. download the released perturbation files (HF dataset zhouxueyang/LIBERO-Pro)
#   3. create the augur-pro env: a clone of augur without hf-libero, with LIBERO-PRO on
#      PYTHONPATH and its own LIBERO config (~/.libero_pro), like augur-plus for LIBERO-Plus
#   4. copy the files, index the Position suites, write the evaluation plan
# Usage: bash scripts/setup_libero_pro.sh   (from ~/augur/repo)
set -eo pipefail
PRO_COMMIT=eafdb809426b13153aa1e4c42d6601844217dfec
PRO=~/augur/code/LIBERO-PRO
HFDIR=~/augur/data/libero_pro_hf
REPO=$(pwd)
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "== 1. clone LIBERO-PRO @ $PRO_COMMIT"
if [ ! -d "$PRO" ]; then git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git "$PRO"; fi
git -C "$PRO" fetch --quiet origin "$PRO_COMMIT" 2>/dev/null || true
git -C "$PRO" checkout --quiet "$PRO_COMMIT"
git -C "$PRO" log -1 --format="   %H %cd"

echo "== 2. download zhouxueyang/LIBERO-Pro"
conda activate augur
env -u HF_HUB_OFFLINE python - <<EOF
from huggingface_hub import snapshot_download
p = snapshot_download("zhouxueyang/LIBERO-Pro", repo_type="dataset", local_dir="$HFDIR")
print("   downloaded to", p)
EOF
du -sh "$HFDIR"
find "$HFDIR" -maxdepth 3 -type d -not -path "*/.cache*" | sed "s|$HFDIR|   .|" | head -40

echo "== 3. augur-pro env"
if ! conda env list | grep -q "^augur-pro "; then
  conda create -y -q -n augur-pro --clone augur
fi
conda activate augur-pro
pip uninstall -y -q hf-libero || true
conda env config vars set PYTHONPATH="$PRO" LIBERO_CONFIG_PATH="$HOME/.libero_pro" > /dev/null
conda deactivate; conda activate augur-pro
echo "   PYTHONPATH=$PYTHONPATH LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH"

echo "== 4. copy, index, plan"
cd "$REPO"
python scripts/libero_pro_setup.py --pro_root "$PRO" --hf_dir "$HFDIR"
python scripts/make_eval_plan_pro.py
echo "== LIBERO-PRO SETUP DONE"
