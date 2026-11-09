# LOG-002：第一週可行性量測與評測設定登記

- 日期：2026-11-09（UTC+8）
- 前一筆：LOG-001（`prereg-v1`）
- 狀態聲明：截至本紀錄，沒有訓練任何 AUGUR 條件。下列成功率都來自官方 checkpoint `HuggingFaceVLA/smolvla_libero`，只在開發切分的 libero_spatial 任務 1 上執行，用途是確認流程與計時，不進任何判定。
- 和 LOG-001 第 8 節的差異：LOG-001 說訓練預算、回合數、K、人類影片清單都寫在 LOG-002。本紀錄先登記評測設定、每格回合數與「訓練步數的決定規則」；預訓練預算、K、人類影片清單、LIBERO-Plus／PRO 的擾動抽樣種子留到 LOG-003，仍須在任何非冒煙測試的訓練之前 commit。

---

## 1. 硬體與環境

- 硬體：RTX 4080 Laptop 12 GB；24 個 CPU 執行緒；WSL2，記憶體 23 GB。
- conda 環境 `augur`（由 `lerobot` 環境複製）：Python 3.12.14、torch 2.11.0+cu130、transformers 5.5.4、peft 0.21.2、timm 1.0.31。
- LeRobot：v0.6.1，commit `7e241bd6`，獨立 clone 於 `~/augur/code/lerobot`，以可編輯模式安裝；不共用 elevate-i 的原始碼。
- LIBERO：hf-libero 0.1.4、robosuite 1.4.0、MuJoCo 3.8.1；官方任務順序 `[0..9]`；設定檔 `~/.libero/config.yaml`。
- 影片解碼：torchcodec 使用系統 FFmpeg 6.1.1（apt）。
- 已知雜項：環境內另有 conda-forge `ffmpeg 9.0.2` 與其帶入的 `libegl`／`libgl`，torchcodec 不使用它（只支援 FFmpeg 4–8）。conda 退版失敗（pip 已改動 `packaging`、`setuptools`），所以保留。安裝前後的渲染時間相同（d3d12 相機每步 78.1 → 80.1／80.8 ms），判定沒有影響。
- 凍結紀錄：`env/augur_conda_explicit.txt`、`env/augur_pip_freeze.txt`、`env/lerobot_commit.txt`。從本紀錄起不升級 `augur` 環境的任何套件；必要的變更另開 LOG。
- 開發切分 libero_spatial 任務 1：`pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`。其餘切分任務名稱在 LOG-003 列出。

## 2. 量測結果（官方 checkpoint，開發切分 spatial 任務 1）

### 2-1 評測速度

| 設定 | 回合數 | 成功 | 每回合秒數（eval_ep_s） |
|---|---|---|---|
| 1 環境、fp32、n_action_steps=1、EGL 預設 | 2 | 2 | 117.1 |
| 10 環境、bf16、n_action_steps=1、EGL 預設 | 10 | 8 | 36.6 |
| 10 環境、bf16、n_action_steps=5、EGL 預設 | 10 | 8 | 18.5 |
| 10 環境、bf16、n_action_steps=5、d3d12 | 20 | 15 | 13.9 |
| 20 環境 | — | — | WSL 記憶體用盡（23/23 GB），中止 |

- 官方 checkpoint 設定：`chunk_size=50`、`n_action_steps=1`、`num_steps=10`、`n_obs_steps=1`。
- 單一回合的時間分解（cProfile，1 環境）：模型推論約 61 s、渲染約 15 s、Hugging Face 連線約 14 s（之後以 `HF_HUB_OFFLINE=1` 消除）。
- 推論時顯存約 2.9–3.1 GB。

### 2-2 模擬器單獨計時（`scripts/profile_env.py`，100 步）

| 設定 | 每步 |
|---|---|
| 只跑物理 | 10.7 ms |
| 兩台相機 256 px，EGL 預設 | 122.9 ms |
| 兩台相機 256 px，`GALLIUM_DRIVER=d3d12` | 78.1 ms |

### 2-3 訓練吞吐量（代理量測：LeRobot 內建 SmolVLA，`load_vlm_weights=true`，60 步，`lerobot/libero`）

| batch | 每步 | 樣本／秒 | 顯存峰值 |
|---|---|---|---|
| 8 | 0.26 s | 30 | 4.3 GB |
| 16 | 約 0.46 s | 約 34 | 6.2 GB |
| 32 | 0.83 s | 38 | 8.6 GB |

- 暖機後 `data_s` 約 0.01 s，讀資料不是瓶頸。
- 訓練資料 `lerobot/libero`：1,693 條示範、273,465 幀、10 fps。

## 3. 登記：評測設定（所有 AUGUR 條件相同）

1. `n_action_steps=5`；`--policy.use_amp=true`（bf16）；`--eval.batch_size=10`。
2. 渲染：`MUJOCO_GL=egl`、`GALLIUM_DRIVER=d3d12`，影像 256 px，與官方 LIBERO 設定相同的兩台相機。
3. `HF_HUB_OFFLINE=1`；評測 seed 使用 LeRobot 預設值 1000，並以固定初始狀態做配對評測。
4. 跨方法參考列（H15）的官方 checkpoint 使用各自的官方設定（例如 SmolVLA LIBERO 為 `n_action_steps=1`），並在表中註明。

## 4. 登記：每格回合數與評測範圍

- 一格 =（任務，軸）。
- **主矩陣**：保留切分 28 個任務，每格 3 回合。
  - T0、T2–T5 的 -TR／-TT、TW-S、最終設計：LIBERO（同分布）、LIBERO-Plus 7 軸、LIBERO-PRO 5 軸。
  - T3-R、T5-R、T0-C：LIBERO 與 LIBERO-Plus 7 軸（H1、H3、H13 只用到這些）。
- **試跑**：開發切分 12 個任務，LIBERO-Plus 7 軸加 LIBERO-PRO Position，每格 5 回合（試跑 seed 較少，回合多一點以降低關卡判定的雜訊）。
- 估計 GPU 時間（每回合約 18 s，含長任務的 1.3 倍）：主矩陣約 8–9 天；試跑約 1 天。
- 每格回合數的統計依據：M1 比的是兩類軸的平均差。每個條件每類約 4 軸 × 28 任務 × 3 回合 × 3 seeds ≈ 1,000 回合，配對評測下足以估計數個百分點的差距。

## 5. 登記：訓練設定與步數的決定規則

- 動作微調的有效 batch size 一律 32。顯存放不下的條件改用 16 × 梯度累積 2，並在結果表註明。
- 所有條件的動作微調步數相同，記為 S。
- **S 的決定規則**：
  1. 用 AUGUR 自己的 T0 實作（與其他條件同一份程式，只是關掉預見），seed 0，訓練到 60k 步。
  2. 每 5k 步只在**開發切分的同分布 LIBERO**（4 個 suite 的 12 個開發任務，每任務 10 回合）上評測，不評任何擾動軸。
  3. S = 成功率首次達到整條曲線最高值 95% 的最小步數，上限 40k 步。
  4. S 寫入 LOG-003 並 commit 之後，才可以訓練任何預見條件。
- 理由：S 只由不含預見的對照組在同分布資料上決定，和 M1–M4、H 系列的比較無關。
- 這次 T0 曲線的 run 不當作主矩陣的 T0 結果；主矩陣的 T0 依 S 重新訓練 3 個 seed。

## 6. 仍待登記（LOG-003，訓練前必須 commit）

1. S（依第 5 節規則）。
2. 預見預訓練的步數與 batch（各目標相同）。
3. 推論時版的預見 token 數 K。
4. 人類影片子集清單（SSv2 影片 ID；是否使用 EgoDex）。
5. LIBERO-Plus、LIBERO-PRO 各軸的擾動變體抽樣規則與種子清單；兩者各自的 conda 環境與 `LIBERO_CONFIG_PATH`。
6. 開發與保留切分全部 40 個任務的名稱。
7. 授權查核：SmolVLA 權重、SSv2、EgoDex、LIBERO-Plus、LIBERO-PRO。
