# LOG-004：訓練配方登記（所有條件共用）與 T0 學習曲線的執行方式

- 日期：2026-11-10（UTC+8）
- 前一筆：LOG-003
- 狀態聲明：截至本紀錄，只跑過 200 步的冒煙測試（`runs/train/smoke_t0`，不進任何判定）。T0 學習曲線在本紀錄 commit 之後才開始。
- 與前面紀錄的差異：LOG-003 第 7 節的待登記項目，本紀錄處理「訓練配方」（S 的決定規則所需的前提）；S、預訓練預算、K、人類影片清單、LIBERO-PRO、評測時間重估、授權查核移到 LOG-005，仍須在 T0 以外任何條件的訓練之前 commit。

---

## 1. 更正：混合精度

- LeRobot 的 `--policy.use_amp=true` 使用 PyTorch 在 CUDA 上的預設 autocast 型別，也就是 **fp16**。所以 2026-11-10 以前的所有評測（LOG-002、LOG-003 的診斷）實際上是 fp16，而不是 LOG-002 寫的 bf16。這些評測都只是診斷，不進任何判定。
- 自本紀錄起，`scripts/train_augur.py` 與 `scripts/eval_mixed.py` 在 LeRobot 讀取設定前呼叫 `torch.set_autocast_dtype("cuda", torch.bfloat16)`（環境變數 `AUGUR_AMP_DTYPE`，預設 `bf16`）。**所有正式訓練與評測一律 bf16。**

## 2. 澄清：開發／保留切分是評測切分

- LOG-001 第 5 節的切分只決定「哪些評測結果可以用來做設計選擇」（開發）與「哪些用來判定假說」（保留）。
- **訓練資料不依切分過濾**：所有條件都用完整的 `lerobot/libero`（1,693 條示範、40 個任務），這是 LIBERO 的標準設定，也和官方 SmolVLA 一致。擾動評測（LIBERO-Plus／PRO）的變體不在訓練資料中。

## 3. 訓練配方（所有 13 個條件相同；預見條件另加的部分在 LOG-005 登記）

| 項目 | 設定 |
|---|---|
| 程式 | `scripts/train_augur.py`（LeRobot v0.6.1 `lerobot-train` 加三個掛鉤） |
| 模型 | `--policy.type=smolvla --policy.load_vlm_weights=true`：VLM 用 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 權重，取前 16 層語言層；動作專家從頭訓練（不使用 `lerobot/smolvla_base`） |
| 凍結 | 視覺編碼器與 VLM 全部凍結（`train_expert_only=true`，LeRobot 預設），只有 LoRA 可訓練 |
| LoRA | VLM 16 層語言層的 `q_proj`、`k_proj`、`v_proj`、`o_proj`，共 64 個投影；rank 16、alpha 32；A 以 Kaiming 均勻分布初始化、B 為 0；共 1,638,400 個參數；存檔時合併回原權重 |
| 可訓練參數 | 101,519,392（動作專家、狀態投影、LoRA） |
| 資料 | `lerobot/libero`，影像 256×256，兩台相機，torchcodec 解碼，8 個 worker |
| 擴增 | LOG-003 第 5 節的基礎配方（`AUGUR_AUG=base`），在 LeRobot 前處理之後於 GPU 上執行。實作細節：色彩抖動依亮度、對比、飽和度、色相的固定順序；state dropout 把**正規化後**的狀態向量設為 0，等於資料集平均值 |
| 動作 | chunk 50；訓練時 `n_action_steps` 不影響 |
| 優化器 | AdamW，lr 1e-4，betas (0.9, 0.95)，eps 1e-8，weight decay 1e-10，梯度裁切 10（SmolVLA 預設） |
| 學習率排程 | warmup 1,000 步，餘弦衰減 30,000 步到 2.5e-6，之後維持常數（SmolVLA 預設）。**排程與總步數無關**，所以學習曲線在第 s 步的 checkpoint，就等於一個只訓練 s 步的 run |
| batch | 32（不用梯度累積） |
| 精度 | bf16 autocast |
| seed | 依條件與 seed 編號；T0 學習曲線用 seed 0 |
| 驗證 | `scripts/test_train_hooks.py` 全部通過（LoRA 初始等價、合併等價、平移為裁切、state dropout 實測 29.6%）；冒煙測試 200 步，loss 2.07 → 1.06；存檔無 LoRA 殘留，第 0 層 `q_proj` 與原始權重最大差 0.0011；評測載入無 missing／unexpected keys |
| 實測 | 每步約 0.97 秒；顯存峰值 8.7 GB |

## 4. T0 學習曲線（執行 LOG-002 第 5 節的 S 決定規則）

- 訓練：上述配方，T0（無預見），seed 0，60,000 步，每 5,000 步存一次 checkpoint，輸出 `runs/train/t0_curve_s0`。預估約 16 小時。
- 評測：訓練結束後，以 `scripts/eval_dev_vanilla.py` 依序評 12 個 checkpoint，每個在開發切分的原版 LIBERO 上跑 12 個任務 × 10 個初始狀態 = 120 回合（256 px、每批 10、`n_action_steps=5`、bf16）。不評任何擾動軸。
- S = 成功率首次達到整條曲線最高值 95% 的最小步數（5,000 步為一格），上限 40,000 步。S 與整條曲線寫入 LOG-005。
- 曲線上第 S 步的 checkpoint，直接作為 LOG-003 第 6 節「T0 地板關卡」的受測模型（因排程與總步數無關，它等同於只訓練 S 步的 T0 seed 0）。
- 若地板關卡進入補救（加強版擴增），S 不變，以加強版配方重新訓練 T0 seed 0 共 S 步再評一次。
- 這條曲線的 checkpoint 不作為主矩陣的 T0；主矩陣的 T0 依 S 另訓練 3 個 seed。
