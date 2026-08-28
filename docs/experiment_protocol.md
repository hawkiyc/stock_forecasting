# 條件式 Alpha 模型實驗協定

## 中文

### 1. 研究問題

主要問題：在嚴格 point-in-time、next-open execution 與 chronological evaluation
下，以金融 K-line 預訓練的 Kronos-base 經 LoRA 微調後，能否比簡單規則、傳統 ML
與小型 causal DL baseline 更準確地預測個別股票／ETF 從可設定的第 1、2 或 3 個
持有交易日起至固定第 14 日、
相對市場 benchmark 的 alpha 條件分布？

本協定不把 validation Sharpe 或單次回測視為「可交易 alpha」的充分證據，也不把模型
輸出直接當交易規則。技術指標 baseline 只用來衡量模型增量，不會加入 production
model input 或 quant head。

### 2. 固定假設與範圍

- 商品層級預測，不是投資組合預測。
- 日線 OHLCV only。
- 美國與台灣股票、ETF；期貨與選擇權不在本 PoC。
- 收盤 `t` 後產生訊號，下一共同交易日 raw open 進場。
- 進場日算第 1 個持有交易日。
- `h_start` 可設定為 1、2 或 3；horizons 是 `h_start` 到固定第 14 日。
- label 固定為 benchmark-relative adjusted execution log return。
- CAPM abnormal return 只作未來 diagnostic/ablation。
- 連續 quantile 預測是唯一訓練 target；方向訊號只作後處理。

### 3. 模型假說

主假說：共享金融預訓練 encoder 加上歷史 benchmark 的動態 gated conditioning，能在
同一資料與切分下，降低所有 `h_start`–14 日 horizon 的 normalized pinball loss，且改善
median correlation 或 calibration，而不是只在單一五日方向分類上看似提升。

必要反證條件：若 zero-return、past-only momentum、GBDT、GRU、DLinear 或 PatchTST
在同一 validation protocol 下持續優於完整模型，則不能以 foundation-model 名義宣稱
有模型增益。

### 4. Dataset identity

每次 run 必須綁定：

- dataset profile 與 `selected_datasets`
- raw/processed Parquet SHA-256、size、row count
- provider、market、symbol、asset type 與 date range
- benchmark mapping SHA-256
- window/preparation spec 與 data-pipeline digest
- `h_start`、固定 `max_horizon=14` 與完整 horizon 序列
- train-only robust scales
- chronological split counts 與 window exclusion audit

正式比較不得混用不同 profile、universe、資料版本或 benchmark mapping。EODHD 與
官方台股資料的來源差異必須在報告中揭露。

### 5. 標籤與因果性

對 cutoff `t` 與 horizon `h`：

```text
entry = next shared trading day's raw regular-session open
exit  = h-th shared holding day's raw close
alpha_h = log(asset adjusted gross return) - log(benchmark adjusted gross return)
```

商品與 benchmark 使用相同 entry/exit timestamps。模型 input 只包含兩者截至
`cutoff_at` 的 point-in-time adjusted OHLCV；未來 entry、exit、benchmark return、
asset return、CAPM diagnostic 與任何 future label 都不得出現在 input tensor。

測試至少覆蓋：

1. 修改 cutoff 後 benchmark 只改 label、不改兩個 contexts。
2. 對 adjusted close 乘共同常數不改 input 或 label。
3. split 後 raw price 跳空被 point-in-time adjustment 消除，volume 方向正確。
4. context 的最大 timestamp 不超過 cutoff。
5. benchmark calendar gap fail closed。

### 6. Split 與 Stage 設計

先建立全部 causal windows，再作 chronological 70/15/15 train/validation/test split，
邊界使用 purge 20 bars 與 effective embargo 14 bars。不要隨機切 asset-day rows，也不要
讓同一 future label interval 跨越 split 邊界。

| 項目 | Stage 1 | Stage 2 |
|---|---|---|
| 目的 | 驗證完整腳本、資料、GPU、loss、checkpoint 與 validation | 完整 train split 的 PoC 結果 |
| train 樣本 | 依 market/asset type 確定性配置、精確 15% target count | 100% train split |
| validation/test | 完整保留 | 完整保留 |
| 模型架構 | Kronos-base + LoRA + shared resampler + conditioner + alpha head | 完全相同 |
| 初始化 | 原始 pretrained base | 原始 pretrained base |
| 接續 Stage 1 checkpoint | 否 | 否 |

Stage 1 不是「最早 15% 時間」，也不是縮短 validation/test。兩個 config 的
`config.model_architecture_digest()` 必須相同；此 digest 包含 `h_start`／輸出維度，
Stage 2 重新從相同 pretrained revision
開始，以免把 Stage 1 script smoke 當成額外訓練資料。

### 7. Objective

唯一 objective 是 q10/q50/q90 pinball loss。每個 horizon 以 train-only robust scale
正規化後等權平均：

```text
scale_h = max(IQR_h, 1.4826 * MAD_h, 1e-4)
selection_score = mean_h(normalized_pinball_h), h=h_start,...,14
```

為保留既有 RunPod checkpoint monitor 路徑，aggregate score 同時寫在
`primary_5d/selection_score`；它仍涵蓋全部 horizons，不是只有 5 日。checkpoint
selection mode 固定為 `min`。沒有 classification loss 或 loss-weight search。

### 8. Comparator suite

所有 baseline 只能讀相同 cutoff-inclusive asset/benchmark contexts，並用相同 train
subset 與 validation split：

- constant distributions：always-buy、zero-return
- rules：relative momentum/reversal、MA crossover、RSI、MACD、volatility-scaled
- traditional ML：per-horizon quantile GBDT
- causal DL：paired GRU、DLinear、compact PatchTST
- full model：Kronos-base LoRA + dynamic benchmark conditioner

規則 baseline 的 residual quantiles、GBDT、neural early stopping 與 robust scales 都只
能使用 train labels；validation 用來選 model/checkpoint；test 不參與調參。

### 9. Validation metrics

主要 selection metric：

- all-horizon mean normalized pinball，lower is better。

每個 `h_start`–14 日 horizon 另報告：

- raw pinball 與 normalized pinball
- q50 MAE 與 normalized MAE
- q50 Pearson correlation
- q50 sign agreement
- q10–q90 interval coverage、width 與 coverage error

研究 diagnostics：

- date-level cross-sectional RankIC/IR
- equal-weight long/short net log return、Sharpe、max drawdown、turnover
- 五級 postprocess signal distribution
- market、asset type、provider、year slices

cross-sectional diagnostics 不是投資組合模型輸出，也不是主要 checkpoint metric。
transaction-cost 假設必須明列；單一 Sharpe 不可解讀為已證明可交易。

### 10. Test policy

自動 RunPod validation 預設只讀 train 與 validation，`test_unlocked=false`。test split
要在 architecture、hyperparameters、threshold、data QA 與 model selection 全部鎖定後
才能一次性解封。若 test 結果導致修改模型，該 test 已成為 validation，下一輪必須用
新的時間區間或 dataset version。

### 11. Checkpoint 與重現性

每個可接受 checkpoint 必須：

1. 由 validation `primary_5d/selection_score` 選出。
2. 保存 trainable parameter union、optimizer、scheduler、RNG state 與 resolved config。
3. 綁定 run ID、dataset artifacts、model architecture、Kronos source/model/tokenizer
   revisions 與 bounded training-source digest。
4. 通過 artifact SHA-256/size 與 strict trainable-key restore。
5. 使用 `model_output_schema_version=5.0`；舊 checkpoint fail closed。

報告成功狀態必須以 terminal lifecycle、run manifest、best-checkpoint pointer、trainer
state 與 raw validation JSON 交叉確認，不能只看 W&B chart 或 README。

### 12. 最低驗收

Stage 1：

- train subset 精確為完整 train split 的 15%。
- 完成 remote ruff/pytest、forward/backward、validation-ranked checkpoint save/reload、
  inference schema smoke 與自動 lifecycle termination。
- 輸出 `[B,15-h_start,3]` ordered alpha quantiles，`h_start ∈ {1,2,3}`；
  沒有 LLM/fact/classifier。

Stage 2：

- 使用相同 architecture digest、100% train split、相同 pretrained revisions。
- 完成所有 baseline 與完整模型的同協定 validation。
- raw JSON 有 per-horizon、aggregate、cross-sectional 與 subgroup metrics。
- 資料 provenance 可追溯，且 training code 無 provider/API import。

研究主張：

- Stage 1 只證明腳本可運作。
- 單一 seed Stage 2 只算 PoC。
- 要主張模型優於 baseline，至少需多 seed、報告 dispersion，並完成預先定義的 ablation。

### 13. 優先 ablation

在不改 output contract與資料切分下，依序測試：

1. gated benchmark conditioner vs asset-only（benchmark gate 固定為 0）。
2. Kronos LoRA vs frozen Kronos + trainable downstream modules。
3. Kronos vs compact PatchTST/DLinear，在相同 histories、horizons、loss 下比較。
4. raw price-index benchmark input vs total-return-adjusted benchmark input。
5. CAPM abnormal-return diagnostic，只作分析，不取代主要 label。

一次只改一個因子，並產生新的 experiment/run identity；不得用同一 checkpoint 名稱
覆寫不同語意。

### 14. 遠端驗證邊界

Python environment、Poetry lock、lint、pytest、data preparation、model prefetch、GPU
training 與 validation 全部在 RunPod 執行。本機只做 source/config 編輯、shell 靜態
檢查、S3 sync、Pod 控制與 artifact 下載。本機 Python/PyTorch/CUDA 狀態不是本專案
runtime 證據。

---

## English

### 1. Research question

Under strict point-in-time inputs, next-open execution, and chronological
evaluation, can a finance-K-line-pretrained Kronos-base with LoRA predict the
`h_start`-through-14 holding-day benchmark-relative alpha distribution of an individual stock
or ETF more accurately than simple rules, traditional ML, and compact causal DL
baselines?

Validation Sharpe or one backtest is not sufficient proof of tradable alpha,
and model output is not a complete trading rule. Technical baselines measure
incremental model value; they do not enter production inputs or the alpha head.

### 2. Fixed assumptions and scope

- Instrument-level, not portfolio-level, forecasting.
- Daily OHLCV only.
- US and Taiwan stocks/ETFs; no futures or options in this PoC.
- Signal after close `t`; entry at the next shared trading day's raw open.
- Entry day counts as holding day one.
- `h_start` is configurable as 1, 2, or 3; horizons run through fixed day 14.
- Label fixed to benchmark-relative adjusted execution log return.
- CAPM abnormal return reserved for a diagnostic/ablation.
- Continuous quantiles are the sole training target; direction is post-processing.

### 3. Model hypothesis

The primary hypothesis is that a shared finance-pretrained encoder with dynamic
gated historical-benchmark conditioning reduces mean normalized pinball across
all horizons and improves median correlation or calibration—not merely one
five-day classification number.

If zero-return, past-only momentum, GBDT, GRU, DLinear, or PatchTST consistently
outperforms the full model under the same protocol, no foundation-model gain may
be claimed.

### 4. Dataset identity

Every run binds dataset profile and selected sources, raw/processed hashes and
sizes, providers/markets/symbols/types/dates, benchmark-mapping hash, preparation
and pipeline digests, `h_start`, fixed `max_horizon=14`, the full horizon sequence,
train-only robust scales, split counts, and exclusion
audit. Formal comparisons cannot mix profiles, universes, dataset versions, or
benchmark mappings. EODHD versus official Taiwan source differences must be
disclosed.

### 5. Labels and causality

For cutoff `t` and horizon `h`:

```text
entry = next shared trading day's raw regular-session open
exit  = h-th shared holding day's raw close
alpha_h = log(asset adjusted gross return) - log(benchmark adjusted gross return)
```

Instrument and benchmark use identical timestamps. Inputs contain only both
point-in-time adjusted histories through `cutoff_at`. Future entries, exits,
returns, CAPM diagnostics, and labels must never enter input tensors.

Tests cover future-benchmark label-only changes, global adjusted-scale
invariance, split continuity and volume direction, cutoff-bounded contexts, and
fail-closed benchmark calendar gaps.

### 6. Splits and stages

Build all causal windows first, then assign chronological 70/15/15 train/
validation/test splits with 20-bar purge and effective 14-bar embargo. Never
randomly split asset-day rows or let one future label interval cross a boundary.

| Item | Stage 1 | Stage 2 |
|---|---|---|
| Purpose | Validate the complete script/data/GPU/loss/checkpoint/validation path | Full-train PoC result |
| Train samples | Exact deterministic 15% target count allocated by market/type | 100% of train |
| Validation/test | Fully retained | Fully retained |
| Architecture | Kronos-base + LoRA + shared resampler + conditioner + alpha head | Identical |
| Initialization | Original pretrained base | Original pretrained base |
| Continue Stage 1 checkpoint | No | No |

Stage 1 is not the earliest 15% of time and does not shrink validation/test.
Both configs require the same architecture digest. Stage 2 restarts from the
same pretrained revisions so the Stage 1 smoke run is not hidden extra training.

### 7. Objective

The sole objective is q10/q50/q90 pinball loss, equally averaged after
train-only per-horizon normalization:

```text
scale_h = max(IQR_h, 1.4826 * MAD_h, 1e-4)
selection_score = mean_h(normalized_pinball_h), h=h_start,...,14
```

The aggregate is also exposed at `primary_5d/selection_score` to preserve the
stable RunPod checkpoint-monitor path. It still covers every horizon.
Checkpoint mode is `min`. There is no classification loss or loss-weight search.

### 8. Comparator suite

All comparators read the same cutoff-inclusive asset/benchmark contexts and use
the same train subset and validation split:

- constant distributions: always-buy and zero-return
- rules: relative momentum/reversal, MA crossover, RSI, MACD, volatility-scaled
- traditional ML: per-horizon quantile GBDT
- causal DL: paired GRU, DLinear, compact PatchTST
- full model: Kronos-base LoRA plus dynamic benchmark conditioning

Rule residual quantiles, learned baselines, early stopping, and robust scales use
train labels only. Validation selects models/checkpoints; test never tunes them.

### 9. Validation metrics

Primary selection metric: all-horizon mean normalized pinball, lower is better.

For every horizon report raw/normalized pinball, q50 raw/normalized MAE, q50
Pearson correlation, q50 sign agreement, q10–q90 coverage, width, and coverage
error. Research diagnostics include date-level RankIC/IR, equal-weight
long/short return, Sharpe, drawdown, turnover, five-level signal distribution,
and slices by market, asset type, provider, and year.

Cross-sectional diagnostics are neither portfolio-model outputs nor checkpoint
selection metrics. Transaction costs must be explicit, and one Sharpe is not
proof of tradability.

### 10. Test policy

Automatic RunPod validation reads train and validation only and records
`test_unlocked=false`. Unlock test once after architecture, hyperparameters,
thresholds, data QA, and selection are frozen. If test results cause a model
change, that test became validation and the next cycle needs a new time range or
dataset version.

### 11. Checkpoint and reproducibility

An acceptable checkpoint is validation-selected, stores the trainable union,
optimizer/scheduler/RNG/resolved config, binds run/data/model/source revisions,
passes artifact and strict-key verification, and declares
`model_output_schema_version=5.0`. Old checkpoints fail closed. Confirm success
using terminal lifecycle, run manifest, best pointer, trainer state, and raw
validation JSON—not a W&B chart or README alone.

### 12. Minimum acceptance

Stage 1 uses exactly 15% of train, passes remote ruff/pytest, forward/backward,
validation-ranked save/reload, inference-schema smoke, and lifecycle
termination. It outputs ordered `[B,15-h_start,3]` alpha quantiles for
`h_start in {1,2,3}`, with no LLM, facts, or classifier.

Stage 2 uses the same architecture and pretrained revisions with 100% train,
evaluates all baselines under one protocol, persists per-horizon/aggregate/
cross-sectional/subgroup raw JSON, retains provenance, and keeps provider code
out of training.

Stage 1 proves script viability only. A single-seed Stage 2 is still a PoC.
Stronger claims require multiple seeds, dispersion, and predefined ablations.

### 13. Priority ablations

Without changing outputs or splits, test one factor at a time:

1. Gated benchmark conditioning versus asset-only (gate fixed to zero).
2. Kronos LoRA versus frozen Kronos plus trainable downstream modules.
3. Kronos versus compact PatchTST/DLinear with identical histories/horizons/loss.
4. Price-index benchmark input versus total-return-adjusted benchmark input.
5. CAPM abnormal-return diagnostic analysis without replacing the main label.

Each ablation needs a new run identity and must never overwrite a checkpoint
with different semantics.

### 14. Remote-validation boundary

Python environment, Poetry lock, lint, pytest, data preparation, model prefetch,
GPU training, and validation all run on RunPod. The local machine only edits
source/config, performs shell-level static checks, syncs S3, controls Pods, and
downloads artifacts. Local Python/PyTorch/CUDA state is not runtime evidence.
