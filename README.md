# 金融 OHLCV 時序基礎模型微調

## 中文

### 專案定位

本專案以美國與台灣股票、ETF 的日線 OHLCV 資料，微調金融領域預訓練的時序基礎模型。系統只處理數值時序：

- 輸入與輸出都是數值張量，不提供自然語言生成或事實重建功能。
- 不把外部 API 放進訓練迴圈。
- 只輸出 3–14 個持有交易日的連續 alpha 條件分布。
- 提供可供下游系統重用的數值 encoder 介面。

這是研究與能力驗證用的 PoC，不是投資建議、交易系統或可保證獲利的模型。

### 數值輸出契約

`MultiHorizonAlphaHead` 的唯一預測輸出是：

- `alpha_quantiles`: `[batch, 12, 3]`。
- 第二維依序是持有 3、4、…、14 個交易日。
- 第三維固定為 q10、q50、q90。
- 單位是商品相對其 benchmark 的 adjusted execution log return。

模型沒有 `forecast_logits`、分類 head、分類 loss 或方向機率。推論時可由每個
horizon 的 q10/q50/q90，使用固定閾值後處理成 `strong_bearish`、`bearish`、
`neutral`、`bullish`、`strong_bullish`；這些訊號不是額外訓練目標，也不會增加
loss 權重。Checkpoint 必須符合 `model_output_schema_version=4.0`；不相容的
output schema 會被拒絕載入。

### 模型架構

```text
Adjusted asset OHLCV through close t ─────┐
                                          ├── shared Kronos-base + LoRA
Adjusted benchmark OHLCV through close t ─┘              │
                                                         ▼
                                           Causal Perceiver Resampler
                                                         │
                                                         ▼
                                           gated benchmark cross-attention
                                                         │
                                                         ▼
                                     3–14d alpha q10/q50/q90 [B,12,3]
```

生產設定使用 `NeoQuasar/Kronos-base` 與
`NeoQuasar/Kronos-Tokenizer-base`。Kronos predictor 的基礎權重凍結，只在
`q_proj`、`k_proj`、`v_proj`、`out_proj`、`w1`、`w2`、`w3`
注入 LoRA；resampler、benchmark conditioner 與 alpha head 可訓練。官方 source 固定為 commit
`67b630e67f6a18c9e9be918d9b4337c960db1e9a`，preflight 與建模都會驗證實際
checkout，不只依賴 setup script 的宣告。模型與 tokenizer 權重另分別固定為
Hugging Face commits `2b554741eca47781b64468546e77fef3e85130e6` 與
`0e0117387f39004a9016484a186a908917e22426`，下載、離線 smoke test、config 與
checkpoint 都會綁定 revisions。

`QuantForecastModel.encode_ohlcv(...)` 顯式輸出：

- `last_hidden_state`
- `attention_mask`
- `latent_tokens`

下游系統可以在這些數值表示之後接入額外的數值模組或跨模態對齊模組，而不改變目前的 alpha 輸出契約。

模型直接學習條件 alpha 分布；不是先預測 raw-return q50 再減 benchmark q50。
benchmark 的歷史只透過動態 gated cross-attention 影響輸出。benchmark 在收盤 t
之後的資料只供離線 label construction 使用，不會出現在模型輸入。

### 為什麼選 Kronos-base

| 候選模型                         | 與金融 OHLCV 的證據                                                   | 本專案優勢                                                                    | 本專案主要限制                                                         | 決策                |
| -------------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------- |
| Kronos-base                      | 論文報告以超過 120 億筆、來自 45 個交易所的金融 K-line 記錄預訓練     | 領域與 OHLCV 高度吻合；公開權重；官方程式含微調流程；約 102M 參數適合單卡 PoC | 論文中的資料組成與比較主要由作者報告；context 上限 512                 | 採用                |
| TimesFM 2.5                      | 原始 TimesFM 語料以 Google Trends、Wikipedia pageviews 等通用序列為主 | 200M 參數、最長 16k context、成熟的 point/quantile forecasting 與 LoRA 範例   | 沒有足夠證據顯示預訓練以金融 K-line 為核心；OHLCV 多變量適配需額外設計 | 不作第一版 backbone |
| Chronos-Bolt / Chronos-2         | 通用公開與合成時序語料；不是金融專用語料的明確證據                    | Bolt 推論快、記憶體需求低；Chronos-2 支援多變量與 covariates；工具鏈成熟      | 領域吻合度低於 Kronos；換 backbone 不能只比較吞吐量                    | 保留為受控 baseline |
| MOIRAI-1.1-R / Moirai 2          | LOTSA 涵蓋九類領域、約 270 億 observations，但不是以金融 OHLCV 為主   | 原生多變量、不同頻率與任意 horizon；有完整微調工具                            | 領域專用性較弱；部分 checkpoint 授權限制需逐一確認                     | 不作第一版 backbone |
| PLUTUS / DELPHYNE 等金融時序模型 | 研究方向與金融相符                                                    | 可作後續研究參考                                                              | 公開權重、可重現微調鏈或與現有 head 的整合成熟度不足                   | 暫不採用            |

選擇 Kronos 不是因為它在所有時序任務都必然最好，而是因為本 PoC 的首要條件是「預訓練確實接觸大量金融 K-line」，同時還要能公開重現、在 RunPod 單卡微調，並保留數值 encoder。後續比較必須在相同資料切分、輸入長度、quant head 與評估指標下進行，避免同時改變多個變因。

### 資料來源與可選資料集

透過 `FIN_TS_DATASET_PROFILE` 或 `fin-ts-download --profile` 選擇資料：

| profile           | 實際來源              | 狀態             | 適用情境                  |
| ----------------- | --------------------- | ---------------- | ------------------------- |
| `tw_only`       | TWSE 官方 + TPEx 官方 | 可用             | 零美股 API 費用的研究路徑 |
| `us_only_eodhd` | EODHD 美國股票/ETF    | 可用             | 先驗證美股能力            |
| `us_tw_eodhd`   | EODHD + TWSE + TPEx   | 預設 PoC         | 美台跨市場完整 PoC        |
| `us_tw_massive` | Massive + TWSE + TPEx | 僅保留型別化介面 | 取得適合授權後再實作      |

EODHD 路徑預設可發現 active 與 delisted 美國股票/ETF，減少只保留存活標的造成的 survivorship bias。若費用或呼叫額度有限，可用 `--symbols`、`--etf-symbols` 或 `--symbol-limit` 縮小 universe。輸出 manifest 會列出 profile、實際 provider、market、symbol、asset type、日期範圍與每個 split 的樣本數。

EODHD 是 PoC 資料，不應被描述成交易所級真實行情。跨 provider 的 adjusted price、公司行動、delisted history、時區與資料修訂可能不同；正式比較前必須先做重疊標的抽樣對帳。

### 交易時間、benchmark 與調整資料契約

每筆樣本在交易日 `t` 收盤後產生訊號；下一個共同交易日的 regular-session raw
open 進場，該日算第 1 個持有交易日，持有 `h` 日時在第 `h` 個共同交易日的
raw close 出場。label 是商品與 benchmark 在完全相同 entry/exit timestamps 的
total-return log return 差，`h ∈ {3,…,14}`。

預設 benchmark policy：

- 美國股票與一般 ETF：`VTI.US`。
- TWSE 股票：`TAIEX.TW`，其 adjusted anchor 使用官方發行量加權股價報酬指數。
- TPEx 股票：`TPEX.TWO`，其 adjusted anchor 使用櫃買報酬指數。
- 無法合理映射的窄基、槓桿、反向、商品型或跨國 ETF 會 fail closed；只有窄幅
  allowlist 或明確的 `benchmark_mapping_path` 映射才進入訓練。

raw O/H/L/C/V 永久保留。模型視窗把 vendor/官方 total-return factor 正規化到
`cutoff_at`，再套用到歷史 O/H/L/C，因此收盤後推論不會因未來公司行動而回寫輸入；
volume 只依 split/share change 調整，不用現金股利調整。EODHD 保留
`adjusted_close`；2015-01-01 起的範圍只抓一次 exchange-wide split calendar，較早的
範圍則改用每個 symbol 的 Historical Splits API，因為 calendar 官方覆蓋只從 2015
年開始。兩種策略與預估/實際呼叫數都寫入 manifest，且在下載前受
`STAGE1_MAX_API_CALLS` 限制。台股使用 TWSE/TPEx 官方除權息資料與官方報酬指數。
這可避免股票分割或除權息造成的人為跳空，同時維持下一日 raw open 的可交易 entry
語意。

目前 raw OHLCV Parquet 會分批寫入，但 processed window 建立與訓練 dataset
仍會載入選定資料集到記憶體。第一次 PoC 應先用明確的 `--symbols`、
`--etf-symbols` 或 `--symbol-limit` 驗證容量，再逐步放大；不能因 API 額度足夠就
假設 CPU RAM 也能容納全美股與全台股的完整歷史。若要把全市場長歷史當成正式
Stage 2 範圍，下一步必須先把 processed windows 改為 partitioned、lazy
讀取；這項 out-of-core 改造不包含在目前 PoC。

### 離線資料管線

資料取得與模型訓練是兩個互斥階段：

```text
外部 API
  │
  ▼
不可變 raw JSON cache
  │
  ▼
canonical daily OHLCV Parquet + download-manifest.json
  │
  ▼
品質檢查、因果視窗、purge/embargo
  │
  ▼
processed windows.parquet + dataset-manifest.json
  │
  ▼
Stage 1 / Stage 2 訓練（完全離線）
```

資料下載器具備：

- provider-specific QPS throttle。
- 指數退避與有限次重試。
- 跨 provider、含 retry 的實際 network attempts 共用 `max_api_calls`
  fail-closed 上限；cache hit 不扣額度。
- API token 不進 cache key、request log 或 manifest。
- raw cache 與直接執行的下載／準備 CLI 拒絕靜默覆寫。
- Parquet 與 manifest 的 SHA-256、row count 與 provenance 綁定。

訓練器只接受落地的 `.parquet` 與 `state=ready` 的 dataset manifest；模型訓練、評估及推論程式不呼叫 EODHD、TWSE、TPEx 或 Massive。

### 遠端執行環境與本機邊界

本專案的 Python dependency resolution、Poetry environment、lint、pytest、資料準備、
模型 cache smoke test、訓練與驗證都在 RunPod 執行。本機只作為 control plane：編輯
source/config/`.env`、透過 shell script 上傳程式碼、建立或停止 Pod，以及下載 artifacts。

不得在本機為本專案執行 `poetry install`、`poetry lock`、pytest、Python preflight 或模型
載入，也不得建立或檢查本機 `.venv`。本機若殘留其他環境產生的 `poetry.lock`，它已被
`.gitignore` 與 source upload allowlist 排除，不是此專案的 runtime 證據。

RunPod workflow 會在 approved image 內使用 Python `>=3.12,<3.13`，建立 persistent
Poetry environment、重新產生 canonical `poetry.lock`，再執行 lint、完整 pytest 與後續
工作。修改 `pyproject.toml` 後只需重新同步 source，讓下一次遠端 CPU preparation 重新
解析 lock；不要在本機嘗試對齊 RunPod 的 Python/PyTorch/CUDA 環境。

### 下載並落地資料

標準流程是依後文操作 RunPod CPU preparation Pod；不要在本機直接執行資料 CLI。以下
命令只是已完成遠端環境設定後，在 RunPod CPU Pod 內除錯資料管線時使用的低階參考。
日期 `--end` 是 exclusive；EODHD token 應由 RunPod Secret 注入，不要在 shell history
手動 export 明文 token。

只使用台股官方資料：

```bash
poetry run fin-ts-download \
  --profile tw_only \
  --start 2010-01-01 \
  --end 2026-07-28 \
  --output data/raw/market.parquet
```

以小型美股 universe 驗證 EODHD：

```bash
poetry run fin-ts-download \
  --profile us_only_eodhd \
  --symbols AAPL MSFT \
  --etf-symbols SPY QQQ \
  --start 2010-01-01 \
  --end 2026-07-28 \
  --output data/raw/market.parquet
```

建立因果訓練視窗：

```bash
poetry run fin-ts-prepare \
  --input data/raw/market.parquet \
  --output data/processed/windows.parquet
```

若要建立新的資料版本，請使用新的資料根目錄或版本化檔名，不要覆寫既有 cache、
Parquet 或 manifest。RunPod CPU preparation wrapper 會把 staging artifacts 發布到
固定的 `DATA_ROOT` 路徑，因此重跑時必須明確指定新的、版本化 `DATA_ROOT`；不能把
舊 dataset 視為自動受保護。

### 兩階段訓練

| 項目                        | Stage 1                                                          | Stage 2                |
| --------------------------- | ---------------------------------------------------------------- | ---------------------- |
| 目的                        | 驗證資料、模型、loss、checkpoint、評估與 RunPod 腳本             | 完整資料微調與正式評估 |
| train 樣本                  | 依 market/asset strata 確定性配置的 15% target count             | 100% train split       |
| validation / test           | 完整保留                                                         | 完整保留               |
| 架構                        | Kronos-base + 同一組 LoRA + resampler + conditioner + alpha head | 完全相同               |
| 初始化                      | 原始 pretrained base                                             | 原始 pretrained base   |
| 是否接續 Stage 1 checkpoint | 否                                                               | 否                     |

設定檔：

- `configs/stage1_kronos_base_lora.yaml`
- `configs/stage2_kronos_base_lora.yaml`

兩份設定的 `model.architecture_digest()` 必須一致。Stage 1 的 15% 是確定性、分層且精確的 train subset，不會縮小 validation/test，也不能用「前 15% 資料」取代。

### RunPod 完整操作手冊

RunPod 操作流程涵蓋 Pod 建立、S3 同步、network volume、readiness marker、
supervisor、checkpoint、驗證與自動終止。資料準備、訓練與驗證都使用
quant-only 設定。

本專案目前**沒有部署 PostgreSQL、SQLite、向量資料庫或其他資料庫服務**。
下文的「遠端資料層」是 RunPod persistent network volume 上的 Parquet、
API raw cache、manifest 與模型 cache。CPU preparation Pod 負責建立這個
離線資料層；GPU Pod 只讀已落地資料，不會在訓練迴圈呼叫外部 API。

#### 1. 建立 RunPod 帳號資源與本機設定

本機控制端需要 `bash`、Python 3、AWS CLI、`curl` 與 `runpodctl`。此處的系統
Python 3 只供無第三方相依的 manifest/JSON control helper 使用，不代表建立、載入或
檢查本機專案 Python environment。RunPod 官方文件：

- [Network volumes](https://docs.runpod.io/storage/network-volumes)
- [S3-compatible API](https://docs.runpod.io/storage/s3-api)
- [RunPod Secrets](https://docs.runpod.io/pods/templates/secrets)
- [runpodctl](https://docs.runpod.io/runpodctl/overview)

在 RunPod Console 完成以下設定：

1. 在支援 S3-compatible API 的 datacenter 建立 persistent network volume。
   記下 volume ID 與 datacenter ID。CPU Pod、GPU Pod、S3 region 與 endpoint
   必須使用同一個 datacenter。
2. 建立 project-scoped RunPod API key，供本機建立與終止 Pod。
3. 另外建立 S3 API key。它與 RunPod API key 是不同的 credential，僅供本機
   上傳、查詢與下載 network volume 物件。
4. 建立下列 RunPod Secrets；名稱需與 `.env` 中的
   `RUNPOD_*_SECRET_NAME` 一致：

   - `huggingface_token`：必要，用來預抓固定 revision 的 Kronos model 與
     tokenizer。
   - `wandb_api_key`：必要，用於訓練與 validation tracking。
   - `eodhd_api_token`：只有 `us_only_eodhd` 或 `us_tw_eodhd` profile
     需要；`tw_only` 不需要。

建立本機設定檔並限制權限：

```bash
cp .env.example .env
chmod 600 .env
```

至少填入：

```text
RUNPOD_API_KEY=<project-scoped RunPod API key>
RUNPOD_NETWORK_VOLUME_ID=<network volume ID>
RUNPOD_DATACENTER_ID=<network volume datacenter>
RUNPOD_S3_ACCESS_KEY_ID=<S3 access key>
RUNPOD_S3_SECRET_ACCESS_KEY=<S3 secret key>
RUNPOD_S3_REGION=<same datacenter>
RUNPOD_S3_ENDPOINT=https://s3api-<lowercase-datacenter>.runpod.io/
WANDB_ENTITY=
RUNPOD_CONFIG=configs/stage1_kronos_base_lora.yaml
```

再選擇資料 profile 與 API 預算。成本受限的最小美股驗證範例：

```text
FIN_TS_DATASET_PROFILE=us_tw_eodhd
STAGE1_US_SYMBOLS=AAPL MSFT
STAGE1_US_ETF_SYMBOLS=SPY QQQ
STAGE1_SYMBOL_LIMIT=
STAGE1_DATA_START=2010-01-01
STAGE1_DATA_END=2026-07-27
STAGE1_MAX_API_CALLS=5000
STAGE1_EODHD_QPS=5
STAGE1_TAIWAN_QPS=0.5
```

若要完全避免美股 API 費用，改用
`FIN_TS_DATASET_PROFILE=tw_only`。若 EODHD profile 的美股 symbol 與 ETF
清單都留空，CPU preparation 會嘗試發現帳號可取得的完整 active/delisted
universe；第一次 PoC 不建議在沒有確認 RAM、API 額度與費用前這樣執行。
若 `STAGE1_DATA_START` 早於 2015-01-01，完整 split history 會讓每個美股 symbol
多一個 API request；若從 2015-01-01 起始，則整個美股 universe 共用一個 calendar
request。先用小型明確清單驗證預估呼叫數，再放大 universe。

`.env` 已被 `.gitignore` 排除。不要 `source .env`，專案 wrapper 會以 allowlist
讀取它；不要把 API key、token 或 secret value 寫進 README、config、shell
script 或提交紀錄。Pod 只會收到 RunPod Secret reference，不會收到本機的
account-level RunPod/S3 credential。

#### 2. 驗證 S3 並上傳程式碼

先執行 read-only S3 權限檢查，再預覽明確的上傳 allowlist：

```bash
bash scripts/verify_runpod_s3_access.sh
bash scripts/sync_project_to_runpod_volume.sh --dry-run
```

確認清單後才實際上傳，並驗證 remote code readiness：

```bash
bash scripts/sync_project_to_runpod_volume.sh --apply
bash scripts/verify_runpod_stage_readiness.sh --code-only
```

上傳器會掃描 allowlisted source、config、script、test、`README.md` 與
`pyproject.toml` 的 secret pattern，逐檔上傳並核對遠端大小，最後才發布
`lifecycle/stage1/code.json`。`.env`、cache、資料、checkpoint 與本機
artifact 不會上傳。`poetry.lock` 也不會上傳；它會依 approved RunPod image
的 Python/PyTorch/CUDA 環境在 network volume 上重新產生。

任何 allowlisted 程式碼或 config 修改後，都要重新執行 `--dry-run`、
`--apply` 與 readiness check。若資料 marker 綁定的是舊 code release，
還必須重跑 CPU preparation，不能略過 GPU gate。

#### 3. 遠端部署模型與離線資料層

Stage 1 第一次準備資料時，確認 `.env` 使用：

```text
RUNPOD_CONFIG=configs/stage1_kronos_base_lora.yaml
```

在本機建立短生命週期 CPU Pod：

```bash
bash scripts/create_runpod_cpu_pod.sh
```

指令會輸出 Pod ID、外部 hard-limit guard 與 SSH 後應執行的 workflow。
由 RunPod Console 的 Connect 頁面取得 SSH 命令。登入 Pod 後執行：

```bash
cd /runpod-volume/ts_multimodal_LLM
bash scripts/runpod_tmux_launch.sh cpu-prepare
```

如需即時查看，可 attach 到 tmux；離開時用 `Ctrl-b d`，不要停止 session：

```bash
tmux -L fin-ts-cpu-prepare attach -t fin-ts-cpu-prepare
```

`cpu-prepare` 會依序：

1. 建立 persistent directory layout、Poetry 2.4.0 與 remote Python 3.12
   `.venv`，並在 RunPod image 內產生 canonical `poetry.lock`。
2. 固定 Kronos source/model/tokenizer revisions，執行完整 pytest。
3. 依 `FIN_TS_DATASET_PROFILE`、symbol、日期、QPS 與 API call 上限下載資料；
   cache hit 不會再次呼叫 provider。
4. 建立並驗證下列 persistent artifacts：
   | 遠端路徑                                                | 內容                                              |
   | ------------------------------------------------------- | ------------------------------------------------- |
   | `/runpod-volume/data/api-cache/`                      | provider raw response cache                       |
   | `/runpod-volume/data/raw/market.parquet`              | canonical daily OHLCV                             |
   | `/runpod-volume/data/processed/windows.parquet`       | 因果訓練視窗                                      |
   | `/runpod-volume/data/download-manifest.json`          | 實際 provider、profile、symbols 與下載 provenance |
   | `/runpod-volume/data/dataset-manifest.json`           | split counts、hash 與資料契約                     |
   | `/runpod-volume/data/manifests/api-request-log.jsonl` | 不含 token 的 request audit                       |
   | `/runpod-volume/cache/huggingface/`                   | 離線 Kronos model/tokenizer cache                 |
   | `/runpod-volume/cache/hf-models.json`                 | 固定 model revisions 與 cache manifest            |
5. 只在全部檢查成功後發布
   `/runpod-volume/lifecycle/stage1/dataset.json`，再自動終止 Pod。

Pod 終止後，以 S3 lifecycle 為準，不要依賴已消失的 SSH session：

```bash
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://<network-volume-id>/lifecycle/stage1/dataset.json" - \
  --only-show-errors
```

若狀態是 `failed` 或 `timed_out`，先從 lifecycle 讀取 `launch_id` 與
`log_path`。tmux log 目錄固定為
`logs/tmux/fin-ts-cpu-prepare/<launch-id>/`，可下載到本機診斷目錄：

```bash
mkdir -p "runpod_error_log_temp/cpu-prepare/<launch-id>"
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://<network-volume-id>/logs/tmux/fin-ts-cpu-prepare/<launch-id>/" \
  "runpod_error_log_temp/cpu-prepare/<launch-id>/" \
  --recursive --only-show-errors
```

只有下列 gate 通過後才租用 GPU：

```bash
bash scripts/verify_runpod_stage_readiness.sh --gpu
```

Stage 2 使用同一份完整 Parquet，但以 100% train split 重新從相同 pretrained
base 開始，不接續 Stage 1 checkpoint。將 `.env` 改為：

```text
RUNPOD_CONFIG=configs/stage2_kronos_base_lora.yaml
```

若 source/config 有修改，先重新上傳。接著建立 CPU Pod，SSH 登入後執行
Stage 2 contract finalization：

```bash
bash scripts/create_runpod_cpu_pod.sh
```

```bash
cd /runpod-volume/ts_multimodal_LLM
bash scripts/runpod_tmux_launch.sh cpu-finalize
```

finalization 完成後，在本機再次執行
`bash scripts/verify_runpod_stage_readiness.sh --gpu`。這一步不重新動態下載訓練
資料；它重新驗證 Stage 2 config、完整資料、model cache 與 code release 的
一致性。

#### 4. 建立 GPU Pod 並訓練

先列出目前可用的完整 `gpuId`；預設是
`NVIDIA GeForce RTX 5090`：

```bash
bash scripts/runpodctl_project.sh gpu list
```

再次確認 `.env` 的 `RUNPOD_CONFIG` 是要執行的 stage，並通過 GPU gate：

```bash
bash scripts/verify_runpod_stage_readiness.sh --gpu
```

以預設 GPU 建立 Pod：

```bash
bash scripts/create_runpod_pod.sh
```

或指定清單中完整的 `gpuId`：

```bash
RUNPOD_GPU_ID="NVIDIA GeForce RTX 5090" \
  bash scripts/create_runpod_pod.sh
```

建立指令會在本機先配置唯一 run ID、掛載同一個 network volume、注入 W&B
Secret reference，並啟動獨立 hard-limit guard。由 Console SSH 登入後執行：

```bash
cd /runpod-volume/ts_multimodal_LLM
bash scripts/runpod_tmux_launch.sh stage1-train
```

`stage1-train` 是為了維持既有部署相容性的 workflow 名稱；實際 Stage 1 或
Stage 2 由 `RUNPOD_CONFIG` 決定。即時查看：

```bash
tmux -L fin-ts-stage1-train attach -t fin-ts-stage1-train
```

訓練會先驗證 image runtime、CUDA、mounted readiness、dataset/model
manifest 與 run identity，再寫入
`/runpod-volume/savedModel/<run-id>/`。兩份 production config 都設定
`validation.auto_run_after_training: true`，因此同一個 GPU workflow 會在訓練
完成後自動執行完整 validation benchmark；成功、失敗或超時後都會留下
lifecycle/artifact，再依既有 supervisor 與外部 guard 終止 Pod。

Pod 終止後，以 S3 查詢最新 terminal state：

```bash
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://<network-volume-id>/lifecycle/stage1/training.json" - \
  --only-show-errors
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://<network-volume-id>/lifecycle/stage1/validation.json" - \
  --only-show-errors
```

`state=ready` 才表示該 lifecycle 完成；`failed` 與 `timed_out` 必須視為未完成。
`wandb_run_id` 是 checkpoint、evaluation、W&B 與 run-scoped log 共用的
`<run-id>`。SSH/tmux 只用於仍存活 Pod 的即時除錯，不是 terminal state 的
權威來源。

若訓練已完成但需要獨立重跑 validation，可在本機建立 validation Pod：
先確認 `.env` 的 `RUNPOD_CONFIG` 與該 run 保存的 config 相同。

```bash
bash scripts/create_runpod_validation_pod.sh <run-id>
```

SSH 登入後執行：

```bash
cd /runpod-volume/ts_multimodal_LLM
bash scripts/runpod_tmux_launch.sh stage1-validate
```

#### 5. 下載 best checkpoint 與 validation 結果

以下命令會下載到已被 `.gitignore` 排除的 `artifacts/runpod/`。先把
`VOLUME_ID` 設成 `.env` 中相同的 network volume ID，下載 training lifecycle，
再從其中取得 run ID：

```bash
VOLUME_ID="<network-volume-id>"
DOWNLOAD_ROOT="artifacts/runpod"
mkdir -p "${DOWNLOAD_ROOT}"

bash scripts/runpod_s3_project.sh s3 cp \
  "s3://${VOLUME_ID}/lifecycle/stage1/training.json" \
  "${DOWNLOAD_ROOT}/training-lifecycle.json" \
  --only-show-errors

RUN_ID="$(python3 -c \
  'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["wandb_run_id"])' \
  "${DOWNLOAD_ROOT}/training-lifecycle.json")"
printf 'Run ID: %s\n' "${RUN_ID}"
mkdir -p "${DOWNLOAD_ROOT}/${RUN_ID}"
```

先下載 run manifest、leaderboard 與 best-checkpoint pointer：

```bash
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://${VOLUME_ID}/savedModel/${RUN_ID}/run-manifest.json" \
  "${DOWNLOAD_ROOT}/${RUN_ID}/run-manifest.json" \
  --only-show-errors
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://${VOLUME_ID}/savedModel/${RUN_ID}/checkpoint-leaderboard.json" \
  "${DOWNLOAD_ROOT}/${RUN_ID}/checkpoint-leaderboard.json" \
  --only-show-errors
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://${VOLUME_ID}/savedModel/${RUN_ID}/best-checkpoint.json" \
  "${DOWNLOAD_ROOT}/${RUN_ID}/best-checkpoint.json" \
  --only-show-errors

BEST_CHECKPOINT="$(python3 -c \
  'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["path"])' \
  "${DOWNLOAD_ROOT}/${RUN_ID}/best-checkpoint.json")"
printf 'Best checkpoint: %s\n' "${BEST_CHECKPOINT}"
```

只下載 validation-selected best checkpoint，不必下載全部 top-k：

```bash
mkdir -p "${DOWNLOAD_ROOT}/${RUN_ID}/${BEST_CHECKPOINT}"
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://${VOLUME_ID}/savedModel/${RUN_ID}/${BEST_CHECKPOINT}/" \
  "${DOWNLOAD_ROOT}/${RUN_ID}/${BEST_CHECKPOINT}/" \
  --recursive --only-show-errors
```

下載 validation benchmark、validation lifecycle 與 immutable training
completion record：

```bash
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://${VOLUME_ID}/evaluations/${RUN_ID}/validation-benchmark.json" \
  "${DOWNLOAD_ROOT}/${RUN_ID}/validation-benchmark.json" \
  --only-show-errors
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://${VOLUME_ID}/lifecycle/stage1/validation.json" \
  "${DOWNLOAD_ROOT}/${RUN_ID}/validation-lifecycle.json" \
  --only-show-errors
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://${VOLUME_ID}/lifecycle/runs/${RUN_ID}/training-completed.json" \
  "${DOWNLOAD_ROOT}/${RUN_ID}/training-completed.json" \
  --only-show-errors
```

確認本機檔案：

```bash
find "${DOWNLOAD_ROOT}/${RUN_ID}" -maxdepth 3 -type f -print | sort
```

best checkpoint 目錄至少應包含 `adapter.safetensors`、
`resolved-config.yaml`、`trainer-state.json` 與它所列出的 optimizer/scheduler
state。`validation-benchmark.json` 是完整數值 validation 與 baseline 比較；
不要只根據 W&B 畫面或 README 宣稱 run 成功，應同時檢查 lifecycle 的
`state`、run ID、result path 與本機下載的原始 JSON。

實際訓練 stage 由 `RUNPOD_CONFIG` 決定。操作命令、tmux session 或 lifecycle
路徑中的 `stage1-*` 名稱不會覆寫 config 所選擇的 stage。

### 訓練與推論產物

每個 run 至少保存：

- `adapter.safetensors`：LoRA、resampler、benchmark conditioner 與 alpha head 的可訓練權重。
- `resolved-config.yaml`
- `trainer-state.json`
- optimizer / scheduler state
- run manifest、checkpoint leaderboard 與 best-checkpoint pointer
- dataset manifest 摘要、architecture digest、Kronos source/model/tokenizer
  revisions 與 bounded training implementation digest
- validation metrics 與 baseline 比較

Checkpoint resume 只接受目前的 quant output schema，並且必須通過 RunPod run
identity、artifact integrity、validation-selection、資料、模型與訓練程式碼契約檢查；
任何不相容的 schema 都會 fail closed。

推論也必須在已建立專案 Poetry environment 且掛載相同 network volume 的 RunPod Pod
內執行，不在本機載入 checkpoint：

```bash
poetry run fin-ts-infer \
  --config configs/stage2_kronos_base_lora.yaml \
  --checkpoint /runpod-volume/savedModel/<run-id>/<checkpoint> \
  --input /runpod-volume/data/raw/market.parquet \
  --symbol AAPL.US
```

推論輸出只包含數值 forecast、資料 provenance、encoder shape 與 checkpoint metadata，不包含自然語言解釋。

### 驗收原則

PoC 最低驗收條件：

1. Stage 1 精確使用 15% train samples，完整 validation/test，能完成 forward/backward、checkpoint reload 與 inference smoke test。
2. Stage 2 與 Stage 1 architecture digest 相同，且由相同 pretrained base 重新開始。
3. 所有 run 都可追溯到 immutable Parquet、dataset profile、provider、symbols、日期範圍與 split counts。
4. `alpha_quantiles` 固定為 `[B,12,3]`，且沒有 classifier、文字或 fact 輸出。
5. 模型 input、時序切分與正規化不使用未來資訊；未來 benchmark 只存在於離線 label construction。
6. 模型至少與 zero-return、momentum、technical、GBDT 與 neural baselines 在同一 validation/test protocol 下比較。
7. 不只報告單一 aggregate loss；同時報告各 horizon 的 normalized pinball、median correlation、方向一致率、區間 coverage/width 與依市場、asset type、年份的切片。

Stage 1 只證明腳本與契約可運作，不用來宣稱模型具備 alpha。Stage 2 若只跑單一 seed，也只能視為 PoC 結果；要做較強的模型優劣主張，需增加多 seed 與受控 ablation。

### 主要資料與模型參考

- [Kronos 論文](https://arxiv.org/abs/2508.02739)
- [Kronos 官方程式庫](https://github.com/shiyu-coder/Kronos)
- [TimesFM 官方程式庫](https://github.com/google-research/timesfm)
- [Google Research: TimesFM](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
- [Chronos 官方程式庫](https://github.com/amazon-science/chronos-forecasting)
- [Chronos 論文](https://arxiv.org/abs/2403.07815)
- [Uni2TS / Moirai 官方程式庫](https://github.com/SalesforceAIResearch/uni2ts)
- [Moirai 論文](https://arxiv.org/abs/2402.02592)
- [EODHD EOD API](https://eodhd.com/financial-apis/api-for-historical-data-and-volumes)
- [EODHD split calendar API](https://eodhd.com/financial-apis/calendar-upcoming-earnings-ipos-and-splits)
- [EODHD Historical Splits API](https://eodhd.com/financial-apis/api-splits-dividends)
- [EODHD API limits](https://eodhd.com/financial-apis/api-limits)
- [EODHD pricing](https://eodhd.com/pricing)
- [TWSE OpenAPI](https://openapi.twse.com.tw/)
- [TWSE 除權除息計算說明](https://www.twse.com.tw/en/announcement/ex-right/twt49u.html)
- [TPEx OpenAPI](https://www.tpex.org.tw/openapi/)
- [TPEx 報酬指數](https://www.tpex.org.tw/web/stock/iNdex_info/reward_index/ROE.php?l=en-us)
- [Massive stocks pricing](https://massive.com/pricing?product=stocks)
- [Massive market-data terms](https://massive.com/legal/market-data-terms-of-service)

---

## English

### Project scope

This project fine-tunes a finance-pretrained time-series foundation model on
daily OHLCV data for US and Taiwan stocks and ETFs. The system processes only
numerical time series:

- Inputs and outputs are numerical tensors; no natural-language generation or
  fact-reconstruction interface is provided.
- The training loop never calls an external market-data API.
- It predicts continuous conditional alpha distributions for holding days 3–14.
- It exposes reusable numerical encoder representations for downstream systems.

This is a research and capability-validation PoC. It is not investment advice,
a production trading system, or a claim of guaranteed profitability.

### Numerical output contract

The only prediction emitted by `MultiHorizonAlphaHead` is:

- `alpha_quantiles`: `[batch, 12, 3]`.
- Dimension two contains holding periods 3, 4, ..., 14 trading days.
- Dimension three is fixed to q10, q50, and q90.
- Units are adjusted execution log return relative to the instrument's benchmark.

There is no `forecast_logits`, classification head, classification loss, or
direction probability. Inference may post-process each horizon's q10/q50/q90
with a fixed threshold into `strong_bearish`, `bearish`, `neutral`, `bullish`,
or `strong_bullish`. These signals are not extra training targets and introduce
no loss weights. Checkpoints must use `model_output_schema_version=4.0`;
incompatible output schemas fail closed.

### Architecture

```text
Adjusted asset OHLCV through close t ─────┐
                                          ├── shared Kronos-base + LoRA
Adjusted benchmark OHLCV through close t ─┘              │
                                                         ▼
                                           Causal Perceiver Resampler
                                                         │
                                                         ▼
                                           gated benchmark cross-attention
                                                         │
                                                         ▼
                                     3–14d alpha q10/q50/q90 [B,12,3]
```

Production configs use `NeoQuasar/Kronos-base` and
`NeoQuasar/Kronos-Tokenizer-base`. Kronos base weights are frozen. LoRA is
injected into `q_proj`, `k_proj`, `v_proj`, `out_proj`, `w1`, `w2`, and `w3`;
the resampler, benchmark conditioner, and alpha head remain trainable. The official source is pinned to
commit `67b630e67f6a18c9e9be918d9b4337c960db1e9a`. Preflight and model
construction verify the actual checkout instead of trusting the setup-script
declaration alone. Model and tokenizer weights are separately pinned to
Hugging Face commits `2b554741eca47781b64468546e77fef3e85130e6` and
`0e0117387f39004a9016484a186a908917e22426`; downloads, offline smoke tests,
configs, and checkpoints bind those revisions.

`QuantForecastModel.encode_ohlcv(...)` explicitly returns:

- `last_hidden_state`
- `attention_mask`
- `latent_tokens`

Downstream systems can attach additional numerical modules or multimodal
alignment modules after these representations without changing the current
alpha-output contract.

The model predicts the conditional alpha distribution directly; it does not
predict a raw-return q50 and subtract a benchmark q50. Historical benchmark
state affects predictions through dynamic gated cross-attention. Benchmark data
after close t is used only for offline label construction and never enters the
model input.

### Why Kronos-base

| Candidate                                      | Financial-OHLCV evidence                                                                                 | Advantages here                                                                                                       | Main limitations here                                                                                             | Decision                  |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------- |
| Kronos-base                                    | The paper reports pretraining on more than 12 billion financial K-line records from 45 exchanges         | Strong domain match, public checkpoints, an official fine-tuning path, and about 102M parameters for a single-GPU PoC | Corpus composition and comparisons are primarily author-reported; context is limited to 512                       | Selected                  |
| TimesFM 2.5                                    | The original TimesFM corpus is dominated by general series such as Google Trends and Wikipedia pageviews | 200M parameters, up to 16k context, mature point/quantile forecasting, and a LoRA example                             | Insufficient evidence that financial K-lines dominate pretraining; multivariate OHLCV needs additional adaptation | Not the first backbone    |
| Chronos-Bolt / Chronos-2                       | General public and synthetic time-series corpora rather than clearly finance-focused pretraining         | Bolt is fast and memory-efficient; Chronos-2 supports multivariate data and covariates; mature tooling                | Lower domain match than Kronos; throughput alone is not a fair backbone criterion                                 | Controlled baseline later |
| MOIRAI-1.1-R / Moirai 2                        | LOTSA spans about 27B observations and nine domains, but is not finance-OHLCV-centric                    | Native multivariate, frequency, and arbitrary-horizon support with a complete fine-tuning toolkit                     | Weaker domain specificity; checkpoint licensing must be checked individually                                      | Not the first backbone    |
| PLUTUS / DELPHYNE and related financial models | Research direction matches finance                                                                       | Useful future research references                                                                                     | Public weights, reproducible tuning paths, or integration maturity with the locked head are insufficient          | Deferred                  |

Kronos is selected because this PoC prioritizes demonstrable exposure to a
large financial K-line corpus while remaining reproducible, single-GPU
fine-tunable on RunPod, and usable as a numerical encoder. This is not a claim
that Kronos is universally superior. Later comparisons must hold the dataset,
splits, input length, quant head, and evaluation protocol constant.

### Data sources and selectable datasets

Select a dataset with `FIN_TS_DATASET_PROFILE` or
`fin-ts-download --profile`:

| profile           | Actual sources                | Status               | Use case                                  |
| ----------------- | ----------------------------- | -------------------- | ----------------------------------------- |
| `tw_only`       | Official TWSE + official TPEx | Available            | Research without US API cost              |
| `us_only_eodhd` | EODHD US stocks/ETFs          | Available            | Validate US-market capability first       |
| `us_tw_eodhd`   | EODHD + TWSE + TPEx           | Default PoC          | Full US/Taiwan PoC                        |
| `us_tw_massive` | Massive + TWSE + TPEx         | Typed interface only | Implement after obtaining suitable rights |

The EODHD path can discover both active and delisted US stocks/ETFs by default,
reducing survivorship bias. Use `--symbols`, `--etf-symbols`, or
`--symbol-limit` when budget or quota is constrained. Manifests record the
profile, actual providers, markets, symbols, asset types, date range, and
per-split sample counts.

EODHD is PoC data and must not be represented as an exchange-grade market feed.
Adjusted prices, corporate actions, delisted history, time zones, and revisions
can differ across providers. Sample reconciliation on overlapping instruments
is required before formal comparisons.

### Execution timing, benchmark, and adjusted-data contract

Each sample emits a signal after trading-day `t` closes. Entry occurs at the
next shared trading day's raw regular-session open, which counts as holding day
one. A horizon `h` exits at the raw close of the `h`th shared trading day. The
label is the difference between instrument and benchmark total-return log
returns over identical entry and exit timestamps, for `h ∈ {3,...,14}`.

Default benchmark policy:

- US stocks and ordinary ETFs: `VTI.US`.
- TWSE stocks: `TAIEX.TW`, whose adjusted anchor uses the official TAIEX total
  return index.
- TPEx stocks: `TPEX.TWO`, whose adjusted anchor uses the official TPEx return
  index.
- Narrow, leveraged, inverse, commodity, or cross-country ETFs fail closed
  unless covered by the narrow allowlist or an explicit `benchmark_mapping_path`.

Raw O/H/L/C/V is retained permanently. Model windows normalize each vendor or
official total-return factor to `cutoff_at` before applying it to historical
O/H/L/C, so future corporate actions cannot rewrite an after-close inference
input. Volume is adjusted only for splits/share changes, never for cash
dividends. EODHD retains `adjusted_close` and fetches the exchange-wide split
calendar once for ranges beginning on or after 2015-01-01. Earlier ranges use
the per-symbol Historical Splits API because the documented calendar coverage
starts in 2015. Both the strategy and estimated/actual request counts are
recorded in the manifest and bounded before download by
`STAGE1_MAX_API_CALLS`. Taiwan uses official TWSE/TPEx ex-right/ex-dividend
data and official return indices. This removes artificial corporate-action
gaps while retaining the tradable next-day raw-open entry semantics.

Raw OHLCV Parquet is written incrementally, but processed-window preparation
and the training dataset currently load the selected dataset into memory. For
the first PoC, validate capacity with explicit `--symbols`, `--etf-symbols`, or
`--symbol-limit` settings before scaling up. API quota alone does not imply that
CPU RAM can hold the complete US and Taiwan histories. A full-market,
long-history Stage 2 first requires partitioned, lazy processed windows; that
out-of-core change is outside the current PoC.

### Offline data pipeline

Acquisition and model training are separate phases:

```text
External APIs
  │
  ▼
Immutable raw JSON cache
  │
  ▼
Canonical daily OHLCV Parquet + download-manifest.json
  │
  ▼
Quality checks, causal windows, purge/embargo
  │
  ▼
processed windows.parquet + dataset-manifest.json
  │
  ▼
Stage 1 / Stage 2 training (fully offline)
```

The downloader provides:

- Provider-specific QPS throttling.
- Bounded retries with exponential backoff.
- One fail-closed `max_api_calls` cap shared by actual network attempts across
  providers, including retries; cache hits do not consume it.
- Cache identities, request logs, and manifests that exclude API tokens.
- A raw cache and direct download/preparation CLIs that refuse silent overwrite.
- SHA-256, row-count, and provenance bindings for artifacts.

Training accepts only materialized `.parquet` data and a dataset manifest whose
state is `ready`. Training, evaluation, and inference never call EODHD, TWSE,
TPEx, or Massive.

### Remote runtime and local boundary

Python dependency resolution, the Poetry environment, lint, pytest, data
preparation, model-cache smoke tests, training, and validation all run on
RunPod. The local machine is only a control plane: edit source, configs, and
`.env`; use shell scripts to upload code and create or stop Pods; and download
artifacts.

Do not run `poetry install`, `poetry lock`, pytest, Python preflight, or model
loading for this project on the local machine, and do not create or inspect a
local `.venv`. If a `poetry.lock` generated by another environment remains in
the working directory, `.gitignore` and the source-upload allowlist exclude it;
it is not runtime evidence for this project.

The RunPod workflow uses Python `>=3.12,<3.13` inside the approved image,
creates the persistent Poetry environment, regenerates the canonical
`poetry.lock`, and then runs lint, the complete pytest suite, and subsequent
work. After changing `pyproject.toml`, resync the source and let the next remote
CPU-preparation run resolve the lock. Do not try to reproduce the RunPod
Python/PyTorch/CUDA environment locally.

### Download and materialize data

The standard path is the RunPod CPU-preparation workflow documented below; do
not execute the data CLI locally. The following commands are low-level
references for debugging the data pipeline inside a RunPod CPU Pod after its
remote environment has been set up. `--end` is exclusive. RunPod Secrets must
inject the EODHD token; do not export a plaintext token into shell history.

Taiwan official data only:

```bash
poetry run fin-ts-download \
  --profile tw_only \
  --start 2010-01-01 \
  --end 2026-07-28 \
  --output data/raw/market.parquet
```

Small EODHD US validation universe:

```bash
poetry run fin-ts-download \
  --profile us_only_eodhd \
  --symbols AAPL MSFT \
  --etf-symbols SPY QQQ \
  --start 2010-01-01 \
  --end 2026-07-28 \
  --output data/raw/market.parquet
```

Create causal training windows:

```bash
poetry run fin-ts-prepare \
  --input data/raw/market.parquet \
  --output data/processed/windows.parquet
```

Create a new versioned data root or filenames for each dataset revision. Do not
overwrite an existing cache, Parquet file, or manifest. The RunPod CPU
preparation wrapper publishes staged artifacts to fixed paths under `DATA_ROOT`;
therefore, a rerun must explicitly use a new, versioned `DATA_ROOT`. Existing
datasets are not automatically protected by that wrapper.

### Two training stages

| Item                  | Stage 1                                                                 | Stage 2                                     |
| --------------------- | ----------------------------------------------------------------------- | ------------------------------------------- |
| Purpose               | Validate data, model, loss, checkpoints, evaluation, and RunPod scripts | Full-data fine-tuning and formal evaluation |
| Train samples         | Deterministic 15% target count allocated by market/asset strata         | 100% of the train split                     |
| Validation / test     | Fully retained                                                          | Fully retained                              |
| Architecture          | Kronos-base + the same LoRA + resampler + conditioner + alpha head      | Identical                                   |
| Initialization        | Original pretrained base                                                | Original pretrained base                    |
| Continue from Stage 1 | No                                                                      | No                                          |

Configs:

- `configs/stage1_kronos_base_lora.yaml`
- `configs/stage2_kronos_base_lora.yaml`

The two configs must have identical `model.architecture_digest()` values. Stage
1 uses an exact, deterministic, stratified 15% train subset. It does not shrink
validation/test and must not be approximated by taking the first 15% of data.

### Complete RunPod operations guide

The RunPod workflow covers Pod creation, S3 synchronization, network volumes,
readiness markers, supervision, checkpoints, validation, and automatic
termination. Data preparation, training, and validation use quant-only configs.

This project currently deploys **no PostgreSQL, SQLite, vector database, or
other database service**. The "remote data layer" below means Parquet, raw API
cache, manifests, and model cache on a persistent RunPod network volume. A CPU
preparation Pod builds this offline data layer. The GPU Pod reads materialized
data and never calls an external market-data API from the training loop.

#### 1. Create RunPod account resources and local configuration

The local control machine needs `bash`, Python 3, AWS CLI, `curl`, and
`runpodctl`. This system Python runs dependency-free manifest and JSON control
helpers only; it does not create, load, or validate a local project Python
environment. Official RunPod references:

- [Network volumes](https://docs.runpod.io/storage/network-volumes)
- [S3-compatible API](https://docs.runpod.io/storage/s3-api)
- [RunPod Secrets](https://docs.runpod.io/pods/templates/secrets)
- [runpodctl](https://docs.runpod.io/runpodctl/overview)

Complete these steps in the RunPod Console:

1. Create a persistent network volume in a datacenter that supports the
   S3-compatible API. Record the volume ID and datacenter ID. The CPU Pod, GPU
   Pod, S3 region, and endpoint must all use that datacenter.
2. Create a project-scoped RunPod API key for local Pod creation and
   termination.
3. Create a separate S3 API key. It is distinct from the RunPod API key and is
   used only by the local machine to upload, inspect, and download
   network-volume objects.
4. Create these RunPod Secrets. Their names must match the
   `RUNPOD_*_SECRET_NAME` entries in `.env`:

   - `huggingface_token`: required to prefetch the pinned Kronos model and
     tokenizer revisions.
   - `wandb_api_key`: required for training and validation tracking.
   - `eodhd_api_token`: required only by the `us_only_eodhd` and
     `us_tw_eodhd` profiles; it is not required by `tw_only`.

Create the local settings file and restrict its permissions:

```bash
cp .env.example .env
chmod 600 .env
```

At minimum, fill in:

```text
RUNPOD_API_KEY=<project-scoped RunPod API key>
RUNPOD_NETWORK_VOLUME_ID=<network volume ID>
RUNPOD_DATACENTER_ID=<network volume datacenter>
RUNPOD_S3_ACCESS_KEY_ID=<S3 access key>
RUNPOD_S3_SECRET_ACCESS_KEY=<S3 secret key>
RUNPOD_S3_REGION=<same datacenter>
RUNPOD_S3_ENDPOINT=https://s3api-<lowercase-datacenter>.runpod.io/
WANDB_ENTITY=
RUNPOD_CONFIG=configs/stage1_kronos_base_lora.yaml
```

Then select the dataset profile and API budget. A cost-bounded small US-market
validation example is:

```text
FIN_TS_DATASET_PROFILE=us_tw_eodhd
STAGE1_US_SYMBOLS=AAPL MSFT
STAGE1_US_ETF_SYMBOLS=SPY QQQ
STAGE1_SYMBOL_LIMIT=
STAGE1_DATA_START=2010-01-01
STAGE1_DATA_END=2026-07-27
STAGE1_MAX_API_CALLS=5000
STAGE1_EODHD_QPS=5
STAGE1_TAIWAN_QPS=0.5
```

Use `FIN_TS_DATASET_PROFILE=tw_only` to eliminate US API cost. If both US stock
and ETF lists are empty for an EODHD profile, CPU preparation attempts to
discover the complete active/delisted universe available to the account. Do not
do this for the first PoC without first confirming RAM, API quota, and cost.
When `STAGE1_DATA_START` is earlier than 2015-01-01, complete split history adds
one request per US symbol. A range beginning on or after 2015-01-01 shares one
calendar request across the US universe. Validate the estimated call count with
a small explicit universe before scaling out.

`.env` is excluded by `.gitignore`. Do not `source .env`; the project wrappers
read it through an allowlist. Never put API keys, tokens, or secret values in
the README, configs, shell scripts, or commit history. Pods receive RunPod
Secret references, not the local account-level RunPod or S3 credentials.

#### 2. Verify S3 and upload source code

Run the read-only S3 access check, then preview the explicit upload allowlist:

```bash
bash scripts/verify_runpod_s3_access.sh
bash scripts/sync_project_to_runpod_volume.sh --dry-run
```

After reviewing the list, upload it and verify remote code readiness:

```bash
bash scripts/sync_project_to_runpod_volume.sh --apply
bash scripts/verify_runpod_stage_readiness.sh --code-only
```

The uploader scans allowlisted source, configs, scripts, tests, `README.md`, and
`pyproject.toml` for secret patterns, uploads and size-checks every file, and
only then publishes `lifecycle/stage1/code.json`. It does not upload `.env`,
caches, data, checkpoints, or local artifacts. It also excludes `poetry.lock`;
the approved RunPod image regenerates that file for its Python/PyTorch/CUDA
environment on the network volume.

After any allowlisted source or config change, rerun `--dry-run`, `--apply`, and
the readiness check. If the dataset marker is bound to an older code release,
rerun CPU preparation as well. Never bypass the GPU gate.

#### 3. Deploy the model and offline data layer remotely

For the first Stage 1 data preparation, set:

```text
RUNPOD_CONFIG=configs/stage1_kronos_base_lora.yaml
```

Create a short-lived CPU Pod from the local machine:

```bash
bash scripts/create_runpod_cpu_pod.sh
```

The command prints the Pod ID, external hard-limit guard, and the workflow to
run after SSH login. Obtain the SSH command from the RunPod Console Connect
page. Inside the Pod, run:

```bash
cd /runpod-volume/ts_multimodal_LLM
bash scripts/runpod_tmux_launch.sh cpu-prepare
```

Attach for live observation if needed. Detach with `Ctrl-b d`; do not stop the
session:

```bash
tmux -L fin-ts-cpu-prepare attach -t fin-ts-cpu-prepare
```

`cpu-prepare` performs the following sequence:

1. Creates the persistent directory layout, Poetry 2.4.0, and the remote Python
   3.12 `.venv`, then generates the canonical `poetry.lock` inside the RunPod
   image.
2. Pins the Kronos source/model/tokenizer revisions and runs the complete pytest
   suite.
3. Downloads according to `FIN_TS_DATASET_PROFILE`, symbols, dates, QPS, and
   the API-call cap. Cache hits do not call the provider again.
4. Creates and verifies these persistent artifacts:
   | Remote path                                             | Contents                                                    |
   | ------------------------------------------------------- | ----------------------------------------------------------- |
   | `/runpod-volume/data/api-cache/`                      | Provider raw-response cache                                 |
   | `/runpod-volume/data/raw/market.parquet`              | Canonical daily OHLCV                                       |
   | `/runpod-volume/data/processed/windows.parquet`       | Causal training windows                                     |
   | `/runpod-volume/data/download-manifest.json`          | Actual providers, profile, symbols, and download provenance |
   | `/runpod-volume/data/dataset-manifest.json`           | Split counts, hashes, and data contract                     |
   | `/runpod-volume/data/manifests/api-request-log.jsonl` | Request audit without tokens                                |
   | `/runpod-volume/cache/huggingface/`                   | Offline Kronos model/tokenizer cache                        |
   | `/runpod-volume/cache/hf-models.json`                 | Pinned model revisions and cache manifest                   |
5. Publishes `/runpod-volume/lifecycle/stage1/dataset.json` only after all
   checks pass, then terminates the Pod automatically.

After the Pod terminates, use the S3 lifecycle as the authority instead of the
now-unavailable SSH session:

```bash
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://<network-volume-id>/lifecycle/stage1/dataset.json" - \
  --only-show-errors
```

If the state is `failed` or `timed_out`, first read `launch_id` and `log_path`
from the lifecycle. The tmux log directory is
`logs/tmux/fin-ts-cpu-prepare/<launch-id>/`; download it into the local
diagnostic directory:

```bash
mkdir -p "runpod_error_log_temp/cpu-prepare/<launch-id>"
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://<network-volume-id>/logs/tmux/fin-ts-cpu-prepare/<launch-id>/" \
  "runpod_error_log_temp/cpu-prepare/<launch-id>/" \
  --recursive --only-show-errors
```

Do not rent a GPU until this gate passes:

```bash
bash scripts/verify_runpod_stage_readiness.sh --gpu
```

Stage 2 uses the same complete Parquet dataset, but trains on 100% of the train
split from the same pretrained base. It does not continue from a Stage 1
checkpoint. Change `.env` to:

```text
RUNPOD_CONFIG=configs/stage2_kronos_base_lora.yaml
```

If source or config changed, upload it first. Then create a CPU Pod and run the
Stage 2 contract finalization after SSH login:

```bash
bash scripts/create_runpod_cpu_pod.sh
```

```bash
cd /runpod-volume/ts_multimodal_LLM
bash scripts/runpod_tmux_launch.sh cpu-finalize
```

After finalization, rerun
`bash scripts/verify_runpod_stage_readiness.sh --gpu` locally. This step does
not dynamically redownload training data; it revalidates the Stage 2 config,
complete dataset, model cache, and code release as one contract.

#### 4. Create a GPU Pod and train

List the currently available complete `gpuId` values. The default is
`NVIDIA GeForce RTX 5090`:

```bash
bash scripts/runpodctl_project.sh gpu list
```

Confirm that `RUNPOD_CONFIG` in `.env` selects the intended stage and pass the
GPU gate:

```bash
bash scripts/verify_runpod_stage_readiness.sh --gpu
```

Create a Pod with the default GPU:

```bash
bash scripts/create_runpod_pod.sh
```

Or specify one complete `gpuId` from the list:

```bash
RUNPOD_GPU_ID="NVIDIA GeForce RTX 5090" \
  bash scripts/create_runpod_pod.sh
```

The creator first allocates one run ID locally, mounts the same network volume,
injects a W&B Secret reference, and arms an independent hard-limit guard. SSH
through the Console and run:

```bash
cd /runpod-volume/ts_multimodal_LLM
bash scripts/runpod_tmux_launch.sh stage1-train
```

`RUNPOD_CONFIG` determines whether Stage 1 or Stage 2 runs; the
`stage1-train` command name does not override that selection. Attach for live
observation:

```bash
tmux -L fin-ts-stage1-train attach -t fin-ts-stage1-train
```

Training verifies the image runtime, CUDA, mounted readiness, dataset/model
manifests, and run identity before writing
`/runpod-volume/savedModel/<run-id>/`. Both production configs set
`validation.auto_run_after_training: true`, so the same GPU workflow
automatically executes the complete validation benchmark after training.
Whether it succeeds, fails, or times out, the lifecycle and artifacts are
persisted before the existing supervisor and external guard terminate the Pod.

After termination, query the latest terminal states through S3:

```bash
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://<network-volume-id>/lifecycle/stage1/training.json" - \
  --only-show-errors
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://<network-volume-id>/lifecycle/stage1/validation.json" - \
  --only-show-errors
```

Only `state=ready` means that lifecycle completed. Treat `failed` and
`timed_out` as incomplete. `wandb_run_id` is the shared `<run-id>` for
checkpoints, evaluations, W&B, and run-scoped logs. SSH/tmux is only for live
debugging while a Pod still exists; it is not the authority for terminal state.

If training completed but validation must be rerun independently, create a
validation Pod locally. First confirm that `RUNPOD_CONFIG` in `.env` matches the
config stored by that run.

```bash
bash scripts/create_runpod_validation_pod.sh <run-id>
```

After SSH login, run:

```bash
cd /runpod-volume/ts_multimodal_LLM
bash scripts/runpod_tmux_launch.sh stage1-validate
```

#### 5. Download the best checkpoint and validation results

The following commands download into `artifacts/runpod/`, which `.gitignore`
excludes. Set `VOLUME_ID` to the same network-volume ID used in `.env`, download
the training lifecycle, and read the run ID from it:

```bash
VOLUME_ID="<network-volume-id>"
DOWNLOAD_ROOT="artifacts/runpod"
mkdir -p "${DOWNLOAD_ROOT}"

bash scripts/runpod_s3_project.sh s3 cp \
  "s3://${VOLUME_ID}/lifecycle/stage1/training.json" \
  "${DOWNLOAD_ROOT}/training-lifecycle.json" \
  --only-show-errors

RUN_ID="$(python3 -c \
  'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["wandb_run_id"])' \
  "${DOWNLOAD_ROOT}/training-lifecycle.json")"
printf 'Run ID: %s\n' "${RUN_ID}"
mkdir -p "${DOWNLOAD_ROOT}/${RUN_ID}"
```

Download the run manifest, leaderboard, and best-checkpoint pointer first:

```bash
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://${VOLUME_ID}/savedModel/${RUN_ID}/run-manifest.json" \
  "${DOWNLOAD_ROOT}/${RUN_ID}/run-manifest.json" \
  --only-show-errors
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://${VOLUME_ID}/savedModel/${RUN_ID}/checkpoint-leaderboard.json" \
  "${DOWNLOAD_ROOT}/${RUN_ID}/checkpoint-leaderboard.json" \
  --only-show-errors
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://${VOLUME_ID}/savedModel/${RUN_ID}/best-checkpoint.json" \
  "${DOWNLOAD_ROOT}/${RUN_ID}/best-checkpoint.json" \
  --only-show-errors

BEST_CHECKPOINT="$(python3 -c \
  'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["path"])' \
  "${DOWNLOAD_ROOT}/${RUN_ID}/best-checkpoint.json")"
printf 'Best checkpoint: %s\n' "${BEST_CHECKPOINT}"
```

Download only the validation-selected best checkpoint instead of every retained
top-k checkpoint:

```bash
mkdir -p "${DOWNLOAD_ROOT}/${RUN_ID}/${BEST_CHECKPOINT}"
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://${VOLUME_ID}/savedModel/${RUN_ID}/${BEST_CHECKPOINT}/" \
  "${DOWNLOAD_ROOT}/${RUN_ID}/${BEST_CHECKPOINT}/" \
  --recursive --only-show-errors
```

Download the validation benchmark, validation lifecycle, and immutable training
completion record:

```bash
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://${VOLUME_ID}/evaluations/${RUN_ID}/validation-benchmark.json" \
  "${DOWNLOAD_ROOT}/${RUN_ID}/validation-benchmark.json" \
  --only-show-errors
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://${VOLUME_ID}/lifecycle/stage1/validation.json" \
  "${DOWNLOAD_ROOT}/${RUN_ID}/validation-lifecycle.json" \
  --only-show-errors
bash scripts/runpod_s3_project.sh s3 cp \
  "s3://${VOLUME_ID}/lifecycle/runs/${RUN_ID}/training-completed.json" \
  "${DOWNLOAD_ROOT}/${RUN_ID}/training-completed.json" \
  --only-show-errors
```

Inspect the local files:

```bash
find "${DOWNLOAD_ROOT}/${RUN_ID}" -maxdepth 3 -type f -print | sort
```

The best-checkpoint directory should contain at least `adapter.safetensors`,
`resolved-config.yaml`, `trainer-state.json`, and the optimizer/scheduler state
listed by that trainer state. `validation-benchmark.json` is the complete
numerical validation and baseline comparison. Do not claim run success from a
W&B chart or README alone; inspect the lifecycle `state`, run ID, result path,
and downloaded raw JSON together.

The selected stage is controlled by `RUNPOD_CONFIG`. Names containing
`stage1-*` in operation commands, tmux sessions, or lifecycle paths do not
override the selected config.

### Training and inference artifacts

Each run stores at least:

- `adapter.safetensors`: trainable LoRA, resampler, benchmark-conditioner, and alpha-head weights.
- `resolved-config.yaml`
- `trainer-state.json`
- Optimizer and scheduler state
- Run manifest, checkpoint leaderboard, and best-checkpoint pointer
- Dataset-manifest summary, architecture digest, Kronos source/model/tokenizer
  revisions, and bounded training-implementation digest
- Validation metrics and baseline comparisons

Checkpoint resume accepts only the current quant output schema and must pass
the RunPod run-identity, artifact-integrity, validation-selection, dataset,
model, and training-source contract checks. Any incompatible schema fails
closed.

Inference also runs inside a RunPod Pod with the project Poetry environment and
the same network volume mounted; do not load the checkpoint locally:

```bash
poetry run fin-ts-infer \
  --config configs/stage2_kronos_base_lora.yaml \
  --checkpoint /runpod-volume/savedModel/<run-id>/<checkpoint> \
  --input /runpod-volume/data/raw/market.parquet \
  --symbol AAPL.US
```

Inference returns only numerical forecasts, data provenance, encoder shapes, and
checkpoint metadata. It produces no natural-language explanation.

### Acceptance principles

Minimum PoC acceptance:

1. Stage 1 uses exactly 15% of train samples, retains full validation/test, and
   completes forward/backward, checkpoint reload, and inference smoke tests.
2. Stage 2 has the same architecture digest and starts again from the same
   pretrained base.
3. Every run is traceable to immutable Parquet, dataset profile, providers,
   symbols, date range, and split counts.
4. `alpha_quantiles` remains fixed at `[B,12,3]`, with no classifier, text, or
   fact output.
5. Model inputs, time splits, and normalization use no future information;
   future benchmark values exist only in offline label construction.
6. The model is compared with zero-return, momentum, technical, GBDT, and neural
   baselines under one validation/test protocol.
7. Reports include per-horizon normalized pinball, median correlation,
   directional agreement, interval coverage/width, and slices by market, asset
   type, and year—not only one aggregate loss.

Stage 1 proves script and contract viability, not alpha. A single-seed Stage 2
run remains a PoC result; stronger model-comparison claims require multiple
seeds and controlled ablations.

### Primary model and data references

- [Kronos paper](https://arxiv.org/abs/2508.02739)
- [Official Kronos repository](https://github.com/shiyu-coder/Kronos)
- [Official TimesFM repository](https://github.com/google-research/timesfm)
- [Google Research: TimesFM](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
- [Official Chronos repository](https://github.com/amazon-science/chronos-forecasting)
- [Chronos paper](https://arxiv.org/abs/2403.07815)
- [Official Uni2TS / Moirai repository](https://github.com/SalesforceAIResearch/uni2ts)
- [Moirai paper](https://arxiv.org/abs/2402.02592)
- [EODHD EOD API](https://eodhd.com/financial-apis/api-for-historical-data-and-volumes)
- [EODHD split calendar API](https://eodhd.com/financial-apis/calendar-upcoming-earnings-ipos-and-splits)
- [EODHD Historical Splits API](https://eodhd.com/financial-apis/api-splits-dividends)
- [EODHD API limits](https://eodhd.com/financial-apis/api-limits)
- [EODHD pricing](https://eodhd.com/pricing)
- [TWSE OpenAPI](https://openapi.twse.com.tw/)
- [TWSE ex-right/ex-dividend calculation](https://www.twse.com.tw/en/announcement/ex-right/twt49u.html)
- [TPEx OpenAPI](https://www.tpex.org.tw/openapi/)
- [TPEx return index](https://www.tpex.org.tw/web/stock/iNdex_info/reward_index/ROE.php?l=en-us)
- [Massive stocks pricing](https://massive.com/pricing?product=stocks)
- [Massive market-data terms](https://massive.com/legal/market-data-terms-of-service)
