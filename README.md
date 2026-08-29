# 金融 OHLCV 時序基礎模型微調

## 中文

### 專案定位

本專案以美國與台灣普通股、ADR／TDR，以及經稽核且可映射的非槓桿股票型 ETF
日線 OHLCV 資料，微調金融領域預訓練的時序基礎模型。系統只處理數值時序：

- 輸入與輸出都是數值張量，不提供自然語言生成或事實重建功能。
- 不把外部 API 放進訓練迴圈。
- 只輸出從可設定的 `h_start`（1、2 或 3）到固定第 14 個持有交易日的連續
  alpha 條件分布。
- 提供可供下游系統重用的數值 encoder 介面。

這是研究與能力驗證用的 PoC，不是投資建議、交易系統或可保證獲利的模型。

### 數值輸出契約

`MultiHorizonAlphaHead` 的唯一預測輸出是：

- `alpha_quantiles`: `[batch, 15-h_start, 3]`。
- 第二維依序是持有 `h_start`、`h_start+1`、…、14 個交易日；`h_start`
  只能是 1、2 或 3，預設為 3。
- 第三維固定為 q10、q50、q90。
- 單位是商品相對其 benchmark 的 adjusted execution log return。

模型沒有 `forecast_logits`、分類 head、分類 loss 或方向機率。推論時可由每個
horizon 的 q10/q50/q90，使用固定閾值後處理成 `strong_bearish`、`bearish`、
`neutral`、`bullish`、`strong_bullish`；這些訊號不是額外訓練目標，也不會增加
loss 權重。Checkpoint 必須符合 `model_output_schema_version=5.0`；不相容的
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
                         h_start–14d alpha q10/q50/q90 [B,15-h_start,3]
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

正常 RunPod workflow 透過 `bash scripts/runpod_workflow.sh configure` 選擇
profile；`FIN_TS_DATASET_PROFILE` 是腳本驗證 selection 後傳入 Pod 的內部值：

| profile           | 實際來源              | 狀態             | 適用情境                  |
| ----------------- | --------------------- | ---------------- | ------------------------- |
| `tw_only`       | TWSE 官方 + TPEx 官方 | 可用             | 零美股 API 費用的研究路徑 |
| `us_only_eodhd` | EODHD 美國股票/ETF    | 可用             | 先驗證美股能力            |
| `us_tw_eodhd`   | EODHD + TWSE + TPEx   | 預設 PoC         | 美台跨市場完整 PoC        |
| `us_tw_massive` | Massive + TWSE + TPEx | 僅保留型別化介面 | 取得適合授權後再實作      |

EODHD 路徑預設可發現 active 與 delisted 美國股票/ETF，減少只保留存活標的造成的 survivorship bias。若費用或呼叫額度有限，可在 `configure` 使用 `--universe explicit` 搭配 `--stocks`、`--etfs`，或在 all 模式使用 `--symbol-limit` 縮小 universe。輸出 manifest 會列出 profile、實際 provider、market、symbol、asset type、日期範圍與每個 split 的樣本數。

`--universe all` 的 discovery 是準備當下 EODHD 回傳的 active/delisted 清單，並非
每個歷史交易日各自重建的 point-in-time constituents。以 `--start 2005-01-01
--end 2026-04-30` 為例，日期契約是 `[2005-01-01, 2026-04-30)`：區間中途上市的
商品只會從 provider 可取得的第一個交易日開始，區間中途下市的商品只會保留到最後
可取得日；同時在區間內上市又下市的商品，只要 EODHD delisted discovery 有回傳且
帳戶有權限，就會納入。`explicit` 只處理明列的 ticker；使用 `--symbol-limit` 時則只
處理限制後的子集。raw row 的 `is_active` 是 discovery 當下狀態，不是逐日上市狀態。

EODHD 官方另註明：2018 年後下市的商品可取得 EOD、fundamentals、dividends 與
splits；2018 年前下市的商品只保證 EOD。因此這些舊下市商品的 EOD rows 仍可能進入
raw／training windows，但 split-adjusted volume 的輔助資料覆蓋不能視為完整。下載
manifest 會列出 `delisted_pre_2018_auxiliary_coverage_warning` 的數量與 symbols，避免
把「有價格歷史」誤寫成「所有公司行動資料都完整」。

EODHD 是 PoC 資料，不應被描述成交易所級真實行情。跨 provider 的 adjusted price、公司行動、delisted history、時區與資料修訂可能不同；正式比較前必須先做重疊標的抽樣對帳。

### 交易時間、benchmark 與調整資料契約

每筆樣本在交易日 `t` 收盤後產生訊號；下一個共同交易日的 regular-session raw
open 進場，該日算第 1 個持有交易日，持有 `h` 日時在第 `h` 個共同交易日的
raw close 出場。label 是商品與 benchmark 在完全相同 entry/exit timestamps 的
total-return log return 差，`h ∈ {h_start,…,14}`，其中 `h_start ∈ {1,2,3}`。

預設 benchmark policy：

- 美國普通股、ADR 與白名單股票型 ETF：`VTI.US`。
- TWSE 普通股、TDR 與白名單股票型 ETF：`TAIEX.TW`，其 adjusted anchor 使用官方發行量加權股價報酬指數。
- TPEx 普通股：`TPEX.TWO`，其 adjusted anchor 使用櫃買報酬指數。
- 只有經稽核白名單內、可映射到既定 benchmark 的非槓桿股票型 ETF 才進入訓練；
  槓桿、反向、債券、商品、波動率與未稽核 ETF 一律 fail closed。
  `benchmark_mapping_path` 只能改變白名單 ETF 的 benchmark，不能擴張訓練 universe。

`VTI.US` 是美國商品建立 benchmark-relative label 與 benchmark context 的必要資料
依賴，不是 `--symbol-limit` 的一般候選商品，也不會成為自己的訓練 target
（`self_benchmark` 會排除它）。因此限制是在 ETF 與 stock 各自選完 N 檔後才確認
VTI：若已選到便不重複，否則額外補入。這讓 N 個 ETF target candidates 不會被
benchmark 占掉一席；raw universe 最多是 `N ETF + N stock + 1 VTI`，但實際可訓練
target 數仍可能因資料長度、benchmark mapping 或品質 gate 而更少。

raw O/H/L/C 永久保留；台灣 `volume` 也是官方 raw field。EODHD 官方定義的
`volume` 已做 split adjustment，因此管線用完整 Historical Splits response 反推出
當時的未調整 `volume`，並把 vendor 值保留為 `split_adjusted_volume`，不會再乘一次
split factor。模型視窗把 vendor/官方 total-return factor 正規化到
`cutoff_at`，再套用到歷史 O/H/L/C，因此收盤後推論不會因未來公司行動而回寫輸入；
volume 只依 split/share change 調整，不用現金股利調整。EODHD 保留
`adjusted_close`；所有日期範圍都對每個 symbol 使用 Historical Splits API。官方將
這個 endpoint 列入 EOD Historical Data — All World 且每個 request 為 1 API call；
管線不使用另屬 Calendar 產品的 `calendar/splits`。每個 symbol 因此需要一個 EOD
history request 加一個 split-history request，兩者都可 cache／續傳，也會列入資訊性
request 估算；估算值不會阻止完整資料集執行。provider 對 2018 年前下市商品的上述
輔助覆蓋例外則依前述 warning 顯式保留。台股使用 TWSE/TPEx 官方除權息資料與官方
報酬指數，並從既有月度官方 benchmark rows 取得實際交易日，不會把一般週一至週五
一律當成開市日。
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

`train`、`validation`、`test` 依全市場共用的交易日期做 70%／15%／15%
時間順序切分，不做隨機 row split。Train 與 validation 樣本的最晚
`label.end_at` 必須嚴格早於下一個 split boundary；20 個交易日 purge 與 14 個
交易日 effective embargo 之外，程式另有直接的 label-boundary guard，避免未來
調整 horizon 或間隔參數時讓 ground truth 跨界。

資料下載器具備：

- provider-specific QPS throttle。
- EODHD、TWSE、TPEx 以獨立迴圈平行抓取；retryable request 使用指數退避。
- 完整計畫的預估 HTTP requests 只作資訊與容量規劃；台灣最終 plan 使用官方
  benchmark sessions，另保留 pre-calendar weekday upper bound；兩者都不作
  dataset admission gate。
- `max_api_calls` 只限制單次 CPU attempt 的 EODHD network attempts（含 retry）；
  TWSE／TPEx 不受 request-count 上限限制。EODHD、TWSE 與 TPEx 都以同一個預設
  1 分鐘的 `--maxBackoff` 作為各自的退避退出邊界。cache hit 不扣額度，未完成時
  保留進度並由下一個 Pod 續接。
- `--max-api-calls`、`--eodhd-qps`、`--taiwan-qps` 與 `--maxBackoff` 是每次 CPU Pod
  launch 的 acquisition policy，由 `runpod_workflow.sh cpu prepare` 設定；它們不屬於
  immutable dataset selection，也不會改變 dataset request identity。
- 任一 provider 先退出都不會取消其他 provider；只有全部 provider 迴圈退出後，流程才
  發布續傳狀態或固定順序合併 provider-local artifacts。
- Dataset contract 改版或日期改變會建立新的 immutable namespace；新 namespace
  可唯讀命中其他 dataset namespace 中相同 cache revision 與 request identity 的 raw JSON cache，
  但新的 Parquet、manifest 與 progress 只會寫入自己的 namespace。
- QPS 與 `max_api_calls` 都不是 provider 的每日／每週 quota，也不代表 EODHD
  不同 endpoint 的計費 call units。
- 暫時性 provider 錯誤、429、單次 request budget 或 acquisition time budget
  耗盡時，保留成功的 raw responses、寫入 `download-progress.json`，並允許下一個
  CPU Pod 只補未快取的 requests。
- CPU workflow 預設保留 max runtime 的 25% 給 canonical data cleaning 與 causal
  window construction（最多 2 小時，亦可用 `--prepareReserve` 明確設定）；下載完成的 raw Parquet、request log
  與 download manifest 會先發布成 durable `downloaded` checkpoint。後續 Pod 可完全
  跳過 API，只有 processed data 與 readiness 都驗證通過才成為 `ready`。
- API token 不進 cache key、request log 或 manifest。
- raw cache 與直接執行的下載／準備 CLI 拒絕靜默覆寫。
- Parquet 與 manifest 的 SHA-256、row count 與 provenance 綁定。

訓練器只接受落地的 `.parquet` 與 `state=ready` 的 dataset manifest；模型訓練、評估及推論程式不呼叫 EODHD、TWSE、TPEx 或 Massive。

### 遠端執行環境與本機邊界

本專案的 Python dependency resolution、Poetry environment、lint、pytest、資料準備、
模型 cache smoke test、訓練與驗證都在 RunPod 執行。本機只作為 control plane：編輯
source，並透過 workflow script 管理 credentials、selection、上傳、Pod lifecycle 與
artifacts；不手動編輯 `.env`、YAML 或 JSON 設定。

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

每個 dataset request 會自動映射到
`/runpod-volume/datasets/<dataset-request-sha256>/`。profile、日期、universe、
symbol limit 或處理契約變動時會使用新的根目錄，不需要手動指定 `DATA_ROOT`，也不會
把不同範圍的資料誤當成同一份 dataset。

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

兩份設定的 `config.model_architecture_digest()` 必須一致；此 digest 同時綁定模型參數與
`h_start`／輸出 horizon 契約。Stage 1 的 15% 是確定性、分層且精確的 train subset，
不會縮小 validation/test，也不能用「前 15% 資料」取代。

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

在 RunPod Console 建立 project-scoped RunPod API key 與另一組 S3 API key，
再建立下列固定名稱的 RunPod Secrets：

   - `huggingface_token`：必要，用來預抓固定 revision 的 Kronos model 與
     tokenizer。
   - `wandb_api_key`：必要，用於訓練與 validation tracking。
   - `eodhd_api_token`：只有 `us_only_eodhd` 或 `us_tw_eodhd` profile
     需要；`tw_only` 不需要。

包含台灣市場的 profile 另需 TPEx Cloudflare Worker。Worker 部署腳本會為每次
部署建立唯一名稱的 `tpex_proxy_token_<timestamp>_<nonce>` RunPod Secret，並把
secret 名稱寫回本機 `.env`；不要手動建立固定名稱的 TPEx secret。

不要複製、開啟或手動修改 `.env`。以隱藏輸入方式建立 credential-only
`.env`；腳本會原子寫入並固定權限為 `600`：

```bash
bash scripts/runpod_workflow.sh credentials
```

##### TPEx Cloudflare Worker

RunPod 機房若被 TPEx data endpoint 以 HTTP 403 拒絕，台灣市場 profile 必須先
部署受限的 Cloudflare Worker。先在 Cloudflare 建立只有 **Workers Scripts Edit**
權限的 API token（API 文件將同一能力稱為 `Workers Scripts Write`），取得 32 字元
account ID，再執行：

在 **Manage account → Account API tokens → Create a token** 畫面採用下列最小權限
設定。不要選擇畫面中的 **Edit Cloudflare Workers** 範本；該範本會加入本專案不需要
的 Workers Routes、KV、R2、Tail 與其他 read 權限。改選 **Start from scratch**：

Cloudflare 新版 Account API token 編輯器不再把 `Account` 顯示為可逐層展開的選單。
進入 **Edit policy** 後，將最上方的 policy scope 保持為 **Entire Account**，在
**Search for permission groups...** 輸入 `Workers Scripts`；也可以展開
**Developer Platform** 後尋找同名權限。最後只對 **Workers Scripts** 選擇
**Edit**。`Developer Platform` 只是介面分類，不是另一項權限。

| 欄位 | 設定 |
| --- | --- |
| Token name | 使用可辨識用途的名稱，例如 `stock-forecasting-tpex-relay-deployer`。 |
| Policy scope | 保持 **Entire Account**。這是目前 account-owned token 的 account scope，不是授權目前使用者名下的所有 Cloudflare accounts。Workers Scripts 權限無法再縮限至單一 script。 |
| Permission policies | 搜尋 `Workers Scripts`，或展開 **Developer Platform**；只對 **Workers Scripts** 選擇 **Edit**，不要另外選 Read。介面應只顯示一項已選 permission group。 |
| Token expiration | 建議選 **90 days**；需要較少輪替時可選 **1 year**，不建議 `No expiration`。 |
| Client IP address filtering | 家用網路、動態 IP、VPN 或會移動使用時留白。只有已知固定 public egress IP 時才設定 Allow，單一 IPv4 使用 `/32`、單一 IPv6 使用 `/128`。 |

不要加入 **Read all resources**、**Write all resources**、**Edit zone DNS**、Workers
Routes、Workers KV Storage、Workers R2 Storage、Workers Tail 或 Account Settings。
`Workers Scripts Edit` 已涵蓋部署器需要的 subdomain 查詢／建立、Worker module 上傳與
`workers.dev` 啟用操作。API token 只在本機執行 `tpex-proxy deploy` 時使用；90 天或
1 年後到期不會關閉已部署的 Worker，也不會中斷 Pod 透過 relay 抓取 TPEx，只會讓
下一次重新部署前必須輪替 token。

按 **Continue to summary** 後，確認摘要只有目前 account 的 **Workers Scripts
Edit**，沒有其他 permission group，再建立 token。Cloudflare 官方權限表將它列為
account-scoped 的 Worker script 寫入權限。token secret 只顯示一次；不要貼進 README、終端指令列、聊天
或截圖，直接在下一個 `configure` 指令的隱藏輸入提示貼上。32 字元 account ID 不是
token secret，可從 Cloudflare account URL 或 dashboard Overview 取得；不要把個人
account ID 寫入專案文件。

```bash
bash scripts/runpod_workflow.sh tpex-proxy configure
bash scripts/runpod_workflow.sh tpex-proxy deploy
```

`configure` 以隱藏輸入保存 Cloudflare token，並在首次設定時自動產生共享 relay
token。若 Cloudflare 帳號尚未設定 `workers.dev` subdomain，請在互動提示輸入一個
全域唯一名稱；既有帳號可留白，部署器會向 Cloudflare 查詢。`deploy` 透過官方
Workers API 完成以下受控步驟：

`deploy` 會使用本機 `.env` 內既有的 `RUNPOD_API_KEY` 呼叫官方 GraphQL
`secretCreate`。`secretCreate` 是 API mutation 名稱，不是建立 RunPod API key 時可
單獨勾選的權限。你不需要為此預先重建金鑰；若目前金鑰無效、只有讀取權限，或
無法在所屬 account/team 建立 Secret，`deploy` 會停止，而且不會把新的 Worker
metadata 啟用到本機 `.env`。

`configure` 會合併更新 `.env`，保留既有的 RunPod network volume 與 S3 設定。若
volume 已部署完成，不要重新執行 `credentials` 或 `volume deploy`；`deploy` 成功後
直接跳到「驗證 S3 並上傳程式碼」執行 `sync --dry-run` 與 `sync --apply`。

- 上傳 [`cloudflare/tpex-proxy/src/index.mjs`](cloudflare/tpex-proxy/src/index.mjs)，
  並將執行位置提示設為 `gcp:asia-east1`。
- 只允許 `GET`、共享 token、固定 TPEx origin、四個專案使用中的 path，以及各
  path 的固定 query schema；TPEx 若回傳 redirect，最多跟隨三次且每一跳都必須維持
  相同 HTTPS origin，跨 origin、缺少 Location 或無限 redirect 都會拒絕。它不是通用
  或開放式 proxy。
- 啟用該 script 的 `workers.dev` URL，把同一個 relay token 寫入新建的 RunPod
  Secret；Cloudflare API token 不會進入 Pod。
- 直接以 `exDailyQ` 執行 smoke test。驗證器會在首次 `workers.dev` 路由尚未傳播時
  進行有限次指數退避重試，並以 HTTP status 與安全的 Worker error code 回報失敗；
  只有 Worker 實際取得含官方資料表的 JSON 才視為部署成功。

此 Worker 不使用 KV、Durable Objects、資料庫或其他付費 binding；TPEx 每個 cache
miss 對應一次 Worker request 與一次 upstream subrequest。實際是否落在免費額度內
仍以 Cloudflare 帳號方案與當下官方 limits 為準。

Cloudflare placement hint 會讓 Worker 靠近指定 region，但不保證特定的固定出口
IP，也不能保證 TPEx 永遠接受該出口。因此 smoke test 才是建立 CPU Pod 前的實際
通過條件；可隨時重新驗證：

```bash
bash scripts/runpod_workflow.sh tpex-proxy verify
```

TPEx client 仍使用原始 `https://www.tpex.org.tw` endpoint 與 public query params
計算 request SHA-256；Worker URL、relay token 與 transport 模式都不會進入 raw
cache key 或 dataset request identity。切換到 Worker 後，既有成功的 TWSE、TPEx
與 EODHD JSON cache 會照常續用，只對缺少的 TPEx response 經 Worker 發出請求。
相關官方文件：

- [Cloudflare API token 建立流程](https://developers.cloudflare.com/fundamentals/api/get-started/create-token/)
- [Cloudflare Account API tokens](https://developers.cloudflare.com/fundamentals/api/get-started/account-owned-tokens/)
- [Cloudflare API token 範本權限](https://developers.cloudflare.com/fundamentals/api/reference/template/)
- [Cloudflare API token 權限表](https://developers.cloudflare.com/fundamentals/api/reference/permissions/)
- [Cloudflare token expiration 與 IP 限制](https://developers.cloudflare.com/fundamentals/api/how-to/restrict-tokens/)
- [Cloudflare Workers API](https://developers.cloudflare.com/api/resources/workers/)
- [Cloudflare Worker placement](https://developers.cloudflare.com/workers/configuration/placement/)
- [Cloudflare Worker secrets](https://developers.cloudflare.com/workers/configuration/secrets/)
- [Cloudflare Workers limits](https://developers.cloudflare.com/workers/platform/limits/)
- [RunPod GraphQL `secretCreate`](https://docs.runpod.io/sdks/graphql/manage-pod-templates)

接著由腳本建立 network volume。成功回傳的 volume ID、datacenter、S3 region
與 endpoint 會自動寫回同一個 `.env`，不需要複製 ID：

```bash
bash scripts/runpod_workflow.sh volume deploy \
  --name stock-forecasting \
  --size-gb 100 \
  --datacenter EU-RO-1
```

若 `.env` 已登記 volume，volume script 預設不會再建立另一個可能計費的
volume；只有刻意使用 `--force-new` 才會建立並改登記新 volume。

在 CPU Pod 建立前，必須先由腳本選定 stage、資料來源、日期與 universe。
不帶參數會進入互動式選單：

```bash
bash scripts/runpod_workflow.sh configure
```

##### `configure` 參數與資料範圍

`--universe`、`--stocks`、`--etfs` 與 `--symbol-limit` **只控制美國資料
範圍**，不會篩選台股。`us_tw_eodhd` 永遠由「依 universe 選出的 EODHD
美國目標證券」加上「指定日期範圍內 TWSE／TPEx 官方端點回傳、且符合
普通股／TDR／白名單非槓桿股票型 ETF 契約的台灣資料」組成。

`--data-profile` 可選值如下：

| 值 | 美國資料 | 台灣資料 | 是否需要 `eodhd_api_token` |
| --- | --- | --- | --- |
| `tw_only` | 無 | TWSE／TPEx 普通股、TDR、經稽核且可映射的非槓桿股票型 ETF，以及官方 benchmark；`--universe` 必須為 `all` | 否 |
| `us_only_eodhd` | 依 `--universe` 選出的 EODHD 普通股（包含 ADR）與經稽核且可映射的非槓桿股票型 ETF，並自動補入 `VTI.US` benchmark | 無 | 是 |
| `us_tw_eodhd` | 與 `us_only_eodhd` 相同的美國證券範圍 | 與 `tw_only` 相同的台灣目標證券範圍 | 是 |

`--universe` 可選值如下：

| 值 | 意義 | 可搭配的選項 |
| --- | --- | --- |
| `all` | 對含美國資料的 profile，透過 EODHD discovery 取得 active 與 delisted 普通股／ADR，再套用非槓桿股票型 ETF 白名單；對 `tw_only`，表示完整的台灣目標證券範圍 | 美國 profile 可選擇搭配 `--symbol-limit`；不得同時提供 `--stocks` 或 `--etfs` |
| `explicit` | **只限制美國資料**；至少要提供一個 `--stocks` 或 `--etfs`。系統仍會自動補入 `VTI.US` | 只能用於 `us_only_eodhd` 或 `us_tw_eodhd`；不得搭配 `--symbol-limit` |

「完整美國目標證券範圍」在此指 EODHD 帳戶可取得且 discovery 回傳的
active／delisted common stocks（包含 ADR），以及程式白名單內的非槓桿股票型
ETF；使用含美國資料的 profile、`--universe all`，並完全省略 `--symbol-limit`。
槓桿、反向、債券、商品與波動率 ETF 不會成為訓練目標，explicit benchmark
mapping 也不能繞過此限制。這不保證涵蓋 provider 未回傳或帳戶未授權的商品。

所有使用者可設定的 `configure` 選項如下：

| 選項 | 可選值／格式 | 意義與限制 |
| --- | --- | --- |
| `--stage` | `stage1`、`stage2` | 選擇固定的訓練 config。非互動模式必填 |
| `--data-profile` | `tw_only`、`us_only_eodhd`、`us_tw_eodhd` | 決定實際使用的 provider 與市場組合。非互動模式必填 |
| `--dataset-revision` | 1～64 字元；英數開頭，之後可用英數、`.`、`_`、`-`；預設 `v1` | provider 修訂歷史資料時，用新 label 強制建立新的 immutable dataset namespace |
| `--start` | `YYYY-MM-DD`；預設 `2005-01-01` | 所有選定市場共用的起始日，包含該日；省略時固定使用 `2005-01-01` |
| `--end` | `YYYY-MM-DD`；無預設值 | 所有選定市場共用的結束邊界，不包含該日；互動與非互動模式都必須由使用者明確提供，避免不知情地改用本機當日 |
| `--h-start` | `1`、`2`、`3`；預設 `3` | 在 `t` 收盤後同時預測從第幾個持有交易日起至第 14 日的累積 alpha；仍於 `t+1` raw open 評估進場。此值屬於 processed dataset 與模型輸出契約 |
| `--universe` | `all`、`explicit` | 控制美國商品選取方式；對 `tw_only` 只能使用 `all`。非互動模式必填 |
| `--stocks` | 逗號或空白分隔的美國 ticker；可重複提供 | `explicit` 模式中的美國普通股／ADR，例如 `"AAPL,BABA"`；會用 EODHD discovery 驗證型別，不影響台股 |
| `--etfs` | 逗號或空白分隔的美國 ticker；可重複提供 | `explicit` 模式中的非槓桿股票型 ETF，例如 `"SPY,QQQ"`；必須在經稽核白名單內，不影響台股 |
| `--symbol-limit` | 正整數 N | **小規模容量／流程驗證用，不是完整美國市場模式。**只適用於含美國資料的 `all` 模式。discovery 後把白名單內的非槓桿股票型 ETF 與普通股／ADR 分開，各自依 active → delisted、ticker 字母順序取最多 N 檔；若某一類少於 N 就全取。這不是隨機或代表性抽樣。接著確認必要的 `VTI.US` benchmark：已在 N 檔 ETF 內就不重複，否則額外補入，所以 raw universe 最多 `2N+1` 檔。要完整美國目標證券範圍就不要提供此選項 |
| `--interactive` | 無值 flag | 明確開啟互動式選單；直接執行 `configure` 而不帶選項時會自動使用此模式 |

互動模式的預設值是 `stage1`、`us_tw_eodhd`、dataset revision `v1`、起始日
`2005-01-01`、`h_start=3` 與 US universe `all`。`--end` 刻意沒有預設值，提示時若留白會繼續
要求輸入，不會自動採用本機當日。Provider acquisition policy 不在 `configure` 設定；
若在此命令提供 `--max-api-calls`、`--eodhd-qps`、`--taiwan-qps` 或 `--maxBackoff`，
會以未知選項拒絕，不會建立另一個 selection。
底層 helper 的 `--project-root` 由 `runpod_workflow.sh` 自動注入，不是使用者
設定資料範圍的選項，不要自行提供。

下列範例的實際資料範圍是：

- 美國：`AAPL.US`、`MSFT.US`、`SPY.US`、`QQQ.US`，以及系統自動補入的
  `VTI.US` benchmark。
- 台灣：不是只有四個美國 ticker，也不是沒有台股；會包含相同日期範圍內
  TWSE／TPEx 普通股、TDR、白名單內非槓桿股票型 ETF 與官方 benchmark。
- 日期：兩個市場都從 `2015-01-01` 開始，並在 `2026-07-27` 之前結束；
  `2026-07-27` 本身不包含在資料內。

非互動式「美國 explicit universe + 完整台灣市場」範例：

```bash
bash scripts/runpod_workflow.sh configure \
  --stage stage1 \
  --data-profile us_tw_eodhd \
  --start 2015-01-01 \
  --end 2026-07-27 \
  --h-start 3 \
  --universe explicit \
  --stocks "AAPL,MSFT" \
  --etfs "SPY,QQQ"
```

Provider acquisition policy 會在建立 CPU Pod 時另外設定。`--max-api-calls 10000`
只限制該次 acquisition attempt 的 EODHD network attempts，不會把台股或美股截成
10,000 筆資料，也不要求整份資料能在 10,000 次 requests 內完成。EODHD 達到上限後
會退出自己的迴圈，但平行執行中的 TWSE／TPEx 仍會繼續。三個 provider 迴圈都退出後，
CPU preparation 才保存 cache 與 `waiting_for_budget` 進度並結束。

供應商額度是另一層限制。EODHD 官方價格頁目前列出 `EOD Historical Data — All
World` 個人方案月繳 USD 19.99；官方限制文件指出付費方案預設每日 100,000 API
calls、每分鐘 1,000 HTTP requests，且訂閱方案的每日額度在午夜 GMT 重置。兩種
單位彼此獨立，不同 endpoint 也可能消耗不同數量的計費 calls；實際訂閱、帳戶已用
額度與 provider 回傳 headers 才是執行時依據。本專案把預設 pacing 向下取整為每秒
16 requests（每分鐘 960 requests），但若同一帳戶還有其他 client 同時使用，仍須再
降低 `--eodhd-qps`。可參考
[EODHD Pricing](https://eodhd.com/pricing)、
[EODHD API Limits](https://eodhd.com/financial-apis/api-limits) 與
[EODHD User API](https://eodhd.com/financial-apis/user-api)。完整美國目標證券範圍的
request 計畫可以大於 `--max-api-calls`；專案與供應商額度邊界都由後述的 CPU Pod
續傳流程跨多次執行處理。

完整 EODHD／台灣目標證券範圍的設定不提供 `--stocks`、`--etfs` 或
`--symbol-limit`：

```bash
bash scripts/runpod_workflow.sh configure \
  --stage stage1 \
  --data-profile us_tw_eodhd \
  --start 2015-01-01 \
  --end 2026-07-27 \
  --universe all
```

discovery 後的完整 HTTP request 計畫即使超過該次 CPU Pod 的 `--max-api-calls`，
也只會記錄為資訊，不會阻止 Pod 建立或縮小資料範圍。EODHD 每個 CPU attempt 最多
送出設定的 network attempts，
達上限後由下一個 Pod 使用 cache 續傳；可依成本與使用情況調高或調低
`cpu prepare --max-api-calls`，但它不會繞過 provider quota。

如果要使用相同的美國 explicit universe、但**完全不下載台股**，必須把 profile
改成 `us_only_eodhd`：

```bash
bash scripts/runpod_workflow.sh configure \
  --stage stage1 \
  --data-profile us_only_eodhd \
  --start 2015-01-01 \
  --end 2026-07-27 \
  --universe explicit \
  --stocks "AAPL,MSFT" \
  --etfs "SPY,QQQ"
```

這個 `us_only_eodhd` 範例只包含上述四個美國 ticker 加上自動補入的
`VTI.US`，不包含 TWSE／TPEx 資料。

台股全市場、不使用 EODHD 的範例：

```bash
bash scripts/runpod_workflow.sh configure \
  --stage stage1 \
  --data-profile tw_only \
  --start 2015-01-01 \
  --end 2026-07-27 \
  --universe all
```

腳本會建立 `.runpod/selections/<selection-id>.json` 與
`.runpod/active-selection.json`。兩者都不含 secret 且被 `.gitignore` 排除。
目前使用 selection schema 3；舊 schema 不含完整 `h_start` 處理契約，因此更新程式碼後
必須重新執行 `configure`，不會自動轉換。修改 `--end`
會按設計建立新的 dataset request namespace；既有 network-volume 檔案不會被刪除。
profile、日期、universe、symbol limit、資料處理契約或 stage config SHA-256
任一不同，都會得到不同 identity。QPS、API budget 與最大退避只記錄在 CPU launch
metadata、download progress 與 download manifest，不會進入 selection identity。

只修改 `h_start` 會產生新的 dataset request SHA，並重新建立 processed windows、labels、
train-only robust scales 與 split audit；它不會改變日期、symbol、provider request 或
`cache_revision`。CPU acquisition 會以唯讀方式掃描其他 dataset namespace 中相同
`cache_revision=v1` 的 `api-cache`，所以既有相同 URL／參數 response 會記為 cache hit，
不會再次送出網路 API call。只有舊快取原本缺少、失敗，或 request 參數本身改變的資料
才需要呼叫 provider。

若 provider 可能修訂歷史資料，且確實要為相同 profile/date/universe 建立新快照，
請在 `configure` 明確加入新的 `--dataset-revision <label>`；CPU workflow 不會
覆寫完整的既有 namespace，部分殘留也會 fail closed。

可用下列指令檢視目前選擇，不需開啟 JSON：

```bash
bash scripts/runpod_workflow.sh selection show
```

`.env` 只保存本機 RunPod/S3/Cloudflare deployment credentials，以及腳本回填的
volume 與 TPEx Worker metadata；
stage、資料範圍、runtime 與 config 不從 `.env` 讀取。不要 `source .env`，也不要
把 API key、token 或 secret value 寫進 README、config、shell script 或提交
紀錄。Pod 只會收到 RunPod Secret reference 與非敏感 Worker URL，不會收到本機
account-level RunPod/S3/Cloudflare credential。

#### 2. 驗證 S3 並上傳程式碼

先執行 read-only S3 權限檢查，再預覽明確的上傳 allowlist：

```bash
bash scripts/verify_runpod_s3_access.sh
bash scripts/runpod_workflow.sh sync --dry-run
```

確認清單後才實際上傳，並驗證 remote code readiness：

```bash
bash scripts/runpod_workflow.sh sync --apply
bash scripts/runpod_workflow.sh readiness --code-only
```

上傳器會掃描 allowlisted source、config、script、test、`README.md` 與
`pyproject.toml` 的 secret pattern，逐檔上傳並核對遠端大小，最後才發布
`lifecycle/stage1/code.json`。`.env`、cache、資料、checkpoint 與本機
artifact 不會上傳。`poetry.lock` 也不會上傳；它會依 approved RunPod image
的 Python/PyTorch/CUDA 環境在 network volume 上重新產生。

任何 allowlisted 程式碼或 config 修改後，都要重新執行 `--dry-run`、
`--apply` 與 readiness check。config 修改也會使 active selection 失效，必須
重新執行 `configure`。若資料 marker 綁定的是舊 code release 或舊 selection，
還必須重跑 CPU preparation，不能略過 GPU gate。

#### 3. 遠端部署模型與離線資料層

CPU Pod 只能使用 active selection；如果尚未執行 `configure`、config SHA 已改變，
或 selection JSON 不完整，建立前就會失敗。在本機不帶任何選項執行時會進入
互動模式，依序詢問 workload 最長執行時間、這次 Pod 可新增的 EODHD network-attempt
上限、EODHD QPS、TWSE／TPEx 每個 provider 的 QPS、data cleaning/window construction
保留時間、三個 provider 共用的最大單次退避、vCPU 數與 CPU flavor。
`--max-api-calls` 沒有可直接按 Enter 接受的預設值，必須依帳戶當下剩餘額度明確輸入；
其他項目按 Enter 分別
使用 6 小時、16 QPS、每個台灣 provider 0.5 QPS、自動保留、1 分鐘、8 vCPU 與
`cpu3g`。自動保留是 max runtime 的 25%，最多 2 小時；預設 6 小時會保留 90 分鐘。
最後還必須輸入 `y` 或 `yes` 才會建立可能計費的 Pod，直接按 Enter、輸入 `n` 或
`no` 都會安全取消：

```bash
bash scripts/runpod_workflow.sh cpu prepare
```

只要提供任一參數，就會使用非互動模式。此模式必須明確提供 `--max-api-calls`；
EODHD／Taiwan QPS 與資源選項未提供時才採用上述預設值。因此自動化腳本可明確提供
全部參數，不需要修改 `.env` 或重新執行 `configure`：

```bash
bash scripts/runpod_workflow.sh cpu prepare \
  --max-api-calls 80000 \
  --eodhd-qps 16 \
  --taiwan-qps 0.5 \
  --maxRuntime 10h \
  --prepareReserve 2h \
  --maxBackoff 1m \
  --cpuNumber 16 \
  --cpuFlavor cpu5g
```

若希望先在命令列提供互動提示的預設值，再由使用者確認或覆寫，可加入
`--interactive`：

```bash
bash scripts/runpod_workflow.sh cpu prepare \
  --interactive \
  --max-api-calls 80000 \
  --eodhd-qps 16 \
  --taiwan-qps 0.5 \
  --maxRuntime 10h \
  --prepareReserve auto \
  --maxBackoff 1m \
  --cpuNumber 16 \
  --cpuFlavor cpu5g
```

`--max-api-calls` 是正整數，只計入本次 CPU Pod 的 EODHD cache miss 與 retry；它不是
帳戶每日總額度，新的 CPU Pod 也不會自動扣除前一次 Pod 的帳戶使用量。`--eodhd-qps`
與 `--taiwan-qps` 必須大於零，預設分別為 `16` 與 `0.5`；Taiwan 值是每個 provider
各自的 limiter，因此 TWSE 與 TPEx 同時執行時是兩個獨立的 0.5 QPS 上限。
`--maxRuntime`、明確的 `--prepareReserve` 與 `--maxBackoff` 都接受正整數加 `m`、`h` 或 `d`；
`--prepareReserve auto` 使用上述自動公式，明確值必須短於 max runtime。
`--maxBackoff` 預設為 `1m`，同時套用於 EODHD、TWSE 與 TPEx：若下一次由指數退避或
`Retry-After` 得到的等待時間**超過**此值，只有該 provider 迴圈會退出（等於上限仍會
等待）。EODHD 會在 `--max-api-calls` 用完或超過這個退避邊界時停止，兩者任一先發生
即生效。
`--cpuNumber` 必須是 1～32。`--cpuFlavor` 只接受下表六個 RunPod 值，其他值或
超過 32 vCPU 會在建立 Pod 前失敗：

| Flavor | 世代 | 類型 | RAM / vCPU | 32 vCPU RAM | Container disk 上限 |
| --- | ---: | --- | ---: | ---: | ---: |
| `cpu3c` | CPU3 | Compute-Optimized | 2 GB | 64 GB | 10 GB/vCPU |
| `cpu3g` | CPU3 | General Purpose | 4 GB | 128 GB | 10 GB/vCPU |
| `cpu3m` | CPU3 | Memory-Optimized | 8 GB | 256 GB | 10 GB/vCPU |
| `cpu5c` | CPU5 | Compute-Optimized | 2 GB | 64 GB | 15 GB/vCPU |
| `cpu5g` | CPU5 | General Purpose | 4 GB | 128 GB | 15 GB/vCPU |
| `cpu5m` | CPU5 | Memory-Optimized | 8 GB | 256 GB | 15 GB/vCPU |

Container disk 預設要求 30 GB；若較小的 vCPU 數使該值超過表中的上限，建立腳本
會自動降到該 flavor 的合法上限。若另以 `RUNPOD_CPU_CONTAINER_DISK_GB` 明確指定
超過上限的值，則會在建立 Pod 前失敗。

指令會輸出 Pod ID、外部 hard-limit guard 與 SSH 後應執行的 workflow。
由 RunPod Console 的 Connect 頁面取得 SSH 命令。登入 Pod 後執行：

```bash
cd /runpod-volume/stock_forecasting
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
3. 從 mounted immutable selection 重新驗證 stage、profile、日期、universe、
   config SHA-256 與 Pod environment，再依該 selection 下載資料；cache hit 不會
   再次呼叫 provider。腳本會同時讀取使用者要求的 vCPU 數、RunPod 提供的
   `RUNPOD_CPU_COUNT` 與容器實際可見核心數，採三者最小值作為 worker 數；
   EODHD、TWSE 與 TPEx 以三個獨立頂層迴圈平行抓取，迴圈內再依商品／官方 benchmark
   所列實際交易日使用 thread pool；因果視窗依商品平行建立，pytest worker 也不會超過
   這個有效核心數。每個 provider 有自己的 QPS limiter。`--max-api-calls` 只限制 EODHD，
   TWSE／TPEx 沒有專案端 request-count ceiling；三個 provider 都由同一個
   `--maxBackoff` 值控制各自的退避邊界。
   任一 provider 先退出都不會取消另外兩個；主流程 join 全部迴圈後才合併 provider-local
   Parquet/request-log 分片或發布續傳狀態。完整 request 估算只作資訊；workflow 自動保留 max runtime 的 25%（最多 2 小時；預設 6 小時即
   90 分鐘）給 data cleaning/window construction，也可用 `--prepareReserve` 調整。
4. 建立並驗證下列 persistent artifacts：
   | 遠端路徑                                                                  | 內容                                              |
   | ------------------------------------------------------------------------- | ------------------------------------------------- |
   | `/runpod-volume/datasets/<dataset-request-sha256>/api-cache/`            | provider raw response cache                       |
   | `/runpod-volume/datasets/<dataset-request-sha256>/download-progress.json` | 續傳 attempt、cache 數量與 provider／budget／runtime 等待狀態 |
   | `/runpod-volume/datasets/<dataset-request-sha256>/raw/market.parquet`    | durable `downloaded` checkpoint 的 canonical daily OHLCV |
   | `/runpod-volume/datasets/<dataset-request-sha256>/processed/windows.parquet` | 因果訓練視窗                                  |
   | `/runpod-volume/datasets/<dataset-request-sha256>/download-manifest.json` | 實際 provider、profile、symbols 與下載 provenance |
   | `/runpod-volume/datasets/<dataset-request-sha256>/dataset-manifest.json` | split counts、hash 與資料契約                     |
   | `/runpod-volume/datasets/<dataset-request-sha256>/manifests/api-request-log.jsonl` | 不含 token 的 request audit             |
   | `/runpod-volume/cache/huggingface/`                                      | 離線 Kronos model/tokenizer cache                 |
   | `/runpod-volume/cache/hf-models.json`                                    | 固定 model revisions 與 cache manifest            |
5. raw Parquet、download manifest 與 request log 完整驗證後先發布 `downloaded`
   lifecycle。沒有 exit code 的 `downloaded` 是同一 Pod 內的中間 checkpoint，外部 guard
   不會誤判為終態；若剩餘時間少於 cleaning reserve，腳本才以 exit code 75 將它標記為
   可續傳終態，下一個 CPU Pod 直接從 checkpoint 執行清理，不再呼叫 provider。其後才將
   selection ID/SHA、dataset request SHA、stage/config SHA、requested
   profile/date/universe 與 resolved artifact hashes 綁入
   `/runpod-volume/lifecycle/stage1/dataset.json`；只有全部檢查成功才發布並
   自動終止 Pod。

任何 CPU 或 GPU 工作在讀寫 volume 前，都會先證明 `/runpod-volume` 是**精確的
mount point**：優先使用 `mountpoint`，否則使用 `findmnt`，最後才檢查
`/proc/self/mountinfo`。建立 Pod 時由本機選定的 volume ID 會以
`RUNPOD_EXPECTED_VOLUME_ID` 傳入，並與 RunPod 自動提供的 `RUNPOD_VOLUME_ID`
逐字比對；路徑正確但 ID 不同也會立即停止。專案固定放在
`/runpod-volume/stock_forecasting`，資料、Kronos cache、W&B transaction 與模型則
位於 volume root 下的獨立子目錄；任何 persistent path 都不得使用會被 Pod
重建清除的 `/workspace`。這個 layout 避免把專案目錄本身當成 mount target，
也避免 network volume 掛載時遮蔽同名的 Pod 內建目錄。

Pod 終止後，以 S3 lifecycle 為準，不要依賴已消失的 SSH session：

```bash
bash scripts/runpod_workflow.sh status
```

##### Provider quota 與跨 CPU Pod 續傳

EODHD、TWSE 與 TPEx 使用彼此獨立的平行迴圈。retryable 的 429、暫時性網路錯誤或
provider 5xx 都採指數退避；台灣官方端點由 Pod IP/WAF 暫時回傳的 403 也視為
retryable。三個 provider 共用同一個 `--maxBackoff` 設定（預設 `1m`），但各自獨立
計算退避並在下一次等待超過該值時退出。EODHD 另外受到該 CPU attempt 的
`--max-api-calls` 限制，兩個 EODHD 邊界任一先發生就停止其迴圈；TWSE／TPEx 則沒有
request-count 上限。共同 acquisition deadline 仍可讓任何迴圈進入
`waiting_for_resume`，以保留 data cleaning 時間。流程遵守下列契約：

1. 已成功取得的每個 raw JSON response 仍保留在該 dataset request 專屬的
   `api-cache/`，不發布不完整 Parquet 或 `state=ready` marker。
2. EODHD 先達 `--max-api-calls` 或先超過最大退避時，都只退出 EODHD 迴圈，
   TWSE／TPEx 繼續；任一台灣 provider 先超過最大退避時，也不會停止 EODHD 或另一個
   台灣 provider。最大退避越界會開啟該 provider client 的共享 circuit breaker；
   已送出的 in-flight request 可能完成，但同一 provider 的其他 worker 不會再啟動新
   request。主流程一定等到所有已選 provider 迴圈退出，才決定下一步與允許 CPU Pod
   結束。
3. `download-progress.json` 記錄 attempt number、各 provider cache／network counts、
   EODHD limited count，以及每個 provider 的 `complete`、`waiting_for_budget`、
   `waiting_for_provider` 或 `waiting_for_resume` outcome；provider 錯誤另記已等待總秒數、
   最後等待、下一次 proposed backoff、三者共用的最大值，以及該次 launch 實際使用的
   `max-api-calls`／EODHD QPS／Taiwan QPS。HTTP status、`Retry-After` 與
   rate-limit headers 只在供應商有回傳時記錄；資料契約錯誤另記安全的 provider、
   operation、symbol/date/month 與 exception type，不記錄 token 或 response body。
4. 若任一迴圈仍未完成，dataset lifecycle 依整體 outcome 進入
   `waiting_for_budget`、`waiting_for_provider` 或 `waiting_for_resume`，GPU readiness
   維持不通過，CPU Pod 才自動終止。若三者都完成，則以固定 provider 順序合併分片，
   繼續 data cleaning，不會提早關閉 Pod。
5. 額度恢復後，**不要重新 `configure`、不要改 `--dataset-revision`、不要刪除
   cache**。在本機再次建立 CPU Pod 時，依新的剩餘額度重新輸入 `--max-api-calls`；
   QPS 也可依當次 provider 狀態調整，兩者都不會改變 selection。登入後重新啟動同一個
   workflow：

   ```bash
   bash scripts/runpod_workflow.sh cpu prepare
   # Run after connecting to the newly created CPU Pod:
   cd /runpod-volume/stock_forecasting
   bash scripts/runpod_tmux_launch.sh cpu-prepare
   ```

6. 新 attempt 會用相同 dataset request identity 重新播放已快取 responses，只對缺少
   的 request 呼叫 provider；Parquet 會由完整 response 集合重新建立。只有全部資料、
   manifest 與 selection gate 都通過後，lifecycle 才會變成 `ready`。`all` 模式的
   discovery response 也屬於同一份 immutable cache，因此跨日續傳不會重新取得一份
   已漂移的商品清單。

EODHD 偶爾會在已上市商品的日資料中回傳全零或其他無法通過 canonical OHLCV contract
的 placeholder row。下載器不補值、不改價，而是只丟棄無效 source rows，將數量記入
`dropped_source_rows`，並繼續保留同檔商品的有效 observations。TWSE 的早期除權資料
有時存在於 `TWT49U` 主表，但 detail 端點回傳「無相關資料」；此時保留可驗證的
price factor，share multiplier 使用 identity `1.0`，並在
`missing_share_multiplier_details_by_provider` 明確記錄 volume-adjustment coverage gap，
不把未知比例偽造成完整資料。

`bash scripts/runpod_workflow.sh status` 會在 dataset lifecycle 下方顯示 download
attempt、已快取 response 數、此次 network request 數、可用的完整 request 估算與
安全的 provider error 摘要，不需要手動開啟 JSON。

這是 request-level 續傳，不是 HTTP response 的 byte-range 續傳。每個成功完成的 API
request 都是續傳單位。若 `download-progress.json` 的 identity 與目前 dataset request
不同，流程會 fail closed，避免混用不同 profile、日期或 universe。過小的
`--max-api-calls` 會產生可續傳的 `waiting_for_budget`，不是失敗，也不要求重新
`configure`；可用相同 selection 直接建立下一個 CPU Pod。401／403 或其他非暫時性
設定錯誤才標記為 `failed`，應先修正 Secret。只有確實要建立新的
provider 資料快照時才改 `--dataset-revision`，新 revision 不會沿用舊 snapshot cache。

若狀態是 `waiting_for_budget`、`waiting_for_provider`、`waiting_for_resume`、`failed`
或 `timed_out`，先從 lifecycle 讀取
`launch_id`、`log_path` 與可用的 `progress_path`。tmux log 目錄固定為
`logs/tmux/fin-ts-cpu-prepare/<launch-id>/`；由腳本解析並下載到本機診斷目錄：

```bash
bash scripts/runpod_workflow.sh cpu-logs
```

只有下列 gate 通過後才租用 GPU：

```bash
bash scripts/runpod_workflow.sh readiness --gpu
```

Stage 2 使用相同 dataset request 時會得到相同 data namespace，但以 100%
train split 重新從相同 pretrained base 開始，不接續 Stage 1 checkpoint。
必須重新執行 `configure --stage stage2` 並明確提供與原 CPU dataset 相同的
profile/date/universe/dataset revision/`h_start`；stage/config identity 不同時，舊 Stage 1 marker 會被 GPU
gate 拒絕。範例：

```bash
bash scripts/runpod_workflow.sh configure \
  --stage stage2 \
  --data-profile tw_only \
  --start 2015-01-01 \
  --end 2026-07-27 \
  --h-start 3 \
  --universe all
bash scripts/runpod_workflow.sh sync --apply
bash scripts/runpod_workflow.sh cpu prepare
```

Stage 2 的 CPU Pod 建立完成後，腳本會提示在 SSH 內執行
`bash scripts/runpod_tmux_launch.sh cpu-finalize`；不要誤用 Stage 1 的
`cpu-prepare` 名稱重新下載資料。

CPU workflow 完成後，在本機再次執行
`bash scripts/runpod_workflow.sh readiness --gpu`。若只把 stage 改成 Stage 2，
但 profile/date/universe/dataset revision/`h_start` 與先前不同，dataset request SHA
也會不同，不會錯用舊
Parquet。

#### 4. 建立 GPU Pod 並訓練

先列出目前可用的完整 `gpuId`；預設是
`NVIDIA GeForce RTX 5090`：

```bash
bash scripts/runpodctl_project.sh gpu list
```

檢視 active selection 並通過 GPU gate；gate 會在租用 GPU 前比對本機 selection、
S3 selection、CPU marker、code release、config SHA 與 namespaced artifacts：

```bash
bash scripts/runpod_workflow.sh selection show
bash scripts/runpod_workflow.sh readiness --gpu
```

以預設 GPU 與 12 小時 workload 上限建立 Pod：

```bash
bash scripts/runpod_workflow.sh train
```

或直接指定 workload 上限與清單中完整的 `gpuId`：

```bash
bash scripts/runpod_workflow.sh train \
  --maxRuntime 18h \
  --gpuId "NVIDIA GeForce RTX 5090"
```

`--maxRuntime` 控制訓練加 validation workflow 的可用時間；外部 guard 與
RunPod `--terminate-after` 另保留一小時，只供 terminal lifecycle 寫入與失敗
清理，不會延長模型工作本身。CLI 的實際 runtime 會透過
`MAX_RUNTIME_SECONDS` 寫入 resolved Pydantic config，因此 W&B 不會仍顯示 YAML
原始的 6 小時預設。

建立指令會在本機先配置唯一 run ID、掛載同一個 network volume、注入 W&B
Secret reference，並啟動獨立 hard-limit guard。本機 guard 在 macOS 會自動以
`caffeinate -is -w <guard-pid>` 防止控制端睡眠，並把 guard、caffeinate 與
keep-awake 狀態寫在提示的 guard log 同目錄；不要關閉 guard process 或讓本機
斷電。Pod 端仍會自行終止，RunPod 的 `--terminate-after` 是第三層上限。
由 Console SSH 登入後執行：

```bash
cd /runpod-volume/stock_forecasting
bash scripts/runpod_tmux_launch.sh stage1-train
```

`stage1-train` 是為了維持既有部署相容性的 workflow 名稱；實際 Stage 1 或
Stage 2 由 immutable selection 所映射的 `RUNPOD_CONFIG` 決定。即時查看：

```bash
tmux -L fin-ts-stage1-train attach -t fin-ts-stage1-train
```

訓練會先驗證 image runtime、CUDA、mounted readiness、dataset/model
manifest 與 run identity，再寫入
`/runpod-volume/savedModel/<run-id>/`。兩份 production config 都設定
`validation.auto_run_after_training: true`，因此同一個 GPU workflow 會在訓練
完成後自動執行完整 validation benchmark。訓練加 validation 正確完成時，GPU
Pod 會自動終止；CPU prepare 正確完成，以及 CPU/GPU 工作失敗或超時時，也會先
發布 terminal lifecycle／tmux status，再自動終止。若 Pod 端 API 呼叫失敗，
本機 guard 會根據同一 Pod ID、run ID 與 lifecycle 接手；掛載 network volume 的
Pod 一律使用 terminate，而不是 stop。

##### W&B 記錄與離線補傳

W&B run config 會保存完整的 resolved YAML／Pydantic 設定、system metadata、
dataset provenance 與 immutable selection identity。`train/loss` 與
`train/pinball_loss` 在**每一個 optimizer step** 以 `trainer/global_step` custom
axis 記錄；使用 gradient accumulation 時記錄該 optimizer step 所含 microbatch
loss 的平均值。loss 與 validation 即使位於相同 optimizer step，也會各自保留
history row，不會因重複使用 W&B internal step 而遺失。訓練期間的 validation
會在每次 evaluation step 上傳所有有限數值指標；訓練後
benchmark validation 會以 `benchmark_validation/global_step` custom axis 對齊
被評估 checkpoint，而不回寫已完成 run 的舊 W&B internal step，並上傳模型與
baseline 的完整數值結果，包括 cross-sectional Sharpe、RankIC、turnover、
drawdown、coverage 與 subgroup 指標，而不只記錄最終 loss。

每個 run 的傳送狀態會原子寫入
`/runpod-volume/lifecycle/runs/<run-id>/wandb.json`，並由
`bash scripts/runpod_workflow.sh status` 顯示 `training` 與 `validation`
component。`online_finished`／`synced` 才表示該 component 已確認完成；
`offline_pending`、`sync_failed`，或 terminal workflow 後仍為 `online_running`
都必須視為尚待補傳。線上初始化失敗時，production config 允許切換為保存在
network volume 的 offline run；若連 offline transaction 都無法建立，工作會標記
`failed` 並中止。W&B 官方支援事後同步，因此單純 server 暫時不可用不需要丟棄
已完成的模型工作。

在一個由本專案建立、已注入 `wandb_api_key` 的 GPU Pod 中，**不要啟動訓練或
validation tmux**，直接執行下列補傳腳本；省略 run ID 會掃描所有 pending run：

```bash
cd /runpod-volume/stock_forecasting
bash scripts/runpod_wandb_sync.sh <run-id>
```

腳本只處理 `online_running`、`offline_pending` 或 `sync_failed` component，使用
W&B transaction directory 執行 `wandb sync --legacy --include-offline
--include-online --append --id <run-id>`。專案將 dependency 固定為 W&B 0.28.0；
`--legacy` 使用該版本的 legacy sync 路徑，因為 `--include-offline` 與 `--append`
在該版只支援 legacy mode。
成功改為 `synced`；失敗改為 `sync_failed` 並保留錯誤
原因。每次續訓或重跑 validation 產生的 transaction 都會分別保留及補傳，不會只
處理最後一段。腳本完成後會自動終止這個補傳 Pod。訓練已完成的 run 可先用 `validate` 建立
這個 Pod；未完成但已有 retained checkpoint 的 run 可先用後述 `resume` 建立，
然後改執行補傳腳本。也可以在下一個已存在的本專案 GPU Pod 開始工作前補傳，
避免另租 Pod。W&B 官方參考：
[offline mode](https://docs.wandb.ai/models/ref/python/functions/init) 與
[`wandb sync`](https://docs.wandb.ai/models/ref/cli/wandb-sync)。

Pod 終止後，以 workflow status 查詢 S3 上的最新 terminal state：

```bash
bash scripts/runpod_workflow.sh status
```

`state=ready` 才表示該 lifecycle 完成；`failed` 與 `timed_out` 必須視為未完成。
`wandb_run_id` 是 checkpoint、evaluation、W&B 與 run-scoped log 共用的
`<run-id>`。SSH/tmux 只用於仍存活 Pod 的即時除錯，不是 terminal state 的
權威來源。

若 training loop 在 `--maxRuntime` 前未完成，training lifecycle 會是 `timed_out`
（或錯誤時為 `failed`）且 `training_completed` 不會為 true。使用下列指令會在
本機選取該 run 最新、完整且通過 hash／resume contract 的 retained checkpoint，
沿用同一個 run ID、optimizer 與 scheduler 狀態建立新 Pod：

```bash
bash scripts/runpod_workflow.sh resume <run-id>
# Optional overrides:
bash scripts/runpod_workflow.sh resume \
  --maxRuntime 18h \
  --gpuId "NVIDIA GeForce RTX 5090" \
  <run-id>
```

省略 `<run-id>` 時只接受最新 `failed`／`timed_out` 且尚未完成 training phase 的
training lifecycle。若 immutable `training-completed.json` 已存在，`resume` 會拒絕
續訓並要求改用獨立 validation。SSH 登入續訓 Pod 後仍執行
`bash scripts/runpod_tmux_launch.sh stage1-train`；訓練完成後會接著跑 validation，
發布 terminal 狀態並關閉 Pod。

若訓練已完成但需要獨立重跑 validation，可在本機建立 validation Pod。active
selection 必須與該 run 保存的 stage 與 dataset identity 相同：

```bash
bash scripts/runpod_workflow.sh validate <run-id>
# Optional overrides; defaults are 12h and NVIDIA GeForce RTX 5090:
bash scripts/runpod_workflow.sh validate \
  --maxRuntime 8h \
  --gpuId "NVIDIA GeForce RTX 5090" \
  <run-id>
```

SSH 登入後執行：

```bash
cd /runpod-volume/stock_forecasting
bash scripts/runpod_tmux_launch.sh stage1-validate
```

獨立 validation 會沿用已完成 training run 的 best checkpoint 與同一個 W&B
run ID；數值 benchmark 完成、失敗或超時後都會發布 validation lifecycle 並自動
終止 GPU Pod。它不能用來替代尚未完成的 training；後者必須使用 `resume`。

#### 5. 下載所有 retained checkpoints 或 best checkpoint 與 validation 結果

不需要手動讀取 `.env` 的 volume ID、解析 lifecycle 或拼接 S3 path。先查看
已知 lifecycle：

```bash
bash scripts/runpod_workflow.sh status
```

不帶 run ID 時，下載腳本只接受最新 `state=ready` 的 training lifecycle，並
自動解析 run ID。`--checkpointScope` 可設為 `all` 或 `best`，預設為 `all`；
`--checkpoint-scope` 是等效別名。預設命令會下載 checkpoint leaderboard 目前
列出的所有 retained checkpoints：

```bash
bash scripts/runpod_workflow.sh download
```

也可以明確選擇全部或僅下載 validation-selected best checkpoint；run ID 可省略以
使用最新 ready run，或由使用者明確指定：

```bash
bash scripts/runpod_workflow.sh download --checkpointScope best
bash scripts/runpod_workflow.sh download --checkpointScope all <run-id>
bash scripts/runpod_workflow.sh download --checkpointScope best <run-id>
```

`all` 的意義是 leaderboard 中仍由 top-k retention policy 保留的完整集合，不包含
訓練期間已被該政策刪除的歷史 checkpoints。下載流程直接使用 immutable leaderboard，
不會以目錄列舉猜測 checkpoint。若本機目錄已存在，必須顯式使用 `--resume` 才會
填補或重新取得所選範圍的已知檔案：

```bash
bash scripts/runpod_workflow.sh download <run-id>
bash scripts/runpod_workflow.sh download --resume <run-id>
bash scripts/runpod_workflow.sh download --resume --checkpointScope best <run-id>
```

成果固定下載到被 `.gitignore` 排除的
`artifacts/runpod/<run-id>/`，包含 run manifest、resolved config、leaderboard、
best-checkpoint pointer、所選範圍的 checkpoint 目錄、validation benchmark、
training/validation lifecycle 與 immutable training completion record。`--resume` 不會
刪除本機既有的其他 checkpoint 目錄；例如從 `all` 切換成 `best` 時不會進行 prune。

每個下載的 checkpoint 目錄至少應包含 `adapter.safetensors`、
`resolved-config.yaml`、`trainer-state.json` 與它所列出的 optimizer/scheduler
state。`validation-benchmark.json` 是完整數值 validation 與 baseline 比較；
不要只根據 W&B 畫面或 README 宣稱 run 成功，應同時檢查 lifecycle 的
`state`、run ID、result path 與本機下載的原始 JSON。

實際訓練 stage 由 active immutable selection 與它綁定的 config SHA 決定。
操作命令、tmux session 或 lifecycle 路徑中的 `stage1-*` 名稱不會覆寫該選擇。

### 訓練與推論產物

每個 run 至少保存：

- `adapter.safetensors`：LoRA、resampler、benchmark conditioner 與 alpha head 的可訓練權重。
- `resolved-config.yaml`
- `trainer-state.json`
- optimizer / scheduler state
- run manifest、checkpoint leaderboard 與 best-checkpoint pointer
- selection ID/SHA、dataset request SHA、stage config SHA 與 requested dataset contract
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
  --input "${DATA_ROOT}/raw/market.parquet" \
  --symbol AAPL.US
```

推論輸出只包含數值 forecast、資料 provenance、encoder shape 與 checkpoint metadata，不包含自然語言解釋。

### 驗收原則

PoC 最低驗收條件：

1. Stage 1 精確使用 15% train samples，完整 validation/test，能完成 forward/backward、checkpoint reload 與 inference smoke test。
2. Stage 2 與 Stage 1 architecture digest 相同，且由相同 pretrained base 重新開始。
3. 所有 run 都可追溯到 immutable Parquet、dataset profile、provider、symbols、日期範圍與 split counts。
4. CPU marker 與 active training selection 必須在 stage、config SHA、profile、日期、requested universe 與 dataset request SHA 完全一致；例如 CPU `tw_only` 對 GPU `us_tw_eodhd` 必須在 Pod 建立前 fail closed。
5. `alpha_quantiles` 固定為 `[B,15-h_start,3]`，`h_start ∈ {1,2,3}`、最大 horizon
   固定為 14，且沒有 classifier、文字或 fact 輸出。
6. 模型 input、時序切分與正規化不使用未來資訊；未來 benchmark 只存在於離線 label construction。
7. 模型至少與 zero-return、momentum、technical、GBDT 與 neural baselines 在同一 validation/test protocol 下比較。
8. 不只報告單一 aggregate loss；同時報告各 horizon 的 normalized pinball、median correlation、方向一致率、區間 coverage/width 與依市場、asset type、年份的切片。

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
- [EODHD delisted data coverage](https://eodhd.com/financial-apis/delisted-stock-companies-data-2)
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
daily OHLCV data for US/Taiwan common stocks, ADRs/TDRs, and audited
benchmark-mappable unleveraged equity ETFs. The system processes only
numerical time series:

- Inputs and outputs are numerical tensors; no natural-language generation or
  fact-reconstruction interface is provided.
- The training loop never calls an external market-data API.
- It predicts continuous conditional alpha distributions from configurable
  `h_start` (1, 2, or 3) through the fixed 14th holding day.
- It exposes reusable numerical encoder representations for downstream systems.

This is a research and capability-validation PoC. It is not investment advice,
a production trading system, or a claim of guaranteed profitability.

### Numerical output contract

The only prediction emitted by `MultiHorizonAlphaHead` is:

- `alpha_quantiles`: `[batch, 15-h_start, 3]`.
- Dimension two contains holding periods `h_start`, `h_start+1`, ..., 14;
  `h_start` is restricted to 1, 2, or 3 and defaults to 3.
- Dimension three is fixed to q10, q50, and q90.
- Units are adjusted execution log return relative to the instrument's benchmark.

There is no `forecast_logits`, classification head, classification loss, or
direction probability. Inference may post-process each horizon's q10/q50/q90
with a fixed threshold into `strong_bearish`, `bearish`, `neutral`, `bullish`,
or `strong_bullish`. These signals are not extra training targets and introduce
no loss weights. Checkpoints must use `model_output_schema_version=5.0`;
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
                         h_start–14d alpha q10/q50/q90 [B,15-h_start,3]
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

The normal RunPod workflow selects a profile through
`bash scripts/runpod_workflow.sh configure`. `FIN_TS_DATASET_PROFILE` is an
internal value passed to the Pod only after the script validates the selection:

| profile           | Actual sources                | Status               | Use case                                  |
| ----------------- | ----------------------------- | -------------------- | ----------------------------------------- |
| `tw_only`       | Official TWSE + official TPEx | Available            | Research without US API cost              |
| `us_only_eodhd` | EODHD US stocks/ETFs          | Available            | Validate US-market capability first       |
| `us_tw_eodhd`   | EODHD + TWSE + TPEx           | Default PoC          | Full US/Taiwan PoC                        |
| `us_tw_massive` | Massive + TWSE + TPEx         | Typed interface only | Implement after obtaining suitable rights |

The EODHD path can discover both active and delisted US stocks/ETFs by default,
reducing survivorship bias. When budget or quota is constrained, use
`--universe explicit` with `--stocks` and `--etfs`, or use `--symbol-limit` in
all-universe mode. Manifests record the profile, actual providers, markets,
symbols, asset types, date range, and per-split sample counts.

Discovery under `--universe all` is the active/delisted snapshot returned by
EODHD at preparation time, not point-in-time constituents reconstructed for
every historical session. For `--start 2005-01-01 --end 2026-04-30`, the date
contract is `[2005-01-01, 2026-04-30)`: an instrument listed during the range
starts at its first provider-available session, and one delisted during the
range ends at its last available session. An instrument both listed and
delisted inside the range is included when EODHD delisted discovery returns it
and the account is entitled to it. `explicit` processes only named tickers;
`--symbol-limit` processes only its selected subset. A raw row's `is_active`
value is the discovery-time state, not a daily listing-state history.

EODHD separately documents EOD, fundamentals, dividends, and splits for
instruments delisted after 2018, but guarantees only EOD for pre-2018
delistings. Those older EOD rows can therefore still enter raw data and training
windows, while their split-adjusted-volume auxiliary coverage cannot be treated
as complete. The download manifest records their count and symbols under
`delisted_pre_2018_auxiliary_coverage_warning`, so price history is not
misrepresented as complete corporate-action history.

EODHD is PoC data and must not be represented as an exchange-grade market feed.
Adjusted prices, corporate actions, delisted history, time zones, and revisions
can differ across providers. Sample reconciliation on overlapping instruments
is required before formal comparisons.

### Execution timing, benchmark, and adjusted-data contract

Each sample emits a signal after trading-day `t` closes. Entry occurs at the
next shared trading day's raw regular-session open, which counts as holding day
one. A horizon `h` exits at the raw close of the `h`th shared trading day. The
label is the difference between instrument and benchmark total-return log
returns over identical entry and exit timestamps, for
`h ∈ {h_start,...,14}` and `h_start ∈ {1,2,3}`.

Default benchmark policy:

- US common stocks, ADRs, and allowlisted equity ETFs: `VTI.US`.
- TWSE common stocks, TDRs, and allowlisted equity ETFs: `TAIEX.TW`, whose adjusted anchor uses the official TAIEX total
  return index.
- TPEx common stocks: `TPEX.TWO`, whose adjusted anchor uses the official TPEx return
  index.
- Only audited allowlisted unleveraged equity ETFs that can map to an approved
  benchmark enter training. Leveraged, inverse, bond, commodity, volatility,
  and unaudited ETFs fail closed. `benchmark_mapping_path` may change the
  benchmark of an allowlisted ETF but cannot expand the training universe.

`VTI.US` is a required data dependency for US benchmark-relative labels and
benchmark context. It is not an ordinary `--symbol-limit` candidate and cannot
be its own training target (`self_benchmark` excludes it). The workflow first
selects N ETF candidates and N stock candidates, then ensures VTI is present:
an already-selected VTI is not duplicated; otherwise it is added. This prevents
the benchmark from consuming one of the N ETF candidate slots. The raw universe
is therefore at most `N ETFs + N stocks + 1 VTI`, while data-length, benchmark
mapping, and quality gates can reduce the actual trainable-target count.

Raw O/H/L/C is retained permanently, as is official Taiwan raw volume. EODHD
defines its EOD `volume` as already split-adjusted, so the pipeline uses the
complete Historical Splits response to reconstruct contemporaneous unadjusted
`volume` and retains the vendor value as `split_adjusted_volume`; it never
multiplies that value by the split factor again. Model windows normalize each vendor or
official total-return factor to `cutoff_at` before applying it to historical
O/H/L/C, so future corporate actions cannot rewrite an after-close inference
input. Volume is adjusted only for splits/share changes, never for cash
dividends. EODHD retains `adjusted_close` and uses the per-symbol Historical
Splits API for every date range. EODHD lists that endpoint under EOD
Historical Data — All World at one API call per request; the pipeline does not
use `calendar/splits`, which belongs to Calendar-enabled products. Each symbol
therefore requires one EOD-history request plus one split-history request. Both
are cacheable/resumable and included in the informational request estimate; the
estimate never blocks a complete dataset. The provider's pre-2018 delisting
exception is retained explicitly in the warning described above. Taiwan uses
official TWSE/TPEx ex-right/ex-dividend data and return indices. Existing monthly
benchmark rows provide the actual trading sessions, so ordinary weekdays are
not blindly treated as open sessions. This removes artificial corporate-action
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

The `train`, `validation`, and `test` partitions use global market-calendar
boundaries in chronological 70%/15%/15% order rather than random row splits.
The latest `label.end_at` for every train and validation sample must be strictly
earlier than the next split boundary. In addition to the 20-trading-day purge
and 14-trading-day effective embargo, a direct label-boundary guard prevents a
future horizon or spacing change from moving ground truth across partitions.

The downloader provides:

- Provider-specific QPS throttling.
- Independent parallel EODHD, TWSE, and TPEx loops with exponential backoff for
  retryable requests.
- Estimated HTTP requests for the complete plan are informational capacity data.
  The final Taiwan plan uses official benchmark sessions and records its
  pre-calendar weekday upper bound separately; neither is an admission gate.
- `max_api_calls` limits only EODHD network attempts, including retries, in one
  CPU attempt. TWSE/TPEx have no request-count ceiling. EODHD, TWSE, and TPEx
  each use the same default one-minute `--maxBackoff` value as their independent
  backoff exit boundary. Cache hits do not consume the EODHD counter; incomplete
  work is saved for another Pod.
- `--max-api-calls`, `--eodhd-qps`, `--taiwan-qps`, and `--maxBackoff` are
  acquisition policy values for one CPU Pod launch. They are set by
  `runpod_workflow.sh cpu prepare`, are not part of the immutable dataset
  selection, and cannot change dataset request identity.
- One provider exiting never cancels another. Only after every provider loop
  exits does the workflow publish a resume state or deterministically merge
  provider-local artifacts.
- A dataset-contract or date change creates a new immutable namespace. That
  namespace may read matching cache revisions and request identities from raw JSON caches in older
  dataset namespaces, while its Parquet files, manifests, and progress remain
  confined to the new namespace.
- Neither QPS nor `max_api_calls` represents a provider's daily/weekly quota or
  EODHD's endpoint-specific billed call units.
- After a temporary provider failure, HTTP 429, per-attempt request-budget
  exhaustion, or acquisition-time exhaustion, successful raw responses remain
  cached, `download-progress.json` is updated, and a later CPU Pod requests only
  missing cache entries.
- The CPU workflow reserves 25% of max runtime for canonical cleaning and causal
  window construction by default, capped at 2 hours and explicitly configurable
  through `--prepareReserve`. Once
  acquisition completes, raw Parquet, the request log, and the download manifest
  become a durable `downloaded` checkpoint. A later Pod can skip every API call;
  only verified processed data and readiness can become `ready`.
- Cache identities, request logs, and manifests that exclude API tokens.
- A raw cache and direct download/preparation CLIs that refuse silent overwrite.
- SHA-256, row-count, and provenance bindings for artifacts.

Training accepts only materialized `.parquet` data and a dataset manifest whose
state is `ready`. Training, evaluation, and inference never call EODHD, TWSE,
TPEx, or Massive.

### Remote runtime and local boundary

Python dependency resolution, the Poetry environment, lint, pytest, data
preparation, model-cache smoke tests, training, and validation all run on
RunPod. The local machine is only a control plane: edit source and use the
workflow script to manage credentials, selections, uploads, Pod lifecycle, and
artifacts. Do not manually edit `.env`, YAML, or JSON configuration.

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

Each dataset request maps automatically to
`/runpod-volume/datasets/<dataset-request-sha256>/`. A profile, date range,
universe, symbol limit, or preparation-contract change selects a new root. No
manual `DATA_ROOT` is required, and different data ranges cannot be mistaken
for the same dataset.

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

The two configs must have identical `config.model_architecture_digest()` values;
this digest binds model parameters and the `h_start`/output-horizon contract. Stage
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

In the RunPod Console, create a project-scoped RunPod API key and a separate S3
API key. Then create these fixed-name RunPod Secrets:

   - `huggingface_token`: required to prefetch the pinned Kronos model and
     tokenizer revisions.
   - `wandb_api_key`: required for training and validation tracking.
   - `eodhd_api_token`: required only by the `us_only_eodhd` and
     `us_tw_eodhd` profiles; it is not required by `tw_only`.

Profiles containing Taiwan data also require the TPEx Cloudflare Worker. Each
Worker deployment creates a uniquely named
`tpex_proxy_token_<timestamp>_<nonce>` RunPod Secret and records that secret
name in the local `.env`; do not create a fixed-name TPEx secret manually.

Do not copy, open, or manually edit `.env`. Create the credential-only file
through hidden input; the script writes it atomically with mode `600`:

```bash
bash scripts/runpod_workflow.sh credentials
```

##### TPEx Cloudflare Worker

When a RunPod datacenter receives HTTP 403 from TPEx data endpoints, deploy the
restricted Cloudflare Worker before using a Taiwan-market profile. Create a
Cloudflare API token with only **Workers Scripts Edit** permission (the API
documentation calls the same capability `Workers Scripts Write`), obtain the
32-character account ID, and run:

Use the following least-privilege settings on **Manage account → Account API
tokens → Create a token**. Do not select the **Edit Cloudflare Workers**
template shown on that page; it adds Workers Routes, KV, R2, Tail, and other
read permissions that this project does not use. Select **Start from scratch**
instead:

The current Account API token editor no longer presents `Account` as a nested
menu. In **Edit policy**, leave the top policy scope set to **Entire Account**,
then enter `Workers Scripts` in **Search for permission groups...**.
Alternatively, expand **Developer Platform** and find the same permission.
Select only **Edit** for **Workers Scripts**. `Developer Platform` is a UI
category, not an additional permission.

| Field | Setting |
| --- | --- |
| Token name | Use a descriptive purpose, such as `stock-forecasting-tpex-relay-deployer`. |
| Policy scope | Keep **Entire Account**. This is the account scope of the current account-owned token; it does not grant access to every Cloudflare account owned by the current user. Workers Scripts cannot be narrowed to a single script. |
| Permission policies | Search for `Workers Scripts`, or expand **Developer Platform**; select only **Edit** for **Workers Scripts** and do not add Read. The UI should show only one selected permission group. |
| Token expiration | **90 days** is recommended; use **1 year** when less frequent rotation is necessary. Avoid `No expiration`. |
| Client IP address filtering | Leave this blank for residential networks, dynamic addresses, VPNs, or mobile use. Restrict it only for a known fixed public egress address, using `/32` for one IPv4 address or `/128` for one IPv6 address. |

Do not add **Read all resources**, **Write all resources**, **Edit zone DNS**,
Workers Routes, Workers KV Storage, Workers R2 Storage, Workers Tail, or Account
Settings. `Workers Scripts Edit` already covers the subdomain lookup/creation,
Worker-module upload, and `workers.dev` activation operations used by the
deployer. The API token is used only by the local `tpex-proxy deploy` control
command. Its expiration does not disable an already deployed Worker or
interrupt a Pod using the relay; it only requires token rotation before a
future deployment.

On **Continue to summary**, verify that the summary contains only **Workers
Scripts Edit** for the current account and no other permission group, then
create the token. Cloudflare's permission reference defines it as the
account-scoped permission that grants write access to Worker scripts. The token
secret is displayed only once. Do not paste it into the README, a shell command
line, chat, or a screenshot; paste it directly into the hidden prompt from the
next `configure` command. The 32-character account ID is not the token secret
and can be found in the Cloudflare account URL or dashboard Overview. Do not
put a personal account ID in project documentation.

```bash
bash scripts/runpod_workflow.sh tpex-proxy configure
bash scripts/runpod_workflow.sh tpex-proxy deploy
```

`configure` stores the Cloudflare token through hidden input and generates the
shared relay token on first setup. If the Cloudflare account has no
`workers.dev` subdomain, enter a globally unique name at the prompt; leave it
blank for an existing account and the deployer queries Cloudflare. `deploy`
enforces these boundaries through the official Workers API:

`deploy` uses the existing `RUNPOD_API_KEY` in the local `.env` to call the
official GraphQL `secretCreate` mutation. `secretCreate` is an API operation,
not a separately selectable permission when creating a RunPod API key. There
is no need to replace the key in advance. If the current key is invalid,
read-only, or cannot create a Secret in its account/team scope, `deploy` stops
without activating the new Worker metadata in the local `.env`.

`configure` merges its updates into `.env` and preserves the existing RunPod
network-volume and S3 settings. If the volume is already deployed, do not rerun
`credentials` or `volume deploy`; after `deploy` succeeds, proceed directly to
"Verify S3 and upload source code" and run `sync --dry-run` and `sync --apply`.

- It uploads [`cloudflare/tpex-proxy/src/index.mjs`](cloudflare/tpex-proxy/src/index.mjs)
  with a `gcp:asia-east1` placement hint.
- It accepts only `GET`, the shared token, the fixed TPEx origin, the four paths
  used by this project, and each path's fixed query schema. If TPEx redirects a
  request, the Worker follows at most three hops and every hop must remain on
  the same HTTPS origin; cross-origin, missing-Location, and looping redirects
  are rejected. It is not a general or open proxy.
- It enables the script's `workers.dev` URL and stores the same relay token in
  a newly created RunPod Secret. The Cloudflare API token never enters a Pod.
- It runs an `exDailyQ` smoke test. The verifier performs bounded exponential
  backoff while a new `workers.dev` route propagates and reports failures using
  the HTTP status plus a safe Worker error code. Deployment passes only when
  the Worker actually returns official JSON containing a data table.

The Worker uses no KV, Durable Objects, database, or other paid binding. Each
TPEx cache miss consumes one Worker request and one upstream subrequest.
Whether usage remains inside the free allowance is governed by the account's
Cloudflare plan and the current official limits.

A Cloudflare placement hint moves execution near the requested region, but it
does not guarantee a fixed egress IP or permanent acceptance by TPEx. The smoke
test is therefore the live gate before CPU Pod creation. Re-run it at any time:

```bash
bash scripts/runpod_workflow.sh tpex-proxy verify
```

The TPEx client still hashes the original `https://www.tpex.org.tw` endpoint
and public query parameters for request identity. The Worker URL, relay token,
and transport mode do not enter raw-cache keys or dataset-request identity.
After switching transports, all successful TWSE, TPEx, and EODHD JSON cache
entries remain reusable; only missing TPEx responses pass through the Worker.
Official references:

- [Create a Cloudflare API token](https://developers.cloudflare.com/fundamentals/api/get-started/create-token/)
- [Cloudflare Account API tokens](https://developers.cloudflare.com/fundamentals/api/get-started/account-owned-tokens/)
- [Cloudflare API token template permissions](https://developers.cloudflare.com/fundamentals/api/reference/template/)
- [Cloudflare API token permission groups](https://developers.cloudflare.com/fundamentals/api/reference/permissions/)
- [Cloudflare token expiration and IP restrictions](https://developers.cloudflare.com/fundamentals/api/how-to/restrict-tokens/)
- [Cloudflare Workers API](https://developers.cloudflare.com/api/resources/workers/)
- [Cloudflare Worker placement](https://developers.cloudflare.com/workers/configuration/placement/)
- [Cloudflare Worker secrets](https://developers.cloudflare.com/workers/configuration/secrets/)
- [Cloudflare Workers limits](https://developers.cloudflare.com/workers/platform/limits/)
- [RunPod GraphQL `secretCreate`](https://docs.runpod.io/sdks/graphql/manage-pod-templates)

Create the network volume through the script. The returned volume ID,
datacenter, S3 region, and endpoint are written back to the same `.env`
automatically:

```bash
bash scripts/runpod_workflow.sh volume deploy \
  --name stock-forecasting \
  --size-gb 100 \
  --datacenter EU-RO-1
```

If `.env` already registers a volume, the script will not create another
potentially billable volume by default. Only an intentional `--force-new`
creates and registers a replacement.

Before creating a CPU Pod, select the stage, data source, date range, and
universe through the script. With no options, it opens an interactive menu:

```bash
bash scripts/runpod_workflow.sh configure
```

##### `configure` options and dataset scope

`--universe`, `--stocks`, `--etfs`, and `--symbol-limit` control **only the US
dataset scope**; they never filter Taiwan instruments. `us_tw_eodhd` always
combines the EODHD US target-security scope selected by `--universe` with the
TWSE/TPEx data that satisfies the common-stock/TDR/audited-unleveraged-equity-
ETF contract over the selected date range.

Available `--data-profile` values are:

| Value | US data | Taiwan data | Requires `eodhd_api_token` |
| --- | --- | --- | --- |
| `tw_only` | None | TWSE/TPEx common stocks, TDRs, audited benchmark-mappable unleveraged equity ETFs, and official benchmarks; `--universe` must be `all` | No |
| `us_only_eodhd` | EODHD common stocks (including ADRs) and audited benchmark-mappable unleveraged equity ETFs selected by `--universe`, plus the automatically added `VTI.US` benchmark | None | Yes |
| `us_tw_eodhd` | The same US security scope as `us_only_eodhd` | The same Taiwan target-security scope as `tw_only` | Yes |

Available `--universe` values are:

| Value | Meaning | Compatible options |
| --- | --- | --- |
| `all` | For profiles containing US data, use EODHD discovery for active and delisted common stocks/ADRs, then apply the audited unleveraged-equity-ETF allowlist. For `tw_only`, use the complete Taiwan target-security scope | US profiles may optionally use `--symbol-limit`; do not provide `--stocks` or `--etfs` |
| `explicit` | Restrict **only the US scope** and require at least one `--stocks` or `--etfs` value. The workflow still adds `VTI.US` automatically | Only `us_only_eodhd` and `us_tw_eodhd`; cannot be combined with `--symbol-limit` |

Here, the “complete US target-security scope” means active/delisted common
stocks (including ADRs) returned by EODHD discovery and the unleveraged equity
ETFs in the audited program allowlist. Use a US-containing profile with
`--universe all` and omit `--symbol-limit` entirely. Leveraged, inverse, bond,
commodity, and volatility ETFs cannot become training targets, and an explicit
benchmark mapping cannot bypass this restriction. The workflow cannot guarantee
instruments that the provider omits or the account cannot access.

All user-facing `configure` options are:

| Option | Values or format | Meaning and restrictions |
| --- | --- | --- |
| `--stage` | `stage1`, `stage2` | Select the fixed training config. Required in non-interactive mode |
| `--data-profile` | `tw_only`, `us_only_eodhd`, `us_tw_eodhd` | Select the actual provider and market combination. Required in non-interactive mode |
| `--dataset-revision` | 1-64 characters; start with an alphanumeric character, followed by alphanumerics, `.`, `_`, or `-`; default `v1` | Use a new label to force a new immutable dataset namespace after a provider revises historical data |
| `--start` | `YYYY-MM-DD`; default `2005-01-01` | Inclusive start date shared by every selected market; omission always selects `2005-01-01` |
| `--end` | `YYYY-MM-DD`; no default | Exclusive end boundary shared by every selected market; both interactive and non-interactive modes require an explicit user value instead of silently selecting the local current date |
| `--h-start` | `1`, `2`, or `3`; default `3` | First cumulative holding-day alpha horizon predicted after the close at `t`, through the fixed day 14; entry is still evaluated at the raw open of `t+1`. This is part of the processed-dataset and model-output contract |
| `--universe` | `all`, `explicit` | Select the US-instrument strategy; `tw_only` accepts only `all`. Required in non-interactive mode |
| `--stocks` | Comma- or space-separated US tickers; repeatable | US common stocks/ADRs in `explicit` mode, such as `"AAPL,BABA"`; provider type is verified against EODHD discovery and does not affect Taiwan data |
| `--etfs` | Comma- or space-separated US tickers; repeatable | Unleveraged US equity ETFs in `explicit` mode, such as `"SPY,QQQ"`; each ticker must be in the audited allowlist and does not affect Taiwan data |
| `--symbol-limit` | Positive integer N | **A bounded capacity/workflow check, not a complete-US-market mode.** Only valid for a US-containing `all` profile. After discovery, split allowlisted unleveraged equity ETFs from common stocks/ADRs, then keep up to N of each by active → delisted and ticker order; take all when a type has fewer than N. This is neither random nor representative sampling. Then ensure the required `VTI.US` benchmark is present: do not duplicate it if it is among the N ETFs, otherwise add it, so the raw universe is at most `2N+1`. Omit this option for the complete US target-security scope |
| `--interactive` | Flag with no value | Explicitly open the interactive prompts; invoking `configure` with no options enables this mode automatically |

Interactive defaults are `stage1`, `us_tw_eodhd`, dataset revision `v1`, start
date `2005-01-01`, `h_start=3`, and US universe `all`. `--end` intentionally has no default:
leaving its prompt blank asks again instead of selecting the local current date.
Provider acquisition policy is not configured here. Supplying `--max-api-calls`,
`--eodhd-qps`, `--taiwan-qps`, or `--maxBackoff` to this command is rejected as
an unknown option instead of creating another selection. The lower-level helper's
`--project-root` is injected by `runpod_workflow.sh`; it is not a user-facing
dataset-scope option and should not be supplied manually.

The following example has this exact scope:

- US: `AAPL.US`, `MSFT.US`, `SPY.US`, `QQQ.US`, plus the automatically added
  `VTI.US` benchmark.
- Taiwan: it is neither absent nor restricted to the four US tickers; it
  contains the complete market data and official benchmarks returned by the
  TWSE/TPEx endpoints over the same date range.
- Dates: both markets start on `2015-01-01` and end before `2026-07-27`;
  `2026-07-27` itself is excluded.

Non-interactive "explicit US universe plus complete Taiwan market" example:

```bash
bash scripts/runpod_workflow.sh configure \
  --stage stage1 \
  --data-profile us_tw_eodhd \
  --start 2015-01-01 \
  --end 2026-07-27 \
  --h-start 3 \
  --universe explicit \
  --stocks "AAPL,MSFT" \
  --etfs "SPY,QQQ"
```

Provider acquisition policy is supplied separately when creating the CPU Pod.
`--max-api-calls 10000` limits only EODHD network attempts in that acquisition.
It neither truncates either market to 10,000 rows nor requires the full dataset
to finish within 10,000 requests. At the ceiling, the EODHD loop exits while the
parallel TWSE/TPEx loops continue. CPU preparation saves cache and
`waiting_for_budget` progress only after all three loops exit.

Provider quotas are a separate boundary. EODHD's current pricing page lists the
personal `EOD Historical Data — All World` plan at USD 19.99 per month. Its
limits documentation gives paid plans a default 100,000 API calls per day and
1,000 HTTP requests per minute, with subscription daily limits resetting at
midnight GMT. The units are independent, and endpoints may consume different
numbers of billed calls. The actual subscription, used account quota, and
provider response headers remain authoritative. This project floors the default
pacing to 16 requests per second (960 per minute); lower `--eodhd-qps` further
if another client uses the same account concurrently. See
[EODHD Pricing](https://eodhd.com/pricing),
[EODHD API Limits](https://eodhd.com/financial-apis/api-limits) and the
[EODHD User API](https://eodhd.com/financial-apis/user-api). A complete US
target-security request plan may exceed `--max-api-calls`; the resumable CPU workflow
handles both project and provider boundaries across multiple attempts.

For the complete EODHD and Taiwan target-security scopes, provide none of
`--stocks`, `--etfs`, or `--symbol-limit`:

```bash
bash scripts/runpod_workflow.sh configure \
  --stage stage1 \
  --data-profile us_tw_eodhd \
  --start 2015-01-01 \
  --end 2026-07-27 \
  --universe all
```

Even when the post-discovery complete HTTP request estimate exceeds that CPU
Pod's `--max-api-calls`, it remains informational: it neither prevents Pod
creation nor shrinks the dataset. Each CPU attempt sends at most the configured
EODHD network attempts and the next Pod resumes from cache. Adjust
`cpu prepare --max-api-calls` for cost and usage control; it never bypasses
provider quotas.

To use the same explicit US universe while downloading **no Taiwan data**, set
the profile to `us_only_eodhd`:

```bash
bash scripts/runpod_workflow.sh configure \
  --stage stage1 \
  --data-profile us_only_eodhd \
  --start 2015-01-01 \
  --end 2026-07-27 \
  --universe explicit \
  --stocks "AAPL,MSFT" \
  --etfs "SPY,QQQ"
```

This `us_only_eodhd` example contains only those four US tickers plus the
automatically added `VTI.US`; it contains no TWSE or TPEx data.

For the complete Taiwan universe without EODHD:

```bash
bash scripts/runpod_workflow.sh configure \
  --stage stage1 \
  --data-profile tw_only \
  --start 2015-01-01 \
  --end 2026-07-27 \
  --universe all
```

The script creates `.runpod/selections/<selection-id>.json` and
`.runpod/active-selection.json`. They contain no secrets and are excluded by
`.gitignore`. Selection schema 3 binds the complete `h_start` preparation
contract. Older selections are not migrated automatically, so rerun `configure`
after updating the code. Changing `--end` intentionally creates a new dataset request
namespace; existing network-volume files are not deleted. A different profile,
date range, universe, symbol limit, data
preparation contract, or stage-config SHA-256 produces a different identity.
QPS, API budget, and maximum backoff are recorded only in CPU launch metadata,
download progress, and the download manifest; they never enter selection identity.

Changing only `h_start` creates a new dataset request SHA and rebuilds processed
windows, labels, train-only robust scales, and split audit. It does not change
dates, symbols, provider requests, or `cache_revision`. CPU acquisition scans
other dataset namespaces with the same `cache_revision=v1` as read-only cache
fallbacks, so an existing response with the same URL and parameters is recorded
as a cache hit and sends no network API call. Only responses that were missing or
failed previously, or whose request parameters changed, require provider access.

When provider history may have been revised and a new snapshot is intentional
for the same profile/date/universe, pass a new explicit
`--dataset-revision <label>` to `configure`. The CPU workflow never overwrites a
complete existing namespace and fails closed on a partial namespace.

Inspect the active selection without opening JSON:

```bash
bash scripts/runpod_workflow.sh selection show
```

`.env` stores only local RunPod/S3/Cloudflare deployment credentials plus
volume and TPEx Worker metadata written by the scripts. Stage, data range,
runtime, and config are never read from
`.env`. Do not `source .env`, and never put API keys, tokens, or secret values
in the README, configs, shell scripts, or commit history. Pods receive RunPod
Secret references and a non-secret Worker URL, not local account-level RunPod,
S3, or Cloudflare credentials.

#### 2. Verify S3 and upload source code

Run the read-only S3 access check, then preview the explicit upload allowlist:

```bash
bash scripts/verify_runpod_s3_access.sh
bash scripts/runpod_workflow.sh sync --dry-run
```

After reviewing the list, upload it and verify remote code readiness:

```bash
bash scripts/runpod_workflow.sh sync --apply
bash scripts/runpod_workflow.sh readiness --code-only
```

The uploader scans allowlisted source, configs, scripts, tests, `README.md`, and
`pyproject.toml` for secret patterns, uploads and size-checks every file, and
only then publishes `lifecycle/stage1/code.json`. It does not upload `.env`,
caches, data, checkpoints, or local artifacts. It also excludes `poetry.lock`;
the approved RunPod image regenerates that file for its Python/PyTorch/CUDA
environment on the network volume.

After any allowlisted source or config change, rerun `--dry-run`, `--apply`, and
the readiness check. A config change also invalidates the active selection, so
rerun `configure`. If the dataset marker is bound to an older code release or
selection, rerun CPU preparation as well. Never bypass the GPU gate.

#### 3. Deploy the model and offline data layer remotely

The CPU Pod accepts only the active selection. Pod creation fails before any
compute is rented when `configure` has not run, the config SHA changed, or the
selection JSON is incomplete. With no options, the local command is interactive:
it prompts for maximum workload runtime, the maximum additional EODHD network
attempts for this Pod, EODHD QPS, per-provider TWSE/TPEx QPS, time reserved for
data cleaning/window construction, the maximum retry backoff shared by all three
providers, vCPU count, and CPU flavor. `--max-api-calls` has no Enter-to-accept
default and must be
entered explicitly from the account's current remaining quota. Pressing Enter
accepts 6 hours, 16 EODHD QPS, 0.5 QPS for each Taiwan provider, an automatic
reserve, 1 minute, 8 vCPUs, and `cpu3g`. The automatic reserve is 25% of max
runtime, capped at 2 hours; the six-hour default reserves 90 minutes.
The final prompt requires `y` or `yes` before creating a potentially billable
Pod; Enter, `n`, or `no` cancels safely:

```bash
bash scripts/runpod_workflow.sh cpu prepare
```

Providing any option selects non-interactive mode. This mode requires an explicit
`--max-api-calls`; omitted QPS and resource options retain their defaults.
Automation can therefore provide all values without editing `.env` or rerunning
`configure`:

```bash
bash scripts/runpod_workflow.sh cpu prepare \
  --max-api-calls 80000 \
  --eodhd-qps 16 \
  --taiwan-qps 0.5 \
  --maxRuntime 10h \
  --prepareReserve 2h \
  --maxBackoff 1m \
  --cpuNumber 16 \
  --cpuFlavor cpu5g
```

Add `--interactive` to use command-line values as prompt defaults that the
user can confirm or replace:

```bash
bash scripts/runpod_workflow.sh cpu prepare \
  --interactive \
  --max-api-calls 80000 \
  --eodhd-qps 16 \
  --taiwan-qps 0.5 \
  --maxRuntime 10h \
  --prepareReserve auto \
  --maxBackoff 1m \
  --cpuNumber 16 \
  --cpuFlavor cpu5g
```

`--max-api-calls` is a positive integer counting only EODHD cache misses and
retries in this CPU Pod. It is not the account's total daily quota, and a new
Pod does not automatically subtract account usage by earlier Pods.
`--eodhd-qps` and `--taiwan-qps` must be positive, defaulting to `16` and `0.5`.
The Taiwan value is per provider, so concurrent TWSE and TPEx loops each have an
independent 0.5-QPS limiter. `--maxRuntime`, an explicit `--prepareReserve`, and
`--maxBackoff` accept a positive integer followed by `m`, `h`, or `d`.
`--prepareReserve auto` uses the
formula above; an explicit reserve must be shorter than max runtime.
`--maxBackoff` defaults to `1m` and applies to EODHD, TWSE, and TPEx. Only the
affected provider loop exits when its next exponential or `Retry-After` delay
would **exceed** this limit; a delay equal to the limit is still performed.
EODHD stops at whichever comes first: `--max-api-calls` exhaustion or this
backoff boundary. `--cpuNumber` must be
between 1 and 32. `--cpuFlavor` accepts only the six RunPod values below; any
other value or more than 32 vCPUs fails before Pod creation:

| Flavor | Generation | Type | RAM / vCPU | RAM at 32 vCPUs | Container-disk limit |
| --- | ---: | --- | ---: | ---: | ---: |
| `cpu3c` | CPU3 | Compute-Optimized | 2 GB | 64 GB | 10 GB/vCPU |
| `cpu3g` | CPU3 | General Purpose | 4 GB | 128 GB | 10 GB/vCPU |
| `cpu3m` | CPU3 | Memory-Optimized | 8 GB | 256 GB | 10 GB/vCPU |
| `cpu5c` | CPU5 | Compute-Optimized | 2 GB | 64 GB | 15 GB/vCPU |
| `cpu5g` | CPU5 | General Purpose | 4 GB | 128 GB | 15 GB/vCPU |
| `cpu5m` | CPU5 | Memory-Optimized | 8 GB | 256 GB | 15 GB/vCPU |

The default container-disk request is 30 GB. If a small vCPU count makes that
value exceed the table's limit, the creator caps it at the legal flavor limit.
An explicit `RUNPOD_CPU_CONTAINER_DISK_GB` value above that limit fails before
Pod creation.

The command prints the Pod ID, external hard-limit guard, and the workflow to
run after SSH login. Obtain the SSH command from the RunPod Console Connect
page. Inside the Pod, run:

```bash
cd /runpod-volume/stock_forecasting
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
3. Revalidates the stage, profile, dates, universe, config SHA-256, and Pod
   environment from the mounted immutable selection before downloading. Cache
   hits do not call the provider again. The script reads the requested vCPU
   count, RunPod's `RUNPOD_CPU_COUNT`, and the cores visible to the container,
   then uses the minimum as its worker count. EODHD, TWSE, and TPEx run as three
   independent top-level acquisition loops; each loop uses a thread pool across
   instruments or actual sessions listed by official benchmark history.
   Causal-window construction runs across symbols, and pytest workers never
   exceed the effective CPU count. Each provider has its own QPS limiter.
   `--max-api-calls` limits only EODHD; TWSE/TPEx have no project-side request
   counter ceiling. All three providers use the same `--maxBackoff` value for
   their independent retry boundary. One provider exiting never
   cancels the other two. The process joins all loops before deterministically
   merging provider-local Parquet/request-log parts or publishing a resume state. The
   complete-plan estimate is informational. The workflow automatically reserves
   25% of max runtime for cleaning/window construction (90 minutes for the
   default six-hour runtime and capped at 2 hours), configurable with
   `--prepareReserve`.
4. Creates and verifies these persistent artifacts:
   | Remote path                                                                        | Contents                                                    |
   | ---------------------------------------------------------------------------------- | ----------------------------------------------------------- |
   | `/runpod-volume/datasets/<dataset-request-sha256>/api-cache/`                     | Provider raw-response cache                                 |
   | `/runpod-volume/datasets/<dataset-request-sha256>/download-progress.json`          | Resume attempt, cache counts, and provider/budget/runtime wait state |
   | `/runpod-volume/datasets/<dataset-request-sha256>/raw/market.parquet`             | Canonical daily OHLCV in the durable `downloaded` checkpoint |
   | `/runpod-volume/datasets/<dataset-request-sha256>/processed/windows.parquet`      | Causal training windows                                     |
   | `/runpod-volume/datasets/<dataset-request-sha256>/download-manifest.json`         | Actual providers, profile, symbols, and download provenance |
   | `/runpod-volume/datasets/<dataset-request-sha256>/dataset-manifest.json`          | Split counts, hashes, and data contract                     |
   | `/runpod-volume/datasets/<dataset-request-sha256>/manifests/api-request-log.jsonl` | Request audit without tokens                               |
   | `/runpod-volume/cache/huggingface/`                                             | Offline Kronos model/tokenizer cache                        |
   | `/runpod-volume/cache/hf-models.json`                                           | Pinned model revisions and cache manifest                   |
5. Publishes a `downloaded` lifecycle after validating raw Parquet, the download
   manifest, and the request log. A `downloaded` marker without an exit code is an
   intermediate checkpoint in the same Pod, so the external guard does not treat it
   as terminal. Only when the remaining time is below the cleaning reserve does exit
   code 75 make it a resumable terminal state; the next CPU Pod then cleans directly
   from the checkpoint without provider calls. It binds the selection ID/SHA,
   dataset request SHA, stage/config SHA, requested
   profile/date/universe, and resolved artifact hashes into
   `/runpod-volume/lifecycle/stage1/dataset.json`. It publishes the marker only
   after every check passes, then terminates automatically.

Before any CPU or GPU workflow reads or writes persistent data, it proves that
`/runpod-volume` is the **exact mount point**: use `mountpoint` first, then
`findmnt`, and finally `/proc/self/mountinfo`. The volume ID selected locally is
passed as `RUNPOD_EXPECTED_VOLUME_ID` and compared byte-for-byte with RunPod's
automatically supplied `RUNPOD_VOLUME_ID`; a correct-looking path with the wrong
ID fails immediately. The project lives at
`/runpod-volume/stock_forecasting`, while data, the Kronos cache, W&B
transactions, and models use separate children of the volume root. Persistent
paths may never use the restart-cleared `/workspace`. This layout keeps the
project itself from becoming the mount target and prevents a network-volume
mount from hiding a same-named image directory.

After the Pod terminates, use the S3 lifecycle as the authority instead of the
now-unavailable SSH session:

```bash
bash scripts/runpod_workflow.sh status
```

##### Provider quotas and cross-Pod resume

EODHD, TWSE, and TPEx use independent parallel loops. Retryable HTTP 429,
temporary network failures, and provider 5xx responses use exponential backoff.
A temporary HTTP 403 from a Taiwan official endpoint due to Pod IP/WAF policy is
also retryable. All three providers share one `--maxBackoff` setting (default
`1m`) but maintain independent backoff state and exit when their next delay
would exceed it. EODHD is additionally bounded by the CPU attempt's
`--max-api-calls`; whichever EODHD boundary is reached first stops its loop.
TWSE/TPEx have no request-count ceiling. The shared acquisition deadline can
still place any loop in `waiting_for_resume` to protect cleaning time. The
workflow follows this contract:

1. Every successful raw JSON response remains in the dataset request's
   `api-cache/`. No incomplete Parquet or `state=ready` marker is published.
2. If EODHD reaches `--max-api-calls` or its backoff boundary first, only its
   loop exits; TWSE/TPEx continue. A Taiwan provider crossing its backoff boundary
   likewise does not stop EODHD or the other Taiwan provider. Crossing the
   boundary opens a shared circuit breaker for that provider client. Requests
   already in flight may finish, but sibling workers cannot start another request
   for the stopped provider. The main process always waits for every selected
   provider loop to exit before the CPU Pod may finish.
3. `download-progress.json` records the attempt, per-provider cache/network
   counts, the limited EODHD count, and each provider's `complete`,
   `waiting_for_budget`, `waiting_for_provider`, or `waiting_for_resume` outcome.
   Provider errors also record cumulative wait, last wait, next proposed backoff,
   the shared maximum, and the launch's effective `max-api-calls`, EODHD QPS,
   and Taiwan QPS. HTTP status, `Retry-After`, and rate-limit headers are stored
   when supplied. Data-contract errors additionally identify the safe provider,
   operation, symbol/date/month, and exception type. Tokens and response bodies
   are not stored.
4. If any loop is incomplete, the aggregate lifecycle becomes
   `waiting_for_budget`, `waiting_for_provider`, or `waiting_for_resume`; GPU
   readiness stays blocked, and only then does the CPU Pod terminate. If all
   loops complete, provider-local parts are merged in a fixed order and cleaning
   continues instead of closing the Pod early.
5. After quota becomes available, **do not reconfigure, change
   `--dataset-revision`, or delete the cache**. When creating the next CPU Pod,
   enter a new `--max-api-calls` value for the current remaining quota. QPS may
   also change for that attempt without changing the selection. Then connect to
   it and start the same workflow:

   ```bash
   bash scripts/runpod_workflow.sh cpu prepare
   # Run after connecting to the newly created CPU Pod:
   cd /runpod-volume/stock_forecasting
   bash scripts/runpod_tmux_launch.sh cpu-prepare
   ```

6. The new attempt replays cached responses under the same dataset request
   identity and calls the provider only for missing requests. It rebuilds
   Parquet from the complete response set. The lifecycle becomes `ready` only
   after all data, manifests, and selection gates pass. The `all`-mode discovery
   response is part of the same immutable cache, so a cross-day resume does not
   replace it with a drifted instrument list.

EODHD occasionally returns all-zero or otherwise invalid placeholder rows inside
an instrument's daily history. The downloader neither imputes nor rewrites those
prices: it drops only source rows that cannot satisfy the canonical OHLCV
contract, counts them in `dropped_source_rows`, and retains the instrument's
valid observations. Early TWSE actions can exist in the `TWT49U` main table while
the detail endpoint returns no record. In that case the verified price factor is
retained, the unknown share multiplier stays at identity `1.0`, and
`missing_share_multiplier_details_by_provider` explicitly records the resulting
volume-adjustment coverage gap instead of inventing a ratio.

`bash scripts/runpod_workflow.sh status` prints the download attempt, cached
response count, network requests in that attempt, the full request estimate when
available, and a safe provider-error summary below the dataset lifecycle, so no
JSON file needs to be opened manually.

This is request-level resume, not byte-range resume within one HTTP response.
Each successfully completed API request is the resume unit. A progress identity
that differs from the current profile, dates, universe, or dataset request
fails closed. An undersized `--max-api-calls` produces resumable
`waiting_for_budget`, not failure; create another CPU Pod with the same selection.
A 401/403 or other non-temporary configuration error produces `failed` and
requires correcting the Secret. Change
`--dataset-revision` only for an intentional new provider snapshot; a new
revision does not reuse the old snapshot cache.

If the state is `waiting_for_budget`, `waiting_for_provider`,
`waiting_for_resume`, `failed`, or `timed_out`, first read
`launch_id`, `log_path`, and any `progress_path` from the lifecycle. The tmux log directory is
`logs/tmux/fin-ts-cpu-prepare/<launch-id>/`; let the script resolve and download
it into the local diagnostic directory:

```bash
bash scripts/runpod_workflow.sh cpu-logs
```

Do not rent a GPU until this gate passes:

```bash
bash scripts/runpod_workflow.sh readiness --gpu
```

Stage 2 uses the same data namespace when its dataset request is identical, but
trains on 100% of the train split from the same pretrained base. It does not
continue from a Stage 1 checkpoint. Run `configure --stage stage2` again and
explicitly supply the same profile/date/universe/dataset revision/`h_start` as
the intended CPU dataset.
The GPU gate rejects an old Stage 1 marker when the stage/config identity has
changed. For example:

```bash
bash scripts/runpod_workflow.sh configure \
  --stage stage2 \
  --data-profile tw_only \
  --start 2015-01-01 \
  --end 2026-07-27 \
  --h-start 3 \
  --universe all
bash scripts/runpod_workflow.sh sync --apply
bash scripts/runpod_workflow.sh cpu prepare
```

After creating a Stage 2 CPU Pod, follow its SSH prompt and run
`bash scripts/runpod_tmux_launch.sh cpu-finalize`; do not use the Stage 1
`cpu-prepare` name to download the data again.

After the CPU workflow completes, rerun
`bash scripts/runpod_workflow.sh readiness --gpu` locally. If only the stage is
changed but profile/date/universe differs from the prior CPU deployment, the
dataset request SHA also differs and the old Parquet cannot be selected.

#### 4. Create a GPU Pod and train

List the currently available complete `gpuId` values. The default is
`NVIDIA GeForce RTX 5090`:

```bash
bash scripts/runpodctl_project.sh gpu list
```

Inspect the active selection and pass the GPU gate. Before renting the GPU, the
gate compares the local selection, S3 selection, CPU marker, code release,
config SHA, and namespaced artifacts:

```bash
bash scripts/runpod_workflow.sh selection show
bash scripts/runpod_workflow.sh readiness --gpu
```

Create a Pod with the default GPU and a 12-hour workload limit:

```bash
bash scripts/runpod_workflow.sh train
```

Or set the workload limit and one complete `gpuId` from the list directly:

```bash
bash scripts/runpod_workflow.sh train \
  --maxRuntime 18h \
  --gpuId "NVIDIA GeForce RTX 5090"
```

`--maxRuntime` bounds the combined training-and-validation workload. The
external guard and RunPod `--terminate-after` retain one additional hour only
for terminal-lifecycle publication and failure cleanup; it is not extra model
runtime. `MAX_RUNTIME_SECONDS` also overrides the resolved Pydantic runtime
config, so W&B records the CLI value instead of the YAML's original six-hour
default.

The creator first allocates one run ID locally, mounts the same network volume,
injects a W&B Secret reference, and arms an independent hard-limit guard. On
macOS, the local launcher automatically runs `caffeinate -is -w <guard-pid>` and
writes guard, caffeinate, and keep-awake state beside the reported guard log.
Do not stop the guard process or power off the control machine. Pod-side
self-termination remains primary, and RunPod `--terminate-after` is a third
deadline. SSH through the Console and run:

```bash
cd /runpod-volume/stock_forecasting
bash scripts/runpod_tmux_launch.sh stage1-train
```

The immutable selection maps to the `RUNPOD_CONFIG` that determines whether
Stage 1 or Stage 2 runs; the `stage1-train` command name does not override that
selection. Attach for live observation:

```bash
tmux -L fin-ts-stage1-train attach -t fin-ts-stage1-train
```

Training verifies the image runtime, CUDA, mounted readiness, dataset/model
manifests, and run identity before writing
`/runpod-volume/savedModel/<run-id>/`. Both production configs set
`validation.auto_run_after_training: true`, so the same GPU workflow
automatically executes the complete validation benchmark after training.
The GPU Pod terminates after training plus validation succeeds. A successful
CPU preparation and any CPU/GPU workflow failure or timeout likewise publish
terminal lifecycle/tmux status before terminating. If the Pod-side API call
fails, the local guard takes over using the same Pod ID, run ID, and lifecycle.
Pods with network volumes are always terminated, never stopped.

##### W&B logging and offline recovery

The W&B run config contains the full resolved YAML/Pydantic configuration,
system metadata, dataset provenance, and immutable selection identity.
`train/loss` and `train/pinball_loss` are logged at **every optimizer step**
against the `trainer/global_step` custom axis. With gradient accumulation, the
value is the mean of the microbatch losses contributing to that optimizer step.
Loss and validation retain separate history rows even when they share an
optimizer step instead of colliding on W&B's internal step. In-training
validation sends every finite numeric metric at each evaluation step. The
post-training benchmark uses `benchmark_validation/global_step` as a custom
axis for the evaluated checkpoint instead of writing to an already committed
internal W&B step. It logs the complete model-and-baseline results, including
cross-sectional Sharpe, RankIC, turnover, drawdown, coverage, and subgroup
metrics—not only the final loss.

Each run atomically records delivery state at
`/runpod-volume/lifecycle/runs/<run-id>/wandb.json`.
`bash scripts/runpod_workflow.sh status` prints separate `training` and
`validation` components. Only `online_finished` or `synced` confirms completion;
`offline_pending`, `sync_failed`, or `online_running` after a terminal workflow
requires recovery. If online initialization fails, production configs permit
an offline transaction on the network volume. If that offline transaction also
cannot be created, the workflow records `failed` and aborts. W&B officially
supports later synchronization, so a temporary server outage does not require
discarding otherwise completed model work.

Inside a project-created GPU Pod with the `wandb_api_key` Secret injected,
**do not start the training or validation tmux workflow**. Run the recovery
script directly; omitting the run ID scans all pending runs:

```bash
cd /runpod-volume/stock_forecasting
bash scripts/runpod_wandb_sync.sh <run-id>
```

The script processes only `online_running`, `offline_pending`, and
`sync_failed` components. It invokes `wandb sync --legacy --include-offline
--include-online --append --id <run-id>` on the recorded transaction directory.
The project pins W&B 0.28.0. The explicit `--legacy` selects that version's
legacy sync path because `--include-offline` and `--append` are legacy-only in
that version. It
then records `synced` or `sync_failed`. Every transaction created by resumed
training or repeated validation is retained and synced separately, rather than
only the final segment. The script then terminates the recovery Pod. For a
completed training run, use `validate` to provision this Pod; for an incomplete
run with a retained checkpoint, use `resume`, then execute the sync script
instead of tmux. It can also run before work starts on the next existing project
GPU Pod, avoiding a separate rental. See W&B's official
[offline-mode reference](https://docs.wandb.ai/models/ref/python/functions/init)
and [`wandb sync` reference](https://docs.wandb.ai/models/ref/cli/wandb-sync).

After termination, use the workflow status command to query the latest terminal
states stored on S3:

```bash
bash scripts/runpod_workflow.sh status
```

Only `state=ready` means that lifecycle completed. Treat `failed` and
`timed_out` as incomplete. `wandb_run_id` is the shared `<run-id>` for
checkpoints, evaluations, W&B, and run-scoped logs. SSH/tmux is only for live
debugging while a Pod still exists; it is not the authority for terminal state.

If the training loop has not completed when `--maxRuntime` expires, the
training lifecycle is `timed_out` (`failed` for an execution error) and
`training_completed` is not true. The command below selects the run's latest
complete retained checkpoint after hash and resume-contract validation, then
creates a new Pod with the same run ID, optimizer state, and scheduler state:

```bash
bash scripts/runpod_workflow.sh resume <run-id>
# Optional overrides:
bash scripts/runpod_workflow.sh resume \
  --maxRuntime 18h \
  --gpuId "NVIDIA GeForce RTX 5090" \
  <run-id>
```

Without `<run-id>`, only the latest `failed` or `timed_out` training lifecycle
whose training phase is incomplete is accepted. If immutable
`training-completed.json` exists, `resume` rejects further training and directs
the operator to standalone validation. After SSH login, run
`bash scripts/runpod_tmux_launch.sh stage1-train` as usual; resumed training is
followed by validation, terminal-state publication, and Pod termination.

If training completed but validation must be rerun independently, create a
validation Pod locally. The active selection must match the stage and dataset
identity stored by that run.

```bash
bash scripts/runpod_workflow.sh validate <run-id>
# Optional overrides; defaults are 12h and NVIDIA GeForce RTX 5090:
bash scripts/runpod_workflow.sh validate \
  --maxRuntime 8h \
  --gpuId "NVIDIA GeForce RTX 5090" \
  <run-id>
```

After SSH login, run:

```bash
cd /runpod-volume/stock_forecasting
bash scripts/runpod_tmux_launch.sh stage1-validate
```

Standalone validation reuses the completed training run's best checkpoint and
W&B run ID. Completion, failure, or timeout publishes the validation lifecycle
and automatically terminates the GPU Pod. It cannot replace unfinished
training; use `resume` for that case.

#### 5. Download all retained checkpoints or the best checkpoint and validation results

There is no need to read the volume ID from `.env`, parse lifecycle JSON, or
assemble S3 paths manually. First inspect known lifecycle markers:

```bash
bash scripts/runpod_workflow.sh status
```

Without a run ID, the download script accepts only the latest training
lifecycle with `state=ready`, then resolves its run ID automatically.
`--checkpointScope` accepts `all` or `best` and defaults to `all`;
`--checkpoint-scope` is an equivalent alias. The default command downloads
every retained checkpoint currently listed by the checkpoint leaderboard:

```bash
bash scripts/runpod_workflow.sh download
```

You may explicitly select all retained checkpoints or only the
validation-selected best checkpoint. Omit the run ID to use the latest ready
run, or provide it explicitly:

```bash
bash scripts/runpod_workflow.sh download --checkpointScope best
bash scripts/runpod_workflow.sh download --checkpointScope all <run-id>
bash scripts/runpod_workflow.sh download --checkpointScope best <run-id>
```

Here, `all` means the complete set still retained by the leaderboard's top-k
retention policy; it does not include historical checkpoints already deleted
during training. The download uses the immutable leaderboard directly rather
than guessing checkpoints from a directory listing. If the local target already
exists, use `--resume` explicitly before the script fills or refreshes the
selected set of known files:

```bash
bash scripts/runpod_workflow.sh download <run-id>
bash scripts/runpod_workflow.sh download --resume <run-id>
bash scripts/runpod_workflow.sh download --resume --checkpointScope best <run-id>
```

Results are written under the ignored
`artifacts/runpod/<run-id>/` directory. They include the run manifest, resolved
config, leaderboard, best-checkpoint pointer, checkpoint directories selected
by the scope, validation benchmark, training/validation lifecycle files, and
immutable training completion record. `--resume` does not delete other local
checkpoint directories; for example, switching from `all` to `best` does not
prune local files.

Each downloaded checkpoint directory should contain at least `adapter.safetensors`,
`resolved-config.yaml`, `trainer-state.json`, and the optimizer/scheduler state
listed by that trainer state. `validation-benchmark.json` is the complete
numerical validation and baseline comparison. Do not claim run success from a
W&B chart or README alone; inspect the lifecycle `state`, run ID, result path,
and downloaded raw JSON together.

The active immutable selection and its bound config SHA control the selected
stage. Names containing `stage1-*` in operation commands, tmux sessions, or
lifecycle paths do not override that selection.

### Training and inference artifacts

Each run stores at least:

- `adapter.safetensors`: trainable LoRA, resampler, benchmark-conditioner, and alpha-head weights.
- `resolved-config.yaml`
- `trainer-state.json`
- Optimizer and scheduler state
- Run manifest, checkpoint leaderboard, and best-checkpoint pointer
- Selection ID/SHA, dataset request SHA, stage-config SHA, and requested dataset contract
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
  --input "${DATA_ROOT}/raw/market.parquet" \
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
4. The CPU marker and active training selection match exactly on stage, config
   SHA, profile, dates, requested universe, and dataset request SHA. For example,
   CPU `tw_only` versus GPU `us_tw_eodhd` fails closed before Pod creation.
5. `alpha_quantiles` remains fixed at `[B,15-h_start,3]`, with
   `h_start in {1,2,3}`, a fixed maximum horizon of 14, and no classifier, text,
   or fact output.
6. Model inputs, time splits, and normalization use no future information;
   future benchmark values exist only in offline label construction.
7. The model is compared with zero-return, momentum, technical, GBDT, and neural
   baselines under one validation/test protocol.
8. Reports include per-horizon normalized pinball, median correlation,
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
- [EODHD delisted data coverage](https://eodhd.com/financial-apis/delisted-stock-companies-data-2)
- [TWSE OpenAPI](https://openapi.twse.com.tw/)
- [TWSE ex-right/ex-dividend calculation](https://www.twse.com.tw/en/announcement/ex-right/twt49u.html)
- [TPEx OpenAPI](https://www.tpex.org.tw/openapi/)
- [TPEx return index](https://www.tpex.org.tw/web/stock/iNdex_info/reward_index/ROE.php?l=en-us)
- [Massive stocks pricing](https://massive.com/pricing?product=stocks)
- [Massive market-data terms](https://massive.com/legal/market-data-terms-of-service)
