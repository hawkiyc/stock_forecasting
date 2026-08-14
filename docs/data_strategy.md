# 日線 OHLCV 資料策略

## 中文

### 1. 目標與非目標

PoC 的資料範圍是美國與台灣股票、ETF 日線 OHLCV。任何外部 API 呼叫都只能發生
在獨立 CPU preparation 階段；資料必須先落地為 immutable Parquet 與 manifest，
GPU 訓練、validation 與 inference 才能讀取。訓練迴圈禁止動態下載。

第一版不追求 tick、分鐘線、期貨、選擇權、新聞或財報。選擇權預訓練資料不是
backbone 的硬性必要條件；本 PoC 只需確認選定 foundation model 曾以大量金融
K-line 預訓練。

### 2. Provider 決策

| 市場／通道 | 選擇 | 理由 | 主要風險與限制 |
|---|---|---|---|
| 美國 PoC | EODHD | 一個 symbol 的日線歷史可用單次 EOD request；提供 adjusted close、symbol discovery、delisted 選項與 split calendar；價格與用量比商業級 feed 適合 side-project | 不是交易所級 truth；方案額度與授權需依實際訂閱確認；跨 provider 需抽樣對帳 |
| 台灣上市 | TWSE 官方 | 官方市場日報、除權息資料與 TAIEX/報酬指數 | endpoint schema 可能調整；逐交易日下載需低 QPS、cache 與重試 |
| 台灣上櫃 | TPEx 官方 | 官方市場日報、除權息資料與櫃買報酬指數 | 同樣需低 QPS、cache、schema 驗證 |
| 美國擴充 | Massive Developer | 保留型別化 provider 與 profile | side-project 目前沒有足夠商業授權，因此 fail closed，不執行下載 |

程式不把 provider 公布上限直接當成安全 QPS。預設客戶端 cap 是 EODHD `5 QPS`、
每個台灣官方 provider `0.5 QPS`，且由 `.env` 調低或依實際方案審慎調整。所有實際
network attempts（包含 retry）共用 `STAGE1_MAX_API_CALLS`；cache hit 不扣新的
network attempt。這些值是專案端的保守流量控制，不是對 provider SLA 的宣稱。

### 3. 可選 dataset profiles

| profile | `selected_datasets` | EODHD token | 狀態 |
|---|---|---|---|
| `tw_only` | `tpex_official`, `twse_official` | 不需要 | 可用、最低 API 成本 |
| `us_only_eodhd` | `eodhd_us` | 需要 | 可用 |
| `us_tw_eodhd` | `eodhd_us`, `tpex_official`, `twse_official` | 需要 | 預設 PoC |
| `us_tw_massive` | `massive_us`, `tpex_official`, `twse_official` | 不適用 | 介面保留、尚未實作 |

profile 會寫入 download manifest、ready dataset manifest、training result 與 inference
provenance。`selected_datasets` 是實際選擇的邏輯來源清單；manifest 另列實際 provider、
market、symbol、asset type、日期與 split counts。日後若退回 `tw_only`，每個輸出都能
清楚區分資料範圍。

### 4. Universe 管理

EODHD profile 可指定 `STAGE1_US_SYMBOLS` 與 `STAGE1_US_ETF_SYMBOLS`，或把兩者留空
以執行 active/delisted discovery。`STAGE1_SYMBOL_LIMIT` 在套用後仍會自動補入
`VTI.US` benchmark，避免目標 symbol 存在但 benchmark 缺失。

第一次 PoC 建議明確指定少量股票與 ETF；完整 discovery 可能同時受到 API 額度、
訂閱權限、CPU preparation 時間、Parquet 大小與記憶體限制。raw Parquet 是串流分批
寫入，但 processed windows 及訓練 dataset 目前仍載入記憶體；全市場長歷史之前應先
實作 partitioned/lazy windows。

台股官方日報會取得全市場可解析股票與 ETF。是否可成為 target 仍由 benchmark policy
決定；不合適的 ETF 會在 window construction fail closed，而不是被錯誤對到大盤。

### 5. Canonical raw schema

必要欄位：

```text
timestamp
symbol
asset_type
open
high
low
close
volume
adjusted_close
split_adjusted_volume
adjustment_source
provider
market
currency
source_symbol
is_active
dataset_profile
```

`open/high/low/close/volume` 永遠是 provider raw fields；不能用 adjusted data 覆寫。
`adjusted_close` 是 total-return anchor，`split_adjusted_volume` 只包含 share-change
調整。所有 timestamps 正規化為 UTC，symbol 使用 canonical suffix，例如
`AAPL.US`、`0050.TW`、`6488.TWO`、`TAIEX.TW`、`TPEX.TWO`。

schema validator 會拒絕 duplicate `(symbol,timestamp)`、非正值 price、不合理 OHLC
range、負 volume、非有限數值與未知 asset type。

### 6. Adjustment 與 benchmark series

EODHD：

- EOD endpoint 提供 raw OHLCV 與 adjusted close。
- 2015-01-01 起的日期範圍使用 calendar/splits，在整個 US universe 只抓一次，再依
  symbol 套用 share multiplier。
- calendar 官方完整度只從 2015 年開始；更早的日期範圍改用每個 symbol 的
  Historical Splits API。此額外成本會在任何歷史資料請求前納入 `max_api_calls`
  預檢，不能用不完整 split history 靜默產生 volume anchor。
- `adjusted_close` 保持 vendor total-return series，`split_adjusted_volume` 由 split
  event 產生。

TWSE/TPEx：

- 每個交易日抓 market-wide raw OHLCV。
- 每段日期各抓官方除權息資料；TWSE 權事件必要時讀 detail 取得無償配股比例。
- 公司行動建立 price factor 與 share multiplier，再保留 raw fields 並新增 adjusted
  anchors。
- `TAIEX.TW` 以官方 TAIEX price-index OHLC 配對官方發行量加權股價報酬指數。
- `TPEX.TWO` 以官方櫃買 price-index OHLC 配對官方櫃買報酬指數。

樣本視窗再把 total-return 與 split factors 正規化到 `cutoff_at`，確保 point-in-time
因果性。這也讓 vendor 對全部 adjusted close 乘上共同常數時，模型 input 與 label
不受影響。

### 7. Benchmark mapping

預設：US → `VTI.US`、TWSE → `TAIEX.TW`、TPEx → `TPEX.TWO`。ETF 使用小型且明確的
本土股票曝險 allowlist。`benchmark_mapping_path` 是 JSON object：

```json
{
  "CUSTOM.US": "VTI.US"
}
```

mapping 內容的 canonical JSON SHA-256 會寫入 preparation spec；修改 mapping 後舊
dataset readiness 不再有效，必須重建 processed data。

### 8. API 與 immutable artifact 流程

```text
provider API
  -> content-addressed raw JSON cache (token excluded from identity)
  -> api-request-log.jsonl (no secret)
  -> raw/market.parquet
  -> download-manifest.json (state=downloaded)
  -> causal windows + chronological purge/embargo
  -> processed/windows.parquet
  -> dataset-manifest.json (state=ready)
```

下載器具備 provider-level throttle、bounded exponential retry、shared fail-closed request
budget、cache reuse、staging writes 與拒絕 silent overwrite。manifest 儲存 SHA-256、
size、row count、profile、providers、symbols、date range、quality summary 與 request-log
artifact。secret-like key 不得寫入 manifest。

RunPod CPU wrapper 發布到固定 `DATA_ROOT`。要建立另一資料版本，應使用新的版本化
`DATA_ROOT` 或新的 network volume；不要刪除或覆寫既有 immutable artifacts。

### 9. Processed record contract

每筆 schema `3.0` record 包含：

- `context`：商品截至 `cutoff_at` 的 point-in-time adjusted OHLCV。
- `benchmark_context`：同日期、同長度的 benchmark adjusted OHLCV。
- `label.alpha_log_returns`：3–14 日 benchmark-relative execution log returns。
- `label.asset_total_returns` 與 `benchmark_total_returns`：只供 audit，不進 input。
- `diagnostics.capm_abnormal_return=null`：預留 diagnostic，不是 target。
- provider、market、benchmark policy、dataset profile 等 metadata。

entry/exit label dates 不會被序列化到任一 context。缺 benchmark 日期、極端 adjusted
transition、歷史不足或 benchmark mapping 不明確的樣本會被排除並寫入 window audit。

### 10. Split、purge 與 robust scale

先依時間建立 windows，再以 chronological 70% train、15% validation、15% test
切分，並在邊界套用 purge 20 bars 與 effective embargo 14 bars。sample stride 是 1。
RunPod 穩定 shell 仍傳入 legacy `stride=5`、`embargo=5` readiness sentinels；ready
manifest 同時記錄 effective values，訓練前會雙重驗證。

每個 horizon 的 loss scale 只從 train split label 計算：

```text
max(IQR, 1.4826 * MAD, 1e-4)
```

validation/test 不參與 scaling。test 保持 sealed，直到研究流程明確 unlock。

### 11. 資料 QA 與已知限制

必查項目：duplicate bars、OHLC consistency、缺值、極端 adjusted transitions、
benchmark calendar gaps、symbol/date coverage、corporate-action 前後連續性、split volume
方向，以及 EODHD 與其他來源重疊標的抽樣對帳。

目前限制：

- EODHD PoC 資料不等於 exchange-grade truth。
- 官方 endpoint schema 可能更動，remote tests 必須驗證 parser。
- `is_active` 無法從台股每日行情單獨證明，因此保持 unknown。
- 全市場 processed data 尚未 out-of-core。
- 點時 universe、下市完整性與正式交易成本研究仍需加強。

### 12. 來源

- [EODHD historical EOD API](https://eodhd.com/financial-apis/api-for-historical-data-and-volumes)
- [EODHD API limits](https://eodhd.com/financial-apis/api-limits)
- [EODHD pricing](https://eodhd.com/pricing)
- [EODHD split calendar](https://eodhd.com/financial-apis/calendar-upcoming-earnings-ipos-and-splits)
- [EODHD Historical Splits API](https://eodhd.com/financial-apis/api-splits-dividends)
- [TWSE OpenAPI](https://openapi.twse.com.tw/)
- [TWSE ex-right/ex-dividend reference](https://www.twse.com.tw/en/announcement/ex-right/twt49u.html)
- [TPEx OpenAPI](https://www.tpex.org.tw/openapi/)
- [TPEx return index](https://www.tpex.org.tw/web/stock/iNdex_info/reward_index/ROE.php?l=en-us)
- [Massive pricing](https://massive.com/pricing?product=stocks)
- [Massive market-data terms](https://massive.com/legal/market-data-terms-of-service)

---

## English

### 1. Goal and non-goals

The PoC covers daily OHLCV for US and Taiwan stocks and ETFs. External APIs may
be called only during a separate CPU-preparation phase. Data must first become
immutable Parquet plus manifests; GPU training, validation, and inference then
read only materialized artifacts. Dynamic acquisition in the training loop is
forbidden.

Version one does not cover ticks, intraday bars, futures, options, news, or
filings. Options data during foundation-model pretraining is not mandatory; the
selected backbone must instead have credible large-scale financial K-line
pretraining evidence.

### 2. Provider decision

| Market/channel | Choice | Rationale | Main risks and constraints |
|---|---|---|---|
| US PoC | EODHD | One full daily history per symbol per EOD request; adjusted close, discovery, delisted option, and split calendar; side-project economics are more suitable than a commercial-grade feed | Not exchange-grade truth; actual quota/license follows the subscription; overlap reconciliation is required |
| Taiwan listed | Official TWSE | Official market-wide daily report, actions, TAIEX and return index | Endpoint schemas can change; daily requests require low QPS, caching, and retries |
| Taiwan OTC | Official TPEx | Official market-wide daily report, actions, and TPEx return index | Same low-QPS, cache, and parser-validation requirements |
| US expansion | Massive Developer | Typed provider/profile remains reserved | No suitable side-project commercial license now, so it fails closed |

Published provider limits are not treated as safe operating QPS automatically.
Project defaults cap EODHD at `5 QPS` and each Taiwan provider at `0.5 QPS`;
`.env` may lower or carefully adapt them to the actual plan. All network
attempts, including retries, share `STAGE1_MAX_API_CALLS`; cache hits create no
new attempt. These are conservative client controls, not provider-SLA claims.

### 3. Selectable dataset profiles

| profile | `selected_datasets` | EODHD token | State |
|---|---|---|---|
| `tw_only` | `tpex_official`, `twse_official` | No | Available, lowest API cost |
| `us_only_eodhd` | `eodhd_us` | Yes | Available |
| `us_tw_eodhd` | `eodhd_us`, `tpex_official`, `twse_official` | Yes | Default PoC |
| `us_tw_massive` | `massive_us`, `tpex_official`, `twse_official` | N/A | Reserved, not implemented |

Profiles flow into download/ready manifests, training results, and inference
provenance. `selected_datasets` records the selected logical sources; manifests
also record actual providers, markets, symbols, asset types, dates, and split
counts. A later `tw_only` run is therefore distinguishable in every output.

### 4. Universe management

EODHD profiles may set `STAGE1_US_SYMBOLS` and `STAGE1_US_ETF_SYMBOLS`, or leave
both empty for active/delisted discovery. After `STAGE1_SYMBOL_LIMIT`, the
pipeline still adds `VTI.US`, ensuring the benchmark is present.

The first PoC should use a small explicit universe. Complete discovery can hit
API quota, subscription, preparation-time, Parquet-size, and memory limits. Raw
Parquet is streamed, but processed windows and training data currently load in
memory. Full-market long history needs partitioned/lazy windows first.

Taiwan daily reports provide every parseable stock and ETF. Benchmark policy
still decides target eligibility; inappropriate ETFs fail closed during window
construction instead of receiving a misleading broad-market benchmark.

### 5. Canonical raw schema

Required fields:

```text
timestamp, symbol, asset_type,
open, high, low, close, volume,
adjusted_close, split_adjusted_volume, adjustment_source,
provider, market, currency, source_symbol, is_active, dataset_profile
```

Raw O/H/L/C/V is never overwritten. `adjusted_close` is the total-return anchor;
`split_adjusted_volume` contains share-change adjustments only. Timestamps are
normalized to UTC. Symbols use canonical suffixes such as `AAPL.US`, `0050.TW`,
`6488.TWO`, `TAIEX.TW`, and `TPEX.TWO`.

Validation rejects duplicate `(symbol,timestamp)`, non-positive prices,
inconsistent OHLC ranges, negative volume, non-finite values, and unknown asset
types.

### 6. Adjustments and benchmark series

EODHD retains EOD raw OHLCV and adjusted close. For ranges beginning on or
after 2015-01-01, it fetches calendar/splits once for the US universe and
applies per-symbol share multipliers. Because the documented calendar history
starts in 2015, earlier ranges use the per-symbol Historical Splits API. Those
extra calls are included in the fail-closed `max_api_calls` estimate before
historical requests begin; incomplete split history is never silently labeled
as a complete volume anchor.

TWSE/TPEx fetch market-wide raw OHLCV by trading date and official action data
for the range. TWSE may query action detail for free-share ratios. Price and
share factors become separate adjusted anchors while raw fields remain.
`TAIEX.TW` aligns official price-index OHLC with the official TAIEX total-return
index; `TPEX.TWO` does the same with the official TPEx return index.

Each sample normalizes total-return and split factors at `cutoff_at`. This
enforces point-in-time causality and makes a common vendor rescaling of adjusted
close irrelevant to model inputs and labels.

### 7. Benchmark mapping

Defaults are US → `VTI.US`, TWSE → `TAIEX.TW`, and TPEx → `TPEX.TWO`. ETFs use a
small explicit domestic-equity allowlist. `benchmark_mapping_path` is a JSON
object such as:

```json
{
  "CUSTOM.US": "VTI.US"
}
```

The canonical mapping SHA-256 is stored in the preparation spec. Changing the
mapping invalidates old readiness and requires rebuilding processed data.

### 8. API and immutable-artifact flow

```text
provider API
  -> content-addressed raw JSON cache (token excluded)
  -> api-request-log.jsonl (no secret)
  -> raw/market.parquet
  -> download-manifest.json (state=downloaded)
  -> causal windows + chronological purge/embargo
  -> processed/windows.parquet
  -> dataset-manifest.json (state=ready)
```

The downloader provides provider throttles, bounded exponential retries, one
shared fail-closed request budget, cache reuse, staging writes, and refusal to
silently overwrite. Manifests bind SHA-256, size, row count, profile, providers,
symbols, dates, quality, and the request log. Secret-like keys are rejected.

The CPU wrapper publishes fixed `DATA_ROOT` paths. Create a versioned
`DATA_ROOT` or separate network volume for a new dataset; do not delete or
overwrite existing immutable artifacts.

### 9. Processed record contract

Each schema `3.0` record contains the instrument and aligned benchmark contexts
through `cutoff_at`, 3–14 day benchmark-relative alpha labels, auditable asset/
benchmark returns outside the input, a reserved null CAPM diagnostic, and
provider/market/benchmark/profile metadata. Future entry/exit values are never
serialized into either context. Missing benchmark dates, extreme adjusted
transitions, inadequate history, and ambiguous ETF mappings are excluded and
counted in the window audit.

### 10. Split, purge, and robust scale

Windows are assigned chronologically to 70% train, 15% validation, and 15% test,
with 20-bar purge and an effective 14-bar embargo. Effective sample stride is
one. Stable RunPod shell passes legacy `stride=5` and `embargo=5` readiness
sentinels; the manifest separately records effective values and validates both.

Each horizon's loss scale uses train labels only:

```text
max(IQR, 1.4826 * MAD, 1e-4)
```

Validation/test never influence scaling. Test remains sealed until explicitly
unlocked by the research protocol.

### 11. QA and known limitations

Required checks include duplicates, OHLC consistency, missingness, extreme
adjusted transitions, benchmark calendar gaps, symbol/date coverage,
corporate-action continuity, split-volume direction, and sampled EODHD overlap
reconciliation.

Current limitations include non-exchange-grade EODHD data, mutable official
endpoint schemas, unknown Taiwan `is_active` status from daily reports,
in-memory processed windows, and incomplete point-in-time universe/delisting/
transaction-cost research.

### 12. Sources

- [EODHD historical EOD API](https://eodhd.com/financial-apis/api-for-historical-data-and-volumes)
- [EODHD API limits](https://eodhd.com/financial-apis/api-limits)
- [EODHD pricing](https://eodhd.com/pricing)
- [EODHD split calendar](https://eodhd.com/financial-apis/calendar-upcoming-earnings-ipos-and-splits)
- [EODHD Historical Splits API](https://eodhd.com/financial-apis/api-splits-dividends)
- [TWSE OpenAPI](https://openapi.twse.com.tw/)
- [TWSE ex-right/ex-dividend reference](https://www.twse.com.tw/en/announcement/ex-right/twt49u.html)
- [TPEx OpenAPI](https://www.tpex.org.tw/openapi/)
- [TPEx return index](https://www.tpex.org.tw/web/stock/iNdex_info/reward_index/ROE.php?l=en-us)
- [Massive pricing](https://massive.com/pricing?product=stocks)
- [Massive market-data terms](https://massive.com/legal/market-data-terms-of-service)
