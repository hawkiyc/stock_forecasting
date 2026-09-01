# 日線 OHLCV 資料策略

## 中文

### 1. 目標與非目標

PoC 的資料範圍是美國與台灣普通股、ADR／TDR，以及經稽核且可映射的非槓桿
股票型 ETF 日線 OHLCV。任何外部 API 呼叫都只能發生
在獨立 CPU preparation 階段；資料必須先落地為 immutable Parquet 與 manifest，
GPU 訓練、validation 與 inference 才能讀取。訓練迴圈禁止動態下載。

第一版不追求 tick、分鐘線、期貨、選擇權、新聞或財報。選擇權預訓練資料不是
backbone 的硬性必要條件；本 PoC 只需確認選定 foundation model 曾以大量金融
K-line 預訓練。

### 2. Provider 決策

| 市場／通道 | 選擇 | 理由 | 主要風險與限制 |
|---|---|---|---|
| 美國 PoC | EODHD | 一個 symbol 的日線歷史可用單次 EOD request；提供 adjusted close、symbol discovery、delisted 選項與 Historical Splits API；價格與用量比商業級 feed 適合 side-project | 不是交易所級 truth；方案額度與授權需依實際訂閱確認；跨 provider 需抽樣對帳 |
| 台灣上市 | TWSE 官方 | 官方市場日報、除權息資料與 TAIEX/報酬指數 | endpoint schema 可能調整；逐交易日下載需低 QPS、cache 與重試 |
| 台灣上櫃 | TPEx 官方 | 官方市場日報、除權息資料與櫃買報酬指數 | 同樣需低 QPS、cache、schema 驗證 |
| 美國擴充 | Massive Developer | 保留型別化 provider 與 profile | side-project 目前沒有足夠商業授權，因此 fail closed，不執行下載 |

EODHD 付費方案的官方預設值是每日 `100,000 API calls` 與每分鐘 `1,000 HTTP
requests`。每次 `runpod_workflow.sh cpu prepare` 都必須依帳戶當下剩餘額度明確設定
`--max-api-calls`；QPS 預設把每分鐘限制向下取整為 `16 QPS`，也就是每分鐘最多
`960 requests`，保留 40 requests（4%）的餘裕。每個台灣官方 provider 預設維持
`0.5 QPS`。這三個值是 per-launch acquisition policy，不寫入 `configure` 建立的
immutable selection，也不手動修改 `.env` 或 config。`max_api_calls` 只計入單次
acquisition attempt 的 EODHD
cache misses/retries；TWSE／TPEx network attempts 會被記錄，但沒有專案端 request-count
上限。完整計畫的 request estimate 只作資訊，不是 dataset admission gate；台灣最終
estimate 使用官方 benchmark sessions，並另外記錄 pre-calendar weekday upper bound。

EODHD、TWSE 與 TPEx 是三個獨立的平行迴圈。三者都採指數退避並共用
`runpod_workflow.sh cpu prepare --maxBackoff`（預設 `1m`）設定；各自的下一次等待超過
此值時只退出自己的迴圈。EODHD 另外以本次 `--max-api-calls` 為退出條件，兩個邊界
任一先發生就停止。任何 provider 先退出都不會取消其他迴圈，主流程 join 全部迴圈後
才發布續傳狀態或固定順序合併 provider-local artifacts。budget、最大退避、暫時性
provider 錯誤或 acquisition time 用完時都保存 cache 並由下一個 CPU Pod 續接。
最大退避越界會開啟該 provider client 的共享 circuit breaker；in-flight request
可以完成，但同一 provider 的其他 worker 不會再啟動新的 request。
台灣官方端點由 Pod IP/WAF 暫時回傳的 HTTP 403 也使用同一個有限退避契約，不會立即
把整體 acquisition 標成不可續傳。tmux finalizer 不得把 worker 已發布的精確
`waiting_for_*` 狀態覆寫成一般 `failed`，同一次 launch 的 terminal rewrite 也必須保留
`progress_path`。
API calls 與 HTTP requests 是不同單位，實際帳戶已用額度與 provider headers 仍是
執行時依據。

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
以執行 active/delisted discovery。`STAGE1_SYMBOL_LIMIT=N` 會在 ETF 與 stock 兩組
各自依 active → delisted、ticker 排序並各取最多 N 檔；不是兩類合計 N 檔。
`VTI.US` 是 label/context 必需、但不會成為 target 的 benchmark dependency，因此
不占 N 個 ETF candidate 名額；若限制結果沒有 VTI 才額外補入，raw universe 最多
`2N+1` 檔。

discovery 是下載當下的 active/delisted 清單，不是逐日 point-in-time constituents。
區間中途上市的商品只有上市後可取得的 rows，中途下市的商品只有下市前可取得的
rows；兩者皆發生於區間內的商品，必須由 EODHD delisted discovery 回傳且帳戶有權限
才會納入。`is_active` 是 discovery 當下狀態，不是每個 timestamp 的狀態。

官方只保證 2018 年前下市商品的 EOD，不保證 splits/dividends 等輔助資料。這些 EOD
rows 仍可進入 eligible cutoff ranges，但 split-adjusted volume 覆蓋可能不完整；download manifest
會以 `delisted_pre_2018_auxiliary_coverage_warning` 列出受影響數量與 symbols。

完整 discovery 仍可能同時受到 API 額度、訂閱權限、CPU preparation 時間與 Parquet
大小限制，但不再要求 RAM 容納全部 bars 或逐-window 資料集。raw Parquet 串流分批
寫入；preparation 以 128 個 bounded hash buckets 建立 symbol-oriented compressed bar
store、索引與有效 cutoff ranges。訓練只按需讀取 symbol row group 並動態建立 context
與 label。

台股官方日報會取得全市場可解析股票與 ETF。是否可成為 target 仍由 benchmark policy
決定；不合適的 ETF 會在 cutoff eligibility 建置時 fail closed，而不是被錯誤對到大盤。

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

`open/high/low/close` 永遠是 provider raw fields；台股 `volume` 也是官方 raw field。
EODHD EOD `volume` 依官方定義已做 split adjustment，管線以完整 Historical Splits
response 反推當時的未調整 `volume`，並把原 vendor 值放在
`split_adjusted_volume`，避免重複調整。`adjusted_close` 是 total-return anchor。
所有 timestamps 正規化為 UTC，symbol 使用 canonical suffix，例如
`AAPL.US`、`0050.TW`、`6488.TWO`、`TAIEX.TW`、`TPEX.TWO`。

schema validator 會拒絕 duplicate `(symbol,timestamp)`、非正值 price、不合理 OHLC
range、負 volume、非有限數值與未知 asset type。

### 6. Adjustment 與 benchmark series

EODHD：

- EOD endpoint 提供 raw OHLC、split-adjusted volume 與 adjusted close。
- 每個 symbol 都使用不帶日期裁切的完整 Historical Splits API response；官方將它列入 EOD
  Historical Data — All World 且每個 request 為 1 API call。管線不使用需要
  Calendar-enabled 產品的 `calendar/splits`。
- 每個商品是一個 EOD request 加一個 split-history request。兩種 response 都能跨
  CPU Pod cache／續傳；總 request 估算只作資訊，`max_api_calls` 只限制單次 attempt
  的 cache misses/retries，不能用不完整 split history 靜默產生 volume anchor。
- `adjusted_close` 保持 vendor total-return series；完整 split factors 用來反推未調整
  `volume`，`split_adjusted_volume` 保留 vendor 已調整值。
- vendor history 內全零、非有限、負 volume、不合理 OHLC 或重複日期的 placeholder rows
  不補值也不改寫；只丟棄無法符合 canonical contract 的 rows，並計入
  `dropped_source_rows`。

TWSE/TPEx：

- 先從本來就必須下載的月度官方 benchmark rows 取得實際交易日，再抓 market-wide
  raw OHLCV；不把一般週一至週五全部視為開市日。
- 每段日期各抓官方除權息資料；TWSE 權事件必要時讀 detail 取得無償配股比例。
- 舊 TWSE 權事件若主表存在但 detail 回覆「無相關資料」，保留可驗證的 price factor，
  未知 share multiplier 使用 `1.0`，並在 manifest 記錄 coverage gap；不推測比例。
- 公司行動建立 price factor 與 share multiplier，再保留 raw fields 並新增 adjusted
  anchors。
- `TAIEX.TW` 以官方 TAIEX price-index OHLC 配對官方發行量加權股價報酬指數。
- `TPEX.TWO` 以官方櫃買 price-index OHLC 配對官方櫃買報酬指數。

樣本視窗再把 total-return 與 split factors 正規化到 `cutoff_at`，確保 point-in-time
因果性。這也讓 vendor 對全部 adjusted close 乘上共同常數時，模型 input 與 label
不受影響。

### 7. Benchmark mapping

預設：US 普通股／ADR → `VTI.US`、TWSE 普通股／TDR → `TAIEX.TW`、TPEx 普通股 →
`TPEX.TWO`。ETF 使用小型、明確且只含非槓桿股票曝險的 allowlist；槓桿、反向、
債券、商品、波動率與未稽核 ETF 都不會成為 target。`benchmark_mapping_path` 是
JSON object，但只能替 allowlist 內的 ETF 選擇另一個 benchmark，不能擴張 universe：

```json
{
  "QQQ.US": "SPY.US"
}
```

mapping 內容的 canonical JSON SHA-256 會寫入 preparation spec；修改 mapping 後舊
dataset readiness 不再有效，必須由既有 raw Parquet 重建 bar store 與 cutoff ranges，
但不需要重新呼叫 provider。

### 8. API 與 immutable artifact 流程

```text
provider API
  -> content-addressed raw JSON cache (token excluded from identity)
  -> api-request-log.jsonl (no secret)
  -> raw/market.parquet
  -> download-manifest.json (state=downloaded)
  -> resumable symbol bar-store buckets
  -> symbol-index.parquet + cutoff-ranges.parquet
  -> dataset-manifest.json (state=ready)
```

下載器具備 provider-level throttle、獨立平行 provider 迴圈、exponential retry、
只限 EODHD 的單次 request budget、三個 provider 共用設定但獨立計算的最大退避邊界、
cache reuse、provider-local staging writes、固定順序合併與拒絕 silent overwrite。
完整計畫 estimate 不阻止執行；
`--max-api-calls`、`--eodhd-qps`、`--taiwan-qps` 與 `--maxBackoff` 都由每次
`runpod_workflow.sh cpu prepare` 設定，不屬於 dataset identity；
CPU workflow 預設保留 max runtime 的 25%（最多 2 小時）給 data cleaning/bar-store construction。
可透過 `runpod_workflow.sh cpu prepare --prepareReserve DURATION` 明確調整，或使用
`--prepareReserve auto` 保留自動值；明確值必須短於 max runtime。
`--maxBackoff DURATION` 預設 `1m`；EODHD、TWSE 或 TPEx 的下一次退避大於此值時，
只有該迴圈退出，其他 provider 仍會繼續到自己的退出條件。EODHD 也會在先達到
`--max-api-calls` 時退出。
raw Parquet、request log 與 download manifest 先成為 durable `downloaded` checkpoint。
bar store 的 raw scan、bucket compaction、quality candidates 與 chronological split ranges
各自以固定 hash 分區，透過 `spawn` process pool 執行並原子發布 checkpoint。worker
數是 vCPU 設定的上限，不是強制平行度：排程器取 cgroup 與系統可用記憶體的較小值，
保留 parent process headroom，最多配置當下可用記憶體的 60%，再以該階段最重 task
的保守記憶體估算降低 process 數。raw scan 將細碎來源 row groups 合併為約
`4 * batch_rows` 的來源分區，每個 process 最多串流 `batch_rows`，在 Pod 本機 `/tmp`
建立 bucket 暫存，最後只向 Network Volume 原子發布一個具 row-group 索引的分區
Parquet；每個 child 的 Arrow／BLAS native threads 固定為 1。單一 task 若已超過安全預算，會在啟動 pool 前
停止並保留 checkpoint，避免由 OOM killer 非預期中止。後續 Pod 可不呼叫 provider，
並透過 `scan-index.json` 完全跳過已完成的來源分區與 compacted buckets，不再列舉數萬個
segment 目錄。若中斷位於某個來源分區內，最多只重做該分區，已完成分區不會重寫。到達安全
截止時間時 lifecycle 為 `waiting_for_preparation`，成功發布 `_SUCCESS.json` 後才回收
`.work` 暫存分區。manifest 儲存 SHA-256、
size、row count、profile、providers、symbols、date range、quality summary 與 request-log
artifact。worker 數與記憶體規劃只屬於 execution metadata，不在 dataset identity；換用
不同 vCPU／RAM 的 Pod 仍能續用既有 raw 與 bucket checkpoints。secret-like key 不得
寫入 manifest。
CPU readiness 與訓練 preflight 會依 bar-store manifest 逐 shard 驗證 size 與 SHA-256；
任何 shard 缺失、截斷或內容不符都不能進入訓練。

RunPod CPU wrapper 發布到固定 `DATA_ROOT`。要建立另一資料版本，應使用新的版本化
`DATA_ROOT` 或新的 network volume；不要刪除或覆寫既有 immutable artifacts。

`h_start` 是 runtime label/model-output 契約，不是 bar-store identity。`h_start=1`、`2`、
`3` 共用同一 dataset request SHA、`DATA_ROOT`、raw Parquet、bar store、cutoff ranges 與
split audit；變更它只會得到新的 training selection SHA。DataLoader 在取樣時動態計算
`h_start...14` labels，train-only robust scales 在每次訓練啟動時由 train split 抽樣估計，
並寫入 checkpoint 供續訓、評估與推論還原；不會重建資料或呼叫 provider。

### 9. Lazy sample contract

磁碟只保存每個 symbol 一份 bars 與連續有效 cutoff ranges，不保存 schema `4.0` window
records。DataLoader 取得一個 `(symbol, cutoff_index)` 後，在記憶體中暫時建立：

- `context`：商品截至 `cutoff_at` 的 point-in-time adjusted OHLCV。
- `benchmark_context`：同日期、同長度的 benchmark adjusted OHLCV。
- `label.alpha_log_returns`：從 `h_start`（1、2 或 3）到固定第 14 日的
  benchmark-relative execution log returns。
- `label.asset_total_returns` 與 `benchmark_total_returns`：只供 audit，不進 input。
- `diagnostics.capm_abnormal_return=null`：預留 diagnostic，不是 target。
- provider、market、benchmark policy、dataset profile 等 metadata。

entry/exit label dates 不會被序列化到任一 context 或磁碟 label artifact。缺 benchmark
日期、極端 adjusted transition、歷史不足或 benchmark mapping 不明確的 cutoff 會在
preparation 時從 ranges 排除並寫入 quality/split audit。

### 10. Split、purge 與 robust scale

先從全市場有效 cutoff dates 決定 chronological 70% train、15% validation、15% test
邊界，再以向量化 bucket assignment 套用 purge 20 bars 與 effective embargo 14 bars。
不建立 windows；sample stride 是 1。
此外，train 與 validation 中每個樣本的最晚 `label.end_at` 必須嚴格早於下一個
split boundary；任何跨界 ground truth 都會被排除並記入 split audit。
RunPod 穩定 shell 仍傳入 legacy `stride=5`、`embargo=5` readiness sentinels；ready
manifest 同時記錄 effective values，訓練前會雙重驗證。

每個 horizon 的 loss scale 只從 train split 的動態 label 抽樣計算：

```text
max(IQR, 1.4826 * MAD, 1e-4)
```

validation/test 不參與 scaling。test 保持 sealed，直到研究流程明確 unlock。

訓練 sampler 以 O(1) 狀態對 valid cutoffs 作 deterministic blockwise permutation。
Stage 1 的 target set 精確為 15%，Stage 2 為 100%；最後不足一個 batch 時只從同一
target set 開頭補齊，補齊數量寫入 training summary，不配置全量 window index。

### 11. 資料 QA 與已知限制

必查項目：duplicate bars、OHLC consistency、缺值、極端 adjusted transitions、
benchmark calendar gaps、symbol/date coverage、corporate-action 前後連續性、split volume
方向，以及 EODHD 與其他來源重疊標的抽樣對帳。

目前限制：

- EODHD PoC 資料不等於 exchange-grade truth。
- 官方 endpoint schema 可能更動，remote tests 必須驗證 parser。
- `is_active` 無法從台股每日行情單獨證明，因此保持 unknown。
- 首次 bar-store 建置期間會暫時同時保留 raw Parquet、已完成 shards 與可續傳工作分區；
  `_SUCCESS.json` 發布後會回收工作分區，但 raw Parquet 仍作為不可變下載證據保留。
- 點時 universe、下市完整性與正式交易成本研究仍需加強。

### 12. 來源

- [EODHD historical EOD API](https://eodhd.com/financial-apis/api-for-historical-data-and-volumes)
- [EODHD API limits](https://eodhd.com/financial-apis/api-limits)
- [EODHD pricing](https://eodhd.com/pricing)
- [EODHD delisted data coverage](https://eodhd.com/financial-apis/delisted-stock-companies-data-2)
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

The PoC covers daily OHLCV for US/Taiwan common stocks, ADRs/TDRs, and audited
benchmark-mappable unleveraged equity ETFs. External APIs may
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
| US PoC | EODHD | One full daily history per symbol per EOD request; adjusted close, discovery, delisted option, and the Historical Splits API; side-project economics are more suitable than a commercial-grade feed | Not exchange-grade truth; actual quota/license follows the subscription; overlap reconciliation is required |
| Taiwan listed | Official TWSE | Official market-wide daily report, actions, TAIEX and return index | Endpoint schemas can change; daily requests require low QPS, caching, and retries |
| Taiwan OTC | Official TPEx | Official market-wide daily report, actions, and TPEx return index | Same low-QPS, cache, and parser-validation requirements |
| US expansion | Massive Developer | Typed provider/profile remains reserved | No suitable side-project commercial license now, so it fails closed |

EODHD's official paid-plan defaults are `100,000 API calls` per day and `1,000
HTTP requests` per minute. Every `runpod_workflow.sh cpu prepare` launch requires
an explicit `--max-api-calls` based on the account's current remaining quota.
QPS defaults floor the minute limit to `16 QPS`, or at most `960 requests` per
minute, leaving 40 requests (4%) of headroom. Each Taiwan provider defaults to
`0.5 QPS`. These three values are per-launch acquisition policy: they are not
stored in the immutable selection created by `configure` and are never set by
editing `.env` or config files. `max_api_calls` counts only EODHD cache misses
and retries in one acquisition; TWSE/TPEx attempts are recorded but have no project request-count
ceiling. The complete-plan estimate is informational, not a dataset-admission
gate. Taiwan's final estimate uses official benchmark sessions and records the
pre-calendar weekday upper bound separately.

EODHD, TWSE, and TPEx run as independent parallel loops. All three use
exponential backoff and the same `runpod_workflow.sh cpu prepare --maxBackoff`
setting (default `1m`); each exits independently when its next delay exceeds the
boundary. EODHD additionally exits when it reaches `--max-api-calls`, whichever
condition occurs first. One provider exiting never cancels another; the main
process joins every loop before publishing a resume state or deterministically
merging provider-local artifacts.
Budget exhaustion, backoff boundaries, temporary provider failures, and
acquisition-time exhaustion preserve cache state for the next CPU Pod.
Crossing the maximum backoff opens a shared circuit breaker for that provider
client. In-flight requests may finish, but sibling workers cannot start another
request for the stopped provider. Temporary HTTP 403 responses from Taiwan
official endpoints due to Pod IP/WAF
policy use the same bounded-backoff contract instead of making the aggregate
failure immediately non-resumable. The tmux finalizer must not replace a precise
worker-published `waiting_for_*` state with generic `failed`, and same-launch
terminal rewrites preserve `progress_path`. API calls
and HTTP requests are different units, so account usage and provider headers
remain authoritative at runtime.

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
both empty for active/delisted discovery. `STAGE1_SYMBOL_LIMIT=N` sorts ETFs and
stocks separately by active → delisted and ticker, then keeps up to N of each;
it is not N instruments across both groups. `VTI.US` is a label/context
dependency but never a training target, so it does not consume one of the N ETF
candidate slots. It is added only when absent, making the raw universe at most
`2N+1` instruments.

Discovery is the active/delisted snapshot at download time, not daily
point-in-time constituents. An instrument listed during the requested range has
rows only from its first available session; one delisted during the range has
rows only through its last available session. An instrument experiencing both
events inside the range is included only when EODHD delisted discovery returns
it and the account is entitled to it. `is_active` is the discovery-time state,
not a timestamp-by-timestamp state.

EODHD guarantees only EOD—not splits/dividends and other auxiliary data—for
instruments delisted before 2018. Those EOD rows can still enter eligible cutoff
ranges, but
split-adjusted-volume coverage may be incomplete. The download manifest records
the affected count and symbols under
`delisted_pre_2018_auxiliary_coverage_warning`.

Complete discovery can still hit API quota, subscription, preparation-time,
and Parquet-size limits, but RAM no longer needs to hold all bars or a
per-window dataset. Raw Parquet is streamed. Preparation uses 128 bounded hash
buckets to build a symbol-oriented compressed bar store, indexes, and valid
cutoff ranges. Training reads symbol row groups on demand and constructs
contexts and labels dynamically.

Taiwan daily reports provide every parseable stock and ETF. Benchmark policy
still decides target eligibility; inappropriate ETFs fail closed during cutoff
eligibility construction instead of receiving a misleading broad-market benchmark.

### 5. Canonical raw schema

Required fields:

```text
timestamp, symbol, asset_type,
open, high, low, close, volume,
adjusted_close, split_adjusted_volume, adjustment_source,
provider, market, currency, source_symbol, is_active, dataset_profile
```

Raw O/H/L/C is never overwritten, and Taiwan volume remains the official raw
field. EODHD defines EOD volume as already split-adjusted; the pipeline uses the
complete Historical Splits response to reconstruct contemporaneous unadjusted
`volume` and stores the vendor value in `split_adjusted_volume`, avoiding a
second multiplication. `adjusted_close` is the total-return anchor. Timestamps are
normalized to UTC. Symbols use canonical suffixes such as `AAPL.US`, `0050.TW`,
`6488.TWO`, `TAIEX.TW`, and `TPEX.TWO`.

Validation rejects duplicate `(symbol,timestamp)`, non-positive prices,
inconsistent OHLC ranges, negative volume, non-finite values, and unknown asset
types.

### 6. Adjustments and benchmark series

EODHD retains EOD raw OHLC, split-adjusted volume, and adjusted close. Every
symbol uses the complete, non-date-truncated Historical Splits response, which
EODHD lists under EOD Historical Data —
All World at one API call per request. The pipeline does not use
`calendar/splits`, which requires a Calendar-enabled product. Each instrument
therefore needs one EOD request and one split-history request. Both responses
are cacheable/resumable. The total estimate is informational, while
`max_api_calls` caps only cache misses/retries in one attempt; incomplete split
history is never silently labeled as a complete volume anchor. Split factors
reconstruct unadjusted volume, while `split_adjusted_volume` retains the vendor
value. All-zero, non-finite, negative-volume, inconsistent-OHLC, or duplicate-date
vendor placeholder rows are neither imputed nor rewritten. Only rows that cannot
satisfy the canonical contract are dropped and counted in `dropped_source_rows`.

TWSE/TPEx first derive actual sessions from the already-required monthly official
benchmark rows, then fetch market-wide raw OHLCV and official action data for
those dates. Ordinary weekdays are not assumed to be open. TWSE may query action
detail for free-share ratios. If an early action exists in the main table but its
detail endpoint returns no record, the verified price factor is retained, the
unknown share multiplier remains `1.0`, and the manifest records the coverage
gap instead of inferring a ratio. Price and
share factors become separate adjusted anchors while raw fields remain.
`TAIEX.TW` aligns official price-index OHLC with the official TAIEX total-return
index; `TPEX.TWO` does the same with the official TPEx return index.

Each sample normalizes total-return and split factors at `cutoff_at`. This
enforces point-in-time causality and makes a common vendor rescaling of adjusted
close irrelevant to model inputs and labels.

### 7. Benchmark mapping

Defaults are US common stocks/ADRs → `VTI.US`, TWSE common stocks/TDRs →
`TAIEX.TW`, and TPEx common stocks → `TPEX.TWO`. ETFs use a small explicit
allowlist containing only unleveraged equity exposures. Leveraged, inverse,
bond, commodity, volatility, and unaudited ETFs cannot become targets.
`benchmark_mapping_path` may choose another benchmark only for an allowlisted
ETF; it cannot expand the universe. It is a JSON object such as:

```json
{
  "QQQ.US": "SPY.US"
}
```

The canonical mapping SHA-256 is stored in the preparation spec. Changing the
mapping invalidates old readiness and requires rebuilding the bar store and
cutoff ranges from existing raw Parquet, but no provider call.

### 8. API and immutable-artifact flow

```text
provider API
  -> content-addressed raw JSON cache (token excluded)
  -> api-request-log.jsonl (no secret)
  -> raw/market.parquet
  -> download-manifest.json (state=downloaded)
  -> resumable symbol bar-store buckets
  -> symbol-index.parquet + cutoff-ranges.parquet
  -> dataset-manifest.json (state=ready)
```

The downloader provides provider throttles, independent parallel provider loops,
exponential retries, an EODHD-only per-attempt request budget, one maximum
backoff setting independently enforced by EODHD/TWSE/TPEx, cache reuse,
provider-local staging, deterministic merging, and
refusal to silently overwrite. `--max-api-calls`, `--eodhd-qps`,
`--taiwan-qps`, and `--maxBackoff` are supplied for every
`runpod_workflow.sh cpu prepare` launch and are not part of dataset identity.
The complete-plan estimate never blocks execution. The CPU workflow reserves
25% of max runtime for cleaning/bar-store
construction by default, capped at 2 hours, with an explicit override available through
`runpod_workflow.sh cpu prepare --prepareReserve DURATION`; `auto` retains the
automatic value, and an explicit reserve must be shorter than max runtime.
`--maxBackoff DURATION` defaults to `1m`; an EODHD, TWSE, or TPEx loop exits when
its next delay exceeds that boundary while other providers continue. EODHD also
exits if it reaches `--max-api-calls` first. The workflow
publishes raw Parquet, the request log, and download manifest as a durable
`downloaded` checkpoint before preparation. Raw scan, bucket compaction,
quality candidates, and chronological split ranges use fixed hash partitions,
run in `spawn` process pools, and each publish atomic checkpoints. Fragmented
source row groups are coalesced into partitions of roughly `4 * batch_rows`;
each worker streams no more than `batch_rows`, uses Pod-local `/tmp` for bucket
intermediates, and atomically publishes one row-group-indexed partition Parquet
to the Network Volume. The configured
vCPU count is a ceiling rather than mandatory parallelism: the planner uses the
lower cgroup/OS available-memory value, reserves parent-process headroom, assigns
at most 60% of currently available memory to workers, and lowers each phase's
process count using a conservative estimate for its largest task. Every child
limits Arrow/BLAS native threads to one. If one task exceeds the safe budget, the workflow stops before
starting the pool and preserves existing checkpoints instead of risking an OOM
kill. A later Pod skips provider calls and uses `scan-index.json` to skip finished
source partitions and compacted buckets without listing per-segment directories.
If interruption occurs inside a source partition, only that partition is redone;
completed partitions are never rewritten. Reaching the safe deadline produces `waiting_for_preparation`;
`.work` is reclaimed only after `_SUCCESS.json` is published.
Manifests bind SHA-256, size, row count, profile, providers,
symbols, dates, quality, and the request log. Worker count and memory planning are
execution metadata, not dataset identity, so a Pod with different vCPU/RAM can
reuse existing raw and bucket checkpoints. Secret-like keys are rejected.
CPU readiness and training preflight validate every shard's size and SHA-256
against the bar-store manifest; missing, truncated, or altered shards cannot train.

The CPU wrapper publishes fixed `DATA_ROOT` paths. Create a versioned
`DATA_ROOT` or separate network volume for a new dataset; do not delete or
overwrite existing immutable artifacts.

`h_start` is a runtime label/model-output contract, not bar-store identity.
Values 1, 2, and 3 share one dataset request SHA, `DATA_ROOT`, raw Parquet, bar
store, cutoff ranges, and split audit; changing it creates only a new training
selection SHA. The DataLoader computes `h_start...14` labels on demand, and
train-only robust scales are sampled from the train split when training starts.
They are persisted in checkpoints for exact resume, evaluation, and inference
restoration. No data rebuild or provider request occurs.

### 9. Lazy sample contract

Disk stores each symbol's bars once plus contiguous valid cutoff ranges; it does
not store schema `4.0` window records. Given `(symbol, cutoff_index)`, the
DataLoader temporarily constructs the instrument and aligned benchmark contexts
through `cutoff_at`, benchmark-relative alpha labels from `h_start` (1, 2, or 3)
through fixed day 14, auditable asset/benchmark returns outside the input, a
reserved null CAPM diagnostic, and provider/market/benchmark/profile metadata.
Future entry/exit values are never serialized into a context or disk label
artifact. Cutoffs with missing benchmark dates, extreme adjusted transitions,
inadequate history, or ambiguous ETF mappings are excluded during preparation
and counted in quality/split audits.

### 10. Split, purge, and robust scale

Global valid cutoff dates determine chronological 70% train, 15% validation,
and 15% test boundaries. Vectorized per-bucket assignment then applies a 20-bar
purge and effective 14-bar embargo without creating windows. Effective sample
stride is one. The latest `label.end_at` of every train and validation sample must also be
strictly earlier than the next split boundary; crossing ground truth is dropped
and counted in the split audit. Stable RunPod shell passes legacy `stride=5` and `embargo=5` readiness
sentinels; the manifest separately records effective values and validates both.

Each horizon's loss scale uses a runtime sample of train-only labels:

```text
max(IQR, 1.4826 * MAD, 1e-4)
```

Validation/test never influence scaling. Test remains sealed until explicitly
unlocked by the research protocol.

The training sampler applies a deterministic blockwise permutation with O(1)
state. Stage 1 targets exactly 15% of valid cutoffs and Stage 2 targets 100%; a
short final batch is filled only from the beginning of the same target set. The
padding count is recorded in the training summary without allocating a full
window-index array.

### 11. QA and known limitations

Required checks include duplicates, OHLC consistency, missingness, extreme
adjusted transitions, benchmark calendar gaps, symbol/date coverage,
corporate-action continuity, split-volume direction, and sampled EODHD overlap
reconciliation.

Current limitations include non-exchange-grade EODHD data, mutable official
endpoint schemas, unknown Taiwan `is_active` status from daily reports, and
incomplete point-in-time universe/delisting/transaction-cost research. During
the first bar-store build, raw Parquet, finished shards, and resumable work
partitions coexist temporarily. Work partitions are reclaimed after
`_SUCCESS.json`; raw Parquet remains as immutable acquisition evidence.

### 12. Sources

- [EODHD historical EOD API](https://eodhd.com/financial-apis/api-for-historical-data-and-volumes)
- [EODHD API limits](https://eodhd.com/financial-apis/api-limits)
- [EODHD pricing](https://eodhd.com/pricing)
- [EODHD delisted data coverage](https://eodhd.com/financial-apis/delisted-stock-companies-data-2)
- [EODHD Historical Splits API](https://eodhd.com/financial-apis/api-splits-dividends)
- [TWSE OpenAPI](https://openapi.twse.com.tw/)
- [TWSE ex-right/ex-dividend reference](https://www.twse.com.tw/en/announcement/ex-right/twt49u.html)
- [TPEx OpenAPI](https://www.tpex.org.tw/openapi/)
- [TPEx return index](https://www.tpex.org.tw/web/stock/iNdex_info/reward_index/ROE.php?l=en-us)
- [Massive pricing](https://massive.com/pricing?product=stocks)
- [Massive market-data terms](https://massive.com/legal/market-data-terms-of-service)
