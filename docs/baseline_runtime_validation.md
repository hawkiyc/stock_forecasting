# Baseline 執行效率與接續訓練驗收 / Baseline runtime and resume acceptance

## 中文

### 版本、範圍與環境

- 日期：2026-09-21；版本：`v0.2.1`。受測功能程式碼為 `9e1af2c`，
  之前的實作與驗證腳本修正為 `9ebff63`、`f4bb115`；版本 tag 另包含本紀錄。
- 驗證 run：`baseline-run-20260921T043022Z-3705`；RTX 4090、約 61 GB RAM。
  CPU affinity 可見 48 cores，但容器 quota 為 13.6 cores，規劃器採 13 cores。
- 同 GPU 最多兩個 deep jobs，另有一個 CPU job；每個 loader 一個 subprocess。
  這是同時保留 train／validation pools、控制程序與 CPU job 的保守資源分配，
  不是 `num_workers=0`，也不是已達到該 GPU 的吞吐上限。
- 驗證 Pod `m2eezovnkt9gro` 於 04:30:57 UTC 建立，約 05:19:40 UTC
  由本機刪除；API 回傳 `deleted: true`，再查 Pod 清單為空。總計約 49 分鐘，
  未超過一小時授權；55 分鐘本機 hard guard 全程保留至確認刪除。
  原效率異常 Pod `twepblqbl3kfqz` 也已先行刪除。Network volume 未刪除。
- 未在本機建立或安裝 Python 訓練環境、未執行正式 baseline／A／B 訓練、
  未下載新行情，也未執行 CPU prepare／finalize。

### 問題與修正

原資料路徑每個 window 都重複經過 pandas、Python list、時間字串轉換及 Parquet
metadata 解析。GPU 程序主要等待資料；僅根據 GPU utilization 或增加 batch size
不足以解決這個問題。舊 CPU 規劃也未正確限制於容器 quota。

本版修正如下：

1. 依股票分組向量化讀取 windows／計算 baseline 特徵，保持原樣本順序、重複索引、
   corporate-action 調整、因果 cutoff 與 target 定義。只解碼必要數值欄位。
2. 各 baseline worker 有獨立、有界 Parquet metadata LRU；預設 128 files、
   512 MiB 保守估計，淘汰時關閉 handle。每 worker 預算為 1 GiB，spawn 不傳
   handles 或完整資料。它不是預先展開的 window 資料集。
3. 依 affinity、cgroup v1／v2 quota、host／GPU／shared memory 分配資源；
   分別實測 train／evaluation batch，保留同 GPU 並行的記憶體分額與安全空間。
   以實際 loader 等待時間與 buffer 預算調整 prefetch，有界探測，不更新權重。
4. 神經模型保存 sample cursor、模型、optimizer、scheduler 與 RNG；第一個 batch
   後、每 300 秒以及完整 validation 前後保存。可在不同 batch size 下接續同一
   隨機 epoch 的剩餘樣本；warmup／validation 邊界以樣本數對齊。
5. Tabular cache 先 flush／fsync 再提交 row cursor；GBDT 每完成一個
   horizon／quantile 的增量 fit 保存。未完成 validation、最終 test 或原生 GBDT
   fit 會重做該單元，不將部分結果視為完成。

更換 batch size 保證續接的樣本次序與位置，不保證不同硬體／batch 的浮點結果或
optimizer 軌跡逐 bit 相同。資源設定不納入 baseline 數值快取識別；資料／baseline
數值參數或相關實作改變才建立新身份。舊 Pod 未保存的 RAM 狀態無法事後還原。

### 回歸與完整性

最終 `final-status.json` 的 `pytest`、`baseline_throughput`、`kronos`、`ruff`
均為 `0`。完整 suite：**573 passed、1 skipped、97 subtests passed**，596.56 秒。
唯一 skip 是 source-only 部署沒有 Git metadata 的 preparation 語意檢查；本機
stdlib 控制平面 suite 的 11 項檢查通過，另有 4 項 runtime 控制測試通過。

回歸涵蓋：

- 真實資料 128 個隨機索引的 target 完全相等，特徵與舊逐筆路徑在容差內相等；
  另測 raw／relative、不同 horizon 起點、重複／亂序／負索引、零交易量與調整價格。
- Tabular cache 中斷後換 batch 接續；神經模型中途保存後由 batch 32 改 47 續訓；
  待完成 validation 重跑；GBDT 不重做已提交的 quantile fit；完成結果快取重用。
- Metadata cache 有界、重用、釋放以及 pickle 不攜帶 handles。
- 主模型原有四個 sampler 的 AST 與 `v0.2.0` 一致。Training 保留動態 sampling；
  validation／test 仍不抽樣、不丟尾 batch、逐產品動態遍歷所有合法 windows。
- 真實離線 Kronos 的 CUDA forward／backward／optimizer step，有限且非零梯度；
  另測 4,096 筆 holdout 輸入管線，不是完整 holdout 預測評分。

本次雲端驗證也完成所有 eligible instruments 的 cutoff 稽核：

| 項目 | 結果 |
| --- | --- |
| 稽核產品數 | 44,784 |
| Validation | 1,232,972 windows，13,538 products |
| Holdout/test | 1,264,861 windows，16,084 products |
| 遍歷方式 | compact cutoff ranges；未展開 windows |
| prepared bar-store manifest | 驗證前後相同 |

Dataset request SHA-256：
`aa49bbbdb9061c74a37eb659c8f8908c03014fc49b4a57f341b61061999eda0a`。
Validation membership SHA-256：
`b5187abd95bee5223841ca21733b3b4904ea449ef20b54f1a06d37ac4b53febb`；
test membership SHA-256：
`67adda03221adf6c6929e84b7043ce55f62ede5f871dc23fcd8c391d6ebb529a`。

另以 **1,264,861 筆合成預測**檢查磁碟 backing 與指標聚合容量：31.44 秒，
程序 peak RSS 約 1.48 GiB，輸出約 1.06 GiB。這不是完整真實 test 的推論或效能指標。
上游 pin-memory deprecated API 等 warnings 有記錄，不能解讀成無任何 warning。

### 吞吐測量

同一 Pod、同一批 16,384 個 windows、batch 256、一個 input worker：

| 資料路徑 | 含首批等待的時間 | 排除首批後的 throughput |
| --- | ---: | ---: |
| 保留的 legacy 逐筆路徑 | 52.51 s | 376 windows/s |
| 新批次數值路徑 | 9.47 s | 18,171 windows/s |

包含首批等待約 **5.54 倍**、排除首批約 **48.29 倍**。這是相同環境的資料路徑
對照，不是兩次完整訓練的端到端比較；legacy 使用目前共用的檔案讀取層，不是整個
舊 commit 的獨立部署。不可把短測速直接乘上全量 epoch 作為完訓時間保證。

第二次 profiling 發現 4,096 筆動態 windows 的 0.539 秒中，0.232 秒花在
反覆開啟 Parquet metadata。新增有界 metadata cache／直接數值解碼後，
同類 32,768 筆並行測試的 GRU／DLinear 從約 6,000 提高至約 14,000 windows/s。

最終自動選擇與短測速結果如下（GPU 已就緒後的訓練 loop，包含 input 等待）：

| 模型 | Training batch | Evaluation batch | Prefetch | 測量 windows | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU | 512 | 256 | 4 | 32,768 | 14,040/s |
| DLinear | 2,048 | 4,096 | 2 | 32,768 | 13,746/s |
| PatchTST | 2,048 | 2,048 | 4 | 16,384 | 14,521/s |

GRU／DLinear 同 GPU 同時執行，CPU input job 同時執行；PatchTST 在它們結束後
測試，未超過兩個 GPU experiment 的上限。這些 batch 是此硬體的實測結果，
不是之後固定套用於所有 GPU 的設定。

延長至每個 neural job **131,072 windows**、仍為 GRU＋DLinear＋CPU input
並行：GRU 18,071 windows/s、DLinear 17,042 windows/s。GRU loop 7.253 秒中
6.135 秒為 input 等待，DLinear 7.691 秒中 7.433 秒為等待。**資料供應仍是限制，
不能宣稱 GPU 完全滿載或 throughput 已達最佳。** 比起單看 utilization，本紀錄以
實際樣本 throughput、等待時間、完整性及中斷復原作為驗收依據。

整個驗證 Pod 觀測到的 cgroup memory 高水位約 11.00 GiB，包含回歸測試與 page
cache，不能當成完整 GBDT 或所有正式 jobs 的峰值記憶體保證。完整 GBDT 仍需依
resource plan 的全量記憶體估計准入，不會為了放進 GPU Pod 而抽樣。

### 證據與使用方式

Network volume 證據根目錄：
`/runpod-volume/diagnostics/full-workflow/baseline-run-20260921T043022Z-3705/`。
關鍵檔案為 `final-status.json`、`pytest-final.xml`、`pytest-final.log`、
`baseline-throughput-final/*.json`、`baseline-extended/*.json`、
`lazy-evaluation-audit.json`、`capacity/capacity.json`、`kronos-smoke.json`。

第一輪 throughput harness 曾誤用 12 個預設 horizons；正式訓練原本已使用設定的
14 個 horizons。`f4bb115` 修正 harness，最終驗證重新通過，未掩蓋原失敗紀錄。

啟動與續訓沿用 README 的同一組 `runpod_workflow.sh baseline`／
`runpod_tmux_launch.sh baseline` 指令；不要求使用者輸入 checkpoint 路徑。
此版本不需要重新 CPU prepare，但新增的 baseline 實作有不同的 code identity，
不會把舊不相容的部分快取或未保存狀態冒充成新版本已完成的結果。

## English

### Version, scope and environment

- Date: 2026-09-21; version: `v0.2.1`. Tested implementation: `9e1af2c`, following
  `9ebff63` and harness fix `f4bb115`. The version tag also includes this record.
- Verification run: `baseline-run-20260921T043022Z-3705`; RTX 4090, approximately
  61 GB RAM. Affinity exposed 48 cores, but the container quota was 13.6 cores;
  resource planning used 13 cores.
- At most two GPU experiments plus one CPU job, with one subprocess per loader.
  This conservative allocation reserves both train/validation pools, control
  processes and CPU work. It is not `num_workers=0` or a claim of peak GPU throughput.
- Pod `m2eezovnkt9gro` was created at 04:30:57 UTC and deleted from the control
  host at approximately 05:19:40 UTC. The API returned `deleted: true`, followed
  by an empty Pod list. Approximately 49 minutes elapsed, within the one-hour
  authorization; the 55-minute local hard guard remained armed until deletion.
  The original inefficient Pod `twepblqbl3kfqz` had already been deleted.
  The network volume was retained.
- No local Python training environment setup, production baseline/A/B training,
  new market-data downloads or CPU prepare/finalize runs were performed.

### Diagnosis and changes

The old path repeatedly used pandas, Python lists, timestamp strings and Parquet
metadata parsing for individual windows. GPU jobs mainly waited for inputs;
increasing batch size alone could not solve this. CPU planning also needed to
respect the actual container quota.

1. Vectorize windows and baseline features by symbol while preserving ordering,
   duplicate indices, corporate-action adjustments, causal cutoffs and targets.
   Decode only the required numerical columns.
2. Give each baseline worker a bounded Parquet metadata LRU: 128 files and a
   conservative 512 MiB estimate by default, with handles closed on eviction.
   Reserve 1 GiB per worker; do not pickle handles or expanded datasets.
3. Plan resources from affinity, cgroup v1/v2 quotas and host/GPU/shared memory.
   Probe train and evaluation batches separately within each concurrent GPU job's
   memory allocation. Bound prefetch by measured loader wait and buffer capacity.
   Probes do not update model weights.
4. Save neural model/optimizer/scheduler/RNG and sample cursor after the first
   batch, every 300 seconds and around full validation. Resume the remaining
   dynamic epoch order even after rebatching; align warmup and validation by samples.
5. Flush/fsync tabular arrays before publishing a durable row cursor. Save GBDT
   after each incremental horizon/quantile fit. Replay an interrupted validation,
   final test or native fit unit rather than accepting partial results.

Rebatching preserves the sample order and cursor, not bitwise floating-point
results or an identical optimizer trajectory. Hardware/resource knobs do not
invalidate numerical baseline identity; relevant data, baseline parameters or
implementation changes do. Unsaved RAM from the deleted old Pod is unrecoverable.

### Regression and completeness

Final `pytest`, `baseline_throughput`, `kronos` and `ruff` statuses are all `0`.
The full suite produced **573 passed, 1 skipped and 97 subtests passed** in
596.56 seconds. The only skip required Git metadata unavailable in the source-only
deployment; it passed in the 11-test local stdlib control-plane suite. Four separate
runtime-control tests also passed locally.

Coverage includes real-data numerical equivalence on 128 random indices; raw and
relative features; horizon offsets; duplicate, unsorted and negative indices;
zero volume and adjusted prices; durable tabular-cache resume; neural resume from
batch 32 to 47; replay of pending validation; committed GBDT-quantile reuse;
completed-artifact reuse; and bounded, closable, non-pickled metadata handles.

The four existing main-model samplers are AST-identical to `v0.2.0`. Training
retains dynamic sampling. Validation/test remain exhaustive, lazy and ordered,
including the final partial batch. Offline real Kronos CUDA forward/backward,
optimizer and finite/nonzero-gradient checks passed, with a separate 4,096-row
holdout input profile rather than full holdout prediction scoring.

The same verification session audited all 44,784 eligible instruments:

| Split | Windows | Products |
| --- | ---: | ---: |
| Validation | 1,232,972 | 13,538 |
| Holdout/test | 1,264,861 | 16,084 |

Compact cutoff ranges were traversed without materializing windows. The prepared
bar-store manifest and dataset request identity remained unchanged. Dataset and
membership hashes are recorded in the Chinese section above and the audit JSON.

A separate **1,264,861-row synthetic prediction** capacity test completed in
31.44 seconds with approximately 1.48 GiB peak process RSS and 1.06 GiB disk output.
It is not real full-test inference or a model-performance result. Upstream
pin-memory deprecation and other warnings remain documented.

### Throughput

For the same 16,384 windows, batch 256 and one input worker on the same Pod:

| Input path | Time including first-batch wait | Throughput after first batch |
| --- | ---: | ---: |
| Retained per-record legacy path | 52.51 s | 376 windows/s |
| New numerical batch path | 9.47 s | 18,171 windows/s |

The improvement was **5.54x including first-batch wait**, or **48.29x excluding
it**. This compares input paths, not two complete training runs. The legacy path
uses the current shared file-reading layer, not a separately deployed old commit.
These measurements must not be extrapolated into a guaranteed epoch duration.

Additional profiling attributed 0.232 of 0.539 seconds for 4,096 dynamic windows
to repeated Parquet metadata opening. Bounded metadata caching and direct numerical
decoding increased comparable 32,768-row concurrent GRU/DLinear probes from roughly
6,000 to roughly 14,000 windows/s.

Final hardware-selected settings and measured training-loop throughput, including
input wait but excluding initialization/autotuning:

| Model | Train batch | Eval batch | Prefetch | Measured rows | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU | 512 | 256 | 4 | 32,768 | 14,040/s |
| DLinear | 2,048 | 4,096 | 2 | 32,768 | 13,746/s |
| PatchTST | 2,048 | 2,048 | 4 | 16,384 | 14,521/s |

GRU and DLinear overlapped on the same GPU while CPU input work also ran. PatchTST
ran afterward, respecting the two-GPU-experiment limit. These are measured settings
for this hardware, not universal fixed defaults.

An extended concurrent probe used **131,072 rows per neural job**: GRU achieved
18,071 windows/s and DLinear 17,042 windows/s. Input waits were 6.135 of 7.253 seconds
for GRU and 7.433 of 7.691 seconds for DLinear. **Input supply remains a bottleneck;
full GPU saturation or globally optimal throughput has not been established.**
Acceptance relies on throughput, wait time, completeness and recovery, not utilization alone.

The observed container memory high-water mark was approximately 11.00 GiB across
the verification session, including regression tests and page cache. It is not a
full-GBDT or production-workload memory guarantee. Full GBDT remains subject to
whole-dataset memory admission and is never silently subsampled.

### Evidence and operation

Evidence root on the network volume:
`/runpod-volume/diagnostics/full-workflow/baseline-run-20260921T043022Z-3705/`.
Key files: `final-status.json`, `pytest-final.xml`, `pytest-final.log`,
`baseline-throughput-final/*.json`, `baseline-extended/*.json`,
`lazy-evaluation-audit.json`, `capacity/capacity.json` and `kronos-smoke.json`.

The initial throughput harness incorrectly used 12 default horizons, while
production training already passed all 14 configured horizons. `f4bb115` corrected
the harness; final reruns passed and the original failure evidence was retained.

Starting and resuming use the existing README `runpod_workflow.sh baseline` and
`runpod_tmux_launch.sh baseline` commands, without user-specified checkpoint paths.
CPU prepare is not required. Relevant implementation changes create a new baseline
code identity rather than treating incompatible old partial artifacts as completed
new-version results. Production full baseline/A/B training is outside this acceptance.
