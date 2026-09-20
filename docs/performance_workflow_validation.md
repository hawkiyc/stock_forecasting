# 全量評估與可重用 baseline：驗收紀錄

## 中文

日期：2026-09-21（Asia/Taipei）。**本輪限定範圍的工程驗收已完成**：完整回歸測試及失敗項目重測、CUDA 小型整合、既有資料全量 cutoff 稽核、百萬筆指標彙總容量測試。這不等於已完成新架構的正式 A／B 訓練、整份真實 holdout 模型推論，或證明預測準確度改善。

### 版本與實作範圍

- 最新 B 組報告：`28c6a1e`；個人限定／禁止商業與組織使用的授權及既有架構快照：`fb8482e`、`v0.1.0`。這些前置提交與 tag 已推送；第三方 Kronos 的 MIT 授權獨立保留。
- 本輪架構版本為 `v0.2.0`，包含完整評估、可重用 baseline、validation-driven plateau、市場 embedding、20 維歷史特徵與明確輸出尺度、同日同市場 ranking loss、LoRA rank 32／alpha 64。不新增 ensemble。
- 固定期間不變：train `< 2025-06-01`；validation `[2025-06-01, 2025-12-01)`；holdout `[2025-12-01, 2026-06-01)`。
- 本輪程式、測試與文件提交只保留在本機，沒有推送效能改版。

### Training 與 evaluation 的明確界線

**Training 保留動態 sampling。** 各 epoch 的隨機順序、同日同市場分組及 batch 讀取時動態建立 windows 均保留。以 Python AST 比對候選實作 `a51da94`，以下四項 sampler 的實作完全相同：

- `BlockwisePermutationSampler`
- `FixedSizeBatchSampler`
- `ResumableFixedSizeBatchSampler`
- `DateMarketSampler`

**Validation／test 不取樣。** 依固定 `symbol → cutoff` 順序讀取每檔商品所有合法 windows；cutoff ranges 是精確區間而非筆數估計。使用 sequential sampler，保留最後不足一個 batch 的資料，不補齊重複樣本。舊的數值型 `evaluation_max_samples` 會警告並忽略，不能讓舊 checkpoint 設定恢復抽樣。

Dataset 僅保留 compact ranges、商品 metadata 與有上限的快取；context／label 在讀取 batch 時產生。預測與 targets 使用磁碟映射及逐塊指標彙總；不製作完整 input-window tensor 資料集。完整、有序 symbol/date membership SHA-256 用於核對主模型與 baseline 的評估母體。日期相同仍不保證不同 universe 的成員相同。

### 回歸與整合測試結果

本次驗證 Pod 為 `0aqhj4q3mrb6nt`，RTX 4090、48 個可見 CPU，沿用既有 network volume 與遠端 Poetry 環境。正式程式受測版本為 `e90256f`；其後 `247139e` 只修正測試 fixtures。

| 檢查 | 結果與邊界 |
| --- | --- |
| 完整遠端 suite | 544 passed、17 failed、1 skipped；616.92 秒，97 subtests passed |
| 失敗修正後的相關完整測試檔與 feature 測試 | 39 passed，26.18 秒；包含全部 17 個失敗及 1 個新 regression test |
| 兩輪 JUnit 依 classname／name 合併核對 | 563 個唯一 cases；562 passed、1 skipped，沒有未解決 failure／error |
| 遠端略過的 Git metadata 測試 | source-only 部署不含 Git；本機標準函式庫 control-plane suite 重跑 11 passed，包含此項 |
| Ruff | 本機及遠端 passed |
| 真實 Kronos offline CUDA smoke | 4 個既有 train windows；forward／backward／optimizer step 通過；LoRA、市場、尺度分支梯度 finite 且 nonzero |
| 真實完整 cutoff 稽核 | 44,784 個 eligible instruments 逐檔核對，318.93 秒，4 個 bounded workers；manifest 未改動 |
| 百萬筆指標容量測試 | 合成 1,264,861 筆 × 14 horizons；60.18 秒；peak RSS 約 1.48 GiB；磁碟暫存約 1.06 GiB |

17 個失敗分為三種測試 fixture 問題，沒有藉由放寬正式檢查消除：

1. 15 項歷史 migration 測試錯將目前已變更的 evaluator 指紋當成舊版本；改用明確的歷史 from／to fixture，另加「舊 migration 不得授權目前 evaluator」拒絕測試。沒有修改 migration allowlist。
2. Lifecycle 測試未 stub marker 讀取，因此先落在 unreadable marker 分支；補齊隔離，繼續驗證不明狀態不得視為終止訊號。
3. 尺度分支 fixture 提供 14 個 robust scales，卻使用不同長度的預設 horizons；明確指定 horizons 1–14。正式模型的長度檢查保留。

已通過的整合項目包含完整合成 train／validation／test 的 rules、GBDT、GRU、DLinear、PatchTST 建置、權重／預測／指標保存、baseline resume、完成快取直接重用、主模型 testing 不重訓／不推論 baseline，以及同一張 GPU 上兩個獨立神經模型程序。這些是小型完整資料集測試，**不是三千萬筆正式 baseline 訓練**。

### 完整真實評估母體

沿用 B 組已準備的 bar-store；未下載行情或重新 prepare／finalize。

| Split | 有 windows 的商品數 | 精確 windows |
| --- | ---: | ---: |
| Validation | 13,538 | 1,232,972 |
| Holdout／test | 16,084 | 1,264,861 |

稽核獨立重算每檔商品的 context、benchmark 覆蓋、異常轉換、完整 14 日 label 與 split 邊界，再比對既有 cutoff ranges；不是只讀 manifest 的總筆數。另有缺失 benchmark 日期、較晚上市商品、非連續 ranges、不同 seed／batch／worker／prefetch 與尾 batch 的回歸測試。

- Validation membership SHA-256：`b5187abd95bee5223841ca21733b3b4904ea449ef20b54f1a06d37ac4b53febb`
- Test membership SHA-256：`67adda03221adf6c6929e84b7043ce55f62ede5f871dc23fcd8c391d6ebb529a`
- Bar-store manifest SHA-256：`d54d6d62dcf8c676adfd38da421cb59ed5b272c0b7e6c23e44a81d94e1517ab6`
- Dataset request SHA-256：`aa49bbbdb9061c74a37eb659c8f8908c03014fc49b4a57f341b61061999eda0a`，保持不變。
- 既有 train windows：30,224,227；此數字是完整可用母體，不表示本次 smoke 訓練使用全部樣本。

### GPU 資料供應與修正

純 inference batch probe 不做 backward，也不沿用另一張 GPU 的 batch plan。prefetch 同時受實測速度、CPU、容器 RAM、shared memory、同時存活 pools 及 pinned copies 約束；使用多 worker、pinned memory 與非同步 transfer。

實測修正兩項啟動成本：baseline train／validation 使用有界 persistent worker pools，完成或例外時關閉；lazy Dataset 的 spawn 狀態只傳設定，由每個 worker 重新開啟唯讀 compact indexes，不再經 pipe 序列化整份索引。此 Dataset 的 pickle 大小為 **296 bytes**。

本次 RTX 4090 自動選擇 **batch 32、17 workers、prefetch 3、pin_memory=true、drop_last=false**。4096 個真實 test windows 的 bounded profile：

- 含 worker 啟動與結束：26.44 秒，154.93 windows／秒。
- 排除前 256 筆 warmup 後：610.96 windows／秒。
- 純 model probe 最佳 batch 約 804 windows／秒；更大的 batch 沒有更快。
- smoke 訓練 peak GPU 約 0.93 GiB；整個 inference batch 搜尋與 profile 的 peak 約 3.73 GiB，後者不是 batch 32 單獨的用量。

這組參數是該硬體的測量結果，不是寫死的跨 GPU 預設。profile 只走 4096 筆以控制驗證費用；正式 validation／testing 使用完整 split，不會套用此 profile 上限。吞吐量也不是每檔商品或整場正式執行的保證，GPU utilization 不能由此宣稱固定為 100%。

### 資料、費用與證據

未建立／檢查本機 Python 訓練環境，未安裝本機訓練套件，未修改 Poetry lockfile。沒有新行情 API 呼叫、CPU prepare／finalize、正式 A／B 訓練或正式 baseline 完成快取。

Pod 於 2026-09-20 17:52:02 UTC 建立，本機 guard 設定 55 分鐘硬期限。完成後約 18:42 UTC 由本機專案 CLI 刪除，API 回覆 `deleted=true`，後續 Pod list 為空；約 50 分鐘，未超過授權的一小時。Network volume 保留。不是依賴 Pod 自我終止，也沒有因驗證重跑延長期限。

18:47 UTC 的 guard 到期紀錄屬於刪除後的第二次終止請求；API 回覆 404 not found，並非 Pod 仍在計費。當時 per-Pod timeout audit 沒有覆寫其他 run 的 singleton marker。

證據根目錄：

- Network volume：`diagnostics/full-workflow/baseline-run-20260920T175140Z-32301/`
- 本機 ignored：`.runpod/verification/baseline-run-20260920T175140Z-32301/`

保留原始失敗與中斷紀錄，沒有覆寫成成功。最終判讀使用 `pytest-final.xml` 加 `pytest-fixtures.xml`、`fixture-status.json`、`kronos-final.log`／`kronos-smoke.json`、`capacity.json` 與 `lazy-evaluation-audit.json`。合成 baseline 完整產物另外保存在 volume 的 `synthetic-baseline/`，不冒充正式 baseline 快取。先前兩次未完成驗證的證據仍保留，不計入本輪通過數。

`verification-summary.json` 記錄跨兩輪測試的最終核對結果，亦已保存到上述 volume 目錄；不能只拿第一輪的非零 exit code 判斷最後狀態。

下一個正式工作是依既有腳本先 build 對應的完整 baseline，再執行主模型 A／B 訓練。現有 CPU-prepared 資料可沿用；不應為本次改版重跑資料下載或 CPU prepare。正式模型是否提升準確度，須由完整、相同 membership 的評估結果判定。

## English

Date: 2026-09-21 (Asia/Taipei). **The bounded engineering acceptance scope is complete**: full regression execution plus targeted failure reruns, small CUDA integrations, an exhaustive audit of existing cutoff eligibility, and million-row metric aggregation. This is not completed production A/B training, complete real-holdout model inference, or evidence of improved predictive accuracy.

### Version and implementation

The latest B report is `28c6a1e`; licensing and the historical architecture snapshot are `fb8482e` and `v0.1.0`. Those prerequisites were pushed; upstream Kronos retains separate MIT terms. The new architecture snapshot is `v0.2.0`, covering exhaustive evaluation, reusable baselines, validation-driven LR plateaus, market embeddings, 20 historical features and explicit output scaling, same-date/market ranking loss, and LoRA rank 32/alpha 64. No ensemble was added. Performance-revision commits remain local and have not been pushed.

Periods remain unchanged: training before 2025-06-01; validation [2025-06-01, 2025-12-01); holdout [2025-12-01, 2026-06-01).

### Sampling boundary

**Training retains dynamic sampling**, epoch-wise random ordering, date/market groups, and windows constructed at batch-read time. AST comparison with candidate commit `a51da94` confirms unchanged implementations of `BlockwisePermutationSampler`, `FixedSizeBatchSampler`, `ResumableFixedSizeBatchSampler`, and `DateMarketSampler`.

**Validation/testing never sample.** They enumerate every eligible instrument/cutoff in fixed order, retain the last short batch, and never duplicate padding rows. A legacy numeric evaluation cap warns and is ignored. Compact exact ranges and bounded instrument caches replace materialized input windows; predictions and targets use disk-backed, chunked reduction. Ordered membership hashes identify the actual evaluated population. Matching dates alone does not establish matching universes.

### Regression and integration evidence

Tests ran on RTX 4090 Pod `0aqhj4q3mrb6nt`, with 48 visible CPUs and the existing remote Poetry environment. Production code was `e90256f`; subsequent commit `247139e` changed only test fixtures.

- Full suite: 544 passed, 17 failed, 1 skipped in 616.92 seconds; 97 subtests passed.
- Affected suites and the feature test after fixture repairs: 39 passed in 26.18 seconds, covering all 17 failures and one added regression.
- JUnit identity reconciliation: 563 unique cases, 562 passed and one skipped, with no unresolved failures/errors.
- The skipped Git-dependent preparation-contract check cannot run in source-only cloud deployments. It passed in the local standard-library control-plane suite, which passed all 11 checks.
- Local and remote Ruff passed.
- Offline real-Kronos CUDA smoke: four existing training windows, a forward/backward/optimizer step, and finite nonzero LoRA, market and scale gradients.
- Full eligibility audit: 44,784 eligible instruments, four bounded workers, 318.93 seconds, unchanged prepared manifest.
- Synthetic capacity: 1,264,861 rows × 14 horizons, 60.18 seconds, approximately 1.48 GiB peak RSS and 1.06 GiB scratch disk.

The failures were fixture defects: 15 historical migration cases incorrectly assumed that today's changed evaluator still had the historical target digest; one lifecycle test did not isolate marker loading; and one feature test supplied 14 scales without explicitly selecting 14 horizons. Historical migration fixtures were fixed and a current-evaluator rejection test added, marker loading was stubbed, and horizons 1–14 were specified. Production migration allowlists and safety checks were not weakened.

Passing integrations include complete synthetic train/validation/test baseline building for rules, GBDT, GRU, DLinear and PatchTST; persisted weights/predictions/metrics; resume; cache reuse without reinitialization; main testing without baseline fitting/inference; and two independent neural experiments on one GPU. These are complete small-fixture integrations, **not a thirty-million-row production baseline run**.

### Existing real-data populations

The B bar store was reused without downloads or preparation:

| Split | Instruments with windows | Exact windows |
| --- | ---: | ---: |
| Validation | 13,538 | 1,232,972 |
| Holdout/test | 16,084 | 1,264,861 |

The audit independently recomputed context validity, benchmark coverage, exceptional transitions, complete 14-day labels and split boundaries per instrument, then compared exact prepared ranges. Regression cases also cover missing benchmark dates, late listings, non-contiguous ranges, differing seeds/batches/workers/prefetch, and short final batches.

The validation/test membership hashes, manifest hash and unchanged dataset-request hash are recorded in the Chinese section above. The existing complete training population contains 30,224,227 windows; only four were used for the real-model smoke test.

### Loader throughput and memory

Inference-only probing avoids backward passes and does not reuse another GPU's batch plan. Prefetch respects measured speed, CPUs, container RAM, shared memory, live pools and pinned copies. Baseline train/validation pools persist within bounded lifetimes and are closed on completion or error. Dataset spawn state contains constructor settings only; workers reopen compact read-only indexes. Its serialized size was 296 bytes, avoiding the observed large-index pipe stall.

The measured RTX 4090 plan was batch 32, 17 workers, prefetch 3, pinned memory, and no dropped final batch. A 4096-window real-data profile took 26.44 seconds including startup/shutdown: 154.93 windows/second overall, or 610.96 after the first 256 warmup samples. The best pure-model probe was approximately 804 windows/second; larger batches were slower. Training smoke peak GPU allocation was approximately 0.93 GiB; the entire inference batch search/profile peaked at 3.73 GiB, not the isolated batch-32 allocation.

These are measured, hardware-specific settings, not fixed defaults or a promise of 100% utilization. The 4096-row bound applies only to the paid verification profile; production evaluation remains exhaustive.

### Cost, artifacts and remaining experimental work

No local training environment was created or inspected, no local training dependencies installed, and no Poetry lockfile changed. No new market-data API, CPU preparation/finalization, production A/B training, or production baseline completion cache was used.

The Pod was created at 17:52:02 UTC on 2026-09-20 with a 55-minute local guard. The local project CLI deleted it around 18:42 UTC, returning `deleted=true`; the subsequent Pod list was empty. Approximately 50 minutes elapsed, within the authorized hour. The network volume remains. No remote self-termination or deadline extension was relied upon.

The guard's 18:47 UTC deadline record is a second termination attempt after deletion: the API returned 404 not found, not an ongoing paid Pod. Its per-Pod timeout audit did not overwrite another run's singleton marker.

Evidence is retained on the volume under `diagnostics/full-workflow/baseline-run-20260920T175140Z-32301/` and locally in the ignored matching `.runpod/verification/` directory. Original failed/interrupted logs remain unchanged. Use both `pytest-final.xml` and `pytest-fixtures.xml`, plus the fixture status, final Kronos log/JSON, capacity JSON and lazy-audit JSON. Complete synthetic baseline artifacts remain under the evidence directory's `synthetic-baseline/`, separate from production caches. Earlier incomplete attempts receive no passing credit.

The volume also stores `verification-summary.json`, reconciling the two test rounds. An initial nonzero exit code alone does not represent the final acceptance state.

Production execution should now build the matching complete baseline through existing scripts before main-model A/B training. Reuse the prepared data; this revision does not require downloads or CPU prepare. Predictive improvement still requires full, membership-matched model evaluation.
