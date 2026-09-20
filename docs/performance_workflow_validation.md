# 全量評估與可重用 baseline：驗收狀態

## 中文

日期：2026-09-21（Asia/Taipei）。此文件記錄候選實作與可重現的驗收邊界；**不是已完成雲端驗收或模型效能提升的聲明**。

### 已完成的版本前置作業

- 最新長歷史 Stage 2 報告：`28c6a1e`。
- 個人限定、禁止商業／組織使用的授權與既有架構快照：`fb8482e`、`v0.1.0`。
- 上述提交與標籤已推送；第三方 Kronos 的 MIT 授權獨立保留。

### 候選實作範圍

- Validation 與 test 使用完整 split，依 canonical stock/date 順序驗證 membership；磁碟映射預測與逐塊指標彙總。
- `runpod_workflow.sh baseline`：本機先檢查快取，再決定是否建立付費 Pod；主模型訓練以前必須完成對應 baseline。
- 完整 train／validation／test 的規則、GBDT、GRU、DLinear、PatchTST；保留權重、規則參數、最佳 validation 指標、test 預測與指標。
- 同 GPU 有容量上限的神經模型多程序；CPU 輸入快取與 GPU 工作重疊，規則與 GBDT 等待快取完成後使用 CPU；依 CPU／RAM／GPU／shared memory 預算准入。
- 相同資料與 baseline 數值契約可重用；主模型或部署腳本的修改不強迫重新訓練 baseline。主模型 testing 不重新訓練或推論 baseline。
- Validation-driven plateau 排程；市場 embedding；20 維歷史數值特徵與正值輸出尺度；同日同市場 ranking 輔助 loss；LoRA rank 32／alpha 64。
- 固定日期不變：train `< 2025-06-01`；validation `[2025-06-01, 2025-12-01)`；holdout `[2025-12-01, 2026-06-01)`。
- 沒有新增 ensemble，也沒有改變 CPU prepare／finalize 或既有 bar-store 的數值語意。

### 已驗證項目

本機只使用標準函式庫控制面測試與獨立 Ruff 0.15.21；未建立、安裝或檢查本機 Python 訓練環境，未修改 lockfile。

| 檢查 | 結果 |
| --- | --- |
| `test_full_workflow_control_plane.py` | 11 passed；包含 7／8／12／16／32 cores 的 CPU 准入檢查 |
| `test_baseline_tmux.py` | 3 passed |
| `test_probe_tmux.py` | 14 passed；sandbox 封鎖 `/dev/fd`，核准後在 sandbox 外重跑 |
| `test_probe_guard.py` | 9 passed |
| `test_probe_download.py` | 13 passed |
| `test_probe_selection.py` | 19 passed；`PYTHONPATH=src` |
| `ruff check .` | passed |
| 變更檔案 Python AST、shell syntax、`git diff --check` | passed |

這些測試合計 69 項，不等於完整 pytest suite。控制面測試確認資料準備語意與 `v0.1.0` 相同，且 `configure --reuse-current` 保留資料集 request SHA。

既有 B 組 bar-store manifest 的樣本數：train 30,224,227；validation 1,232,972；test 1,264,861。此數字來自既有 manifest 的唯讀檢查，不代表候選版本已跑完這些樣本。

### 雲端驗收阻擋與費用邊界

驗證 Pod `91ceus874mtvur` 於 12:48:31 UTC 建立。非互動 SSH 驗證失敗；其後確認使用的 key 檔案正確，但有 passphrase，而 agent 尚未載入已解鎖 key，不能據此判定公鑰未獲授權。**沒有執行 CUDA、模型訓練、全量評估或下載新行情**。原背景 guard 程序不再存在後，改以本機受控前景程序持有 guard，未延長期限。

本機 guard 於 13:41:44 UTC 成功刪除第一個 Pod，總時間約 53 分鐘；network volume 未刪除。

使用者載入加密 SSH key 並另外授權最多一小時後，`ssh-add -T` 成功。RTX 5090 在原機房沒有庫存，建立請求失敗且沒有產生 Pod；之後改用同機房、同 volume 的 RTX 4090（API 當時回報 US$0.74／小時）。第二個驗證 Pod `l6afw4wirchle8` 於 2026-09-20 15:28:23 UTC 建立，SSH 與 tmux 啟動成功。本機 guard 於 **16:19:59 UTC** 刪除 Pod，運行約 51 分 36 秒，未超過一小時。後續 API Pod 清單為空，沒有建立第三個 Pod。

第二次驗證的持久化 JUnit 記錄有 **140 項：134 passed、5 failed、1 skipped**，不是整個 suite 完成。失敗與後續修正如下：

- 四項 checkpoint-evaluation 契約測試仍期待舊的 DataLoader 呼叫參數；已加入 `ranking_sampling=False` 的新預期，確保純評估不建立 training ranking index。
- 一項 neural baseline 測試遭遇 `DataLoader timed out after 300 seconds`。獨立診斷堆疊確認，驗證腳本過長的 `TMPDIR` 導致 multiprocessing 的 `AF_UNIX path too long`；不是已證明的模型數值錯誤。正式 baseline 使用短暫存路徑；驗證腳本已改用短的 Pod-local `/tmp/fin-ts-qa.*`，日誌仍持久化到 volume。
- 在執行中的 shell 驗證腳本上覆寫檔案，另外造成讀取位置錯位與 `baseline: command not found`；重新執行沒有產生完整的成功記錄，不能計入驗收。後續驗證腳本先將主體解析成函式，且執行中的 shell 腳本不得 hot-edit。
- CPU 准入公式已修正，將 GPU 工作父程序、兩個可能同時存活的 DataLoader pools、control cores 與 CPU input workers 一併計入；上述本機標準函式庫測試通過。新增 GPU 整合測試仍待重跑。

已在雲端通過的新功能測試包含：runtime date/market index 的完整樣本保留，以及 full tabular cache 重用且不改 bar-store。**尚無完整 neural baseline builder、同 GPU 雙模型、真實 Kronos 梯度或百萬筆評估容量測試通過的證據。** 沒有跑正式 A/B 訓練、建立正式 baseline 完成快取、重新準備正式行情資料或呼叫行情供應商下載 API。

證據保留在 network volume 的 `diagnostics/full-workflow/baseline-run-20260920T152758Z-32557/` 與該 run 的 tmux logs；已下載的 `pytest.xml`、`pytest.log`、`neural-debug.log`、`combined.log` 位於本機 ignored `.runpod/verification/baseline-run-20260920T152758Z-32557/`。修正後的驗證腳本會自動依序執行 pytest、Ruff、Kronos smoke 與 capacity check，分別設 timeout 並保存 `acceptance-status.json`；此自動串接本身仍待雲端驗證。

仍須在取得新的雲端執行許可後完成：

1. 遠端既有 Poetry 環境中的完整 `pytest tests`，包括新增 full-data baseline builder、resume、CPU/GPU 同時執行、快取重用與不重跑 baseline 的整合測試。
2. `verify_kronos_full_workflow.py`：僅四筆既有 train windows，offline Kronos／LoRA／尺度與市場分支的 forward/backward 梯度檢查；不保存正式模型權重。
3. `verify_full_evaluation_capacity.py`：1,264,861 筆、14 horizons 的合成資料評估，記錄 peak RSS、耗時與磁碟量；不是行情回測。
4. 修復雲端回歸發現的問題、重新執行測試，再發布下一個正式模型版本標籤。

候選版尚未通過上述驗收，不能據此宣稱準確度改善、全量訓練可在特定時限內完成，或建議直接啟動正式 A/B 訓練。不發布 `v0.2.0` 正式 tag，也不推送效能改版。Network volume 上的程式碼包含本輪的部分修正，與最後本機版本尚未重新同步；後續測試前必須重新 `sync --apply`，不需要重新執行 CPU prepare。

## English

Date: 2026-09-21 (Asia/Taipei). This document records the candidate implementation and its verification boundary. **It is not a claim of completed cloud acceptance or improved predictive performance.**

### Completed release prerequisites

- Latest long-history Stage 2 report: `28c6a1e`.
- Personal-only, noncommercial/nonorganizational terms and the existing architecture snapshot: `fb8482e`, `v0.1.0`.
- These commits and the tag were pushed. Kronos retains its separate third-party MIT terms.

### Candidate implementation

- Full validation/test populations, canonical stock/date membership checks, disk-mapped predictions and chunked metric reductions.
- An independent `runpod_workflow.sh baseline` entrypoint with a control-host cache gate before paid Pod creation; matching completed baselines are required before main-model training.
- Full-train/full-validation/full-test rules, GBDT, GRU, DLinear and PatchTST, retaining weights, rule parameters, selected validation metrics, test predictions and metrics.
- Bounded same-GPU neural processes overlapping CPU input-cache preparation; CPU rules/GBDT wait for complete inputs. Admission considers CPUs, host RAM, GPU memory and shared memory.
- Reuse keyed by immutable data and baseline numerical contracts. Main-model/deployment-only changes do not rebuild baselines. Main-model testing neither fits nor runs baseline inference again.
- Validation-driven LR plateaus; market embeddings; 20 historical features with explicit positive output scaling; a same-date/same-market ranking auxiliary loss; LoRA rank 32/alpha 64.
- Unchanged periods: train before 2025-06-01; validation from 2025-06-01 through 2025-11-30; holdout from 2025-12-01 through 2026-05-31.
- No ensemble and no numerical changes to CPU prepare/finalize or existing prepared bar stores.

### Verified checks

Local checks used standard-library control-plane tests and standalone Ruff 0.15.21. No local Python training environment was created, installed or inspected; no lockfile changed.

The six control-plane suites listed above passed 11, 3, 14, 9, 13 and 19 tests respectively: 69 total. The CPU admission test covers 7/8/12/16/32 cores. The probe tmux suite required approved execution outside the sandbox because `/dev/fd` process substitution was blocked. Ruff, changed-file AST/shell syntax, and Git whitespace checks passed. These results are **not** the full pytest suite. Tests verify unchanged preparation semantics against `v0.1.0` and preservation of the dataset request SHA when refreshing configuration.

The existing B manifest reports 30,224,227 training rows, 1,232,972 validation rows and 1,264,861 test rows. This read-only manifest inspection does not establish execution over those populations.

### Cloud acceptance blocker and spending boundary

Verification Pod `91ceus874mtvur` was created at 12:48:31 UTC. Noninteractive SSH authentication failed. The selected key was subsequently confirmed to be correct but passphrase-protected, without an unlocked identity in the agent; this does not establish missing public-key authorization. No CUDA tests, model training, full evaluation or market-data downloads ran. After the initial detached guard disappeared, a controlled foreground guard was held on the local control host without extending the deadline.

The local guard deleted the first Pod successfully at 13:41:44 UTC, approximately 53 minutes after creation. The network volume was preserved.

After the user unlocked the key and separately authorized another hour, `ssh-add -T` succeeded. The RTX 5090 creation request failed because that datacenter had no stock; it created no Pod. An RTX 4090 in the same datacenter, attached to the same volume, was then used (the API quoted US$0.74/hour). The second verification Pod, `l6afw4wirchle8`, was created at 2026-09-20 15:28:23 UTC. SSH and detached tmux succeeded. The local guard deleted it at **16:19:59 UTC**, about 51 minutes 36 seconds after creation, within the authorized hour. The subsequent API Pod list was empty; no third Pod was created.

Its persisted JUnit report records **140 cases: 134 passed, 5 failed and 1 skipped**, not a completed suite:

- Four checkpoint-evaluation tests expected the old DataLoader arguments. Their expectations now include `ranking_sampling=False`, ensuring evaluation does not build a training ranking index.
- One neural baseline test reported a 300-second DataLoader timeout. An isolated diagnostic exposed `AF_UNIX path too long`: the verification script's long `TMPDIR` prevented multiprocessing tensor transport. This does not establish a model numerical defect. Production baseline execution uses short scratch paths; verification now uses Pod-local `/tmp/fin-ts-qa.*` while persisting evidence on the volume.
- Overwriting the actively executing shell verification file also displaced its read offset and produced `baseline: command not found`. The rerun has no complete successful record and receives no acceptance credit. The revised verifier parses its body as a function first; running shell scripts must not be hot-edited.
- CPU admission now includes GPU parents, two potentially live DataLoader pools, control cores and CPU input workers. The standard-library resource-planning checks passed locally; the added GPU integration checks remain unverified.

The new cloud tests that did pass cover complete-population runtime date/market indexing and reusable full tabular inputs without modifying the bar store. **There is no passing evidence yet for the complete neural baseline builder, two simultaneous GPU experiments, the real Kronos gradient check, or million-row evaluation capacity.** No production A/B training, official baseline completion cache, production market-data preparation or provider download request was run.

Evidence is retained under the volume's `diagnostics/full-workflow/baseline-run-20260920T152758Z-32557/` and the run's tmux logs. Downloaded `pytest.xml`, `pytest.log`, `neural-debug.log` and `combined.log` are in ignored local `.runpod/verification/baseline-run-20260920T152758Z-32557/`. The revised verifier automatically sequences pytest, Ruff, Kronos smoke and capacity checks with individual deadlines and an `acceptance-status.json`; this orchestration itself still requires cloud verification.

After renewed cloud authorization, acceptance still requires:

1. The full remote pytest suite in the existing Poetry environment, including complete synthetic baseline building/resume, CPU/GPU overlap, cache reuse and no baseline recomputation during main testing.
2. The offline four-training-window Kronos/LoRA/scale/market forward-backward smoke test, without production weight publication.
3. A synthetic 1,264,861-row, 14-horizon evaluation capacity test recording peak RSS, elapsed time and disk usage.
4. Fixes and reruns for any runtime regressions before the next formal architecture release tag.

Until then, the candidate is not acceptance-complete and cannot establish better accuracy, a full-training runtime guarantee, or readiness for production A/B training. No formal `v0.2.0` tag or performance-revision push is issued. The network volume contains only some of this run's fixes and has not been resynced with the final local source: run `sync --apply` before further tests, but do not rerun CPU prepare.
