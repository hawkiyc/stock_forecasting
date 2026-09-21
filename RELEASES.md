# 版本紀錄

## 中文

### v0.2.1 2026-09-21

- Baseline 向量化動態 windows／數值特徵，新增有界 Parquet metadata cache 與必要欄位
  解碼；同機 input 路徑對照由 52.51 秒降至 9.47 秒，不代表全量訓練加速倍數。
- cgroup CPU quota 與 host／GPU／shared-memory 准入；每個模型自動選擇 train／eval
  batch 與 prefetch，維持同 GPU／CPU 工作有界並行。
- 神經模型 epoch 內 sample-cursor 接續、tabular cache durable row cursor，以及
  GBDT horizon／quantile 中途保存；沿用既有 baseline／tmux CLI。
- 最終完整雲端回歸 573 passed、97 subtests passed；1 項 Git-dependent 檢查於
  本機通過。CPU prepare 與資料身份不變，不需新行情 API call。
- 已驗證數值等價與吞吐改善；仍有 input wait，未宣稱 GPU 完全滿載或正式訓練已完成。
  詳見 [baseline 執行效率驗收](docs/baseline_runtime_validation.md)。

### v0.2.0 2026-09-21

- Training 保留動態 sampling；validation／test 動態完整遍歷精確 cutoff ranges，
  不抽樣、不丟尾 batch、不預先展開巨大 input-window 資料集。
- 獨立 full-data baseline 建置、權重／指標快取及本機付費 Pod 啟動前的重用檢查；
  同 GPU 多模型與 CPU 工作有界並行。主模型訓練需先完成對應 baseline。
- Validation-driven plateau、市場感知、20 維尺度特徵與明確輸出尺度、同日同市場
  ranking loss、LoRA rank 32／alpha 64。沒有新增 ensemble。
- Inference-only batch tuning、多 worker／prefetch／pinned-memory 預算、輕量 spawn
  狀態及 baseline persistent pools；CPU prepare 的資料語意與既有資料集保持不變。
- 工程驗收：完整 suite 與 fixture 修正重測合併後，562 項雲端通過；1 項 Git-dependent
  檢查在本機通過。真實 Kronos CUDA 梯度、全 cutoff 稽核及百萬筆指標容量測試通過。
- 這是已完成限定工程驗收的架構快照，不是新模型準確度改善或實盤可用性的保證；
  尚未執行新架構的正式 A／B 全量訓練。詳見 [驗收紀錄](docs/performance_workflow_validation.md)。

### v0.1.0 2026-09-20

此 tag 保存 full-size baseline 改版前的模型架構與已完成的 A／B 評估證據。

- Kronos-base、LoRA rank 8／alpha 16、8 維歷史尺度分支與 benchmark 直接連接。
- 模型輸出 schema 5.0：1–14 個持有交易日的 q10／q50／q90 alpha。
- Train 截止 2025-06-01 exclusive；validation 為 2025-06-01 至
  2025-12-01 exclusive；holdout 為 2025-12-01 至 2026-06-01 exclusive。
- A：`run-20260915T091153Z-918316277`，2021 年起始，最佳 `checkpoint-043773`。
- B：`run-20260917T110811Z-1986920595`，2016 年起始，最佳 `checkpoint-094452`。
- 兩份報告位於 `reports/`。此版本的 benchmark 仍使用抽樣集合，不能解讀為完整
  validation／test 評估，也不能把兩個 run 當成相同股票—日期成員的配對比較。
- 自有程式碼與明示發布的新增模型成果採個人非商業授權；上游 Kronos 保留 MIT。

版本 tag 表示可追溯的程式與報告快照，不表示已通過實盤部署驗收。下載的權重與行情
資料不納入 Git，須以 run ID、checkpoint 與各自 manifest 追溯。

## English

### v0.2.1 2026-09-21

- Vectorized lazy baseline windows/features, bounded Parquet metadata caching and
  numerical-column decoding. The same-machine input-path comparison improved from
  52.51 to 9.47 seconds; this is not a whole-training speedup measurement.
- cgroup CPU quotas and host/GPU/shared-memory admission; per-model train/eval
  batch and prefetch tuning, with bounded same-GPU/CPU concurrency.
- Mid-epoch neural sample-cursor resume, durable tabular-cache row cursors and
  GBDT horizon/quantile checkpointing, using the existing baseline/tmux CLI.
- Final full cloud regression: 573 passed and 97 subtests passed; one Git-dependent
  check passed locally. Prepared-data identity is unchanged; no new market calls.
- Numerical equivalence and throughput improvements are verified. Input wait remains;
  this is not evidence of full GPU saturation or completed production training.
  See [baseline runtime acceptance](docs/baseline_runtime_validation.md).

### v0.2.0 2026-09-21

- Preserve dynamic training sampling; exhaust exact validation/test cutoff ranges
  lazily, without sampling, dropping the final batch, or materializing input windows.
- Independent full-data baseline building, reusable weights/metrics and a local
  cache gate before paid Pod creation; bounded same-GPU experiments and CPU work.
  Main-model training requires a matching completed baseline.
- Validation-driven LR plateaus, market awareness, 20 scale features with explicit
  output scaling, same-date/market ranking loss, and LoRA rank 32/alpha 64; no ensemble.
- Inference-only batch tuning, memory-budgeted workers/prefetch/pinning, lightweight
  spawn state and persistent baseline pools. Prepared-data semantics remain unchanged.
- Bounded acceptance reconciles the full suite and repaired-fixture reruns: 562
  cloud passes and one Git-dependent check passed locally. Real Kronos CUDA gradients,
  exhaustive cutoff auditing and million-row metric aggregation passed.
- This is an engineering-validated architecture snapshot, not evidence of improved
  accuracy or live-trading readiness. Production A/B training remains outstanding;
  see the [acceptance record](docs/performance_workflow_validation.md).

### v0.1.0 2026-09-20

This tag preserves the architecture and completed A/B evaluation evidence before
the full-size baseline workflow revision.

- Kronos-base, LoRA rank 8 / alpha 16, eight historical-scale features, and a
  direct benchmark connection.
- Model output schema 5.0: q10/q50/q90 alpha for holding horizons 1–14.
- Train ends before 2025-06-01; validation is [2025-06-01, 2025-12-01);
  holdout is [2025-12-01, 2026-06-01).
- A: `run-20260915T091153Z-918316277`, history from 2021, best `checkpoint-043773`.
- B: `run-20260917T110811Z-1986920595`, history from 2016, best `checkpoint-094452`.
- Both reports are in `reports/`. This version still benchmarks sampled sets;
  these are not full validation/test evaluations or paired A/B evaluations of
  identical symbol-date members.
- Project-owned code and expressly released model additions use the personal
  non-commercial license. Upstream Kronos retains MIT licensing.

A version tag identifies a reproducible code/report snapshot, not production
trading approval. Downloaded weights and market data are excluded from Git;
identify them by run ID, checkpoint, and their respective manifests.
