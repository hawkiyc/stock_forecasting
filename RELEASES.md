# 版本紀錄

## 中文

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
