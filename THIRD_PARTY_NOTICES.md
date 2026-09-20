# 第三方授權聲明

## 中文

本專案自有部分適用 [個人非商業使用授權](LICENSE)。第三方元件不受本專案自訂
授權重新限制；套件、模型與資料必須分別遵循其原有條款。

### Kronos

- 上游：[shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)。
- 本機 vendored 原始碼：`src/stock_forecasting/_vendor/kronos/`。
- 著作權：Copyright (c) 2025 ShiYu。
- 原始 MIT 條文保留於 [vendored LICENSE](src/stock_forecasting/_vendor/kronos/LICENSE)。
- [Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base) 與
  [Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base)
  模型卡於 2026-09-20 核對為 MIT；部署仍須核對實際使用 revision 的條款。

本專案不撤銷原始 Kronos 元件的 MIT 權利。此例外不代表本專案自有的 pipeline、head、
特徵工程或新增模型參數也採 MIT。本專案發布的模型新增部分另見 [MODEL_LICENSE](MODEL_LICENSE)。

### 套件與行情資料

第三方相依套件保留各自授權。本專案不將 EODHD、TWSE、TPEx 或其他供應商的資料納入
軟體授權；使用、存取及再散布資料仍須依供應商契約與適用規範取得權利。
本聲明不代替每個已安裝套件或另行取得資料的完整授權文件。

## English

Project-owned material uses the [Personal Non-Commercial License](LICENSE).
Third-party components are not relicensed or restricted by that custom license.
Dependencies, models, and data remain subject to their respective terms.

### Kronos

- Upstream: [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos).
- Vendored source: `src/stock_forecasting/_vendor/kronos/`.
- Copyright: Copyright (c) 2025 ShiYu.
- The original MIT text is retained in the [vendored LICENSE](src/stock_forecasting/_vendor/kronos/LICENSE).
- The [Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base) and
  [Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base)
  model cards were checked as MIT on 2026-09-20. Verify the terms of the actual
  revision used in a deployment.

Original Kronos components retain their MIT rights. This exception does not make
project-owned pipelines, heads, feature engineering, or added trained parameters
MIT-licensed. See [MODEL_LICENSE](MODEL_LICENSE) for project-released model additions.

### Dependencies and market data

Third-party dependencies retain their own licenses. The software license does not
grant rights to EODHD, TWSE, TPEx, or other provider data. Access, use, and
redistribution require appropriate provider and legal permissions. This notice
does not replace the complete terms of installed packages or separately obtained data.
