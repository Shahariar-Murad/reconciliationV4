# Revenue Reconciliation Control Center

GitHub-ready Streamlit dashboard for the full Revenue Operations reconciliation process.

## Workflows

1. **Orchestrator → Backend API**
   - Backend Order List against BridgerPay, PayProcc, ZEN, Confirmo, Binance Pay, Coinsbuy, and TC Pay.
   - Backend rows sharing one transaction ID are aggregated before comparison, so add-to-cart orders reconcile correctly.
   - CFD/Futures classification is retained from the backend plan fields.

2. **PSP → BridgerPay**
   - Payabl, TrustPayments, Unlimit, PayPal, Nuvei EU, Nuvei AU, Paysafe, and Paystra.
   - Each provider uses its established success population, timezone conversion, route filter, and matching key.

3. **PSP → PayProcc**
   - dLocal, Skrill, and Paysafe local methods.
   - PayProcc rows are isolated by MID/brand.
   - Skrill fee rows are excluded from gross transaction matching.
   - Paysafe local transactions are reconciled in their original currency; local currencies are not added to USD totals.

## Output design

- Shared GMT+3 date range and optional cutoff time.
- Compact KPI cards for matched transactions, actionable exceptions, coverage gaps, outside-period rows, match rate, and USD totals.
- Provider-level summary.
- Daily status trend.
- Separate actionable exception, full detail, control, source inventory, currency, and CFD/Futures views.
- Search and status filters for exceptions and detail.
- Downloadable Excel control workbook and filtered exception CSV.
- CSV, XLS, XLSX, and ZIP uploads.

## Reconciliation behavior

- Period filtering uses an **inclusive start and exclusive end**.
- Transactions with a counterpart outside the selected period are shown as boundary rows and are not treated as missing.
- File coverage gaps are separated from true missing transactions where the source extract begins or ends inside the selected period.
- Amount tolerances:
  - Standard card/payment routes: ±$0.02.
  - Skrill: ±$0.01.
  - Confirmo and Coinsbuy against Backend: ±$3.00.
- Reporting timezone: GMT+3.

## Key provider rules

| Workflow | Provider | Provider time | Primary match |
| --- | --- | --- | --- |
| PSP → BridgerPay | Payabl | GMT+2 → GMT+3 | `Tx-Id` ↔ BridgerPay `transactionId` |
| PSP → BridgerPay | TrustPayments | GMT+0 → GMT+3 | `Reference` ↔ `transactionId` |
| PSP → BridgerPay | Unlimit | GMT+6 → GMT+3 | `Payment ID` ↔ `pspOrderId` |
| PSP → BridgerPay | PayPal | Pacific daylight time → GMT+3 | `Transaction ID` ↔ `transactionId` |
| PSP → BridgerPay | Nuvei EU/AU | GMT+6 → GMT+3 | `Client Unique ID` ↔ `transactionId` |
| PSP → BridgerPay | Paysafe | GMT+0 → GMT+3 | `Transaction ID` ↔ `transactionId` |
| PSP → BridgerPay | Paystra | UTC transaction date | Last four + gross amount; date/sequence resolves duplicates |
| PSP → PayProcc | dLocal | GMT+0 → GMT+3 | `Reference` ↔ `Gateway ID`; `Invoice` ↔ `Payment Public ID` |
| PSP → PayProcc | Skrill | Central-European local → GMT+3 | `Reference` ↔ `Gateway ID` / `Payment Public ID` |
| PSP → PayProcc | Paysafe local | GMT+0 → GMT+3 | `Transaction ID` ↔ `Gateway ID`; merchant ID ↔ public ID |

Backend reconciliation aggregates `Grand Total` by gateway + transaction ID and then compares the relevant successful orchestrator population. ZEN uses all successful non-card USD purchases. PayProcc uses `Applied Amount` when `Applied Currency` is USD; otherwise it uses the transaction amount.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Test

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Deploy with GitHub and Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload the contents of this folder to the repository root. `app.py` must remain at the root.
3. Commit and push to the `main` branch.
4. Open [Streamlit Community Cloud](https://share.streamlit.io/) and select **Create app**.
5. Select the GitHub repository and branch.
6. Set the main file path to `app.py`.
7. Deploy. No secrets are required for this version.

Streamlit installs the packages in `requirements.txt` automatically. Uploaded reconciliation files are processed in the active Streamlit session; the app does not write them to permanent storage.

## Repository structure

```text
.
├── app.py
├── recon_engine.py
├── requirements.txt
├── requirements-dev.txt
├── .streamlit/config.toml
├── .github/workflows/tests.yml
└── tests/test_engine.py
```
