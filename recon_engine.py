from __future__ import annotations

import csv
import io
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd


REPORT_TIMEZONE = "GMT+3"

BACKEND = "Backend API"
BRIDGER = "BridgerPay"
PAYPROCC = "PayProcc"
ZEN = "ZEN"
CONFIRMO = "Confirmo"
BINANCE = "Binance Pay"
COINSBUY = "Coinsbuy"
TCPAY = "TC Pay"

PAYABL = "Payabl"
TRUST = "TrustPayments"
UNLIMIT = "Unlimit"
PAYPAL = "PayPal"
NUVEI_EU = "Nuvei EU"
NUVEI_AU = "Nuvei AU"
PAYSAFE = "Paysafe"
PAYSTRA = "Paystra"
DLOCAL = "dLocal"
SKRILL = "Skrill"

BACKEND_TOLERANCES = {
    BRIDGER: 0.02,
    ZEN: 0.02,
    CONFIRMO: 3.00,
    BINANCE: 0.02,
    PAYPROCC: 0.02,
    TCPAY: 0.02,
    COINSBUY: 3.00,
}

BRIDGER_PROVIDERS = {PAYABL, TRUST, UNLIMIT, PAYPAL, NUVEI_EU, NUVEI_AU, PAYSAFE, PAYSTRA}
PAYPROCC_PROVIDERS = {DLOCAL, SKRILL, PAYSAFE}


class ReconciliationError(ValueError):
    pass


@dataclass
class LoadedSource:
    source: str
    frame: pd.DataFrame
    files: list[str]


def clean_id(value) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def clean_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def parse_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def utc_to_gmt3(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce", format="mixed").dt.tz_convert("Etc/GMT-3").dt.tz_localize(None)


def _require(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ReconciliationError(f"{source}: missing required columns: {', '.join(missing)}")


def _read_csv_flexible(blob: bytes, **kwargs) -> pd.DataFrame:
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(blob), encoding=encoding, low_memory=False, **kwargs)
        except Exception as exc:  # pragma: no cover - only used for fallback diagnostics
            errors.append(f"{encoding}: {exc}")
    raise ReconciliationError("Unable to read CSV. " + " | ".join(errors[-2:]))


def _read_paystra_csv(blob: bytes) -> pd.DataFrame:
    text = blob.decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return pd.DataFrame()
    headers = rows[0] + ["Unlabeled Extra"]
    data = []
    for row in rows[1:]:
        if not row or row[0] == "====END OF FILE====":
            continue
        data.append((row + [""] * len(headers))[: len(headers)])
    return pd.DataFrame(data, columns=headers)


def _detect_from_columns(columns: Iterable[str]) -> str | None:
    cols = set(map(str, columns))
    if {"Gateway", "Grand Total", "Created At", "Transaction ID"}.issubset(cols):
        return BACKEND
    if {"merchantOrderId", "transactionId", "pspName", "processing_date"}.issubset(cols):
        return BRIDGER
    if {"Payment Public ID", "Merchant Order ID", "MID", "Gateway ID"}.issubset(cols):
        return PAYPROCC
    if {"merchant_transaction_id", "accepted_at", "transaction_type", "payment_method"}.issubset(cols):
        return ZEN
    if {"CreatedAt", "MerchantAmount", "MerchantCurrency", "Reference"}.issubset(cols):
        return CONFIRMO
    if {"Transaction status", "TransactionId", "Transaction Amount", "Transaction date"}.issubset(cols):
        return BINANCE
    if {"Tracking ID", "Operation ID", "Target amount", "Created"}.issubset(cols):
        return COINSBUY
    if {"TransactionType", "InvoiceNumber", "TransactionDateUtc", "ChangeStatus"}.issubset(cols):
        return TCPAY
    if {"Tx-Id", "Tx-Type", "Custom 3", "Order No."}.issubset(cols):
        return PAYABL
    if {"Request", "Reference", "Order Reference", "Timestamp (GMT)"}.issubset(cols):
        return TRUST
    if {"Payment ID", "Order type", "Payment date", "CUR"}.issubset(cols):
        return UNLIMIT
    if {"Transaction ID", "Balance Impact", "Gross", "Fee", "Net"}.issubset(cols):
        return PAYPAL
    if {"Client Unique ID", "Transaction Result", "Custom Data", "Transaction Type"}.issubset(cols):
        return "Nuvei"
    if {"Merchant Transaction ID", "Transaction Time (GMT)", "Payment Method", "Account ID"}.issubset(cols):
        return PAYSAFE
    if {"Trans. Curr", "Acct. Amount Gross", "Card No.", "Merch Tran Ref."}.issubset(cols):
        return PAYSTRA
    if {"Creation date", "Validated date", "Invoice", "Status code", "Balance currency"}.issubset(cols):
        return DLOCAL
    if {"Time (CET)", "[-]", "[+]", "ID of the corresponding Skrill transaction"}.issubset(cols):
        return SKRILL
    return None


def _nuvei_region(filename: str, metadata: pd.DataFrame | None) -> str:
    lower_name = filename.casefold()
    if "176688" in lower_name:
        return NUVEI_AU
    if "196908" in lower_name:
        return NUVEI_EU
    if metadata is not None and not metadata.empty:
        joined = " ".join(metadata.fillna("").astype(str).to_numpy().ravel()).casefold()
        if "growthnext f.z.c" in joined:
            return NUVEI_AU
        if "incenteco" in joined:
            return NUVEI_EU
    raise ReconciliationError(
        f"{filename}: Nuvei schema detected, but EU/AU could not be identified. "
        "Keep the standard account number in the filename or report header."
    )


def detect_and_read(filename: str, blob: bytes) -> tuple[str, pd.DataFrame]:
    suffix = filename.rsplit(".", 1)[-1].casefold() if "." in filename else ""
    if suffix == "csv":
        first_line = blob[:5000].decode("utf-8-sig", errors="ignore")
        if "Acct. Amount Gross" in first_line and "Merch Tran Ref." in first_line:
            frame = _read_paystra_csv(blob)
            return PAYSTRA, frame
        frame = _read_csv_flexible(blob)
        source = _detect_from_columns(frame.columns)
        if source is None:
            skipped = _read_csv_flexible(blob, skiprows=1)
            skipped_source = _detect_from_columns(skipped.columns)
            if skipped_source is not None:
                return skipped_source, skipped
        if source is None:
            raise ReconciliationError(f"{filename}: file schema is not supported.")
        return source, frame

    if suffix in {"xlsx", "xls"}:
        excel = pd.ExcelFile(io.BytesIO(blob))
        metadata = None
        candidates: list[tuple[str, dict]] = []
        for sheet in excel.sheet_names:
            candidates.append((sheet, {}))
            candidates.append((sheet, {"skiprows": 1}))
            candidates.append((sheet, {"header": 12, "dtype": str}))
        for sheet, options in candidates:
            try:
                frame = pd.read_excel(excel, sheet_name=sheet, **options)
            except Exception:
                continue
            source = _detect_from_columns(frame.columns)
            if source == "Nuvei":
                try:
                    metadata = pd.read_excel(excel, sheet_name=sheet, header=None, nrows=12, dtype=str)
                except Exception:
                    metadata = None
                return _nuvei_region(filename, metadata), frame
            if source is not None:
                return source, frame
        raise ReconciliationError(f"{filename}: Excel schema is not supported.")

    raise ReconciliationError(f"{filename}: only CSV, XLS, XLSX, and ZIP files are supported.")


def expand_file_tuples(files: Iterable[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
    expanded: list[tuple[str, bytes]] = []
    for filename, blob in files:
        if filename.casefold().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                for member in archive.infolist():
                    if member.is_dir() or member.filename.startswith("__MACOSX/"):
                        continue
                    if member.filename.casefold().endswith((".csv", ".xlsx", ".xls")):
                        expanded.append((member.filename.rsplit("/", 1)[-1], archive.read(member)))
        else:
            expanded.append((filename, blob))
    return expanded


def load_sources(files: Iterable[tuple[str, bytes]]) -> tuple[dict[str, LoadedSource], pd.DataFrame]:
    grouped: dict[str, list[pd.DataFrame]] = defaultdict(list)
    names: dict[str, list[str]] = defaultdict(list)
    inventory = []
    for filename, blob in expand_file_tuples(files):
        source, frame = detect_and_read(filename, blob)
        grouped[source].append(frame)
        names[source].append(filename)
        inventory.append({"Source": source, "File": filename, "Rows": len(frame), "Status": "Ready"})
    loaded = {
        source: LoadedSource(source, pd.concat(frames, ignore_index=True, sort=False), names[source])
        for source, frames in grouped.items()
    }
    return loaded, pd.DataFrame(inventory)


def _standard_frame(frame: pd.DataFrame, source_file: str) -> pd.DataFrame:
    result = frame.copy()
    result["source_file"] = source_file
    for column in [
        "provider", "key", "secondary_id", "merchant_id", "time_gmt3", "amount", "currency",
        "source_amount", "source_currency", "channel", "email", "entity", "record_id", "details",
    ]:
        if column not in result.columns:
            result[column] = "" if column not in {"amount", "source_amount", "time_gmt3"} else np.nan
    return result[
        ["provider", "key", "secondary_id", "merchant_id", "time_gmt3", "amount", "currency",
         "source_amount", "source_currency", "channel", "email", "entity", "record_id", "details", "source_file"]
    ]


def normalize_backend(source: LoadedSource) -> pd.DataFrame:
    frame = source.frame.copy()
    _require(frame, ["Transaction ID", "Gateway", "Grand Total", "Created At", "Status"], BACKEND)
    frame = frame[frame["Status"].astype(str).str.casefold().eq("enabled")].copy()
    frame["provider"] = frame["Gateway"].replace({
        "Bridger Pay": BRIDGER, "Pay Procc": PAYPROCC, "Zen Pay": ZEN,
        "Binance": BINANCE, "Crypto": COINSBUY, "TC PAY": TCPAY,
    })
    frame["key"] = frame["Transaction ID"].map(clean_id)
    frame["time_gmt3"] = pd.to_datetime(frame["Created At"], errors="coerce")
    frame["amount"] = parse_number(frame["Grand Total"])
    frame["currency"] = "USD"
    frame["source_amount"] = frame["amount"]
    frame["source_currency"] = "USD"
    plan = frame.get("Plan Name", pd.Series("", index=frame.index)).fillna("").astype(str)
    plan_type = frame.get("Plan Type", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame["entity"] = np.where(plan.str.contains("future", case=False) | plan_type.str.contains("future", case=False), "Futures", "CFD")
    frame["channel"] = frame["provider"]
    frame["email"] = frame.get("Customer Email", "")
    frame["record_id"] = frame.get("Order ID", "").map(clean_id) if "Order ID" in frame else ""
    frame["details"] = plan
    return _standard_frame(frame, " | ".join(source.files)).loc[lambda x: x["key"].ne("") & x["amount"].notna()]


def normalize_backend_orchestrator(source: LoadedSource) -> pd.DataFrame:
    frame = source.frame.copy()
    name = source.source
    source_file = " | ".join(source.files)

    if name == BRIDGER:
        _require(frame, ["status", "merchantOrderId", "processing_date", "amount", "currency"], BRIDGER)
        mask = frame["status"].astype(str).str.casefold().eq("approved")
        if "type" in frame:
            mask &= frame["type"].astype(str).str.casefold().eq("payment")
        frame = frame[mask].copy()
        frame["provider"] = BRIDGER
        frame["key"] = frame["merchantOrderId"].map(clean_id)
        frame["secondary_id"] = frame.get("transactionId", "").map(clean_id)
        frame["time_gmt3"] = utc_to_gmt3(frame["processing_date"])
        frame["amount"] = parse_number(frame["amount"])
        frame["source_amount"] = frame["amount"]
        frame["source_currency"] = frame["currency"]
        frame["channel"] = frame.get("pspName", "").fillna("").astype(str) + " / " + frame.get("paymentMethod", "").fillna("").astype(str)
        frame["email"] = frame.get("email", "")
        frame["record_id"] = frame.get("id", "").map(clean_id)

    elif name == ZEN:
        _require(frame, ["transaction_type", "transaction_currency", "payment_method", "merchant_transaction_id", "accepted_at", "transaction_amount"], ZEN)
        frame = frame[
            frame["transaction_type"].astype(str).eq("Purchase")
            & frame["transaction_currency"].astype(str).str.upper().eq("USD")
        ].copy()
        frame["provider"] = ZEN
        frame["key"] = frame["merchant_transaction_id"].map(clean_id)
        frame["time_gmt3"] = utc_to_gmt3(frame["accepted_at"])
        frame["amount"] = parse_number(frame["transaction_amount"])
        frame["currency"] = "USD"
        frame["source_amount"] = frame["amount"]
        frame["source_currency"] = frame["transaction_currency"]
        frame["channel"] = frame["payment_method"].fillna("").astype(str) + " / " + frame.get("payment_channel", "").fillna("").astype(str)
        frame["record_id"] = frame.get("transaction_id", "").map(clean_id)

    elif name == CONFIRMO:
        _require(frame, ["Status", "MerchantCurrency", "ID", "CreatedAt", "MerchantAmount"], CONFIRMO)
        frame = frame[frame["Status"].astype(str).str.upper().eq("PAID") & frame["MerchantCurrency"].astype(str).str.upper().eq("USD")].copy()
        frame["provider"] = CONFIRMO
        frame["key"] = frame["ID"].map(clean_id)
        parsed = pd.to_datetime(frame["CreatedAt"], format="%d.%m.%Y %H:%M:%S", utc=True, errors="coerce")
        frame["time_gmt3"] = parsed.dt.tz_convert("Etc/GMT-3").dt.tz_localize(None)
        frame["amount"] = parse_number(frame["MerchantAmount"])
        frame["currency"] = "USD"
        frame["source_amount"] = frame["amount"]
        frame["source_currency"] = frame["MerchantCurrency"]
        frame["channel"] = frame.get("CustomerCurrency", "")
        frame["record_id"] = frame.get("Reference", "").map(clean_id)

    elif name == BINANCE:
        _require(frame, ["Transaction status", "TransactionId", "Transaction date", "Transaction Amount", "Transaction Currency"], BINANCE)
        frame = frame[frame["Transaction status"].astype(str).str.upper().eq("PAID")].copy()
        frame["provider"] = BINANCE
        frame["key"] = frame["TransactionId"].map(clean_id)
        frame["time_gmt3"] = utc_to_gmt3(frame["Transaction date"])
        frame["amount"] = parse_number(frame["Transaction Amount"])
        frame["currency"] = "USD"
        frame["source_amount"] = frame["amount"]
        frame["source_currency"] = frame["Transaction Currency"]
        frame["channel"] = BINANCE
        description = frame.get("Product Description", pd.Series("", index=frame.index)).fillna("").astype(str)
        frame["entity"] = np.where(description.str.contains("future", case=False), "Futures", "CFD")
        frame["record_id"] = frame.get("Merchant Reference", "").map(clean_id)

    elif name == PAYPROCC:
        _require(frame, ["Status", "Type", "Merchant Order ID", "Transaction Date", "Amount", "Currency"], PAYPROCC)
        frame = frame[frame["Status"].astype(str).str.casefold().eq("success") & frame["Type"].astype(str).str.casefold().eq("sale")].copy()
        frame["provider"] = PAYPROCC
        frame["key"] = frame["Merchant Order ID"].map(clean_id)
        frame["secondary_id"] = frame.get("Payment Public ID", "").map(clean_id)
        frame["time_gmt3"] = pd.to_datetime(frame["Transaction Date"], format="%Y-%m-%d %I:%M:%S %p", errors="coerce")
        raw_amount = parse_number(frame["Amount"])
        applied = parse_number(frame.get("Applied Amount", pd.Series(np.nan, index=frame.index)))
        applied_currency = frame.get("Applied Currency", pd.Series("", index=frame.index)).astype(str).str.upper()
        frame["amount"] = np.where(applied_currency.eq("USD") & applied.notna(), applied, raw_amount)
        frame["currency"] = "USD"
        frame["source_amount"] = raw_amount
        frame["source_currency"] = frame["Currency"]
        frame["channel"] = frame.get("Brand", "").fillna("").astype(str) + " / " + frame.get("Payment Method", "").fillna("").astype(str)
        frame["email"] = frame.get("Payer Email", "")
        description = frame.get("Description", pd.Series("", index=frame.index)).fillna("").astype(str)
        frame["entity"] = np.where(description.str.contains("future", case=False), "Futures", "CFD")
        frame["record_id"] = frame.get("Transaction ID", "").map(clean_id)
        frame["details"] = description

    elif name == TCPAY:
        _require(frame, ["TransactionType", "ChangeStatus", "CurrencyCode", "InvoiceNumber", "TransactionDateUtc", "Amount"], TCPAY)
        frame = frame[
            pd.to_numeric(frame["TransactionType"], errors="coerce").eq(14)
            & frame["ChangeStatus"].astype(str).str.casefold().eq("increase")
            & frame["CurrencyCode"].astype(str).str.upper().eq("USD")
            & frame["InvoiceNumber"].notna()
        ].copy()
        frame["provider"] = TCPAY
        frame["key"] = frame["InvoiceNumber"].map(clean_id)
        frame["time_gmt3"] = utc_to_gmt3(frame["TransactionDateUtc"])
        frame["amount"] = parse_number(frame["Amount"])
        frame["currency"] = "USD"
        frame["source_amount"] = frame["amount"]
        frame["source_currency"] = frame["CurrencyCode"]
        frame["channel"] = TCPAY
        frame["details"] = frame.get("Description", "")

    elif name == COINSBUY:
        _require(frame, ["Status", "Type", "Tracking ID", "Operation ID", "Created", "Target amount", "Currency", "Amount"], COINSBUY)
        frame = frame[
            frame["Status"].astype(str).str.casefold().eq("confirmed")
            & frame["Type"].astype(str).str.casefold().eq("deposit")
            & frame["Tracking ID"].fillna("").astype(str).str.startswith("B2B_")
        ].copy()
        frame["provider"] = COINSBUY
        frame["key"] = frame["Operation ID"].fillna("").astype(str).str.extract(r"(\d+)", expand=False).fillna("")
        frame["time_gmt3"] = utc_to_gmt3(frame["Created"])
        frame["amount"] = parse_number(frame["Target amount"])
        frame["source_amount"] = parse_number(frame["Amount"])
        frame["source_currency"] = frame["Currency"]
        frame["currency"] = "USD"
        frame["channel"] = frame["Currency"]
        frame["record_id"] = frame["Tracking ID"].map(clean_id)
    else:
        raise ReconciliationError(f"{name} is not an Orchestrator → Backend source.")

    standard = _standard_frame(frame, source_file)
    return standard.loc[standard["key"].ne("") & standard["amount"].notna()].copy()


def _last_four(series: pd.Series) -> pd.Series:
    return series.map(clean_id).str.replace(r"\D", "", regex=True).str[-4:].str.zfill(4)


def normalize_psp_for_bridger(source: LoadedSource) -> pd.DataFrame:
    frame = source.frame.copy()
    name = source.source
    source_file = " | ".join(source.files)

    if name == PAYABL:
        _require(frame, ["Tx-Type", "Status", "Error-Code", "Currency", "Tx-Id", "Order No.", "Custom 3", "Date", "Time", "Amount"], PAYABL)
        frame = frame[
            frame["Tx-Type"].astype(str).str.casefold().eq("capture")
            & frame["Status"].astype(str).str.casefold().eq("successful")
            & pd.to_numeric(frame["Error-Code"], errors="coerce").eq(0)
            & frame["Currency"].astype(str).str.upper().eq("USD")
        ].copy()
        frame["key"] = frame["Tx-Id"].map(clean_id)
        frame["secondary_id"] = frame["Order No."].map(clean_id)
        frame["merchant_id"] = frame["Custom 3"].map(clean_id)
        # Payabl's Date/Time fields follow Central European local time. Use an
        # IANA timezone so CET/CEST daylight-saving changes are handled before
        # converting to the report timezone (GMT+3).
        local = pd.to_datetime(
            frame["Date"].astype(str) + " " + frame["Time"].astype(str),
            format="%d.%m.%Y %H:%M:%S",
            errors="coerce",
        )
        frame["time_gmt3"] = (
            local.dt.tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="shift_forward")
            .dt.tz_convert("Etc/GMT-3")
            .dt.tz_localize(None)
        )
        frame["amount"] = parse_number(frame["Amount"])
        frame["email"] = frame.get("EMail", "")
        frame["channel"] = frame.get("Credit Card Brand", "")
        frame["record_id"] = frame["Tx-Id"].map(clean_id)

    elif name == TRUST:
        _require(frame, ["Request", "Error Code", "Settle Status", "Currency", "Reference", "Order Reference", "Timestamp (GMT)", "Settle Amount"], TRUST)
        frame = frame[
            frame["Request"].astype(str).str.upper().eq("AUTH")
            & pd.to_numeric(frame["Error Code"], errors="coerce").eq(0)
            & pd.to_numeric(frame["Settle Status"], errors="coerce").isin([0, 100])
            & frame["Currency"].astype(str).str.upper().eq("USD")
        ].copy()
        frame["key"] = frame["Reference"].map(clean_id)
        frame["secondary_id"] = frame["Reference"].map(clean_id)
        frame["merchant_id"] = frame["Order Reference"].map(clean_id)
        frame["time_gmt3"] = pd.to_datetime(frame["Timestamp (GMT)"], errors="coerce") + pd.Timedelta(hours=3)
        frame["amount"] = parse_number(frame["Settle Amount"])
        frame["channel"] = frame.get("Payment type", "")
        frame["details"] = frame.get("Billing Name", "")

    elif name == UNLIMIT:
        _require(frame, ["Status", "Order type", "CUR", "Payment ID", "Payment date", "Amount"], UNLIMIT)
        frame = frame[
            frame["Status"].astype(str).str.casefold().eq("captured")
            & frame["Order type"].astype(str).str.casefold().eq("payment")
            & frame["CUR"].astype(str).str.upper().eq("USD")
        ].copy()
        frame["key"] = frame["Payment ID"].map(clean_id)
        frame["secondary_id"] = _last_four(frame.get("Card number", pd.Series("", index=frame.index)))
        frame["time_gmt3"] = pd.to_datetime(frame["Payment date"], errors="coerce") - pd.Timedelta(hours=3)
        frame["amount"] = parse_number(frame["Amount"])
        frame["email"] = frame.get("Email", "")
        frame["channel"] = "Card"

    elif name == PAYPAL:
        _require(frame, ["Type", "Status", "Currency", "Balance Impact", "Transaction ID", "Date", "Time", "Gross"], PAYPAL)
        frame = frame[
            frame["Type"].astype(str).str.casefold().eq("express checkout payment")
            & frame["Status"].astype(str).str.casefold().eq("completed")
            & frame["Currency"].astype(str).str.upper().eq("USD")
            & frame["Balance Impact"].astype(str).str.casefold().eq("credit")
        ].copy()
        frame["key"] = frame["Transaction ID"].map(clean_id)
        local = pd.to_datetime(frame["Date"].astype(str) + " " + frame["Time"].astype(str), format="%d/%m/%Y %H:%M:%S", errors="coerce")
        frame["time_gmt3"] = local.dt.tz_localize("America/Los_Angeles", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("Etc/GMT-3").dt.tz_localize(None)
        frame["amount"] = parse_number(frame["Gross"])
        frame["source_amount"] = parse_number(frame.get("Net", pd.Series(np.nan, index=frame.index)))
        frame["channel"] = PAYPAL
        frame["email"] = frame.get("From Email Address", frame.get("Email Address", ""))

    elif name in {NUVEI_EU, NUVEI_AU}:
        _require(frame, ["Transaction Type", "Transaction Result", "Currency", "Client Unique ID", "Transaction ID", "Custom Data", "Date", "Amount"], name)
        frame = frame[
            frame["Transaction Type"].astype(str).str.casefold().eq("sale")
            & frame["Transaction Result"].astype(str).str.casefold().eq("approved")
            & frame["Currency"].astype(str).str.upper().eq("USD")
        ].copy()
        frame["key"] = frame["Client Unique ID"].map(clean_id)
        frame["secondary_id"] = frame["Transaction ID"].map(clean_id)
        frame["merchant_id"] = frame["Custom Data"].map(clean_id)
        frame["time_gmt3"] = pd.to_datetime(frame["Date"], errors="coerce") - pd.Timedelta(hours=3)
        frame["amount"] = parse_number(frame["Amount"])
        frame["channel"] = frame.get("Payment Method", "")
        frame["email"] = frame.get("Email Address", "")

    elif name == PAYSAFE:
        _require(frame, ["Transaction Type", "Status", "Payment Method", "Currency", "Merchant Transaction ID", "Transaction ID", "Transaction Date", "Transaction Time (GMT)", "Amount"], PAYSAFE)
        frame = frame[
            frame["Transaction Type"].astype(str).str.casefold().eq("authorization")
            & frame["Status"].astype(str).str.casefold().eq("completed")
            & frame["Payment Method"].astype(str).str.casefold().eq("card")
            & frame["Currency"].astype(str).str.upper().eq("USD")
            & frame["Merchant Transaction ID"].astype(str).str.startswith("BP_", na=False)
        ].copy()
        frame["key"] = frame["Transaction ID"].map(clean_id)
        frame["merchant_id"] = frame["Merchant Transaction ID"].map(clean_id)
        frame["time_gmt3"] = pd.to_datetime(frame["Transaction Date"].astype(str) + " " + frame["Transaction Time (GMT)"].astype(str), format="%m-%d-%Y %I:%M:%S %p", errors="coerce") + pd.Timedelta(hours=3)
        frame["amount"] = parse_number(frame["Amount"])
        frame["channel"] = "Card"

    elif name == PAYSTRA:
        _require(frame, ["Type", "Status", "Trans. Curr", "Transaction Date", "Trans Amount", "Card No."], PAYSTRA)
        frame = frame[
            frame["Type"].astype(str).eq("Purchase")
            & frame["Status"].astype(str).eq("Cleared")
            & frame["Trans. Curr"].astype(str).str.upper().eq("USD")
        ].copy()
        frame["amount"] = parse_number(frame["Trans Amount"])
        frame["secondary_id"] = _last_four(frame["Card No."])
        frame["date_utc"] = pd.to_datetime(frame["Transaction Date"], format="%Y%m%d", errors="coerce")
        frame["time_gmt3"] = frame["date_utc"] + pd.Timedelta(hours=3)
        frame["base_key"] = frame["secondary_id"] + "|" + frame["amount"].map(lambda value: f"{value:.2f}" if pd.notna(value) else "")
        frame = frame.sort_values(["date_utc", "base_key"]).copy()
        frame["occurrence"] = frame.groupby("base_key").cumcount() + 1
        frame["key"] = frame["base_key"] + "|" + frame["occurrence"].astype(str)
        frame["merchant_id"] = frame.get("Merch Tran Ref.", "")
        frame["channel"] = "Card"
        frame["details"] = frame.get("Acquirer Ref.", "")
    else:
        raise ReconciliationError(f"{name} is not supported for PSP → BridgerPay.")

    frame["provider"] = name
    if "currency" not in frame:
        frame["currency"] = "USD"
    frame["source_currency"] = frame["currency"]
    if "source_amount" not in frame:
        frame["source_amount"] = frame["amount"]
    standard = _standard_frame(frame, source_file)
    return standard.loc[standard["key"].ne("") & standard["amount"].notna()].copy()


def normalize_bridger_for_provider(bridger: LoadedSource, provider: str) -> pd.DataFrame:
    frame = bridger.frame.copy()
    _require(frame, ["pspName", "status", "currency", "processing_date", "amount", "transactionId", "pspOrderId", "merchantOrderId"], BRIDGER)
    psp_names = {
        # Payabl card, Apple Pay, and Google Pay are reported under separate
        # BridgerPay PSP labels but reconcile to the same Payabl export.
        PAYABL: {"innatech", "applepayviapayabl", "googlepayviapayabl"},
        TRUST: {"trustpayments"}, UNLIMIT: {"cardpay"}, PAYPAL: {"paypal"},
        NUVEI_EU: {"safecharge"}, NUVEI_AU: {"safecharge"}, PAYSAFE: {"paysafe"}, PAYSTRA: {"paystra"},
    }[provider]
    mask = (
        frame["pspName"].astype(str).str.casefold().isin(psp_names)
        & frame["status"].astype(str).str.casefold().eq("approved")
        & frame["currency"].astype(str).str.upper().eq("USD")
    )
    mid_alias = {
        NUVEI_EU: "SafeCharge-CreditCard-MID-EU",
        NUVEI_AU: "SafeCharge-CreditCard-MID-AE",
        PAYSTRA: "Paystra-CreditCard-MID-1",
    }.get(provider)
    if mid_alias and "midAlias" in frame:
        mask &= frame["midAlias"].astype(str).eq(mid_alias)
    frame = frame[mask].copy()
    frame["provider"] = provider
    frame["time_gmt3"] = utc_to_gmt3(frame["processing_date"])
    frame["amount"] = parse_number(frame["amount"])
    frame["currency"] = "USD"
    frame["source_currency"] = frame["currency"]
    frame["source_amount"] = frame["amount"]
    frame["email"] = frame.get("email", "")
    frame["channel"] = frame.get("midAlias", "")
    frame["record_id"] = frame.get("id", "").map(clean_id)

    if provider == UNLIMIT:
        frame["key"] = frame["pspOrderId"].map(clean_id)
        frame["secondary_id"] = _last_four(frame.get("lastFourDigits", pd.Series("", index=frame.index)))
        frame["merchant_id"] = frame["merchantOrderId"].map(clean_id)
    elif provider == PAYSTRA:
        utc_date = pd.to_datetime(frame["processing_date"], utc=True, errors="coerce").dt.strftime("%Y%m%d")
        last4 = _last_four(frame.get("lastFourDigits", pd.Series("", index=frame.index)))
        frame["secondary_id"] = last4
        frame["base_key"] = last4 + "|" + frame["amount"].map(lambda value: f"{value:.2f}" if pd.notna(value) else "")
        frame = frame.sort_values(["time_gmt3", "base_key"]).copy()
        frame["occurrence"] = frame.groupby("base_key").cumcount() + 1
        frame["key"] = frame["base_key"] + "|" + frame["occurrence"].astype(str)
        frame["merchant_id"] = frame["merchantOrderId"].map(clean_id)
    else:
        frame["key"] = frame["transactionId"].map(clean_id)
        frame["secondary_id"] = frame["pspOrderId"].map(clean_id)
        frame["merchant_id"] = frame["merchantOrderId"].map(clean_id)

    return _standard_frame(frame, " | ".join(bridger.files)).loc[lambda x: x["key"].ne("") & x["amount"].notna()]


def normalize_psp_for_payprocc(source: LoadedSource) -> pd.DataFrame:
    frame = source.frame.copy()
    name = source.source
    source_file = " | ".join(source.files)

    if name == DLOCAL:
        _require(frame, ["Status", "Status code", "Transaction type", "Balance currency", "Validated date", "Reference", "Invoice", "Amount.1"], DLOCAL)
        frame = frame[
            frame["Status"].astype(str).eq("PAID")
            & frame["Status code"].astype(str).eq("200")
            & frame["Transaction type"].astype(str).eq("PAYMENT")
            & frame["Balance currency"].astype(str).str.upper().eq("USD")
        ].copy()
        frame["key"] = frame["Reference"].map(clean_id)
        frame["secondary_id"] = frame["Invoice"].map(clean_id)
        frame["time_gmt3"] = utc_to_gmt3(frame["Validated date"])
        frame["amount"] = parse_number(frame["Amount.1"])
        frame["currency"] = "USD"
        frame["source_currency"] = "USD"
        frame["source_amount"] = frame["amount"]
        frame["email"] = frame.get("Email", "")
        frame["channel"] = frame.get("Payment method name", "")
        frame["details"] = frame.get("Description", "")

    elif name == SKRILL:
        _require(frame, ["Time (CET)", "Type", "Status", "Reference", "[+]", "Currency Sent", "ID of the corresponding Skrill transaction"], SKRILL)
        credits = parse_number(frame["[+]"])
        frame = frame[
            frame["Reference"].astype(str).str.strip().ne("")
            & credits.notna()
            & frame["Status"].astype(str).str.casefold().eq("processed")
            & frame["Type"].astype(str).eq("Receive Money")
            & frame["Currency Sent"].astype(str).str.upper().eq("USD")
        ].copy()
        frame["key"] = frame["Reference"].map(clean_id)
        frame["secondary_id"] = frame["Reference"].map(clean_id)
        local = pd.to_datetime(frame["Time (CET)"], format="%d %b %y %H:%M", errors="coerce")
        frame["time_gmt3"] = local.dt.tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("Etc/GMT-3").dt.tz_localize(None)
        frame["amount"] = parse_number(frame["[+]"])
        frame["currency"] = "USD"
        frame["source_currency"] = "USD"
        frame["source_amount"] = frame["amount"]
        frame["email"] = frame.get("Transaction Details", "").astype(str).str.replace(r"^from\s+", "", regex=True)
        frame["channel"] = "Skrill Wallet"
        frame["record_id"] = frame.get("ID", "").map(clean_id)
        frame["details"] = frame.get("ID of the corresponding Skrill transaction", "").map(clean_id)

    elif name == PAYSAFE:
        _require(frame, ["Transaction Type", "Status", "Payment Method", "Merchant Transaction ID", "Transaction ID", "Transaction Date", "Transaction Time (GMT)", "Amount", "Currency"], PAYSAFE)
        frame = frame[
            frame["Transaction Type"].astype(str).str.casefold().eq("authorization")
            & frame["Status"].astype(str).str.casefold().eq("completed")
            & frame["Payment Method"].astype(str).str.casefold().eq("card")
            & ~frame["Merchant Transaction ID"].astype(str).str.startswith("BP_", na=False)
        ].copy()
        frame["key"] = frame["Transaction ID"].map(clean_id)
        frame["secondary_id"] = frame["Merchant Transaction ID"].map(clean_id)
        frame["time_gmt3"] = pd.to_datetime(frame["Transaction Date"].astype(str) + " " + frame["Transaction Time (GMT)"].astype(str), format="%m-%d-%Y %I:%M:%S %p", errors="coerce") + pd.Timedelta(hours=3)
        frame["amount"] = parse_number(frame["Amount"])
        frame["currency"] = frame["Currency"].astype(str).str.upper()
        frame["source_amount"] = frame["amount"]
        frame["source_currency"] = frame["Currency"]
        frame["channel"] = "Paysafe Local Card"
        frame["details"] = frame.get("Account ID", "")
    else:
        raise ReconciliationError(f"{name} is not supported for PSP → PayProcc.")

    frame["provider"] = name
    standard = _standard_frame(frame, source_file)
    return standard.loc[standard["key"].ne("") & standard["amount"].notna()].copy()


def normalize_payprocc_for_provider(payprocc: LoadedSource, provider: str) -> pd.DataFrame:
    frame = payprocc.frame.copy()
    _require(frame, ["Payment Public ID", "Gateway ID", "MID", "Brand", "Type", "Status", "Currency", "Amount", "Transaction Date"], PAYPROCC)
    base = frame["Type"].astype(str).str.casefold().eq("sale") & frame["Status"].astype(str).str.casefold().eq("success")
    if provider == DLOCAL:
        mask = base & frame["MID"].astype(str).eq("GL-DL-FUNDEDNEXT-01") & frame["Brand"].astype(str).str.casefold().eq("dl") & frame["Currency"].astype(str).str.upper().eq("USD")
    elif provider == SKRILL:
        mask = base & frame["MID"].astype(str).eq("GL-SKRILL-FUNDEDNEXT-01") & frame["Brand"].astype(str).str.casefold().eq("skrill") & frame["Currency"].astype(str).str.upper().eq("USD")
    elif provider == PAYSAFE:
        mask = base & frame["MID"].astype(str).str.endswith("-PS-FUNDEDNEXT-01", na=False)
    else:
        raise ReconciliationError(f"{provider} is not supported for PSP → PayProcc.")
    frame = frame[mask].copy()
    frame["provider"] = provider
    frame["key"] = frame["Gateway ID"].map(clean_id)
    frame["secondary_id"] = frame["Payment Public ID"].map(clean_id)
    frame["merchant_id"] = frame.get("Merchant Order ID", "").map(clean_id)
    frame["time_gmt3"] = pd.to_datetime(frame["Transaction Date"], format="%Y-%m-%d %I:%M:%S %p", errors="coerce")
    frame["amount"] = parse_number(frame["Amount"])
    frame["source_amount"] = frame["amount"]
    frame["source_currency"] = frame["Currency"]
    frame["currency"] = frame["Currency"].astype(str).str.upper()
    frame["channel"] = frame["MID"]
    frame["email"] = frame.get("Payer Email", "")
    frame["record_id"] = frame.get("Transaction ID", "").map(clean_id)
    frame["details"] = frame.get("Description", "")
    return _standard_frame(frame, " | ".join(payprocc.files)).loc[lambda x: x["key"].ne("") & x["amount"].notna()]


def _join_unique(series: pd.Series, limit: int = 8) -> str:
    values = []
    for value in series:
        text = clean_text(value)
        if text and text not in values:
            values.append(text)
        if len(values) >= limit:
            break
    return " | ".join(values)


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[
            "key", "time_gmt3", "last_time_gmt3", "amount", "currency", "secondary_id", "merchant_id",
            "rows", "channel", "email", "entity", "record_id", "details", "source_file",
        ])
    rows = []
    for key, group in frame.groupby("key", sort=False):
        currencies = [value for value in group["currency"].dropna().astype(str).unique() if value]
        entities = [value for value in group["entity"].dropna().astype(str).unique() if value]
        rows.append({
            "key": key,
            "time_gmt3": group["time_gmt3"].min(),
            "last_time_gmt3": group["time_gmt3"].max(),
            "amount": round(float(group["amount"].sum()), 6),
            "currency": currencies[0] if len(currencies) == 1 else ("Mixed" if currencies else ""),
            "secondary_id": _join_unique(group["secondary_id"]),
            "merchant_id": _join_unique(group["merchant_id"]),
            "rows": int(len(group)),
            "channel": _join_unique(group["channel"]),
            "email": _join_unique(group["email"]),
            "entity": entities[0] if len(entities) == 1 else ("Mixed" if entities else "Unclassified"),
            "record_id": _join_unique(group["record_id"]),
            "details": _join_unique(group["details"]),
            "source_file": _join_unique(group["source_file"]),
        })
    return pd.DataFrame(rows)


def _is_in_period(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return series.ge(start) & series.lt(end)


def _align_paystra_keys(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = left.copy().reset_index(drop=True)
    right = right.copy().reset_index(drop=True)
    left["_base"] = left["secondary_id"] + "|" + left["amount"].map(lambda value: f"{value:.2f}")
    right["_base"] = right["secondary_id"] + "|" + right["amount"].map(lambda value: f"{value:.2f}")
    left["_utc_date"] = (left["time_gmt3"] - pd.Timedelta(hours=3)).dt.strftime("%Y%m%d")
    right["_utc_date"] = (right["time_gmt3"] - pd.Timedelta(hours=3)).dt.strftime("%Y%m%d")
    left["_aligned_key"] = ""
    right["_aligned_key"] = ""
    used_right: set[int] = set()
    pair_number = 0
    for left_index, left_row in left.sort_values(["_utc_date", "_base"]).iterrows():
        candidates = right[(right["_base"].eq(left_row["_base"])) & ~right.index.isin(used_right)]
        same_date = candidates[candidates["_utc_date"].eq(left_row["_utc_date"])]
        if not same_date.empty:
            candidates = same_date
        candidates = candidates.sort_values("time_gmt3")
        if candidates.empty:
            continue
        right_index = int(candidates.index[0])
        used_right.add(right_index)
        pair_number += 1
        aligned = f"PAYSTRA-{pair_number:06d}"
        left.at[left_index, "_aligned_key"] = aligned
        right.at[right_index, "_aligned_key"] = aligned
        left.at[left_index, "time_gmt3"] = right.at[right_index, "time_gmt3"]
    for index in left.index[left["_aligned_key"].eq("")]:
        left.at[index, "_aligned_key"] = f"PAYSTRA-PSP-ONLY-{index:06d}"
    for index in right.index[right["_aligned_key"].eq("")]:
        right.at[index, "_aligned_key"] = f"PAYSTRA-BRIDGER-ONLY-{index:06d}"
    left["key"] = left["_aligned_key"]
    right["key"] = right["_aligned_key"]
    return left.drop(columns=["_base", "_utc_date", "_aligned_key"]), right.drop(columns=["_base", "_utc_date", "_aligned_key"])


def _pair_reconcile(
    left: pd.DataFrame,
    right: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    pair_label: str,
    left_name: str,
    right_name: str,
    tolerance: float,
    coverage_gap: bool = False,
    post_classifier=None,
) -> dict:
    left = left[left["time_gmt3"].notna()].copy()
    right = right[right["time_gmt3"].notna()].copy()
    left_in = _aggregate(left[_is_in_period(left["time_gmt3"], start, end)])
    right_in = _aggregate(right[_is_in_period(right["time_gmt3"], start, end)])
    left_all = _aggregate(left)
    right_all = _aggregate(right)

    li = left_in.set_index("key", drop=False).to_dict("index")
    ri = right_in.set_index("key", drop=False).to_dict("index")
    la = left_all.set_index("key", drop=False).to_dict("index")
    ra = right_all.set_index("key", drop=False).to_dict("index")

    detail_rows = []
    for key in sorted(set(li) | set(ri)):
        l_in, r_in = li.get(key), ri.get(key)
        l_any, r_any = la.get(key), ra.get(key)
        if l_in is not None and r_in is not None:
            l_cmp, r_cmp, status = l_in, r_in, "Matched"
            left_window = right_window = "In Period"
        elif l_in is not None and r_any is not None:
            l_cmp, r_cmp, status = l_in, r_any, f"{right_name} Outside Period"
            left_window, right_window = "In Period", "Outside Period"
        elif r_in is not None and l_any is not None:
            l_cmp, r_cmp, status = l_any, r_in, f"{left_name} Outside Period"
            left_window, right_window = "Outside Period", "In Period"
        elif l_in is not None:
            l_cmp, r_cmp, status = l_in, None, f"{left_name} Only"
            left_window, right_window = "In Period", "Not Found"
        else:
            l_cmp, r_cmp, status = None, r_in, f"{right_name} Only"
            left_window, right_window = "Not Found", "In Period"

        left_amount = float(l_cmp["amount"]) if l_cmp is not None else None
        right_amount = float(r_cmp["amount"]) if r_cmp is not None else None
        difference = round((left_amount or 0) - (right_amount or 0), 6)
        left_currency = clean_text(l_cmp["currency"]) if l_cmp is not None else ""
        right_currency = clean_text(r_cmp["currency"]) if r_cmp is not None else ""
        currency_match = None if l_cmp is None or r_cmp is None else left_currency == right_currency
        secondary_match = None
        merchant_match = None
        if l_cmp is not None and r_cmp is not None:
            if clean_text(l_cmp["secondary_id"]) and clean_text(r_cmp["secondary_id"]):
                secondary_match = clean_text(l_cmp["secondary_id"]) == clean_text(r_cmp["secondary_id"])
            if clean_text(l_cmp["merchant_id"]) and clean_text(r_cmp["merchant_id"]):
                merchant_match = clean_text(l_cmp["merchant_id"]) == clean_text(r_cmp["merchant_id"])
        if status == "Matched":
            if currency_match is False:
                status = "Currency Mismatch"
            elif abs(difference) > tolerance + 1e-9:
                status = "Amount Mismatch"

        detail_rows.append({
            "Provider": pair_label,
            "Status": status,
            "Action Required": status in {"Amount Mismatch", "Currency Mismatch", "Secondary Key Mismatch", f"{left_name} Only", f"{right_name} Only"},
            "Reconciliation Key": key,
            f"{left_name} Window": left_window,
            f"{right_name} Window": right_window,
            f"{left_name} Time GMT+3": l_cmp["time_gmt3"] if l_cmp is not None else pd.NaT,
            f"{right_name} Time GMT+3": r_cmp["time_gmt3"] if r_cmp is not None else pd.NaT,
            f"{left_name} Amount": left_amount,
            f"{right_name} Amount": right_amount,
            "Amount Difference": difference,
            "Tolerance": tolerance,
            "Currency": left_currency or right_currency,
            "Currency Match": currency_match,
            "Secondary Key Match": secondary_match,
            "Merchant Key Match": merchant_match,
            f"{left_name} Secondary ID": l_cmp["secondary_id"] if l_cmp is not None else "",
            f"{right_name} Secondary ID": r_cmp["secondary_id"] if r_cmp is not None else "",
            f"{left_name} Merchant ID": l_cmp["merchant_id"] if l_cmp is not None else "",
            f"{right_name} Merchant ID": r_cmp["merchant_id"] if r_cmp is not None else "",
            f"{left_name} Rows": l_cmp["rows"] if l_cmp is not None else 0,
            f"{right_name} Rows": r_cmp["rows"] if r_cmp is not None else 0,
            "Entity": l_cmp["entity"] if l_cmp is not None and l_cmp["entity"] != "Unclassified" else (r_cmp["entity"] if r_cmp is not None else "Unclassified"),
            "Channel": l_cmp["channel"] if l_cmp is not None else (r_cmp["channel"] if r_cmp is not None else ""),
            "Email": l_cmp["email"] if l_cmp is not None and l_cmp["email"] else (r_cmp["email"] if r_cmp is not None else ""),
            f"{left_name} Record ID": l_cmp["record_id"] if l_cmp is not None else "",
            f"{right_name} Record ID": r_cmp["record_id"] if r_cmp is not None else "",
            "Details": l_cmp["details"] if l_cmp is not None and l_cmp["details"] else (r_cmp["details"] if r_cmp is not None else ""),
            f"{left_name} Source File": l_cmp["source_file"] if l_cmp is not None else "",
            f"{right_name} Source File": r_cmp["source_file"] if r_cmp is not None else "",
        })

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        detail = pd.DataFrame({
            "Provider": pd.Series(dtype="object"),
            "Status": pd.Series(dtype="object"),
            "Action Required": pd.Series(dtype="bool"),
            "Reconciliation Key": pd.Series(dtype="object"),
            f"{left_name} Window": pd.Series(dtype="object"),
            f"{right_name} Window": pd.Series(dtype="object"),
            f"{left_name} Time GMT+3": pd.Series(dtype="datetime64[ns]"),
            f"{right_name} Time GMT+3": pd.Series(dtype="datetime64[ns]"),
            f"{left_name} Amount": pd.Series(dtype="float64"),
            f"{right_name} Amount": pd.Series(dtype="float64"),
            "Amount Difference": pd.Series(dtype="float64"),
            "Tolerance": pd.Series(dtype="float64"),
            "Currency": pd.Series(dtype="object"),
            "Currency Match": pd.Series(dtype="object"),
            "Secondary Key Match": pd.Series(dtype="object"),
            "Merchant Key Match": pd.Series(dtype="object"),
            "Entity": pd.Series(dtype="object"),
        })
    elif coverage_gap:
        reconciled = detail[~detail["Status"].isin([f"{left_name} Only", f"{right_name} Only"])]
        right_time_col = f"{right_name} Time GMT+3"
        if not reconciled.empty and right_time_col in reconciled:
            first_match = reconciled[right_time_col].dropna().min()
            last_match = reconciled[right_time_col].dropna().max()
            mask = detail["Status"].eq(f"{right_name} Only") & (
                detail[right_time_col].lt(first_match) | detail[right_time_col].gt(last_match)
            )
            detail.loc[mask, "Status"] = f"{left_name} File Coverage Gap"
            detail.loc[mask, "Action Required"] = False

    if post_classifier is not None and not detail.empty:
        detail = post_classifier(detail)

    outside_token = "Outside Period"
    detail["Status Group"] = np.select(
        [
            detail["Status"].eq("Matched"),
            detail["Status"].str.contains("Mismatch", na=False),
            detail["Status"].str.contains("Coverage Gap", na=False),
            detail["Status"].str.contains(outside_token, na=False),
            detail["Status"].str.endswith(" Only", na=False),
        ],
        ["Matched", "Mismatch", "Coverage Gap", "Outside Period", "Missing"],
        default="Review",
    )
    left_time_col = f"{left_name} Time GMT+3"
    right_time_col = f"{right_name} Time GMT+3"
    detail["Reconciliation Date"] = detail[left_time_col].fillna(detail[right_time_col]).dt.date
    priority = {"Mismatch": 0, "Missing": 1, "Coverage Gap": 2, "Outside Period": 3, "Matched": 4, "Review": 5}
    detail["_priority"] = detail["Status Group"].map(priority).fillna(9)
    detail = detail.sort_values(["_priority", "Reconciliation Date", "Provider", "Reconciliation Key"]).drop(columns="_priority").reset_index(drop=True)

    exceptions = detail[detail["Status"].ne("Matched")].copy()
    left_amount_col = f"{left_name} Amount"
    right_amount_col = f"{right_name} Amount"
    summary = {
        "Provider": pair_label,
        f"{left_name} Transactions": len(left_in),
        f"{right_name} Transactions": len(right_in),
        "Matched": int(detail["Status"].eq("Matched").sum()),
        "Amount Mismatches": int(detail["Status"].eq("Amount Mismatch").sum()),
        "Currency Mismatches": int(detail["Status"].eq("Currency Mismatch").sum()),
        f"{left_name} Only": int(detail["Status"].eq(f"{left_name} Only").sum()),
        f"{right_name} Only": int(detail["Status"].eq(f"{right_name} Only").sum()),
        "Coverage Gaps": int(detail["Status"].str.contains("Coverage Gap", na=False).sum()),
        "Outside Period": int(detail["Status"].str.contains("Outside Period", na=False).sum()),
        "Action Exceptions": int(detail["Action Required"].sum()),
        f"{left_name} USD": round(float(detail.loc[(detail[f"{left_name} Window"] == "In Period") & detail["Currency"].eq("USD"), left_amount_col].fillna(0).sum()), 2),
        f"{right_name} USD": round(float(detail.loc[(detail[f"{right_name} Window"] == "In Period") & detail["Currency"].eq("USD"), right_amount_col].fillna(0).sum()), 2),
    }
    summary["USD Difference"] = round(summary[f"{left_name} USD"] - summary[f"{right_name} USD"], 2)

    currency_rows = []
    for currency in sorted(value for value in detail["Currency"].dropna().unique() if value):
        rows = detail[detail["Currency"].eq(currency)]
        currency_rows.append({
            "Provider": pair_label,
            "Currency": currency,
            f"{left_name} Amount": round(float(rows.loc[rows[f"{left_name} Window"] == "In Period", left_amount_col].fillna(0).sum()), 2),
            f"{right_name} Amount": round(float(rows.loc[rows[f"{right_name} Window"] == "In Period", right_amount_col].fillna(0).sum()), 2),
            "Difference": round(float(rows["Amount Difference"].sum()), 2),
            "Transactions": len(rows),
            "Matched": int(rows["Status"].eq("Matched").sum()),
            "Exceptions": int(rows["Action Required"].sum()),
        })

    return {
        "summary": summary,
        "detail": detail,
        "exceptions": exceptions,
        "currency": pd.DataFrame(currency_rows),
        "left_name": left_name,
        "right_name": right_name,
    }


def _combine_results(results: list[dict], workflow: str, start: pd.Timestamp, end: pd.Timestamp, inventory: pd.DataFrame) -> dict:
    if not results:
        raise ReconciliationError("No supported reconciliation routes were found in the uploaded files.")
    summary = pd.DataFrame([result["summary"] for result in results])
    detail = pd.concat([result["detail"] for result in results], ignore_index=True, sort=False)
    exceptions = pd.concat([result["exceptions"] for result in results], ignore_index=True, sort=False)
    currency_frames = [result["currency"] for result in results if not result["currency"].empty]
    currency = pd.concat(currency_frames, ignore_index=True, sort=False) if currency_frames else pd.DataFrame()
    left_name = results[0]["left_name"]
    right_name = results[0]["right_name"]

    daily = detail.groupby(["Reconciliation Date", "Status Group"], dropna=False).size().unstack(fill_value=0).reset_index()
    for column in ["Matched", "Mismatch", "Missing", "Coverage Gap", "Outside Period"]:
        if column not in daily:
            daily[column] = 0
    daily["Total"] = daily[["Matched", "Mismatch", "Missing", "Coverage Gap", "Outside Period"]].sum(axis=1)

    overall = {
        "Workflow": workflow,
        "Routes": len(summary),
        "Matched": int(detail["Status"].eq("Matched").sum()),
        "Action Exceptions": int(detail["Action Required"].sum()),
        "Coverage Gaps": int(detail["Status Group"].eq("Coverage Gap").sum()),
        "Outside Period": int(detail["Status Group"].eq("Outside Period").sum()),
        "Total Compared": len(detail),
        "Match Rate": float(detail["Status"].eq("Matched").mean()) if len(detail) else 0.0,
        f"{left_name} USD": round(float(summary[f"{left_name} USD"].sum()), 2),
        f"{right_name} USD": round(float(summary[f"{right_name} USD"].sum()), 2),
    }
    overall["Absolute USD Difference"] = round(abs(overall[f"{left_name} USD"] - overall[f"{right_name} USD"]), 2)

    controls = pd.DataFrame([
        {"Control": "Actionable exception count", "Expected": 0, "Actual": overall["Action Exceptions"], "Status": "PASS" if overall["Action Exceptions"] == 0 else "REVIEW"},
        {"Control": "Duplicate normalized reconciliation keys", "Expected": 0, "Actual": int(detail.duplicated(["Provider", "Reconciliation Key"]).sum()), "Status": "PASS" if not detail.duplicated(["Provider", "Reconciliation Key"]).any() else "REVIEW"},
        {"Control": "Rows with missing reconciliation dates", "Expected": 0, "Actual": int(pd.isna(detail["Reconciliation Date"]).sum()), "Status": "PASS" if not pd.isna(detail["Reconciliation Date"]).any() else "REVIEW"},
    ])

    return {
        "workflow": workflow,
        "period_start": start,
        "period_end": end,
        "report_timezone": REPORT_TIMEZONE,
        "left_name": left_name,
        "right_name": right_name,
        "overall": overall,
        "summary": summary,
        "daily": daily,
        "currency": currency,
        "exceptions": exceptions,
        "detail": detail,
        "controls": controls,
        "inventory": inventory,
    }


def reconcile_orchestrator_to_backend(sources: dict[str, LoadedSource], start: datetime, end: datetime, inventory: pd.DataFrame) -> dict:
    if BACKEND not in sources:
        raise ReconciliationError("Upload the Backend API Order List file.")
    backend = normalize_backend(sources[BACKEND])
    results = []
    source_to_gateway = {
        BRIDGER: BRIDGER, ZEN: ZEN, CONFIRMO: CONFIRMO, BINANCE: BINANCE,
        PAYPROCC: PAYPROCC, TCPAY: TCPAY, COINSBUY: COINSBUY,
    }
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    def infer_time(gateway: str, key: str, provider_record_id: str = ""):
        try:
            if gateway in {BRIDGER, PAYPROCC, ZEN}:
                parts = str(key).split("_", 2)
                if len(parts) > 1 and parts[1][:10].isdigit():
                    return pd.Timestamp(int(parts[1][:10]), unit="s", tz="UTC").tz_convert("Etc/GMT-3").tz_localize(None)
            if gateway == TCPAY and str(key)[:12].isdigit():
                return pd.to_datetime(str(key)[:12], format="%y%m%d%H%M%S", utc=True).tz_convert("Etc/GMT-3").tz_localize(None)
            if gateway == COINSBUY:
                match = re.search(r"B2B_(\d{14})_TRACKING", str(provider_record_id))
                if match:
                    return pd.to_datetime(match.group(1), format="%Y%m%d%H%M%S", utc=True).tz_convert("Etc/GMT-3").tz_localize(None)
        except Exception:
            return pd.NaT
        return pd.NaT

    for source_name, gateway in source_to_gateway.items():
        if source_name not in sources:
            continue
        right = normalize_backend_orchestrator(sources[source_name])
        left = backend[backend["provider"].eq(gateway)].copy()
        coverage_start = pd.NaT
        if source_name == TCPAY and "TransactionDateUtc" in sources[TCPAY].frame:
            raw_times = pd.to_datetime(sources[TCPAY].frame["TransactionDateUtc"], utc=True, errors="coerce", format="mixed")
            if raw_times.notna().any():
                coverage_start = raw_times.min().normalize().tz_convert("Etc/GMT-3").tz_localize(None)

        def classify_boundaries(detail, gateway=gateway, coverage_start=coverage_start):
            detail = detail.copy()
            backend_time = "Backend Time GMT+3"
            orchestrator_time = "Orchestrator Time GMT+3"
            provider_record = "Orchestrator Record ID"
            for index, row in detail.iterrows():
                if row["Status"] == "Backend Only":
                    inferred = infer_time(gateway, row["Reconciliation Key"])
                    if pd.notna(inferred):
                        detail.at[index, orchestrator_time] = inferred
                        if gateway == TCPAY and pd.notna(coverage_start) and inferred < coverage_start:
                            detail.at[index, "Status"] = "Orchestrator File Coverage Gap"
                            detail.at[index, "Action Required"] = False
                        elif inferred < start_ts or inferred >= end_ts:
                            detail.at[index, "Status"] = "Orchestrator Outside Period"
                            detail.at[index, "Action Required"] = False
                elif row["Status"] == "Orchestrator Only":
                    inferred = infer_time(gateway, row["Reconciliation Key"], row.get(provider_record, ""))
                    if pd.notna(inferred):
                        detail.at[index, backend_time] = inferred
                        if inferred < start_ts or inferred >= end_ts:
                            detail.at[index, "Status"] = "Backend Outside Period"
                            detail.at[index, "Action Required"] = False
            return detail
        results.append(_pair_reconcile(
            left, right, start_ts, end_ts, gateway,
            "Backend", "Orchestrator", BACKEND_TOLERANCES[gateway],
            coverage_gap=gateway in {TCPAY},
            post_classifier=classify_boundaries,
        ))
    return _combine_results(results, "Orchestrator → Backend API", start_ts, end_ts, inventory)


def reconcile_psp_to_bridger(sources: dict[str, LoadedSource], start: datetime, end: datetime, inventory: pd.DataFrame) -> dict:
    if BRIDGER not in sources:
        raise ReconciliationError("Upload the BridgerPay Transactions Report.")
    results = []
    for provider in [PAYABL, TRUST, UNLIMIT, PAYPAL, NUVEI_EU, NUVEI_AU, PAYSAFE, PAYSTRA]:
        if provider not in sources:
            continue
        left = normalize_psp_for_bridger(sources[provider])
        right = normalize_bridger_for_provider(sources[BRIDGER], provider)
        if provider == PAYSTRA:
            left, right = _align_paystra_keys(left, right)
        results.append(_pair_reconcile(
            left, right, pd.Timestamp(start), pd.Timestamp(end), provider,
            "PSP", "BridgerPay", 0.02,
            coverage_gap=provider in {PAYSAFE, PAYSTRA},
        ))
    return _combine_results(results, "PSP → BridgerPay", pd.Timestamp(start), pd.Timestamp(end), inventory)


def reconcile_psp_to_payprocc(sources: dict[str, LoadedSource], start: datetime, end: datetime, inventory: pd.DataFrame) -> dict:
    if PAYPROCC not in sources:
        raise ReconciliationError("Upload the PayProcc Transactions file.")
    results = []
    for provider in [DLOCAL, SKRILL, PAYSAFE]:
        if provider not in sources:
            continue
        left = normalize_psp_for_payprocc(sources[provider])
        right = normalize_payprocc_for_provider(sources[PAYPROCC], provider)
        tolerance = 0.01 if provider == SKRILL else 0.02
        results.append(_pair_reconcile(
            left, right, pd.Timestamp(start), pd.Timestamp(end), provider,
            "PSP", "PayProcc", tolerance,
            coverage_gap=provider in {DLOCAL, PAYSAFE},
        ))
    return _combine_results(results, "PSP → PayProcc", pd.Timestamp(start), pd.Timestamp(end), inventory)


def _safe_excel_frame(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    for column in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[column]):
            if getattr(clean[column].dt, "tz", None) is not None:
                clean[column] = clean[column].dt.tz_localize(None)
    return clean.replace({np.nan: None, np.inf: None, -np.inf: None})


def build_excel_export(result: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm:ss") as writer:
        overview = pd.DataFrame([{"Metric": key, "Value": value} for key, value in result["overall"].items()])
        overview.to_excel(writer, sheet_name="Overview", index=False, startrow=1)
        result["summary"].to_excel(writer, sheet_name="Provider Summary", index=False)
        result["daily"].to_excel(writer, sheet_name="Daily", index=False)
        if not result["currency"].empty:
            result["currency"].to_excel(writer, sheet_name="Currency Summary", index=False)
        _safe_excel_frame(result["exceptions"]).to_excel(writer, sheet_name="Exceptions", index=False)
        _safe_excel_frame(result["detail"]).to_excel(writer, sheet_name="Recon Detail", index=False)
        result["controls"].to_excel(writer, sheet_name="Controls", index=False)
        result["inventory"].to_excel(writer, sheet_name="Source Files", index=False)

        workbook = writer.book
        header = workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#0F766E", "border": 1, "text_wrap": True, "valign": "vcenter"})
        title = workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#15324B", "font_size": 16})
        money = workbook.add_format({"num_format": "$#,##0.00;[Red]($#,##0.00);-"})
        percent = workbook.add_format({"num_format": "0.00%"})
        pass_fmt = workbook.add_format({"bg_color": "#DCFCE7", "font_color": "#15803D", "bold": True})
        review_fmt = workbook.add_format({"bg_color": "#FEE2E2", "font_color": "#B91C1C", "bold": True})

        overview_sheet = writer.sheets["Overview"]
        overview_sheet.merge_range("A1:B1", f"{result['workflow']} Reconciliation", title)
        overview_sheet.set_column("A:A", 34)
        overview_sheet.set_column("B:B", 24)
        if "Match Rate" in result["overall"]:
            match_row = list(result["overall"].keys()).index("Match Rate") + 2
            overview_sheet.set_row(match_row, None, percent)

        for sheet_name, frame in [
            ("Provider Summary", result["summary"]), ("Daily", result["daily"]),
            ("Exceptions", result["exceptions"]), ("Recon Detail", result["detail"]),
            ("Controls", result["controls"]), ("Source Files", result["inventory"]),
        ] + ([] if result["currency"].empty else [("Currency Summary", result["currency"])]):
            sheet = writer.sheets[sheet_name]
            sheet.freeze_panes(1, 0)
            sheet.autofilter(0, 0, max(len(frame), 1), max(len(frame.columns) - 1, 0))
            for col_idx, column in enumerate(frame.columns):
                width = min(max(len(str(column)) + 2, 12), 42)
                if len(frame):
                    width = min(max(width, int(frame[column].astype(str).str.len().quantile(0.9)) + 2), 42)
                sheet.set_column(col_idx, col_idx, width)
                sheet.write(0, col_idx, column, header)
                if "Amount" in str(column) or "USD" in str(column) or "Difference" in str(column):
                    sheet.set_column(col_idx, col_idx, width, money)
            if "Status" in frame.columns:
                status_col = frame.columns.get_loc("Status")
                sheet.conditional_format(1, status_col, max(len(frame), 1), status_col, {"type": "text", "criteria": "containing", "value": "Matched", "format": pass_fmt})
                sheet.conditional_format(1, status_col, max(len(frame), 1), status_col, {"type": "text", "criteria": "containing", "value": "Mismatch", "format": review_fmt})
    output.seek(0)
    return output.getvalue()
