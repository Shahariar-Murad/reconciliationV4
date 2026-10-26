from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pandas as pd
import streamlit as st

from recon_engine import (
    ReconciliationError,
    build_excel_export,
    load_sources,
    reconcile_orchestrator_to_backend,
    reconcile_psp_to_bridger,
    reconcile_psp_to_payprocc,
)


st.set_page_config(
    page_title="Revenue Reconciliation Control Center",
    page_icon="↔",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --navy:#15324B; --teal:#0F766E; --muted:#64748B; --line:#E2E8F0; }
      .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1600px; }
      h1, h2, h3 { color: var(--navy); letter-spacing: -0.02em; }
      [data-testid="stMetric"] { border:1px solid var(--line); border-radius:12px; padding:14px 16px; background:#FFFFFF; }
      [data-testid="stMetricLabel"] { color:var(--muted); }
      [data-testid="stFileUploader"] { border:1px dashed #94A3B8; border-radius:12px; padding:4px 10px; }
      .period-pill { display:inline-block; padding:6px 12px; border-radius:999px; background:#E8F1FB; color:#15324B; font-weight:700; font-size:.86rem; }
      .workflow-note { border-left:4px solid var(--teal); background:#F8FAFC; padding:10px 14px; border-radius:0 10px 10px 0; color:#334155; }
      .small-muted { color:var(--muted); font-size:.86rem; }
      div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
      button[kind="primary"] { background:var(--teal); border-color:var(--teal); }
    </style>
    """,
    unsafe_allow_html=True,
)


def uploaded_tuples(files) -> tuple[tuple[str, bytes], ...]:
    if files is None:
        return tuple()
    if not isinstance(files, list):
        files = [files]
    return tuple((file.name, file.getvalue()) for file in files)


@st.cache_data(show_spinner=False)
def run_backend_cached(files: tuple[tuple[str, bytes], ...], start_iso: str, end_iso: str):
    sources, inventory = load_sources(files)
    return reconcile_orchestrator_to_backend(sources, pd.Timestamp(start_iso), pd.Timestamp(end_iso), inventory)


@st.cache_data(show_spinner=False)
def run_bridger_cached(files: tuple[tuple[str, bytes], ...], start_iso: str, end_iso: str):
    sources, inventory = load_sources(files)
    return reconcile_psp_to_bridger(sources, pd.Timestamp(start_iso), pd.Timestamp(end_iso), inventory)


@st.cache_data(show_spinner=False)
def run_payprocc_cached(files: tuple[tuple[str, bytes], ...], start_iso: str, end_iso: str):
    sources, inventory = load_sources(files)
    return reconcile_psp_to_payprocc(sources, pd.Timestamp(start_iso), pd.Timestamp(end_iso), inventory)


def money(value) -> str:
    return f"${float(value):,.2f}"


def render_inventory(inventory: pd.DataFrame) -> None:
    st.dataframe(
        inventory,
        use_container_width=True,
        hide_index=True,
        column_config={"Rows": st.column_config.NumberColumn(format="%d")},
    )


def render_provider_summary(result: dict) -> None:
    summary = result["summary"].copy()
    money_columns = [column for column in summary.columns if "USD" in str(column)]
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
        column_config={column: st.column_config.NumberColumn(format="$%.2f") for column in money_columns},
    )


def local_currency_gaps(result: dict) -> pd.DataFrame:
    """Return non-USD reconciliation gaps without mixing currency amounts."""
    currency = result["currency"].copy()
    if currency.empty:
        return currency
    difference = pd.to_numeric(currency["Difference"], errors="coerce").fillna(0)
    return currency[
        ~currency["Currency"].astype(str).str.upper().eq("USD")
        & difference.abs().gt(0.01)
    ].copy()


def filter_table(frame: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    left, right = st.columns([1, 2])
    with left:
        statuses = sorted(frame["Status"].dropna().astype(str).unique()) if "Status" in frame else []
        selected = st.multiselect("Status", statuses, default=statuses, key=f"{key_prefix}_statuses")
    with right:
        search = st.text_input("Search IDs, email, channel, or details", key=f"{key_prefix}_search", placeholder="Type to filter the table")
    filtered = frame.copy()
    if statuses:
        filtered = filtered[filtered["Status"].isin(selected)]
    if search:
        mask = filtered.astype(str).apply(lambda column: column.str.contains(search, case=False, na=False)).any(axis=1)
        filtered = filtered[mask]
    return filtered


def render_result(result: dict, key_prefix: str) -> None:
    overall = result["overall"]
    left_name = result["left_name"]
    right_name = result["right_name"]

    if overall["Action Exceptions"] == 0:
        st.success(
            f"Reconciliation passed: {overall['Matched']:,} matched transactions and no actionable exceptions. "
            f"Coverage-boundary and outside-period rows remain visible separately."
        )
    else:
        st.warning(
            f"Review required: {overall['Action Exceptions']:,} actionable exceptions across "
            f"{overall['Routes']:,} reconciled route(s)."
        )

    first = st.columns(6)
    first[0].metric("Routes", f"{overall['Routes']:,}")
    first[1].metric("Matched", f"{overall['Matched']:,}")
    first[2].metric("Action Exceptions", f"{overall['Action Exceptions']:,}")
    first[3].metric("Coverage Gaps", f"{overall['Coverage Gaps']:,}")
    first[4].metric("Outside Period", f"{overall['Outside Period']:,}")
    first[5].metric("Match Rate", f"{overall['Match Rate']:.1%}")

    has_usd = abs(overall[f"{left_name} USD"]) > 0.01 or abs(overall[f"{right_name} USD"]) > 0.01
    local_gaps = local_currency_gaps(result)
    second = st.columns(3)
    if has_usd:
        second[0].metric(f"{left_name} USD", money(overall[f"{left_name} USD"]))
        second[1].metric(f"{right_name} USD", money(overall[f"{right_name} USD"]))
        second[2].metric("Absolute USD Difference", money(overall["Absolute USD Difference"]))
    else:
        second[0].metric("USD transactions", "Not available")
        second[1].metric("Local-currency gaps", f"{len(local_gaps):,}")
        if local_gaps.empty:
            second[2].metric("Local-currency difference", "0")
        else:
            largest_gap = local_gaps.loc[local_gaps["Difference"].abs().idxmax()]
            second[2].metric(
                "Largest local gap",
                f"{largest_gap['Currency']} {abs(float(largest_gap['Difference'])):,.2f}",
            )

    executive, exception_tab, detail_tab, control_tab = st.tabs(
        ["Executive Summary", f"Exceptions & Boundaries ({len(result['exceptions']):,})", "Reconciliation Detail", "Controls & Sources"]
    )

    with executive:
        st.subheader("Provider summary")
        render_provider_summary(result)

        left, right = st.columns([1.15, 0.85])
        with left:
            st.subheader("Daily reconciliation status")
            daily = result["daily"].copy()
            if not daily.empty:
                chart_columns = [column for column in ["Matched", "Mismatch", "Missing", "Coverage Gap", "Outside Period"] if column in daily]
                st.bar_chart(daily.set_index("Reconciliation Date")[chart_columns], use_container_width=True, height=310)
            else:
                st.info("No daily rows are available for the selected period.")
        with right:
            st.subheader("Status composition")
            status = result["detail"]["Status Group"].value_counts().rename_axis("Status").reset_index(name="Transactions")
            st.dataframe(status, use_container_width=True, hide_index=True, height=310)

        if not result["currency"].empty:
            with st.expander("Currency breakdown", expanded=False):
                currency = result["currency"].copy()
                amount_columns = [column for column in currency.columns if "Amount" in str(column) or column == "Difference"]
                st.dataframe(
                    currency,
                    use_container_width=True,
                    hide_index=True,
                    column_config={column: st.column_config.NumberColumn(format="%.2f") for column in amount_columns},
                )

        if not has_usd:
            st.subheader("Local-currency gaps")
            if local_gaps.empty:
                st.success("No local-currency amount gaps were identified.")
            else:
                st.caption("USD is not available in the selected data. Amounts are shown separately by local currency and are not combined.")
                local_amount_columns = [column for column in local_gaps.columns if "Amount" in str(column) or column == "Difference"]
                st.dataframe(
                    local_gaps,
                    use_container_width=True,
                    hide_index=True,
                    column_config={column: st.column_config.NumberColumn(format="%.2f") for column in local_amount_columns},
                )

        if result["workflow"] == "Orchestrator → Backend API" and "Entity" in result["detail"]:
            entity = (
                result["detail"]
                .groupby(["Provider", "Entity", "Status Group"], dropna=False)
                .size()
                .rename("Transactions")
                .reset_index()
            )
            with st.expander("CFD / Futures breakdown", expanded=False):
                st.dataframe(entity, use_container_width=True, hide_index=True)

    with exception_tab:
        if result["exceptions"].empty:
            st.success("No exceptions or boundary items were identified.")
        else:
            st.caption("Action Required distinguishes true missing, amount, currency, and key issues from coverage or date-boundary rows.")
            filtered = filter_table(result["exceptions"], f"{key_prefix}_exceptions")
            st.dataframe(
                filtered,
                use_container_width=True,
                hide_index=True,
                height=520,
                column_config={
                    "Action Required": st.column_config.CheckboxColumn(),
                    "Amount Difference": st.column_config.NumberColumn(format="%.2f"),
                    "Tolerance": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            st.download_button(
                "Download filtered exceptions (CSV)",
                filtered.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{key_prefix}_exceptions.csv",
                mime="text/csv",
                key=f"{key_prefix}_exception_download",
            )

    with detail_tab:
        filtered = filter_table(result["detail"], f"{key_prefix}_detail")
        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
            height=620,
            column_config={
                "Action Required": st.column_config.CheckboxColumn(),
                "Amount Difference": st.column_config.NumberColumn(format="%.2f"),
                "Tolerance": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    with control_tab:
        left, right = st.columns([0.9, 1.1])
        with left:
            st.subheader("Control checks")
            st.dataframe(result["controls"], use_container_width=True, hide_index=True)
        with right:
            st.subheader("Detected source files")
            render_inventory(result["inventory"])
        st.caption(
            "All timestamps are normalized to GMT+3 before filtering. The selected period uses an inclusive start and exclusive end. "
            "Files are processed in the current Streamlit session and are not written to application storage."
        )

    export_bytes = build_excel_export(result)
    st.download_button(
        "Download complete reconciliation workbook",
        export_bytes,
        file_name=f"{key_prefix}_{result['period_start']:%Y%m%d}_{result['period_end']:%Y%m%d_%H%M}_GMT3.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        key=f"{key_prefix}_excel_download",
    )


def run_workflow(files: tuple[tuple[str, bytes], ...], runner, start_dt: datetime, end_dt: datetime, key_prefix: str) -> None:
    if not files:
        st.info("Upload the required files to start this reconciliation workflow.")
        return
    try:
        with st.spinner("Detecting source schemas, normalizing timestamps, and reconciling transactions…"):
            result = runner(files, start_dt.isoformat(), end_dt.isoformat())
        render_result(result, key_prefix)
    except (ReconciliationError, KeyError, ValueError) as exc:
        st.error(str(exc))
        with st.expander("Troubleshooting guidance"):
            st.markdown(
                "- Confirm the required orchestrator/API file is included.\n"
                "- Keep original provider headers and sheet names unchanged.\n"
                "- Upload the raw report rather than a manually edited copy.\n"
                "- CSV, XLS, XLSX, and ZIP are supported."
            )


with st.sidebar:
    st.header("Reconciliation period")
    today = date.today()
    selected = st.date_input(
        "Date range",
        value=(today - timedelta(days=3), today),
        help="Example: select 14 August to 16 August.",
    )
    if isinstance(selected, tuple) and len(selected) == 2:
        start_date, end_date = selected
    else:
        start_date = end_date = selected if isinstance(selected, date) else today

    with st.expander("Exact cutoff time", expanded=True):
        start_time = st.time_input("Start time", value=time(0, 0), step=timedelta(minutes=15))
        custom_cutoff = st.checkbox("Use a custom end cutoff", value=True)
        if custom_cutoff:
            end_time = st.time_input("End cutoff", value=time(5, 0), step=timedelta(minutes=15))
            end_dt = datetime.combine(end_date, end_time)
        else:
            end_dt = datetime.combine(end_date + timedelta(days=1), time(0, 0))
    start_dt = datetime.combine(start_date, start_time)
    if end_dt <= start_dt:
        st.error("The period end must be later than the start.")
        st.stop()

    st.markdown(
        f'<span class="period-pill">{start_dt:%d %b %Y %H:%M} → {end_dt:%d %b %Y %H:%M} · GMT+3</span>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("Supported uploads: CSV, XLS, XLSX, ZIP")
    st.caption("Reporting timezone: GMT+3")


st.title("Revenue Reconciliation Control Center")
st.markdown(
    "Upload the raw reports, review actionable exceptions first, and export a controlled reconciliation workbook. "
    "Source schemas and provider routes are detected automatically."
)

workflow_backend, workflow_bridger, workflow_payprocc = st.tabs(
    ["1 · Orchestrator → Backend API", "2 · PSP → BridgerPay", "3 · PSP → PayProcc"]
)

with workflow_backend:
    st.markdown(
        '<div class="workflow-note"><b>Routes:</b> BridgerPay, PayProcc, ZEN, Confirmo, Binance Pay, Coinsbuy, and TC Pay. '
        'Backend add-to-cart orders sharing one transaction ID are aggregated before comparison.</div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([0.42, 0.58])
    with col1:
        backend_file = st.file_uploader(
            "Backend API · Order List",
            type=["csv", "xlsx", "xls", "zip"],
            key="backend_api_file",
        )
    with col2:
        orchestrator_files = st.file_uploader(
            "Orchestrator reports · upload one or several",
            type=["csv", "xlsx", "xls", "zip"],
            accept_multiple_files=True,
            key="orchestrator_files",
        )
    files = uploaded_tuples(backend_file) + uploaded_tuples(orchestrator_files)
    run_workflow(files, run_backend_cached, start_dt, end_dt, "orchestrator_backend")

with workflow_bridger:
    st.markdown(
        '<div class="workflow-note"><b>Supported PSPs:</b> Payabl, TrustPayments, Unlimit, PayPal, Nuvei EU, Nuvei AU, Paysafe, and Paystra. '
        'Only approved/successful PSP and BridgerPay payment populations are reconciled.</div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([0.42, 0.58])
    with col1:
        bridger_file = st.file_uploader(
            "BridgerPay · Transactions Report",
            type=["csv", "xlsx", "xls", "zip"],
            key="bridger_file",
        )
    with col2:
        bridger_psp_files = st.file_uploader(
            "PSP reports · upload one or several",
            type=["csv", "xlsx", "xls", "zip"],
            accept_multiple_files=True,
            key="bridger_psp_files",
        )
    files = uploaded_tuples(bridger_file) + uploaded_tuples(bridger_psp_files)
    run_workflow(files, run_bridger_cached, start_dt, end_dt, "psp_bridgerpay")

with workflow_payprocc:
    st.markdown(
        '<div class="workflow-note"><b>Supported PSPs:</b> dLocal, Skrill, and Paysafe local methods. '
        'PayProcc routes are isolated by MID/brand; local-currency amounts are not mixed into USD totals.</div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([0.42, 0.58])
    with col1:
        payprocc_file = st.file_uploader(
            "PayProcc · Transactions",
            type=["csv", "xlsx", "xls", "zip"],
            key="payprocc_file",
        )
    with col2:
        payprocc_psp_files = st.file_uploader(
            "PSP reports · upload one or several",
            type=["csv", "xlsx", "xls", "zip"],
            accept_multiple_files=True,
            key="payprocc_psp_files",
        )
    files = uploaded_tuples(payprocc_file) + uploaded_tuples(payprocc_psp_files)
    run_workflow(files, run_payprocc_cached, start_dt, end_dt, "psp_payprocc")

st.divider()
st.markdown('<div class="small-muted">Revenue Operations · controlled reconciliation · Streamlit/GitHub deployment package</div>', unsafe_allow_html=True)
