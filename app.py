"""Financial Controller dashboard. AI results are session-local; reports follow CSV versions."""
from io import BytesIO
import logging
import re
from decimal import Decimal

import pandas as pd
import streamlit as st

from tools import finance_tools as finance

logger = logging.getLogger(__name__)
SECTIONS = ("Overview", "AI Assistant", "Transactions", "Anomalies", "Reconciliation")
REPORT_FILES = {
    "overview": ("transactions.csv", "invoices.csv", "bank_transactions.csv"),
    "transactions": ("transactions.csv",),
    "anomalies": ("transactions.csv",),
    "reconciliation": ("invoices.csv", "bank_transactions.csv"),
    "missing_bank": ("transactions.csv", "bank_transactions.csv"),
    "missing_supplier": ("transactions.csv", "bank_transactions.csv"),
}
AI_FILES = ("transactions.csv", "invoices.csv", "bank_transactions.csv", "suppliers.csv", "customers.csv")


def file_versions(names):
    """Small stat-only cache keys; no CSV reads needed on a cache hit."""
    versions = []
    for name in names:
        path = finance.DATA_PATH / name
        stat = path.stat()
        versions.append((name, stat.st_mtime_ns, stat.st_size))
    return tuple(versions)


@st.cache_data(show_spinner=False, max_entries=24)
def load_report(name, versions):
    # versions is deliberately hashed, so changed files invalidate derived reports.
    if name == "overview":
        transactions = finance.get_transactions()
        activity = transactions.assign(activity=transactions["type"].str.lower())
        totals = activity.groupby("activity")["amount"].sum().reindex(["income", "expense"], fill_value=0)
        dates = pd.to_datetime(transactions["date"], errors="coerce")
        valid = dates.notna()
        trend = pd.DataFrame()
        if valid.any():
            monthly = activity.loc[valid].assign(month=dates.loc[valid].dt.to_period("M").dt.to_timestamp())
            trend = monthly.groupby(["month", "activity"])["amount"].sum().unstack(fill_value=0)
            trend = trend.reindex(columns=["income", "expense"], fill_value=0)
        return {"transaction_count": len(transactions), "totals": totals, "trend": trend,
                "invalid_dates": int((~valid).sum()), "invoices": len(finance.get_invoices()),
                "bank_records": len(finance.get_bank_transactions()),
                "anomalies": len(finance.detect_large_transactions())}
    loaders = {"transactions": finance.get_transactions,
               "anomalies": finance.detect_large_transactions,
               "reconciliation": finance.reconcile_invoices,
               "missing_bank": finance.detect_missing_bank_records,
               "missing_supplier": finance.detect_missing_supplier_records}
    return loaders[name]()


def report(name):
    return load_report(name, file_versions(REPORT_FILES[name]))


def money(value):
    """Indian digit grouping for display; underlying financial values stay numeric."""
    if value is None or pd.isna(value):
        return "Unavailable"
    amount = Decimal(str(value))
    if not amount.is_finite():
        return "Unavailable"
    whole, fraction = f"{abs(amount):.2f}".split(".")
    groups = [whole[-3:]]
    remaining = whole[:-3]
    while remaining:
        groups.insert(0, remaining[-2:])
        remaining = remaining[:-2]
    decimals = "" if fraction == "00" else "." + fraction
    return ("-" if amount < 0 else "") + "\u20b9" + ",".join(groups) + decimals


def display_text(text):
    """Format explicit currency amounts in generated prose without changing records."""
    return re.sub(r"(?:\bINR\s*|\u20b9\s*)(-?\d[\d,]*(?:\.\d+)?)",
                  lambda match: money(match.group(1).replace(",", "")), str(text))


def kpis(items):
    # Two columns remain readable at smaller browser widths.
    for offset in range(0, len(items), 2):
        columns = st.columns(2)
        for column, (label, value) in zip(columns, items[offset:offset + 2]):
            with column.container(border=True):
                st.metric(label, value)


def table(records, empty="No records found."):
    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if frame.empty:
        st.info(empty)
    else:
        amount_columns = {column: money for column in frame.columns
                          if column == "amount" or column.endswith("_amount")
                          or column in ("difference", "total_due")}
        display = frame.style.format(amount_columns) if amount_columns else frame
        st.dataframe(display, width="stretch", hide_index=True)



def excel_bytes(records):
    """Export every displayed record, never an LLM sample or a fresh date query."""
    frame = pd.DataFrame(records)
    if len(frame) > 1048575:
        raise ValueError("Too many rows for a single Excel sheet")
    output = BytesIO()
    frame.to_excel(output, index=False, engine="openpyxl", sheet_name="Transactions")
    return output.getvalue()


def overview():
    st.header("Financial overview")
    st.caption("A clear view of your records, cash activity, and review signals.")
    data = report("overview")
    kpis([("Total transactions", f"{data['transaction_count']:,}"),
          ("Invoices", f"{data['invoices']:,}"),
          ("Bank records", f"{data['bank_records']:,}"),
          ("Detected anomalies", f"{data['anomalies']:,}")])
    st.caption("Detected anomalies are expense transactions above \u20b92,00,000; they are review signals.")
    totals = data["totals"]
    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader("Income & expenses")
        st.caption("Total recorded amounts | INR | all available dates")
        st.bar_chart(totals.rename("Amount (INR)"), color="#536DFE")
    with right.container(border=True):
        st.subheader("Transaction trend")
        trend = data["trend"]
        if not trend.empty:
            st.caption("Monthly recorded amounts | INR")
            st.line_chart(trend, color=["#10B981", "#536DFE"])
        else:
            st.info("No valid dates available for a trend chart.")
        if data["invalid_dates"]:
            st.caption(f"{data['invalid_dates']:,} records have invalid dates and are excluded from the trend only.")



def render_investigation(investigation):
    if investigation.get("error"):
        st.error(investigation["error"])
        st.caption(f"Analysis status: {investigation.get('analysis_status', 'unavailable')}")
        return
    st.subheader("Transaction investigation")
    risk = investigation.get("risk_level", "UNAVAILABLE")
    indicator = {"LOW": st.success, "MEDIUM": st.warning, "HIGH": st.error}.get(risk, st.info)
    indicator(f"Risk assessment: {risk}")
    kpis([("Transaction ID", investigation.get("transaction_id", "Unavailable")),
          ("Amount", money(investigation.get("amount")))])
    st.write(f"**Supplier:** {investigation.get('supplier', 'Unknown')} "
             f"({investigation.get('supplier_id') or 'Not available'})")
    st.caption(f"Confidence: {investigation.get('confidence', 'UNAVAILABLE')} | "
               f"Analysis status: {investigation.get('analysis_status', 'unavailable')}")
    st.subheader("Findings")
    for finding in investigation.get("findings", []):
        st.write(f"- {display_text(finding)}")
    with st.container(border=True):
        st.markdown("**Reasoning / explanation**")
        st.write(display_text(investigation.get("why") or "Not available."))
        st.markdown("**Recommended action**")
        st.write(display_text(investigation.get("recommended_action") or "Not available."))
    st.caption("An anomaly is a review signal and does not establish fraud.")


def assistant():
    st.header("AI financial assistant")
    st.caption("Ask about your records or investigate a transaction with verified financial evidence.")
    with st.form("financial_question"):
        query = st.text_input("Your question", placeholder="Show transactions from August 1 to August 15, 2026",
                              value=st.session_state.get("last_financial_query", ""))
        submitted = st.form_submit_button("Ask AI", type="primary")
    st.caption("Try: Investigate transaction T008 | Find unusually large transactions | Show invoices for supplier S001")
    if submitted:
        if not query.strip():
            st.warning("Enter a financial question to continue.")
        else:
            st.session_state["last_financial_query"] = query.strip()
            st.session_state.pop("financial_excel", None)
            with st.spinner("Reviewing financial evidence..."):
                try:
                    # Lazy import: other dashboard sections need neither an API key nor an AI client.
                    from agents.financial_agent import ask_financial_agent
                    versions = file_versions(AI_FILES)
                    result = ask_financial_agent(query.strip())
                    st.session_state["financial_result"] = result
                    st.session_state["financial_result_versions"] = versions
                except Exception as exc:
                    logger.warning("Assistant UI request failed error_type=%s", type(exc).__name__)
                    st.session_state["financial_result"] = {"answer": "Analysis could not be completed. Please try again.",
                        "transactions": [], "start_date": None, "end_date": None, "investigation": None}
                    st.session_state.pop("financial_result_versions", None)
    result = st.session_state.get("financial_result")
    if not result:
        st.info("Your latest response will stay here when you switch sections.")
        return
    st.divider()
    st.caption(f"Latest question: {st.session_state.get('last_financial_query', 'Previous session question')}")
    previous = st.session_state.get("financial_result_versions")
    if previous and previous != file_versions(AI_FILES):
        st.warning("Source data has changed since this response. Ask again for an updated analysis. Downloads retain the displayed records.")
    investigation = result.get("investigation")
    if investigation:
        render_investigation(investigation)
    else:
        with st.container(border=True):
            st.markdown(display_text(result.get("answer") or "No answer was returned."))
    records = result.get("transactions", [])
    if records:
        st.subheader("Retrieved transactions")
        st.caption(f"{len(records):,} records | complete retrieved result")
        table(records)
        start, end = result.get("start_date"), result.get("end_date")
        if start and end:
            if st.button("Prepare Excel", key="prepare_financial_excel"):
                try:
                    with st.spinner("Preparing all retrieved records..."):
                        st.session_state["financial_excel"] = excel_bytes(records)
                except Exception as exc:
                    logger.warning("Excel export failed error_type=%s", type(exc).__name__)
                    st.error("Excel export could not be prepared. No records were truncated.")
            if st.session_state.get("financial_excel"):
                st.download_button("Download Excel", data=st.session_state["financial_excel"],
                    file_name=f"transactions_{start}_to_{end}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_financial_excel")
                st.caption(f"Export includes all {len(records):,} retrieved records.")


def transactions_page():
    st.header("Transactions")
    st.caption("Search and filter the complete transaction ledger.")
    frame = report("transactions")
    left, right = st.columns([2, 1])
    search = left.text_input("Search records", placeholder="Transaction ID, description, supplier or customer")
    kinds = right.multiselect("Transaction type", sorted(frame["type"].dropna().unique()))
    filtered = frame
    if search.strip():
        text = frame.astype("string").fillna("")
        mask = text.apply(lambda column: column.str.contains(search.strip(), case=False, regex=False)).any(axis=1)
        filtered = filtered.loc[mask]
    if kinds:
        filtered = filtered.loc[filtered["type"].isin(kinds)]
    dates = pd.to_datetime(frame["date"], errors="coerce")
    valid = dates.notna()
    if valid.any():
        if st.checkbox("Filter by date", key="filter_transaction_dates"):
            bounds = (dates.min().date(), dates.max().date())
            chosen = st.date_input("Inclusive date range", value=bounds, min_value=bounds[0], max_value=bounds[1])
            if len(chosen) != 2:
                st.info("Choose both a start date and an end date.")
                return
            mask = dates.dt.date.between(chosen[0], chosen[1])
            filtered = filtered.loc[mask.reindex(filtered.index)]
    else:
        st.info("Date filtering is unavailable because the ledger contains no valid dates.")
    if not valid.all():
        st.caption(f"{int((~valid).sum()):,} records have invalid dates; they remain visible unless a date filter is applied.")
    st.caption(f"Showing {len(filtered):,} of {len(frame):,} transactions | no sampling")
    table(filtered, "No transactions match these filters.")


def anomalies():
    st.header("Anomalies")
    st.caption("Existing rule: expense transactions strictly above \u20b92,00,000.")
    frame = report("anomalies")
    kpis([("Flagged transactions", f"{len(frame):,}"),
          ("Flagged expense amount", money(frame["amount"].sum()))])
    st.info("These records require review. The threshold alone does not establish fraud.")
    table(frame, "No expense transactions exceed the threshold.")


def reconciliation():
    st.header("Reconciliation")
    st.caption("Invoice comparisons and record consistency checks from the existing financial tools.")
    choice = st.radio("Report", ["Invoice reconciliation", "Missing bank records", "Missing supplier records"],
                      horizontal=True, key="reconciliation_report")
    key = {"Invoice reconciliation": "reconciliation", "Missing bank records": "missing_bank",
           "Missing supplier records": "missing_supplier"}[choice]
    frame = pd.DataFrame(report(key))
    if key == "reconciliation":
        mismatch_count = int(frame["status"].eq("MISMATCH").sum()) if not frame.empty else 0
        kpis([("Invoices checked", f"{len(frame):,}"), ("Amount mismatches", f"{mismatch_count:,}")])
        st.info("Supplier and closest-amount comparisons are indicative; they do not establish a direct payment link.")
    else:
        st.metric("Records to review", f"{len(frame):,}")
        st.caption("Results reflect the existing supplier-and-amount matching rules.")
    table(frame, "No records to report.")


def main():
    st.set_page_config(page_title="Financial Controller | Buildathon", page_icon="\U0001f4ca", layout="wide")
    st.markdown("""<style>
    .block-container {max-width: 1440px; padding-top: 2rem;}
    [data-testid="stMetricValue"] {font-size: clamp(1.25rem, 2.5vw, 2rem);}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.18);}
    </style>""", unsafe_allow_html=True)
    with st.sidebar:
        st.title("Financial Controller")
        st.caption("RAZORPAY AGENTIC AI BUILDATHON")
        st.divider()
        section = st.radio("Workspace", SECTIONS, key="dashboard_section", label_visibility="collapsed")
        st.divider()
        st.caption("Financial intelligence backed by your records.")
        st.caption("Amounts in INR | All available dataset dates")
    st.caption(f"FINANCE WORKSPACE / {section.upper()}")
    pages = {"Overview": overview, "AI Assistant": assistant, "Transactions": transactions_page,
             "Anomalies": anomalies, "Reconciliation": reconciliation}
    try:
        pages[section]()
    except Exception as exc:
        logger.warning("Dashboard render failed section=%s error_type=%s", section, type(exc).__name__)
        st.error("This section could not be loaded. Check that the source data is available and try again.")


if __name__ == "__main__":
    main()
