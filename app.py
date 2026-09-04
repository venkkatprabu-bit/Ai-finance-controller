import streamlit as st
import pandas as pd

from agents.financial_agent import ask_financial_agent

from tools.finance_tools import (
    get_transactions,
    get_invoices,
    get_bank_transactions,
    detect_large_transactions,
    detect_missing_bank_records,
    detect_missing_supplier_records,
    reconcile_invoices,
    export_transactions_to_excel
)


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI Financial Controller",
    page_icon="💰",
    layout="wide"
)


# ==================================================
# TITLE
# ==================================================

st.title("💰 AI Financial Controller")

st.caption(
    "Agentic AI for financial investigation, "
    "anomaly detection and reconciliation"
)


# ==================================================
# LOAD FINANCIAL DATA
# ==================================================

transactions = get_transactions()

invoices = get_invoices()

bank_transactions = get_bank_transactions()

large_transactions = detect_large_transactions()

missing_bank = detect_missing_bank_records()

missing_supplier = detect_missing_supplier_records()

reconciliation = reconcile_invoices()


# ==================================================
# DASHBOARD METRICS
# ==================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Transactions",
        len(transactions)
    )


with col2:

    st.metric(
        "Large Anomalies",
        len(large_transactions)
    )


with col3:

    mismatch_count = sum(
        1
        for item in reconciliation
        if item.get("status") == "MISMATCH"
    )

    st.metric(
        "Invoice Mismatches",
        mismatch_count
    )


with col4:

    st.metric(
        "Missing Bank Records",
        len(missing_bank)
    )


# ==================================================
# FINANCIAL ANOMALIES
# ==================================================

st.divider()

st.header(
    "⚠️ Financial Anomalies"
)


if large_transactions.empty:

    st.success(
        "No large transaction anomalies detected."
    )

else:

    st.dataframe(
        large_transactions,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# INVOICE RECONCILIATION
# ==================================================

st.divider()

st.header(
    "🔄 Invoice Reconciliation"
)


reconciliation_df = pd.DataFrame(
    reconciliation
)


if not reconciliation_df.empty:

    st.dataframe(
        reconciliation_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No reconciliation records available."
    )


# ==================================================
# RECORD SYNCHRONIZATION
# ==================================================

st.divider()

st.header(
    "🏦 Record Synchronization"
)


col1, col2 = st.columns(2)


# --------------------------------------------------
# BANK RECORDS MISSING
# --------------------------------------------------

with col1:

    st.subheader(
        "Bank Records Missing"
    )


    if missing_bank:

        missing_bank_df = pd.DataFrame(
            missing_bank
        )

        st.dataframe(
            missing_bank_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "No missing bank records."
        )


# --------------------------------------------------
# SUPPLIER RECORDS
# --------------------------------------------------

with col2:

    st.subheader(
        "Supplier Records"
    )


    if missing_supplier:

        missing_supplier_df = pd.DataFrame(
            missing_supplier
        )

        st.dataframe(
            missing_supplier_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "No missing supplier records."
        )


# ==================================================
# ALL TRANSACTION DATA
# ==================================================

st.divider()

st.header(
    "📊 Transaction History"
)


st.dataframe(
    transactions,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# AI FINANCIAL ASSISTANT
# ============================================================

st.header("🤖 AI Financial Assistant")

user_query = st.text_input(
    "Ask a financial question",
    placeholder="Example: Show me transactions from August 1 to August 15"
)

if st.button("Ask AI", key="ask_financial_ai"):

    if not user_query.strip():
        st.warning("Please enter a question.")
    else:

        with st.spinner("AI Financial Controller is analyzing..."):

            result = ask_financial_agent(user_query)

            st.session_state["financial_result"] = result


# ============================================================
# DISPLAY AI RESULT
# ============================================================

if "financial_result" in st.session_state:

    result = st.session_state["financial_result"]

    # ========================================================
    # CHECK IF INVESTIGATION RESULT
    # ========================================================

    investigation = result.get("investigation")

    if investigation and "error" not in investigation:

        # Display investigation result
        st.subheader("🔍 Transaction Investigation")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Transaction",
                investigation.get("transaction_id", "N/A")
            )

        with col2:
            amount = investigation.get("amount", 0)
            st.metric(
                "Amount",
                f"₹{amount:,.0f}" if amount else "N/A"
            )

        with col3:
            st.metric(
                "Risk Level",
                investigation.get("risk_level", "N/A")
            )

        st.write(
            f"**Supplier:** "
            f"{investigation.get('supplier', 'Unknown')} "
            f"({investigation.get('supplier_id', 'N/A')})"
        )

        # Findings
        st.subheader("Findings")
        findings = investigation.get("findings", [])
        for finding in findings:
            st.write(f"• {finding}")

        # Why
        st.subheader("Why")
        st.write(investigation.get("why", "Not available."))

        # Recommended Action
        st.subheader("Recommended Action")
        st.info(investigation.get("recommended_action", "Not available."))

        # Confidence
        st.write(
            f"**Confidence:** {investigation.get('confidence', 'N/A')}"
        )

    else:

        # Display normal transaction query response
        st.subheader("AI Response")

        st.markdown(result["answer"])

        # ========================================================
        # TRANSACTION HISTORY RESULTS
        # ========================================================

        history_records = result.get("transactions", [])

        if history_records:

            st.subheader("📊 Transaction Records")

            st.dataframe(
                history_records,
                use_container_width=True
            )


            # Get date range used by the agent
            start_date = result.get("start_date")
            end_date = result.get("end_date")


            # ====================================================
            # EXCEL EXPORT
            # ====================================================

            st.subheader("📥 Export Transaction History")

            if st.button(
                "Prepare Excel",
                key="prepare_financial_excel"
            ):

                if start_date and end_date:

                    excel_file = export_transactions_to_excel(
                        start_date,
                        end_date
                    )

                    with open(excel_file, "rb") as f:
                        st.session_state["excel_data"] = f.read()

                    st.session_state["excel_filename"] = (
                        f"transactions_{start_date}_to_{end_date}.xlsx"
                    )

                    st.success("Excel file is ready.")


            # ====================================================
            # DOWNLOAD BUTTON
            # ====================================================

            if st.session_state.get("excel_data"):

                st.download_button(
                    label="⬇️ Download Excel",
                    data=st.session_state["excel_data"],
                    file_name=st.session_state.get(
                        "excel_filename",
                        "transactions.xlsx"
                    ),
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    key="download_financial_excel"
                )