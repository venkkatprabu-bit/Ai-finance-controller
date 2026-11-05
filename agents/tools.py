from tools.finance_tools import (
    get_transaction,
    get_supplier,
    get_supplier_transactions,
    detect_large_transactions,
    get_supplier_invoices,
    get_supplier_bank_transactions,
    detect_invoice_mismatches,
    get_transactions_by_date,
    export_transactions_to_excel
)




# ==================================================
# TRANSACTION TOOL
# ==================================================

def transaction_tool(transaction_id):

    result = get_transaction(
        transaction_id
    )

    return result.to_dict(
        orient="records"
    )


# ==================================================
# SUPPLIER TOOL
# ==================================================

def supplier_tool(supplier_id):

    result = get_supplier(
        supplier_id
    )

    return result.to_dict(
        orient="records"
    )


# ==================================================
# SUPPLIER TRANSACTION HISTORY
# ==================================================

def supplier_history_tool(supplier_id):

    result = get_supplier_transactions(
        supplier_id
    )

    return result.to_dict(
        orient="records"
    )


# ==================================================
# LARGE TRANSACTION TOOL
# ==================================================

def large_transaction_tool():

    result = detect_large_transactions()

    return result.to_dict(
        orient="records"
    )


# ==================================================
# SUPPLIER INVOICE TOOL
# ==================================================

def supplier_invoices_tool(supplier_id):

    result = get_supplier_invoices(
        supplier_id
    )

    return result.to_dict(
        orient="records"
    )


# ==================================================
# SUPPLIER BANK TRANSACTION TOOL
# ==================================================

def supplier_bank_transactions_tool(supplier_id):

    result = get_supplier_bank_transactions(
        supplier_id
    )

    return result.to_dict(
        orient="records"
    )


# ==================================================
# INVOICE MISMATCH TOOL
# ==================================================

def invoice_mismatch_tool():

    result = detect_invoice_mismatches()

    return result


# ==================================================
# TRANSACTION HISTORY BY DATE
# ==================================================

def transaction_history_tool(
    start_date,
    end_date
):

    result = get_transactions_by_date(
        start_date,
        end_date
    )

    return result.to_dict(
        orient="records"
    )


# ==================================================
# EXCEL EXPORT TOOL
# ==================================================

def export_transaction_history_tool(
    start_date,
    end_date
):

    return export_transactions_to_excel(
        start_date,
        end_date
    )


# ==================================================
# INVESTIGATE TRANSACTION TOOL
# ==================================================


def investigation_evidence_tool(transaction_id):
    """Return compact verified evidence without materializing complete histories as records."""
    from tools.finance_tools import get_investigation_evidence
    return get_investigation_evidence(transaction_id)
