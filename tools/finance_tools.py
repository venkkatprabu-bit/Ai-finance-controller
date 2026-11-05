"""Financial data access and reconciliation helpers.

Compatibility: preserves public function names and return shapes from finance_tools.py.
Matching is still supplier + closest amount; it is NOT proof of payment linkage.
"""
from functools import lru_cache
from pathlib import Path
import unicodedata
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / 'data'

@lru_cache(maxsize=10)
def _cached_data(filename, modified_ns):
    frame = pd.read_csv(DATA_PATH / filename)
    frame.columns = frame.columns.str.strip()
    for column in frame.select_dtypes(include=['object']).columns:
        frame[column] = frame[column].map(
            lambda v: unicodedata.normalize('NFKC', v).strip() if isinstance(v, str) else v
        )
    for column in ('amount', 'total_due'):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors='raise')
    return frame

def _read_data(filename):
    path = DATA_PATH / filename
    return _cached_data(filename, path.stat().st_mtime_ns).copy()

def get_transactions(): return _read_data('transactions.csv')
def get_invoices(): return _read_data('invoices.csv')
def get_bank_transactions(): return _read_data('bank_transactions.csv')
def get_suppliers(): return _read_data('suppliers.csv')
def get_customers(): return _read_data('customers.csv')

def get_transaction(transaction_id):
    df = get_transactions()
    return df.loc[df['transaction_id'] == transaction_id]

def get_supplier(supplier_id):
    df = get_suppliers()
    return df.loc[df['supplier_id'] == supplier_id]

def get_supplier_transactions(supplier_id):
    df = get_transactions()
    return df.loc[df['supplier_id'] == supplier_id]

def detect_large_transactions(threshold=200000):
    df = get_transactions()
    return df.loc[df['type'].str.lower().eq('expense') & df['amount'].gt(threshold)]

def get_unpaid_invoices():
    df = get_invoices()
    return df.loc[df['status'].str.lower().eq('unpaid')]

def get_supplier_invoices(supplier_id):
    df = get_invoices()
    return df.loc[df['supplier_id'] == supplier_id]

def get_supplier_bank_transactions(supplier_id):
    df = get_bank_transactions()
    return df.loc[df['supplier_id'] == supplier_id]

def get_overdue_customers():
    df = get_customers()
    return df.loc[df['status'].str.lower().eq('overdue')]

def get_expenses():
    df = get_transactions()
    return df.loc[df['type'].str.lower().eq('expense')]

def get_income():
    df = get_transactions()
    return df.loc[df['type'].str.lower().eq('income')]

def _reconcile(missing_status):
    invoices = get_invoices()
    bank = get_bank_transactions()
    # Sorting each supplier's bank amounts allows logarithmic closest-amount lookup.
    bank['_original_order'] = range(len(bank))
    groups = {}
    for supplier_id, group in bank.groupby('supplier_id', sort=False):
        groups[supplier_id] = group.sort_values('amount', kind='stable').reset_index(drop=True)
    results = []
    for inv in invoices.itertuples(index=False):
        supplier_id = inv.supplier_id
        group = groups.get(supplier_id)
        if group is None or group.empty:
            results.append({'invoice_id': inv.invoice_id, 'supplier_id': supplier_id,
                            'invoice_amount': inv.amount, 'status': missing_status})
            continue
        # searchsorted finds the two nearest amount candidates, preserving the
        # previous preference for the earliest bank row when tied.
        amounts = group['amount']
        pos = int(amounts.searchsorted(inv.amount, side='left'))
        candidates = [] 
        if pos < len(group): candidates.append(pos)
        if pos > 0: candidates.append(pos - 1)
        # When duplicate amounts exist, choose the first original bank row.
        nearest_distance = min(abs(amounts.iloc[i] - inv.amount) for i in candidates)
        nearest_amounts = {amounts.iloc[i] for i in candidates
                           if abs(amounts.iloc[i] - inv.amount) == nearest_distance}
        tied = group.loc[group['amount'].isin(nearest_amounts)]
        # Match old idxmin tie-breaking: earliest row in original CSV.
        bank_row = tied.loc[tied['_original_order'].idxmin()]
        difference = bank_row['amount'] - inv.amount
        results.append({'invoice_id': inv.invoice_id, 'supplier_id': supplier_id,
                        'invoice_amount': inv.amount,
                        'bank_transaction_id': bank_row['bank_transaction_id'],
                        'bank_amount': bank_row['amount'], 'difference': difference,
                        'status': 'MATCH' if difference == 0 else 'MISMATCH'})
    return results

def reconcile_invoices():
    return _reconcile('No bank transaction found')

def detect_invoice_mismatches():
    return _reconcile('NO_BANK_TRANSACTION')

def detect_missing_bank_records():
    transactions, bank = get_transactions(), get_bank_transactions()
    known = set(zip(bank['supplier_id'].dropna(), bank.loc[bank['supplier_id'].notna(), 'amount']))
    results = []
    for tx in transactions.itertuples(index=False):
        if pd.isna(tx.supplier_id) or (tx.supplier_id, tx.amount) in known:
            continue
        results.append({'transaction_id': tx.transaction_id, 'supplier_id': tx.supplier_id,
                        'transaction_amount': tx.amount, 'transaction_date': tx.date,
                        'status': 'BANK_RECORD_MISSING'})
    return results

def detect_missing_supplier_records():
    transactions, bank = get_transactions(), get_bank_transactions()
    known = set(zip(transactions.loc[transactions['supplier_id'].notna(), 'supplier_id'],
                    transactions.loc[transactions['supplier_id'].notna(), 'amount']))
    supplier_ids = set(transactions['supplier_id'].dropna())
    results = []
    for row in bank.itertuples(index=False):
        if pd.isna(row.supplier_id):
            results.append({'bank_transaction_id': row.bank_transaction_id,
                            'bank_amount': row.amount, 'bank_date': row.date,
                            'status': 'UNIDENTIFIED_BANK_RECORD'})
        elif (row.supplier_id, row.amount) not in known:
            results.append({'bank_transaction_id': row.bank_transaction_id,
                            'supplier_id': row.supplier_id, 'bank_amount': row.amount,
                            'bank_date': row.date,
                            'status': 'AMOUNT_MISMATCH' if row.supplier_id in supplier_ids
                                      else 'SUPPLIER_RECORD_MISSING'})
    return results

def get_transactions_by_date(start_date, end_date):
    df = get_transactions()
    dates = pd.to_datetime(df['date'], errors='raise')
    start, end = pd.to_datetime(start_date), pd.to_datetime(end_date)
    result = df.loc[dates.between(start, end)].copy()
    result['date'] = dates.loc[result.index].dt.strftime('%Y-%m-%d')
    return result

def export_transactions_to_excel(start_date, end_date, output_path='transaction_history.xlsx'):
    get_transactions_by_date(start_date, end_date).to_excel(output_path, index=False)
    return output_path

def get_investigation_evidence(transaction_id):
    """Build bounded evidence; supplier aggregates never establish payment linkage."""
    transactions = get_transactions()
    selected = transactions.loc[transactions['transaction_id'].eq(transaction_id)]
    if selected.empty:
        return None
    if len(selected) != 1:
        raise ValueError("Transaction ID is not unique")
    transaction = selected.astype(object).where(pd.notna(selected), None).iloc[0].to_dict()
    amount = float(transaction['amount'])
    if not pd.notna(amount) or abs(amount) == float('inf'):
        raise ValueError("Invalid transaction amount")
    flagged = str(transaction['type']).lower() == 'expense' and amount > 200000
    evidence = {
        'transaction': transaction,
        'direct_evidence': {'large_expense_threshold': 200000,
                            'large_expense_threshold_exceeded': flagged},
        'supplier': None,
        'supplier_context': None,
        'limitations': 'Supplier aggregates are context only. No transaction-to-invoice or '
                      'transaction-to-bank linkage is established. Amount equality is not proof of payment. '
                      'No invoice reconciliation was performed for this investigation. '
                      'Anomalies are not proof of fraud.'
    }
    supplier_id = transaction.get('supplier_id')
    if not supplier_id:
        return evidence
    supplier = get_supplier(supplier_id)
    evidence['supplier'] = (supplier.astype(object).where(pd.notna(supplier), None)
                            .iloc[0].to_dict()) if not supplier.empty else None
    history = transactions.loc[transactions['supplier_id'].eq(supplier_id)]
    invoices = get_supplier_invoices(supplier_id)
    bank = get_supplier_bank_transactions(supplier_id)

    def summary(frame):
        amounts = frame['amount']
        return {'count': len(frame), 'total_amount': float(amounts.sum()),
                'mean_amount': float(amounts.mean()) if len(frame) else None,
                'max_amount': float(amounts.max()) if len(frame) else None}

    expenses = history.loc[history['type'].str.lower().eq('expense')]
    evidence['supplier_context'] = {
        'scope': 'All available records for this supplier, including the investigated transaction.',
        'transactions': summary(history), 'expenses': summary(expenses),
        'large_expense_count': int(expenses['amount'].gt(200000).sum()),
        'invoices': summary(invoices), 'bank_transactions': summary(bank),
        'same_amount_invoice_count': int(invoices['amount'].eq(amount).sum()),
        'same_amount_bank_record_count': int(bank['amount'].eq(amount).sum())
    }
    return evidence
