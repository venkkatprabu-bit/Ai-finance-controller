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