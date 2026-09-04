import pandas as pd


DATA_PATH = "data/"

def get_transactions():
    return pd.read_csv(
        DATA_PATH + "transactions.csv"
    )


def get_invoices():
    return pd.read_csv(
        DATA_PATH + "invoices.csv"
    )


def get_bank_transactions():
    return pd.read_csv(
        DATA_PATH + "bank_transactions.csv"
    )


def get_suppliers():
    return pd.read_csv(
        DATA_PATH + "suppliers.csv"
    )


def get_customers():
    return pd.read_csv(
        DATA_PATH + "customers.csv"
    )

def get_transaction(transaction_id):

    transactions = get_transactions()

    result = transactions[
        transactions["transaction_id"] == transaction_id
    ]

    return result

def get_supplier(supplier_id):

    suppliers = get_suppliers()

    result = suppliers[
        suppliers["supplier_id"] == supplier_id
    ]

    return result


def get_supplier_transactions(supplier_id):

    transactions = get_transactions()

    return transactions[
        transactions["supplier_id"] == supplier_id
    ]



def detect_large_transactions(threshold=200000):

    transactions = get_transactions()

    return transactions[
        (transactions["type"].str.lower() == "expense")
        &
        (transactions["amount"] > threshold)
    ]

def get_unpaid_invoices():

    invoices = get_invoices()

    return invoices[
        invoices["status"].str.lower() == "unpaid"
    ]


def get_supplier_invoices(supplier_id):

    invoices = get_invoices()

    return invoices[
        invoices["supplier_id"] == supplier_id
    ]

def get_supplier_bank_transactions(supplier_id):

    bank = get_bank_transactions()

    return bank[
        bank["supplier_id"] == supplier_id
    ]

def get_overdue_customers():

    customers = get_customers()

    return customers[
        customers["status"].str.lower() == "overdue"
    ]

def get_expenses():

    transactions = get_transactions()

    return transactions[
        transactions["type"].str.lower() == "expense"
    ]


def get_income():

    transactions = get_transactions()

    return transactions[
        transactions["type"].str.lower() == "income"
    ]



def reconcile_invoices():

    invoices = get_invoices()

    bank = get_bank_transactions()

    results = []


    for _, invoice in invoices.iterrows():

        matches = bank[
            bank["supplier_id"]
            == invoice["supplier_id"]
        ].copy()



        if matches.empty:

            results.append({

                "invoice_id":
                    invoice["invoice_id"],

                "supplier_id":
                    invoice["supplier_id"],

                "invoice_amount":
                    invoice["amount"],

                "status":
                    "No bank transaction found"

            })

            continue


        matches["amount_difference"] = (

            matches["amount"]
            - invoice["amount"]

        ).abs()


        best_match = matches.loc[

            matches["amount_difference"].idxmin()

        ]


        difference = (

            best_match["amount"]
            - invoice["amount"]

        )

        if difference == 0:

            status = "MATCH"

        else:

            status = "MISMATCH"


        results.append({

            "invoice_id":
                invoice["invoice_id"],

            "supplier_id":
                invoice["supplier_id"],

            "invoice_amount":
                invoice["amount"],

            "bank_transaction_id":
                best_match[
                    "bank_transaction_id"
                ],

            "bank_amount":
                best_match["amount"],

            "difference":
                difference,

            "status":
                status

        })


    return results

def detect_invoice_mismatches():

    invoices = get_invoices()

    bank = get_bank_transactions()

    results = []


    for _, invoice in invoices.iterrows():

        matches = bank[
            bank["supplier_id"]
            == invoice["supplier_id"]
        ].copy()

        if matches.empty:

            results.append({

                "invoice_id":
                    invoice["invoice_id"],

                "supplier_id":
                    invoice["supplier_id"],

                "invoice_amount":
                    invoice["amount"],

                "status":
                    "NO_BANK_TRANSACTION"

            })

            continue


        matches["amount_difference"] = (

            matches["amount"]
            - invoice["amount"]

        ).abs()


        best_match = matches.loc[

            matches["amount_difference"].idxmin()

        ]


        difference = (

            best_match["amount"]
            - invoice["amount"]

        )


        if difference == 0:

            status = "MATCH"

        else:

            status = "MISMATCH"


        results.append({

            "invoice_id":
                invoice["invoice_id"],

            "supplier_id":
                invoice["supplier_id"],

            "invoice_amount":
                invoice["amount"],

            "bank_transaction_id":
                best_match[
                    "bank_transaction_id"
                ],

            "bank_amount":
                best_match["amount"],

            "difference":
                difference,

            "status":
                status

        })


    return results

def detect_missing_bank_records():

    transactions = get_transactions()
    bank = get_bank_transactions()

    results = []

    for _, transaction in transactions.iterrows():

        supplier_id = transaction["supplier_id"]

        if pd.isna(supplier_id):
            continue

        supplier_matches = bank[
            bank["supplier_id"] == supplier_id
        ]

        # No bank records for this supplier
        if supplier_matches.empty:

            results.append({
                "transaction_id":
                    transaction["transaction_id"],

                "supplier_id":
                    supplier_id,

                "transaction_amount":
                    transaction["amount"],

                "transaction_date":
                    transaction["date"],

                "status":
                    "BANK_RECORD_MISSING"
            })

            continue

        # Check exact amount
        exact_match = supplier_matches[
            supplier_matches["amount"]
            == transaction["amount"]
        ]

        if not exact_match.empty:
            continue

        # Supplier exists in bank records,
        # but no transaction with matching amount
        results.append({
            "transaction_id":
                transaction["transaction_id"],

            "supplier_id":
                supplier_id,

            "transaction_amount":
                transaction["amount"],

            "transaction_date":
                transaction["date"],

            "status":
                "BANK_RECORD_MISSING"
        })

    return results
def detect_missing_supplier_records():

    transactions = get_transactions()
    bank = get_bank_transactions()

    results = []

    for _, bank_transaction in bank.iterrows():

        supplier_id = bank_transaction["supplier_id"]

        # We cannot identify the supplier
        if pd.isna(supplier_id):

            results.append({
                "bank_transaction_id":
                    bank_transaction["bank_transaction_id"],

                "bank_amount":
                    bank_transaction["amount"],

                "bank_date":
                    bank_transaction["date"],

                "status":
                    "UNIDENTIFIED_BANK_RECORD"
            })

            continue

        supplier_matches = transactions[
            transactions["supplier_id"]
            == supplier_id
        ]

        # No transactions for this supplier
        if supplier_matches.empty:

            results.append({
                "bank_transaction_id":
                    bank_transaction["bank_transaction_id"],

                "supplier_id":
                    supplier_id,

                "bank_amount":
                    bank_transaction["amount"],

                "bank_date":
                    bank_transaction["date"],

                "status":
                    "SUPPLIER_RECORD_MISSING"
            })

            continue

        # Check exact amount
        exact_match = supplier_matches[
            supplier_matches["amount"]
            == bank_transaction["amount"]
        ]

        if not exact_match.empty:
            continue

        # Supplier exists, but amount doesn't match
        results.append({
            "bank_transaction_id":
                bank_transaction["bank_transaction_id"],

            "supplier_id":
                supplier_id,

            "bank_amount":
                bank_transaction["amount"],

            "bank_date":
                bank_transaction["date"],

            "status":
                "AMOUNT_MISMATCH"
        })

    return results

def get_transactions_by_date(start_date, end_date):
    transactions = get_transactions()

    transactions["date"] = pd.to_datetime(
        transactions["date"]
    )

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    result = transactions[
        (transactions["date"] >= start_date) &
        (transactions["date"] <= end_date)
    ].copy()

    # Convert date back to string so it can be
    # safely returned as JSON
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")

    return result
def export_transactions_to_excel(
    start_date,
    end_date,
    output_path="transaction_history.xlsx"
):
    transactions = get_transactions_by_date(
        start_date,
        end_date
    )

    transactions.to_excel(
        output_path,
        index=False
    )

    return output_path