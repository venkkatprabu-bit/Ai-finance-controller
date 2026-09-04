from agents.financial_agent import ask_financial_agent


print("\n===== TEST 1: TRANSACTION HISTORY =====")

result = ask_financial_agent(
    "Show me transactions from August 1 to August 15"
)

print(result)


print("\n===== TEST 2: LARGE TRANSACTIONS =====")

result = ask_financial_agent(
    "Show me the large transactions"
)

print(result)


print("\n===== TEST 3: SUPPLIER INVOICES =====")

result = ask_financial_agent(
    "Show me the invoices for supplier S001"
)

print(result)


print("\n===== TEST 4: INVOICE MISMATCHES =====")

result = ask_financial_agent(
    "Which invoices have mismatches with the bank?"
)

print(result)