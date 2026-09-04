import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from agents.anomaly_agent import investigate_transaction

from agents.tools import (
    transaction_history_tool,
    large_transaction_tool,
    supplier_invoices_tool,
    supplier_bank_transactions_tool,
    invoice_mismatch_tool,
    
)


# ==================================================
# LOAD ENVIRONMENT
# ==================================================

load_dotenv()


# ==================================================
# NVIDIA CLIENT
# ==================================================

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    timeout=60.0
)


# ==================================================
# AI TOOLS
# ==================================================

tools = [

    # --------------------------------------------------
    # TRANSACTION HISTORY
    # --------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "get_transactions_by_date",
            "description": (
                "Get financial transaction history between "
                "a start date and end date."
            ),
            "parameters": {
                "type": "object",
                "properties": {

                    "start_date": {
                        "type": "string",
                        "description": (
                            "Start date in YYYY-MM-DD format."
                        )
                    },

                    "end_date": {
                        "type": "string",
                        "description": (
                            "End date in YYYY-MM-DD format."
                        )
                    }

                },
                "required": [
                    "start_date",
                    "end_date"
                ]
            }
        }
    },


    # --------------------------------------------------
    # LARGE TRANSACTIONS
    # --------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "detect_large_transactions",
            "description": (
                "Find unusually large expense transactions "
                "from the financial transaction data."
            ),
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },


    # --------------------------------------------------
    # SUPPLIER INVOICES
    # --------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "get_supplier_invoices",
            "description": (
                "Get all invoices belonging to a supplier."
            ),
            "parameters": {
                "type": "object",
                "properties": {

                    "supplier_id": {
                        "type": "string",
                        "description": (
                            "Supplier ID such as S001."
                        )
                    }

                },
                "required": [
                    "supplier_id"
                ]
            }
        }
    },


    # --------------------------------------------------
    # SUPPLIER BANK TRANSACTIONS
    # --------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "get_supplier_bank_transactions",
            "description": (
                "Get bank transactions belonging to a supplier."
            ),
            "parameters": {
                "type": "object",
                "properties": {

                    "supplier_id": {
                        "type": "string",
                        "description": (
                            "Supplier ID such as S001."
                        )
                    }

                },
                "required": [
                    "supplier_id"
                ]
            }
        }
    },


    # --------------------------------------------------
    # INVOICE MISMATCHES
    # --------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "detect_invoice_mismatches",
            "description": (
                "Find invoice and bank transaction mismatches."
            ),
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },


    # --------------------------------------------------
    # INVESTIGATE TRANSACTION
    # --------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "investigate_transaction",
            "description": (
                "Investigate a specific transaction for anomalies, risk, "
                "and findings. Use this when the user asks to investigate, "
                "analyze, audit, examine, or assess a specific transaction. "
                "Extract the transaction ID (e.g., T001, T008) from the user's request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": (
                            "Transaction ID to investigate, e.g., T008"
                        )
                    }
                },
                "required": [
                    "transaction_id"
                ]
            }
        }
    }

]


# ==================================================
# FINANCIAL AGENT
# ==================================================

def ask_financial_agent(user_query):
    retrieved_transactions = []

    retrieved_start_date = None

    retrieved_end_date = None

    investigation_result = None

    messages = [

        {
            "role": "system",

            "content": """
You are an AI Financial Controller.

You help finance employees analyze financial data.

You have access to verified financial tools.

IMPORTANT RULES:

1. Use the appropriate tool whenever the
   user's question requires financial data.

2. Never invent transactions, invoices,
   suppliers, amounts or dates.

3. Always use the exact information returned
   by the financial tools.

4. All monetary values are in Indian Rupees (₹).

5. When the user asks for transaction history
   between dates, use get_transactions_by_date.

6. Convert natural-language dates into
   YYYY-MM-DD format.

Examples:

"August 1 to August 15"
→ 2026-08-01 to 2026-08-15

"August 8"
→ 2026-08-08 to 2026-08-08

"from August 1st to August 8th"
→ 2026-08-01 to 2026-08-08

7. When the user asks about unusually large
   transactions, use detect_large_transactions.

8. When the user asks about supplier invoices,
   use get_supplier_invoices.

9. When the user asks about supplier bank
   transactions, use get_supplier_bank_transactions.

10. When the user asks about invoice discrepancies,
    mismatches, or reconciliation problems,
    use detect_invoice_mismatches.

11. When the user asks to INVESTIGATE, ANALYZE,
    AUDIT, EXAMINE, or ASSESS a specific transaction,
    use investigate_transaction. Extract the
    transaction ID from the request (e.g., T008).

12. Do not say:
    "The user wants..."

13. Do not describe your internal reasoning.

14. Do not describe which tools you selected.

15. If the tool returns no records, clearly
    tell the user that no records were found.

16. Never claim that a transaction is fraudulent
    simply because it is unusual.

17. An anomaly is a warning signal and should
    be investigated further.

18. Keep responses concise and useful for
    a finance employee.

RESPONSE RULES:

- Always provide a useful financial answer.
- If the requested data was retrieved by a tool, base the answer strictly on that data.
- Never output random, repeated, corrupted, or meaningless text.
- Never output internal reasoning, tool-selection thoughts, or hidden analysis.
- If the request cannot be answered from the available tools/data, clearly say:
  "I don't have enough information in the available financial records to answer that."
- Keep responses concise and professional.
- Use tables when displaying multiple financial records.
- All monetary values must be in INR (₹).
"""
        },

        {
            "role": "user",
            "content": user_query
        }

    ]


    # ==================================================
    # AGENT LOOP
    # ==================================================

    while True:
        request_kwargs = {
            "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
            "max_tokens": 1000,
        }

        try:
            response = client.chat.completions.create(**request_kwargs)
        except Exception as e:
            # Do not log request content: it may contain sensitive financial data.
            print("===== FINANCIAL AGENT API ERROR =====")
            print("ERROR TYPE:", type(e).__name__)
            print("ERROR:", repr(e))
            print("MODEL:", request_kwargs["model"])
            print("MESSAGE COUNT:", len(messages))
            print("MESSAGE ROLES:", [
                message["role"] if isinstance(message, dict) else message.role
                for message in messages
            ])
            print("TOOL COUNT:", len(tools))
            print("===== END ERROR =====")
            raise


        message = response.choices[0].message


        # --------------------------------------------------
        # FINAL RESPONSE
        # --------------------------------------------------

        if not message.tool_calls:

            return {
            "answer": message.content,
            "transactions": retrieved_transactions,
            "start_date": retrieved_start_date,
            "end_date": retrieved_end_date,
            "investigation": investigation_result
        }


        # --------------------------------------------------
        # ADD AI MESSAGE
        # --------------------------------------------------

        messages.append(message)


        # --------------------------------------------------
        # VARIABLES FOR TRANSACTION HISTORY
        # --------------------------------------------------

        transactions = []

        start_date = None

        end_date = None


        # --------------------------------------------------
        # EXECUTE TOOL CALLS
        # --------------------------------------------------

        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )


            # ==============================================
            # TRANSACTION HISTORY
            # ==============================================

            if tool_name == "get_transactions_by_date":

                retrieved_start_date = arguments["start_date"]

                retrieved_end_date = arguments["end_date"]

                result = transaction_history_tool(
                    retrieved_start_date,
                    retrieved_end_date
                )

                retrieved_transactions = result


            # ==============================================
            # LARGE TRANSACTIONS
            # ==============================================

            elif tool_name == "detect_large_transactions":

                result = large_transaction_tool()


            # ==============================================
            # SUPPLIER INVOICES
            # ==============================================

            elif tool_name == "get_supplier_invoices":

                result = supplier_invoices_tool(
                    arguments["supplier_id"]
                )


            # ==============================================
            # SUPPLIER BANK TRANSACTIONS
            # ==============================================

            elif tool_name == "get_supplier_bank_transactions":

                result = supplier_bank_transactions_tool(
                    arguments["supplier_id"]
                )


            # ==============================================
            # INVOICE MISMATCHES
            # ==============================================

            elif tool_name == "detect_invoice_mismatches":

                result = invoice_mismatch_tool()


            # ==============================================
            # INVESTIGATE TRANSACTION
            # ==============================================

            elif tool_name == "investigate_transaction":

                result = investigate_transaction    (
                    arguments["transaction_id"]
                )

                investigation_result = result


            # ==============================================
            # UNKNOWN TOOL
            # ==============================================

            else:

                result = {
                    "error": "Unknown financial tool."
                }


            # --------------------------------------------------
            # SEND TOOL RESULT BACK TO AI
            # --------------------------------------------------

            messages.append({

                "role": "tool",

                "tool_call_id": tool_call.id,

                "content": json.dumps(
                    result,
                    default=str
                )

            })


        # --------------------------------------------------
        # IMPORTANT
        # --------------------------------------------------
        #
        # The loop now runs again.
        #
        # The AI sees the tool result and produces
        # the final natural-language response.
        #
        # --------------------------------------------------

        if transactions:

            # Store transaction information temporarily
            # so the final response can expose it.

            pass
