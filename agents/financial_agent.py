"""Financial assistant with bounded tool calls and safe, compact tool responses."""
import json
import logging
import os
import re

import pandas as pd
from datetime import date
from decimal import Decimal

from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, APIStatusError

from agents.anomaly_agent import investigate_transaction
from agents.tools import (
    transaction_history_tool, large_transaction_tool,
    supplier_invoices_tool, supplier_bank_transactions_tool,
    invoice_mismatch_tool,
)

load_dotenv()
logger = logging.getLogger(__name__)
MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
MAX_TOOL_ROUNDS = 3
MAX_TOOL_ROWS_FOR_LLM = 20
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    timeout=60.0,
    max_retries=1,
)


def _tool(name, description, properties=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {},
                       "required": required or [], "additionalProperties": False}
    }}


def _string_arg(description):
    return {"type": "string", "description": description}


tools = [
    _tool("get_transactions_by_date", "Retrieve ALL transactions within inclusive dates, including a single day (equal start and end). Always use this for transaction history by date; dataset dates are independent of today.",
          {"start_date": _string_arg("Start date YYYY-MM-DD"),
           "end_date": _string_arg("End date YYYY-MM-DD")}, ["start_date", "end_date"]),
    _tool("detect_large_transactions", "Find unusually large expense transactions."),
    _tool("get_supplier_invoices", "Retrieve invoices for a supplier.",
          {"supplier_id": _string_arg("Supplier ID, e.g. S001")}, ["supplier_id"]),
    _tool("get_supplier_bank_transactions", "Retrieve bank transactions for a supplier.",
          {"supplier_id": _string_arg("Supplier ID, e.g. S001")}, ["supplier_id"]),
    _tool("detect_invoice_mismatches", "Find invoice/bank reconciliation mismatches and missing matches."),
    _tool("investigate_transaction", "Investigate, audit, or assess a transaction by ID.",
          {"transaction_id": _string_arg("Transaction ID, e.g. T008")}, ["transaction_id"]),
]

SYSTEM_PROMPT = """You are an AI Financial Controller working with verified financial tools.
Use a tool for questions requiring financial records. For transaction history by date, always call get_transactions_by_date before answering. Never infer dataset coverage from today or describe requested dates as unavailable without checking the tool. Tool errors are not empty results. Never invent records, amounts, dates, or fraud conclusions.
Amounts are INR (₹). Dates for tools are YYYY-MM-DD; infer a year only when unambiguous,
otherwise ask for clarification. Never reveal internal reasoning or tool selection.
Tool results may contain a SAMPLE rather than all rows: state the total count and never imply
that a sample is exhaustive. If no records are available, say so. Be concise and professional.
Use tables for small sets of records. If data is insufficient, say so. Anomalies are not proof of fraud.
"""


def _validate_arguments(name, args):
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be an object")
    expected = {
        "get_transactions_by_date": ("start_date", "end_date"),
        "get_supplier_invoices": ("supplier_id",),
        "get_supplier_bank_transactions": ("supplier_id",),
        "investigate_transaction": ("transaction_id",),
        "detect_large_transactions": (),
        "detect_invoice_mismatches": (),
    }
    if name not in expected:
        raise ValueError("Unknown financial tool")
    if set(args) != set(expected[name]):
        raise ValueError("Invalid or missing tool arguments")
    for key in expected[name]:
        if not isinstance(args[key], str) or not args[key].strip():
            raise ValueError(f"Invalid {key}")
    if name == "get_transactions_by_date":
        if any(not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args[key]) for key in expected[name]):
            raise ValueError("Dates must use YYYY-MM-DD")
        start, end = date.fromisoformat(args["start_date"]), date.fromisoformat(args["end_date"])
        if start > end:
            raise ValueError("Start date must not be after end date")
    return args


def _jsonable(value):
    """Normalize common pandas/numpy objects for JSON and UI use."""
    if hasattr(value, "to_dict") and hasattr(value, "columns"):
        value = value.astype(object).where(pd.notna(value), None).to_dict(orient="records")
    elif hasattr(value, "to_dict") and not isinstance(value, dict):
             value = value.to_dict()
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, list):
            return [clean(val) for val in item]
        return None if pd.isna(item) else item
    return json.loads(json.dumps(clean(value), default=str, allow_nan=False))


def _compact_for_llm(result):
    """Preserve full result in Python; send bounded sample to LLM."""
    normalized = _jsonable(result)
    if isinstance(normalized, list):
        return {"total_records": len(normalized), "total_amount": str(sum((Decimal(str(row["amount"])) for row in normalized if isinstance(row, dict) and row.get("amount") is not None), Decimal(0))), "sample_records": normalized[:MAX_TOOL_ROWS_FOR_LLM],
                "sample_only": len(normalized) > MAX_TOOL_ROWS_FOR_LLM}
    if isinstance(normalized, dict):
        compact = {}
        for key, value in normalized.items():
            if isinstance(value, list) and len(value) > MAX_TOOL_ROWS_FOR_LLM:
                compact[key] = {"total_records": len(value),
                                "sample_records": value[:MAX_TOOL_ROWS_FOR_LLM], "sample_only": True}
            else:
                compact[key] = value
        return compact
    return normalized


def _response(answer, transactions=None, start_date=None, end_date=None, investigation=None):
    return {"answer": answer, "transactions": transactions if transactions is not None else [],
            "start_date": start_date, "end_date": end_date, "investigation": investigation}


def _explicit_history_dates(query):
    """Route only plain history requests with explicit, unambiguous dates."""
    if not re.search(r"\btransactions?\b", query, re.I):
        return None
    if re.search(r"\b(invoice|supplier|large|unusual|investigate|mismatch|expense|income)\b", query, re.I):
        return None
    month = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    natural = rf"{month}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?"
    tokens = re.findall(rf"\b(?:\d{{4}}-\d{{2}}-\d{{2}}|{natural})\b", query, re.I)
    if len(tokens) not in (1, 2) or not re.search(r"\b(from|between|on)\b", query, re.I):
        return None
    years = re.findall(r"\b\d{4}\b", " ".join(tokens))
    if not years:
        return None
    dates = []
    for token in tokens:
        token = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", token, flags=re.I)
        if not re.search(r"\b\d{4}\b", token):
            token += " " + years[-1]
        dates.append(pd.to_datetime(token, errors="raise").date().isoformat())
    return _validate_arguments("get_transactions_by_date", {"start_date": dates[0], "end_date": dates[-1]})


def _history_response(records, start, end):
    total = sum((Decimal(str(row["amount"])) for row in records), Decimal(0))
    answer = (f"Found {len(records)} transactions from {start} to {end} (inclusive). "
              f"Total recorded amount: INR {total:,.2f}.") if records else (
              f"No transactions found from {start} to {end} (inclusive).")
    logger.info("Transaction history retrieved rows=%d", len(records))
    return _response(answer, records, start, end)


def ask_financial_agent(user_query):
    if not isinstance(user_query, str) or not user_query.strip():
        return _response("Please enter a financial question.")

    try:
        history_dates = _explicit_history_dates(user_query)
    except (ValueError, TypeError):
        return _response("Please provide valid dates with the start on or before the end.")
    if history_dates:
        try:
            records = _jsonable(transaction_history_tool(**history_dates))
            return _history_response(records, history_dates["start_date"], history_dates["end_date"])
        except Exception:
            logger.warning("Transaction history retrieval failed")
            return _response("Financial data could not be retrieved. Please try again.", **history_dates)

    transactions, start_date, end_date, investigation = [], None, None, None
    verified_tool = False
    tool_failed = False
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query.strip()}]
    for round_index in range(MAX_TOOL_ROUNDS + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL, messages=messages,
                tools=tools if round_index < MAX_TOOL_ROUNDS else None,
                tool_choice="auto" if round_index < MAX_TOOL_ROUNDS else None,
                temperature=0.2, max_tokens=1000,
            )
        except (APIConnectionError, APITimeoutError, RateLimitError, APIStatusError) as exc:
            logger.warning("Financial agent API failure: %s", type(exc).__name__)
            return _response("The AI service is temporarily unavailable. Please try again.",
                             transactions, start_date, end_date, investigation)
        except Exception:
            logger.warning("Unexpected financial agent failure")
            return _response("An unexpected error occurred. Please try again.",
                             transactions, start_date, end_date, investigation)

        if not response.choices:
            return _response("The AI service returned no answer. Please try again.",
                             transactions, start_date, end_date, investigation)
        message = response.choices[0].message
        logger.info("Financial agent round=%d tool_calls=%d", round_index, len(message.tool_calls or []))
        if not message.tool_calls:
            if tool_failed:
                return _response("Financial data could not be retrieved reliably. Please check your request and try again.",
                                 transactions, start_date, end_date, investigation)
            if not verified_tool:
                return _response("No financial records were checked. Please provide the dates (including year), supplier ID, or transaction ID needed for your request.")
            return _response((message.content or "I couldn't generate an answer. Please try again.").strip(),
                             transactions, start_date, end_date, investigation)

        messages.append(message)
        for call in message.tool_calls:
            name = call.function.name
            try:
                args = _validate_arguments(name, json.loads(call.function.arguments or "{}"))
                if name == "get_transactions_by_date":
                    start_date, end_date = args["start_date"], args["end_date"]
                    result = transaction_history_tool(start_date, end_date)
                    transactions = _jsonable(result)
                    return _history_response(transactions, start_date, end_date)
                elif name == "detect_large_transactions":
                    result = large_transaction_tool()
                elif name == "get_supplier_invoices":
                    result = supplier_invoices_tool(args["supplier_id"])
                elif name == "get_supplier_bank_transactions":
                    result = supplier_bank_transactions_tool(args["supplier_id"])
                elif name == "detect_invoice_mismatches":
                    result = invoice_mismatch_tool()
                else:
                    result = investigate_transaction(args["transaction_id"])
                    investigation = _jsonable(result)
                tool_payload = _compact_for_llm(result)
                verified_tool = True
                tool_failed = isinstance(tool_payload, dict) and "error" in tool_payload
            except (ValueError, KeyError, TypeError) as exc:
                tool_failed = True
                logger.info("Invalid financial tool request: %s", name if name in {t["function"]["name"] for t in tools} else "unknown")
                tool_payload = {"error": "Invalid tool request. Check the tool schema and provide valid arguments."}
            except Exception:
                logger.warning("Financial tool execution failed: %s", name)
                tool_failed = True
                tool_payload = {"error": "Financial data could not be retrieved."}
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(tool_payload, ensure_ascii=False, default=str)})

    return _response("I couldn't complete that request within the allowed processing steps.",
                     transactions, start_date, end_date, investigation)