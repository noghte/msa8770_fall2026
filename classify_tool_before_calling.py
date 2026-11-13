"""Use Jev via OpenRouter to select which of eight ISBNs require metadata lookup."""

import asyncio
import os
from importlib import import_module

import httpx

from optimization_common import measure, read_rows, save_comparison

metadata = import_module("09_read_structured_metadata_directly")

JEV_MODEL = "typesafe/jev-1.13"
QUESTIONS = {"relevance": {
    "type": "choice",
    "instructions": "Classify the book using only its supplied `note` for a data/AI reading list.",
    "criteria": {
        "relevant": "Practical data analysis, data engineering, distributed data systems, or machine learning.",
        "irrelevant": "Other subjects, including fiction, politics, relationships, and fantasy.",
    },
}}


async def classify_note(http, meter, note):
    response = await http.post(
        "https://openrouter.ai/api/alpha/decisions",
        json={"model": JEV_MODEL, "state": {"note": note}, "questions": QUESTIONS},
    )
    response.raise_for_status()
    body = response.json()
    usage = body.get("usage") or {}
    meter.record(usage, usage.get("cost"))
    answer = body["answers"]["relevance"]
    if answer.get("type") != "choice" or answer.get("choice") not in QUESTIONS["relevance"]["criteria"]:
        raise ValueError("Jev returned an invalid relevance decision.")
    return answer["choice"] == "relevant"


def enriched_row(row, relevant, raw):
    fields = metadata.metadata_fields(raw) if relevant else {
        name: None for name in metadata.BookMetadata.model_fields
    }
    return row | {"relevant": relevant} | fields


async def before(rows, decisions, books_http, meter):
    results = []
    for row in rows:
        raw = await metadata.lookup_book(books_http, meter, row["isbn"])
        relevant = await classify_note(decisions, meter, row["note"])
        results.append(enriched_row(row, relevant, raw))
        print(f"Before {row['id']}: lookup=yes, relevant={relevant}")
    return results


async def after(rows, decisions, books_http, meter):
    results = []
    for row in rows:
        relevant = await classify_note(decisions, meter, row["note"])
        raw = await metadata.lookup_book(books_http, meter, row["isbn"]) if relevant else None
        results.append(enriched_row(row, relevant, raw))
        print(f"After  {row['id']}: lookup={relevant}, relevant={relevant}")
    return results


async def main():
    rows = read_rows("books.csv", ("id", "isbn", "note", "expected_relevant"))
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("Set OPENROUTER_API_KEY in .env.")
    print(f"Decision model: {JEV_MODEL}; cost reported by OpenRouter.")
    async with httpx.AsyncClient(timeout=30, headers={
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
    }) as decisions, httpx.AsyncClient(timeout=15, follow_redirects=True) as books_http:
        original, bm = await measure("Before", lambda m: before(rows, decisions, books_http, m))
        optimized, am = await measure("After", lambda m: after(rows, decisions, books_http, m))
    for label, results in (("Before", original), ("After", optimized)):
        matches = sum(str(r["relevant"]).lower() == r["expected_relevant"] for r in results)
        print(f"{label} relevance matches: {matches}/{len(results)}")
    print("Before/after decisions and metadata identical:", original == optimized)
    print(f"Expected model calls: {len(rows)} per run; lookups: {len(rows)} before, "
          f"{sum(row['relevant'] for row in optimized)} after.")
    save_comparison("classify_before_tool", rows, original, optimized, bm, am)
    print("Tool calls decrease; classification cost should remain approximately unchanged.")


if __name__ == "__main__":
    asyncio.run(main())