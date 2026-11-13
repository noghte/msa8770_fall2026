"""Eight ISBNs: ask a model to copy API fields versus read the JSON fields directly."""

# Standard-library tools: run asynchronous work and check ISBN format.
import asyncio
import re

# HTTP client for fetching book metadata from Open Library.
import httpx

# Shared helpers provide Pydantic validation, model calls, timing, and saved results.
from optimization_common import (
    Output, ROUTER_MODEL, ask, measure, read_rows, router_client, save_comparison,
)

# Instructions for the original workflow's model call: copy existing facts only.
COPY_TASK = (
    "Copy title, publish_date, and number_of_pages exactly from the supplied JSON object. "
    "Use null for absent fields. Preserve dates as strings. Do not infer or add information."
)


# Output extends Pydantic's BaseModel; this schema defines the three fields we keep.
# Each field accepts None, which represents a missing value (null in JSON).
class BookMetadata(Output):
    title: str | None
    publish_date: str | None
    number_of_pages: int | None


def metadata_fields(raw):
    # model_fields lists the schema's fields; raw.get(name) returns None if absent.
    # model_validate() checks their types; strict=True rejects conversions such as "100" to 100.
    # model_dump() converts the validated Pydantic object into a plain Python dict,
    # such as {"title": "Dune", "publish_date": None, "number_of_pages": 412}, not a JSON string.
    return BookMetadata.model_validate(
        {name: raw.get(name) for name in BookMetadata.model_fields}, strict=True,
    ).model_dump()


async def lookup_book(http, meter, isbn):
    # Reject malformed input before including it in an API URL.
    if not re.fullmatch(r"[0-9]{13}", isbn):
        raise ValueError("Expected a 13-digit ISBN.")
    # Calls in these examples are sequential; pace the public API below one request/second.
    await asyncio.sleep(1.05)
    # Count the lookup and await its response without blocking the event loop.
    meter.tool_calls += 1
    response = await http.get(f"https://openlibrary.org/isbn/{isbn}.json")
    # Stop on an HTTP error, then decode the response's JSON into Python values.
    response.raise_for_status()
    raw = response.json()
    # Require a JSON object and validate the selected fields before returning the full response.
    if not isinstance(raw, dict):
        raise ValueError("Expected an Open Library JSON object.")
    metadata_fields(raw)
    return raw


async def before(rows, client, http, meter):
    # Original workflow: fetch each book, then ask a model to copy the three fields.
    results = []
    for row in rows:
        raw = await lookup_book(http, meter, row["isbn"])
        fields = await ask(client, meter, COPY_TASK, raw, BookMetadata)
        # Check the model's answer against the API values so changed facts fail the comparison.
        if fields != metadata_fields(raw):
            raise ValueError(f"{row['id']}: model changed the source metadata.")
        # The dict union operator (|) adds metadata to the CSV row; right-hand values win.
        results.append(row | fields)
    return results


async def after(rows, http, meter):
    # Optimized workflow: keep the same lookups, but read validated fields without a model call.
    results = []
    for row in rows:
        raw = await lookup_book(http, meter, row["isbn"])
        results.append(row | metadata_fields(raw))
    return results


async def main():
    # Load the CSV rows and check that id, isbn, and note have values.
    rows = read_rows("books.csv", ("id", "isbn", "note"))
    print(f"Model: {ROUTER_MODEL}; live Open Library metadata; missing fields stay null.")
    # Open both clients for this block and close them automatically when it ends.
    async with router_client() as client, httpx.AsyncClient(timeout=15, follow_redirects=True) as http:
        # measure() returns results and metrics; each lambda passes its meter into the workflow.
        original, bm = await measure("Before", lambda m: before(rows, client, http, m))
        optimized, am = await measure("After", lambda m: after(rows, http, m))
    # Report whether the live metadata stayed the same, alongside expected call counts.
    print("Before/after metadata identical:", original == optimized)
    print(f"Expected model calls: {len(rows)} before, 0 after; lookups: {len(rows)} per run.")
    # Save both result sets as CSVs and their metrics as JSON for inspection.
    save_comparison("direct_metadata", rows, original, optimized, bm, am)
    print("Only the model-copying stage is removed; HTTP lookups still take time.")


# Start the event loop only when this file is run directly, not when imported.
if __name__ == "__main__":
    asyncio.run(main())

