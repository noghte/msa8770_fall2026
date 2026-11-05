# Expense recommendation workflow

## Goal

We are building a small system that takes an expense, selects the policy that applies to its date, asks a model for a structured recommendation, and saves that recommendation for review. Then we test whether the system produced the right result and whether its explanation is supported.

The output is a **review item**, not an approved expense or a payment. Even an `eligible` recommendation remains pending approval.

The learning goals are to:

- Separate application responsibilities from model responsibilities.
- Understand how state moves through a LangGraph workflow.
- See how selecting the wrong evidence can produce a plausible but wrong answer.
- Evaluate the saved result, including both exact fields and the explanation.

This expense rule is deliberately simple. We use it because the correct answer is easy to check while we study context selection, model output, workflow execution, and evaluation.

## 2. What exists, and what remains

We already have the path from input to a saved review item. We need to make that path understandable and then demonstrate whether it works across cases. One successful example is not an evaluation.


## 3. Example

Use the first case, `feb-120`:

| Evidence | Value |
|---|---|
| Expense date | February 10, 2026 |
| Amount | 12,000 cents = $120 |
| Receipt verified | `true` |
| January policy | $150 limit; January 1 up to, but excluding, March 1 |
| March policy | $100 limit; March 1 onward |
| Approval status | `PENDING` |

The correct reference outcome is `eligible`, with `policy_id = "policy-jan"`.

The two variants change **only the policy-selection method**:

- `latest`: choose the policy with the greatest `valid_from`. This selects the March policy even for a February expense. It is an intentionally flawed baseline.
- `dated`: select policies satisfying `valid_from <= expense.date < valid_to`, with no upper bound when `valid_to` is `None`. Return a policy only when exactly one matches.

For this February expense, the March limit would make $120 appear ineligible. The model can follow the supplied evidence faithfully and still give the wrong business answer because the application supplied the wrong policy.

**Question:** If the model correctly applies the wrong policy, which component should we fix?  
**Answer:** The context/policy-selection code.

## 4. The flow

```text
cases.json
    |
    v
load_cases(): validate the fixtures
    |
    v
main(): choose the first case, model, and latest/dated variant
    |
    v
run_workflow(): build the graph, then invoke it
    |
    v
START
    |
    v
assemble_context       Python selects evidence
    |
    v
 draft_recommendation  One model request; validate the proposal
    |
    v
save_review            Python writes one local JSON review item
    |
    v
END
    |
    v
Return artifact path + request metadata + elapsed time
    |
    v
evaluate.py / Jev      Separate evaluation after generation
```

There are three nodes, one model request per successful invocation, and no loop or conditional graph edge in `workflow.py`. The `latest`/`dated` choice is an ordinary Python branch inside policy selection. Both variants follow the same graph edges.

LangGraph controls execution order and carries state. It does not choose the policy, guarantee a correct explanation, or approve the expense.

## 5. The minimum example

In `minimum.py`, the pattern was:

```text
{expense} -> draft node -> {expense, answer}
```

In `workflow.py`, the same pattern is expanded:

```text
{case, variant}
    -> add context
    -> add draft and response_meta
    -> add result
```

A node is a Python function that reads state and returns a dictionary of updates. In this graph, returning `{'context': ...}` adds or replaces `context`; it does not erase `case` and `variant`.


| Term | Meaning here |
|---|---|
| State | The application's working data across nodes. |
| Context | The selected evidence sent to the model; only part of state. |
| Draft | The model's structured recommendation. |
| Result | The artifact path and API metadata returned by the workflow; not a review decision. |

`WorkflowState(TypedDict, total=False)` describes state keys, allowing some to be absent initially. It is not runtime validation. The Pydantic models validate the case data and parsed recommendation.

## 6. The `workflow.py` in execution order

### A. `main()` chooses a run

Start at the bottom of the file, not at the imports.

`main()` reads `--variant`, opens the OpenAI client, loads the cases, and passes `load_cases()[0]` to `run_workflow()`.


### B. `run_workflow()` builds, invokes, and returns

```python
graph = build_workflow(client=client, model=model, output=output)
final_state = graph.invoke(
    {'case': case, 'variant': variant},
    config={'recursion_limit': 10},
)
```

- `build_workflow()` defines the nodes and edges and returns a compiled graph.
- `compile()` prepares the graph; it does not make the model request.
- `invoke()` runs the graph with the initial state.
- After `END`, `final_state` contains the accumulated state.
- `run_workflow()` returns `final_state['result']` plus elapsed time, rather than returning all of that state.

`recursion_limit` limits graph super-steps, not seconds or tokens. The API timeout and output-token ceiling are separate controls.

### C. `assemble_context()` chooses what the model sees

```python
return {'context': build_context(state['case'], state['variant'])}
```

`build_context()` calls `select_policy()` and returns:

```json
{
  "expense": {
    "date": "2026-02-10",
    "amount_cents": 12000,
    "receipt_verified": true,
    "note": "Illustrative local transport expense. Approval is pending."
  },
  "policy": {
    "id": "policy-jan",
    "valid_from": "2026-01-01",
    "valid_to": "2026-03-01",
    "limit_cents": 15000
  },
  "approval_status": "PENDING"
}
```

This is the `dated` context for `feb-120`. `model_dump(mode='json')` converts values such as Python dates into JSON-compatible representations.

The case also contains `expected_decision` and `expected_policy`. Those are evaluation references. They remain in the application's case object but are omitted from the model input. Sending the answer key would undermine the evaluation.

With zero or multiple date matches, `policy` is `null`. The workflow still proceeds to the model with instructions to return `needs_review`.

### D. `draft_recommendation()` calls `propose()`

`propose(client, model, context)` sends:

- `PROMPT` as task instructions.
- Serialized `context` as evidence.
- `Recommendation` as the structured-output schema.

`Recommendation` has three fields:

```json
{
  "decision": "eligible",
  "policy_id": "policy-jan",
  "explanation": "The verified $120 expense is within the $150 limit. Approval remains pending."
}
```

The function requires a completed, parsed response; a nonempty explanation of at most 2,000 characters; and a policy ID matching the supplied context. When no policy is supplied, it also requires `needs_review`.

**What these checks do not prove:** When a policy exists, the function does not independently recompute eligibility, verify every factual claim, or enforce pending-approval wording. A schema-valid answer can still be wrong. Those gaps motivate evaluation.

`propose()` returns both the parsed draft and the API response. The node converts the draft to a dictionary and extracts response ID, requested/served model, and token usage into `response_meta`. It does not put the complete SDK response object into shared state.

### E. `save_review()` creates the handoff record

The node builds a document containing:

- Case ID, draft, and selected context.
- `status = "WAITING_FOR_REVIEW"`.
- The illustrative reviewer and local destination labels.
- SHA-256 fingerprints of the draft and context.
- The API response ID.

`write_json()` writes a temporary file in the destination directory and replaces the destination with it. This avoids exposing a partially written destination file during normal operation.

The output filename includes a fresh UUID, so separate runs create separate review items. `result['artifact']` points to that saved file.

**A saved review item is the endpoint of this graph.** No reviewer is contacted, no human response is awaited, and no payment occurs. 

Fingerprints let the grader detect content/hash mismatches. They are not signatures, access controls, or proof of approval.

### F. Why are the node functions nested?

`assemble_context`, `draft_recommendation`, and `save_review` are defined inside `build_workflow()` so they can use its `client`, `model`, and `output` values. This is a Python closure.

LangGraph calls each node with state. The closure supplies configuration without putting the client or API key in shared state.

## 7. Trace the state

| Point in execution | Keys present | Newly produced information |
|---|---|---|
| Before first node | `case`, `variant` | Validated case and selector choice |
| After `assemble_context` | Previous keys + `context` | Expense, selected policy or `null`, pending status |
| After `draft_recommendation` | Previous keys + `draft`, `response_meta` | Recommendation and API metadata |
| After `save_review` | Previous keys + `result` | Saved artifact path and API metadata |


1. **Does the model see every state key?** No. The request sends `context` and the prompt.
2. **Does `return {'draft': ...}` delete the input?** No. It updates those keys in this graph.
3. **Where is the external model call?** Inside `propose()`, called by `draft_recommendation`.
4. **Where is the file written?** Inside `save_review()`, through `write_json()`.
5. **What happens if proposal validation raises an error?** Execution stops before `save_review`; that attempt does not create a successful review artifact.
6. **Can this graph resume after a process restart?** No. It has no checkpointer. The saved JSON file is an output artifact, not a resumable graph checkpoint.

## 8. Run and Evaluate

