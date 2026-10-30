"""
This script creates a simple LangGraph workflow, with a node that 
sends an expense to OpenAI
"""

import os
from openai import OpenAI
from langgraph.graph import START, END, StateGraph
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

# shared state: carry information between nodes in langgraph
# example:
# {
# expense: "$120 expense with a verified receipt",
# answer: "Eligible under the $150 limit."
# }
class DraftState(TypedDict, total=False):
    expense: str
    answer: str

def build_workflow(client, model):
    def draft(state: DraftState):
        expense = ""
        if "expense" in state:
            expense = state["expense"]
        response = client.responses.create(
            model=model,
            instructions="",
            input=expense,
            max_output_tokens=2000, # including reasoning tokens
            store=False # Do not retain this response for later API calls
        )
        if response.status != "completed" or not response.output_text.strip():
            raise ValueError("No output available")
        return {"answer": response.output_text}
    
    builder = StateGraph(DraftState)
    builder.add_node('draft', draft)
    builder.add_edge(START, 'draft')
    builder.add_edge('draft', END)
    return builder.compile()

def main():
    model_name = os.getenv("OPEN_AI_MODEL", "gpt-5.4-mini")
    with OpenAI(
                api_key=os.getenv("OPEN_AI_KEY"),
                timeout=60, # HTTP timeout in seconds
                max_retries=0 # by default it is 2; the value 0 is fail fast so errors stay visible
                ) as client:
        graph = build_workflow(client,model_name)
        result = graph.invoke(
            {"expense": "POLICY: up to $150 with a receipt. EXPENSE: $120; receipt verified; approval pending."},
            config={'recursion_limit': 5} # cap graph steps
            )
        print(result["answer"])
if __name__ == "__main__":
    main()