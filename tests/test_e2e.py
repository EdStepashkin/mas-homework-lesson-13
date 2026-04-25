import json
import pytest
import uuid
import os
from deepeval import evaluate, assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from supervisor import supervisor_graph
from config import settings

answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini")

correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradict 'expected output'",
        "Penalize omission of critical details",
        "Different wording of the same concept is acceptable",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model="gpt-4o-mini",
    threshold=0.6,
)

def run_supervisor(user_input: str) -> str:
    """Helper to run supervisor graph and extract generated report."""
    thread_id = str(uuid.uuid4())[:8]
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.max_iterations,
    }
    
    # Run the graph until it hits Human-in-the-loop (save_report)
    try:
        # stream_mode=\"updates\" is optional, we just invoke it until interrupt
        for chunk in supervisor_graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config
        ):
            pass
            
        # Graph paused, get state
        state = supervisor_graph.get_state(config)
        messages = state.values.get("messages", [])
        
        # Extract content from save_report tool call
        report_content = "Failed to generate report"
        if messages:
            for msg in reversed(messages):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for call in msg.tool_calls:
                        if call["name"] == "save_report":
                            report_content = call["args"].get("content", "Empty Report")
                            return report_content
                            
        return report_content
    except Exception as e:
        return f"Error running graph: {e}"

# Load Golden Dataset
dataset_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
try:
    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
except Exception:
    golden_data = []

test_cases = []
for item in golden_data:
    test_case = LLMTestCase(
        input=item["input"],
        actual_output="This will be filled dynamically in test.",
        expected_output=item["expected_output"]
    )
    test_cases.append((item["input"], test_case))

@pytest.mark.parametrize("query,case", test_cases)
def test_golden_dataset(query, case):
    # Run the E2E simulation to get actual system output
    actual_output = run_supervisor(query)
    
    # Update test case
    case.actual_output = actual_output
    
    # Evaluate
    assert_test(case, [correctness, answer_relevancy])
