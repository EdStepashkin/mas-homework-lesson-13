"""
End-to-end LLM-as-a-Judge тест.

Критерій: Фінальний контент відповідає початковому брифу.
Повний pipeline від брифу до approved контенту.
"""

import json
import os
import pytest
import uuid
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langgraph.types import Command

from supervisor import pipeline_graph
from config import settings


# ── Metrics ──────────────────────────────────────────────────

answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini")

e2e_quality = GEval(
    name="E2E Content Quality",
    evaluation_steps=[
        "Check that the final content addresses the topic from the input brief",
        "Check that the content matches the expected output in terms of key themes",
        "Check that the content is well-structured (headers, paragraphs, formatting)",
        "Check that the content is substantive (not empty or placeholder text)",
        "Penalize if the content is completely off-topic from the brief",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model="gpt-4o-mini",
    threshold=0.6,
)


def run_pipeline(topic: str, channel: str = "blog", tone: str = "впевнений", word_count: int = 800) -> str:
    """Run the full pipeline until hitl_gate, auto-approve, and return final content."""
    thread_id = str(uuid.uuid4())[:8]
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.max_iterations,
    }

    initial_state = {
        "messages": [],
        "topic": topic,
        "target_audience": "Tech Leads",
        "channel": channel,
        "tone": tone,
        "word_count": word_count,
        "content_plan": "",
        "draft": "",
        "edit_feedback": "",
        "iteration": 0,
        "final_content": "",
    }

    try:
        # First invoke — will stop at HITL gate
        result = pipeline_graph.invoke(initial_state, config=config)

        # Check if stopped at interrupt
        state = pipeline_graph.get_state(config)
        if state.next:
            # Auto-approve the plan
            result = pipeline_graph.invoke(
                Command(resume={"action": "approve"}),
                config=config,
            )

        if isinstance(result, dict):
            return result.get("final_content", result.get("draft", "No content generated"))

        return "No content generated"
    except Exception as e:
        return f"Error: {e}"


# ── Golden Dataset ───────────────────────────────────────────

dataset_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
try:
    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
except Exception:
    golden_data = []

test_cases = []
for item in golden_data:
    test_cases.append((item["input"], item["expected_output"]))


@pytest.mark.parametrize("brief,expected", test_cases)
def test_e2e_golden_dataset(brief, expected):
    """E2E тест: повний run від брифу до approved контенту."""
    actual_output = run_pipeline(brief)

    test_case = LLMTestCase(
        input=brief,
        actual_output=actual_output,
        expected_output=expected,
    )

    assert_test(test_case, [e2e_quality, answer_relevancy])
