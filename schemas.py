"""
Pydantic models for the Content Creation Pipeline.

ContentPlan  — output of Content Strategist
DraftContent — output of Writer
EditFeedback — output of Editor
"""

from typing import Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class ContentPlan(BaseModel):
    """Structured content plan produced by the Content Strategist."""
    outline: list[str] = Field(description="Content outline — ordered list of sections/points")
    keywords: list[str] = Field(description="SEO / topic keywords to use in the content")
    key_messages: list[str] = Field(description="Core messages the content must convey")
    target_audience: str = Field(description="Who this content is for")
    tone: str = Field(description="Desired tone of voice for the content")


class DraftContent(BaseModel):
    """Draft content produced by the Writer."""
    content: str = Field(description="Full text of the article/post in Markdown")
    word_count: int = Field(description="Number of words in the content")
    keywords_used: list[str] = Field(description="Keywords actually used in the content")


class EditFeedback(BaseModel):
    """Structured feedback produced by the Editor."""
    verdict: Literal["APPROVED", "REVISION_NEEDED"] = Field(
        description="Whether the content is approved or needs revision"
    )
    issues: list[str] = Field(description="Specific issues found in the content")
    tone_score: float = Field(description="Tone of voice adherence score (0.0–1.0)")
    accuracy_score: float = Field(description="Factual accuracy score (0.0–1.0)")
    structure_score: float = Field(description="Structure and readability score (0.0–1.0)")


# ─────────────────────────────────────────────
# LangGraph Pipeline State
# ─────────────────────────────────────────────
from typing import TypedDict

class PipelineState(TypedDict):
    """State for the LangGraph content creation pipeline."""
    messages: Annotated[list, add_messages]
    # Brief from user
    topic: str
    target_audience: str
    channel: str
    tone: str
    word_count: int
    # Pipeline data
    content_plan: str          # serialized ContentPlan
    draft: str                 # current draft content
    edit_feedback: str         # serialized EditFeedback
    iteration: int             # current Writer↔Editor iteration
    final_content: str         # approved final content
