"""
会议流水线共享数据结构（Pydantic + 少量类型别名）

供 Graph、各 Agent、WebSocket 序列化共用。
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field, computed_field


class MeetingStatus(str, Enum):
    """会议在流水线中的阶段（写入 state['status']，需可 JSON 序列化）"""

    PENDING = "pending"
    TRANSCRIBING = "transcribing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    speaker: str = "Unknown"
    text: str = ""
    start: float = 0.0
    end: float = 0.0
    confidence: float = 0.0


class TranscriptResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    meeting_id: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    language: str = "zh"
    duration_seconds: float = 0.0
    full_text: str = ""


class TopicSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    discussion_points: list[str] = Field(default_factory=list)
    participants: list[str] = Field(default_factory=list)
    conclusion: str = ""


class MeetingSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    date: str = ""
    participants: list[str] = Field(default_factory=list)
    topics: list[TopicSummary] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


class ActionItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: f"act-{uuid.uuid4().hex[:8]}")
    assignee: str = ""
    task: str = ""
    deadline: str = ""
    priority: Priority = Priority.MEDIUM
    status: str = "pending"
    context: str = ""
    jira_issue_key: str | None = None
    feishu_task_id: str | None = None


class ActionResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    meeting_id: str
    action_items: list[ActionItem] = Field(default_factory=list)
    sync_status: dict[str, str] = Field(default_factory=dict)


class SpeakerStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    speaker: str
    duration_s: float = 0.0
    percentage: float = 0.0
    word_count: int = 0
    segment_count: int = 0


class MeetingInsight(BaseModel):
    model_config = ConfigDict(extra="ignore")

    meeting_id: str
    overall_sentiment: SentimentType = SentimentType.NEUTRAL
    sentiment_score: float = 0.5
    speaker_stats: list[SpeakerStats] = Field(default_factory=list)
    efficiency_score: float = 0.0
    keywords: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    meeting_pace: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def sentiment(self) -> dict[str, Any]:
        """API 文档对齐：情绪分析对象（overall + score + details）"""
        details = [
            {
                "speaker": s.speaker,
                "sentiment": self.overall_sentiment.value,
                "score": self.sentiment_score,
            }
            for s in self.speaker_stats
        ]
        return {
            "overall": self.overall_sentiment.value,
            "score": self.sentiment_score,
            "details": details,
        }


class FollowUpResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    meeting_id: str
    summary_sent: bool = False
    recipients: list[str] = Field(default_factory=list)
    jira_issues_created: list[str] = Field(default_factory=list)
    feishu_tasks_created: list[str] = Field(default_factory=list)
    reminders_scheduled: int = 0
    report_url: str = ""
    email_sent: bool = False
    email_recipients: list[str] = Field(default_factory=list)
    stored_in_vector_db: bool = False


class MeetingState(TypedDict, total=False):
    """与 LangGraph GraphState 对齐的 state 形状（供类型标注）"""

    meeting_id: str
    status: str
    audio_data: bytes
    transcript: Any
    transcript_text: str
    summary: Any
    actions: Any
    insights: Any
    followup: Any
    errors: list[str]


def create_initial_state(
    meeting_id: str,
    audio_data: bytes = b"",
    title: str = "",
    participants: list[str] | None = None,
    language: str = "zh",
) -> dict[str, Any]:
    """LangGraph ainvoke 使用的初始 state"""
    return {
        "meeting_id": meeting_id,
        "audio_data": audio_data,
        "title": title,
        "participants": participants or [],
        "language": language,
        "errors": [],
    }
