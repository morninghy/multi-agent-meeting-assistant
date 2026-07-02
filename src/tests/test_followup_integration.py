"""Follow-up Agent 集成测试 -- 验证 email 和 chroma 集成"""

from __future__ import annotations

from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from ..agents.followup_agent import FollowUpAgent
from ..models.schemas import (
    ActionResult,
    FollowUpResult,
    MeetingInsight,
    MeetingStatus,
    MeetingSummary,
    SentimentType,
)


def make_state() -> dict:
    """创建测试用的初始状态"""
    return {
        "meeting_id": "test-001",
        "status": MeetingStatus.PROCESSING,
        "summary": MeetingSummary(
            title="测试会议",
            participants=["alice@test.com", "bob@test.com"],
        ),
        "actions": ActionResult(meeting_id="test-001"),
        "insights": MeetingInsight(
            meeting_id="test-001",
            overall_sentiment=SentimentType.POSITIVE,
            efficiency_score=8.0,
        ),
        "errors": [],
    }


class TestFollowUpAgentEmail:
    """邮件发送集成测试"""

    @pytest.mark.asyncio
    async def test_email_sent_when_enabled(self):
        """EmailClient 启用时应发送邮件"""
        feishu = MagicMock()
        feishu.is_enabled = False

        email = MagicMock()
        email.is_enabled = True
        email.send_meeting_report = AsyncMock(return_value=True)

        chroma = MagicMock()
        chroma.is_enabled = False

        agent = FollowUpAgent(feishu, email, chroma)
        state = make_state()

        result_state = await agent.process(state)
        followup: FollowUpResult = result_state["followup"]

        assert followup.email_sent is True
        assert followup.email_recipients == ["alice@test.com", "bob@test.com"]
        email.send_meeting_report.assert_awaited_once_with(
            title="测试会议",
            participants=["alice@test.com", "bob@test.com"],
            summary_md=ANY,
            actions_md=ANY,
            insights_md=ANY,
        )

    @pytest.mark.asyncio
    async def test_email_skipped_when_disabled(self):
        """EmailClient 禁用时应跳过邮件"""
        feishu = MagicMock()
        feishu.is_enabled = False

        email = MagicMock()
        email.is_enabled = False

        chroma = MagicMock()
        chroma.is_enabled = False

        agent = FollowUpAgent(feishu, email, chroma)
        state = make_state()

        result_state = await agent.process(state)
        followup: FollowUpResult = result_state["followup"]

        assert followup.email_sent is False
        assert followup.email_recipients == []


class TestFollowUpAgentChroma:
    """向量存储集成测试"""

    @pytest.mark.asyncio
    async def test_chroma_stored_when_enabled(self):
        """ChromaClient 启用时应存储数据"""
        feishu = MagicMock()
        feishu.is_enabled = False

        email = MagicMock()
        email.is_enabled = False

        chroma = MagicMock()
        chroma.is_enabled = True
        chroma.store_meeting = AsyncMock(return_value=True)

        agent = FollowUpAgent(feishu, email, chroma)
        state = make_state()

        result_state = await agent.process(state)
        followup: FollowUpResult = result_state["followup"]

        assert followup.stored_in_vector_db is True
        chroma.store_meeting.assert_awaited_once_with(
            meeting_id="test-001",
            summary=ANY,
            actions=ANY,
            insights=ANY,
        )

    @pytest.mark.asyncio
    async def test_chroma_skipped_when_disabled(self):
        """ChromaClient 禁用时应跳过存储"""
        feishu = MagicMock()
        feishu.is_enabled = False

        email = MagicMock()
        email.is_enabled = False

        chroma = MagicMock()
        chroma.is_enabled = False

        agent = FollowUpAgent(feishu, email, chroma)
        state = make_state()

        result_state = await agent.process(state)
        followup: FollowUpResult = result_state["followup"]

        assert followup.stored_in_vector_db is False


class TestFollowUpAgentErrorPaths:
    """异常路径测试 -- 验证各集成步骤互不阻塞"""

    @pytest.mark.asyncio
    async def test_email_exception_does_not_block_chroma(self):
        """邮件发送异常不应阻塞向量存储"""
        feishu = MagicMock()
        feishu.is_enabled = False

        email = MagicMock()
        email.is_enabled = True
        email.send_meeting_report = AsyncMock(side_effect=RuntimeError("SMTP error"))

        chroma = MagicMock()
        chroma.is_enabled = True
        chroma.store_meeting = AsyncMock(return_value=True)

        agent = FollowUpAgent(feishu, email, chroma)
        state = make_state()

        result_state = await agent.process(state)
        followup: FollowUpResult = result_state["followup"]

        # 邮件失败但不应阻塞 chroma
        assert followup.email_sent is False
        assert followup.stored_in_vector_db is True
        chroma.store_meeting.assert_awaited_once()
        # 错误应被记录
        assert any("email" in e.lower() or "smtp" in e.lower() for e in result_state["errors"])
