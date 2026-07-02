"""ChromaClient 单元测试"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from ..integrations.chroma_client import ChromaClient
from ..models.schemas import (
    ActionResult,
    ActionItem,
    MeetingInsight,
    MeetingSummary,
    SentimentType,
    TopicSummary,
    TranscriptSegment,
)


class TestChromaClientInit:
    """初始化和配置测试"""

    def test_enabled_with_default_config(self):
        """默认配置（本地持久化）应启用"""
        with patch.dict(os.environ, {"CHROMA_HOST": ""}, clear=False):
            client = ChromaClient()
            assert client.is_enabled is True

    def test_enabled_with_custom_persist_dir(self, tmp_path):
        """自定义持久化目录应启用"""
        client = ChromaClient(persist_dir=str(tmp_path / "chroma"))
        assert client.is_enabled is True

    def test_collection_name_from_env(self):
        """collection 名称应从环境变量读取"""
        with patch.dict(os.environ, {"CHROMA_COLLECTION": "test_collection"}, clear=False):
            client = ChromaClient()
            assert client._collection_name == "test_collection"

    def test_default_collection_name(self):
        """默认 collection 名称为 meetings"""
        with patch.dict(os.environ, {"CHROMA_COLLECTION": ""}, clear=False):
            client = ChromaClient()
            assert client._collection_name == "meetings"


class TestChromaClientStoreMeeting:
    """存储会议数据测试"""

    def test_store_meeting_with_summary(self, tmp_path):
        """存储包含摘要的会议数据"""
        client = ChromaClient(persist_dir=str(tmp_path / "chroma"))
        assert client.is_enabled is True

    def test_store_meeting_disabled(self, tmp_path):
        """禁用状态下存储应返回 False"""
        client = ChromaClient(persist_dir=str(tmp_path / "chroma"))
        client._enabled = False
        assert client.is_enabled is False


class TestChromaClientStoreMeetingAsync:
    """异步存储会议数据测试"""

    @pytest.mark.asyncio
    async def test_store_meeting_returns_false_when_disabled(self, tmp_path):
        """禁用状态下存储应返回 False"""
        client = ChromaClient(persist_dir=str(tmp_path / "chroma"))
        client._enabled = False
        result = await client.store_meeting(meeting_id="test-001")
        assert result is False

    @pytest.mark.asyncio
    async def test_store_meeting_with_summary(self, tmp_path):
        """存储包含摘要的会议数据应返回 True"""
        client = ChromaClient(persist_dir=str(tmp_path / "chroma"))

        summary = MeetingSummary(
            title="周会",
            date="2024-01-01",
            participants=["Alice", "Bob"],
            topics=[
                TopicSummary(
                    title="项目进度",
                    discussion_points=["进度正常"],
                    conclusion="继续推进",
                )
            ],
            decisions=["按计划执行"],
            next_steps=["下周跟进"],
        )

        with patch.object(client, "_get_or_create_collection") as mock_get_col:
            mock_collection = MagicMock()
            mock_get_col.return_value = mock_collection
            result = await client.store_meeting(
                meeting_id="test-001", summary=summary
            )
            assert result is True
            mock_collection.upsert.assert_called_once()
            call_kwargs = mock_collection.upsert.call_args[1]
            doc = call_kwargs["documents"][0]
            meta = call_kwargs["metadatas"][0]
            assert "周会" in doc
            assert meta["meeting_id"] == "test-001"

    @pytest.mark.asyncio
    async def test_store_meeting_with_actions(self, tmp_path):
        """存储包含待办事项的会议数据应返回 True"""
        client = ChromaClient(persist_dir=str(tmp_path / "chroma"))

        actions = ActionResult(
            meeting_id="test-001",
            action_items=[
                ActionItem(assignee="Alice", task="完成报告"),
                ActionItem(assignee="Bob", task="安排会议"),
            ],
        )

        with patch.object(client, "_get_or_create_collection") as mock_get_col:
            mock_collection = MagicMock()
            mock_get_col.return_value = mock_collection
            result = await client.store_meeting(
                meeting_id="test-001", actions=actions
            )
            assert result is True
            mock_collection.upsert.assert_called_once()
            call_kwargs = mock_collection.upsert.call_args[1]
            doc = call_kwargs["documents"][0]
            meta = call_kwargs["metadatas"][0]
            assert "完成报告" in doc
            assert meta["meeting_id"] == "test-001"

    @pytest.mark.asyncio
    async def test_store_meeting_with_insights(self, tmp_path):
        """存储包含洞察的会议数据应返回 True"""
        client = ChromaClient(persist_dir=str(tmp_path / "chroma"))

        insights = MeetingInsight(
            meeting_id="test-001",
            overall_sentiment=SentimentType.POSITIVE,
            efficiency_score=0.85,
            keywords=["项目", "进度", "计划"],
        )

        with patch.object(client, "_get_or_create_collection") as mock_get_col:
            mock_collection = MagicMock()
            mock_get_col.return_value = mock_collection
            result = await client.store_meeting(
                meeting_id="test-001", insights=insights
            )
            assert result is True
            mock_collection.upsert.assert_called_once()
            call_kwargs = mock_collection.upsert.call_args[1]
            doc = call_kwargs["documents"][0]
            meta = call_kwargs["metadatas"][0]
            assert "进度" in doc
            assert meta["meeting_id"] == "test-001"

    @pytest.mark.asyncio
    async def test_store_meeting_handles_exception(self, tmp_path):
        """存储异常时应返回 False"""
        client = ChromaClient(persist_dir=str(tmp_path / "chroma"))

        with patch.object(client, "_get_or_create_collection") as mock_get_col:
            mock_get_col.side_effect = RuntimeError("ChromaDB error")
            result = await client.store_meeting(meeting_id="test-001")
            assert result is False


class TestChromaClientStoreTranscript:
    """存储转写文本测试"""

    @pytest.mark.asyncio
    async def test_store_transcript_returns_false_when_disabled(self, tmp_path):
        """禁用状态下存储转写应返回 False"""
        client = ChromaClient(persist_dir=str(tmp_path / "chroma"))
        client._enabled = False
        result = await client.store_transcript(
            meeting_id="test-001", transcript_text="hello"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_store_transcript_without_segments(self, tmp_path):
        """无段落时应存储完整文本"""
        client = ChromaClient(persist_dir=str(tmp_path / "chroma"))

        with patch.object(client, "_get_or_create_collection") as mock_get_col:
            mock_collection = MagicMock()
            mock_get_col.return_value = mock_collection
            result = await client.store_transcript(
                meeting_id="test-001", transcript_text="完整转写文本"
            )
            assert result is True
            mock_collection.upsert.assert_called_once()
            call_kwargs = mock_collection.upsert.call_args[1]
            assert call_kwargs["ids"] == ["transcript_test-001"]
            assert "完整转写文本" in call_kwargs["documents"][0]
            assert call_kwargs["metadatas"][0]["meeting_id"] == "test-001"

    @pytest.mark.asyncio
    async def test_store_transcript_with_segments(self, tmp_path):
        """有段落时应按段落分 chunk 存储"""
        client = ChromaClient(persist_dir=str(tmp_path / "chroma"))

        segments = [
            TranscriptSegment(speaker="Alice", text="你好", start=0.0, end=1.0),
            TranscriptSegment(speaker="Bob", text="你好啊", start=1.0, end=2.0),
        ]

        with patch.object(client, "_get_or_create_collection") as mock_get_col:
            mock_collection = MagicMock()
            mock_get_col.return_value = mock_collection
            result = await client.store_transcript(
                meeting_id="test-001",
                transcript_text="你好\n你好啊",
                segments=segments,
            )
            assert result is True
            mock_collection.upsert.assert_called_once()
            call_kwargs = mock_collection.upsert.call_args[1]
            assert len(call_kwargs["ids"]) == 2
            assert call_kwargs["documents"] == ["你好", "你好啊"]
            assert call_kwargs["metadatas"][0]["speaker"] == "Alice"
            assert call_kwargs["metadatas"][1]["speaker"] == "Bob"

    @pytest.mark.asyncio
    async def test_store_transcript_handles_exception(self, tmp_path):
        """存储异常时应返回 False"""
        client = ChromaClient(persist_dir=str(tmp_path / "chroma"))

        with patch.object(client, "_get_or_create_collection") as mock_get_col:
            mock_get_col.side_effect = RuntimeError("ChromaDB error")
            result = await client.store_transcript(
                meeting_id="test-001", transcript_text="text"
            )
            assert result is False


class TestChromaClientClose:
    """关闭连接测试"""

    @pytest.mark.asyncio
    async def test_close_clears_client_and_collection(self, tmp_path):
        """close 应清除 client 和 collection 引用"""
        client = ChromaClient(persist_dir=str(tmp_path / "chroma"))
        client._client = MagicMock()
        client._collection = MagicMock()

        await client.close()

        assert client._client is None
        assert client._collection is None
