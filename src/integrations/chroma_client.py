"""ChromaDB 向量存储客户端"""

from __future__ import annotations

import os
from typing import Any

import chromadb
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from ..models.schemas import (
    ActionResult,
    MeetingInsight,
    MeetingSummary,
    TranscriptSegment,
)


class ChromaClient:
    """
    ChromaDB 向量存储客户端

    职责:
    - 将会议数据向量化存入 ChromaDB
    - 存储转写文本（按段落分 chunk）

    配置（环境变量）:
    - CHROMA_HOST: 远程服务地址（为空则使用本地持久化）
    - CHROMA_PORT: 远程服务端口
    - CHROMA_COLLECTION: Collection 名称
    - CHROMA_PERSIST_DIR: 本地持久化目录
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
        persist_dir: str | None = None,
    ):
        self._host = host or os.getenv("CHROMA_HOST", "")
        self._port = port or int(os.getenv("CHROMA_PORT", "8000"))
        self._collection_name = collection_name or os.getenv("CHROMA_COLLECTION") or "meetings"
        self._persist_dir = persist_dir or os.getenv(
            "CHROMA_PERSIST_DIR", "./data/chroma"
        )
        self._enabled = True
        self._client = None
        self._collection = None

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def _get_or_create_client(self):
        """延迟初始化 ChromaDB 客户端"""
        if self._client is not None:
            return self._client

        if self._host:
            self._client = chromadb.HttpClient(
                host=self._host, port=self._port
            )
            logger.info(f"ChromaDB HttpClient connected: {self._host}:{self._port}")
        else:
            os.makedirs(self._persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            logger.info(f"ChromaDB PersistentClient: {self._persist_dir}")

        return self._client

    def _get_or_create_collection(self):
        """获取或创建 collection"""
        if self._collection is not None:
            return self._collection

        client = self._get_or_create_client()
        self._collection = client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB collection ready: {self._collection_name}")
        return self._collection

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def store_meeting(
        self,
        meeting_id: str,
        summary: MeetingSummary | None = None,
        actions: ActionResult | None = None,
        insights: MeetingInsight | None = None,
    ) -> bool:
        """
        将会议数据向量化存入 ChromaDB

        Args:
            meeting_id: 会议ID
            summary: 会议摘要
            actions: 待办事项
            insights: 会议洞察

        Returns:
            是否存储成功
        """
        if not self._enabled:
            logger.warning("ChromaDB not enabled, skipping store")
            return False

        try:
            collection = self._get_or_create_collection()
            doc_id = f"meeting_{meeting_id}"

            # 拼接文档内容
            parts = []
            if summary:
                parts.append(f"标题: {summary.title}")
                if summary.participants:
                    parts.append(f"参会人: {', '.join(summary.participants)}")
                for topic in summary.topics:
                    parts.append(f"议题: {topic.title}")
                    parts.extend(topic.discussion_points)
                    if topic.conclusion:
                        parts.append(f"结论: {topic.conclusion}")
                parts.extend(summary.decisions)
                parts.extend(summary.next_steps)

            if actions:
                for item in actions.action_items:
                    parts.append(f"待办: {item.assignee} - {item.task}")

            if insights:
                parts.append(f"整体氛围: {insights.overall_sentiment.value}")
                parts.append(f"效率评分: {insights.efficiency_score}")
                parts.extend(insights.keywords)

            document = "\n".join(parts) if parts else f"会议 {meeting_id}"

            # 构建 metadata
            metadata: dict[str, Any] = {"meeting_id": meeting_id}
            if summary:
                metadata["title"] = summary.title
                metadata["date"] = summary.date
                metadata["participants"] = ", ".join(summary.participants)

            collection.upsert(ids=[doc_id], documents=[document], metadatas=[metadata])
            logger.info(f"Meeting stored in ChromaDB: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"ChromaDB store error: {e}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def store_transcript(
        self,
        meeting_id: str,
        transcript_text: str,
        segments: list[TranscriptSegment] | None = None,
    ) -> bool:
        """
        存储转写文本（按段落分 chunk）

        Args:
            meeting_id: 会议ID
            transcript_text: 完整转写文本
            segments: 转写段落列表

        Returns:
            是否存储成功
        """
        if not self._enabled:
            logger.warning("ChromaDB not enabled, skipping store")
            return False

        try:
            collection = self._get_or_create_collection()

            if segments:
                ids = [
                    f"transcript_{meeting_id}_{i}"
                    for i in range(len(segments))
                ]
                documents = [seg.text for seg in segments]
                metadatas = [
                    {
                        "meeting_id": meeting_id,
                        "speaker": seg.speaker,
                        "start": seg.start,
                        "end": seg.end,
                    }
                    for seg in segments
                ]
                collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            else:
                doc_id = f"transcript_{meeting_id}"
                collection.upsert(
                    ids=[doc_id],
                    documents=[transcript_text],
                    metadatas=[{"meeting_id": meeting_id}],
                )

            logger.info(f"Transcript stored in ChromaDB: {meeting_id}")
            return True

        except Exception as e:
            logger.error(f"ChromaDB store transcript error: {e}")
            return False

    async def close(self) -> None:
        """关闭连接"""
        self._client = None
        self._collection = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
