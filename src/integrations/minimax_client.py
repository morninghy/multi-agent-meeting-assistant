"""MiniMax LLM 客户端 - 支持 M2.7 模型调用"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class MiniMaxClient:
    """
    MiniMax API 客户端

    支持 MiniMax M2.7 模型，兼容 OpenAI 接口格式。
    文档: https://platform.minimaxi.com/document/guides/chat-model/chat/api
    """

    BASE_URL = "https://api.minimax.chat/v1"

    def __init__(
        self,
        api_key: str | None = None,
        group_id: str | None = None,
        model: str = "abab6.5s-chat",
    ):
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY", "")
        self.group_id = group_id or os.getenv("MINIMAX_GROUP_ID", "")
        self.model = os.getenv("MINIMAX_MODEL", model)
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._client = httpx.AsyncClient(timeout=60.0, headers=headers)

    @retry(
        retry=retry_if_exception_type(httpx.RequestError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
    )
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> str:
        """
        调用 MiniMax 聊天接口

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大生成token数
            response_format: 输出格式约束 (如 {"type": "json_object"})

        Returns:
            模型生成的文本
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        url = f"{self.BASE_URL}/text/chatcompletion_v2"
        if self.group_id:
            url = f"{url}?GroupId={self.group_id}"

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = e.response.text[:500]
            raise RuntimeError(
                f"MiniMax HTTP {e.response.status_code}: {body}"
            ) from e

        data = response.json()
        base_resp = data.get("base_resp") or data.get("baseResponse") or {}
        status_code = base_resp.get("status_code") or base_resp.get("statusCode")
        status_msg = base_resp.get("status_msg") or base_resp.get("statusMsg")
        if status_code not in (None, 0) or status_msg:
            raise RuntimeError(
                f"MiniMax API error: code={status_code}, message={status_msg}, "
                f"model={self.model}, group_id_configured={bool(self.group_id)}"
            )

        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]

            message = choice.get("message")
            if isinstance(message, dict) and message.get("content"):
                return message["content"]

            messages = choice.get("messages")
            if isinstance(messages, list) and messages:
                for msg in reversed(messages):
                    if isinstance(msg, dict) and msg.get("text"):
                        return msg["text"]

            if choice.get("text"):
                return choice["text"]

        logger.error(f"MiniMax API unexpected response: {data}")
        raise RuntimeError(
            f"MiniMax unexpected response: keys={list(data.keys())}, "
            f"choices_type={type(choices).__name__}"
        )

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict:
        """调用聊天接口并解析 JSON 输出"""
        if not self.api_key:
            raise RuntimeError("MINIMAX_API_KEY 未配置，跳过 LLM 调用")

        text = await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            raise

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
