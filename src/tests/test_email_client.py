"""EmailClient 单元测试"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

from ..integrations.email_client import EmailClient


class TestEmailClientInit:
    """初始化和配置测试"""

    def test_disabled_when_no_host(self):
        """SMTP_HOST 为空时客户端应禁用"""
        with patch.dict(os.environ, {"SMTP_HOST": ""}, clear=False):
            client = EmailClient()
            assert client.is_enabled is False

    def test_enabled_when_host_set(self):
        """SMTP_HOST 非空时客户端应启用"""
        with patch.dict(os.environ, {
            "SMTP_HOST": "smtp.test.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "user@test.com",
            "SMTP_PASSWORD": "pass",
            "SMTP_FROM": "from@test.com",
        }, clear=False):
            client = EmailClient()
            assert client.is_enabled is True

    def test_custom_params_override_env(self):
        """构造函数参数应覆盖环境变量"""
        client = EmailClient(
            host="custom.smtp.com",
            port=465,
            username="custom@test.com",
            password="custom_pass",
            from_addr="custom_from@test.com",
        )
        assert client.is_enabled is True


class TestEmailClientSend:
    """发送邮件测试"""

    @pytest.mark.asyncio
    async def test_send_email_skips_when_disabled(self):
        """禁用状态下发送应返回 False"""
        with patch.dict(os.environ, {"SMTP_HOST": ""}, clear=False):
            client = EmailClient()
            result = await client.send_email(
                to=["test@test.com"],
                subject="Test",
                body="Hello",
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_send_email_skips_when_no_recipients(self):
        """收件人为空时应返回 False"""
        with patch.dict(os.environ, {
            "SMTP_HOST": "smtp.test.com",
            "SMTP_USER": "user@test.com",
            "SMTP_PASSWORD": "pass",
            "SMTP_FROM": "from@test.com",
        }, clear=False):
            client = EmailClient()
            result = await client.send_email(to=[], subject="Test", body="Hello")
            assert result is False


class TestEmailClientSendHappyPath:
    """Happy path 发送测试"""

    @pytest.mark.asyncio
    async def test_send_email_calls_smtp(self):
        """启用状态下应调用 aiosmtplib.send"""
        with patch.dict(os.environ, {
            "SMTP_HOST": "smtp.test.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "user@test.com",
            "SMTP_PASSWORD": "pass",
            "SMTP_FROM": "from@test.com",
        }, clear=False):
            client = EmailClient()
            with patch("src.integrations.email_client.aiosmtplib.send", new_callable=AsyncMock) as mock_send:
                result = await client.send_email(
                    to=["to@test.com"],
                    subject="Test Subject",
                    body="Test Body",
                )
                assert result is True
                mock_send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_meeting_report_composes_correctly(self):
        """会议报告应包含正确的主题和内容"""
        with patch.dict(os.environ, {
            "SMTP_HOST": "smtp.test.com",
            "SMTP_USER": "user@test.com",
            "SMTP_PASSWORD": "pass",
            "SMTP_FROM": "from@test.com",
        }, clear=False):
            client = EmailClient()
            with patch("src.integrations.email_client.aiosmtplib.send", new_callable=AsyncMock) as mock_send:
                result = await client.send_meeting_report(
                    title="周会",
                    participants=["a@test.com"],
                    summary_md="## 摘要内容",
                    actions_md="## 待办内容",
                    insights_md="## 洞察内容",
                )
                assert result is True
                # 验证 send 被调用
                mock_send.assert_awaited_once()
                # 验证邮件主题
                call_args = mock_send.call_args
                msg = call_args[0][0]
                assert "周会" in msg["Subject"]


class TestEmailClientMeetingReport:
    """会议报告邮件测试"""

    @pytest.mark.asyncio
    async def test_send_meeting_report_skips_when_disabled(self):
        """禁用状态下发送会议报告应返回 False"""
        with patch.dict(os.environ, {"SMTP_HOST": ""}, clear=False):
            client = EmailClient()
            result = await client.send_meeting_report(
                title="测试会议",
                participants=["a@test.com"],
                summary_md="## 摘要",
                actions_md="## 待办",
                insights_md="## 洞察",
            )
            assert result is False
