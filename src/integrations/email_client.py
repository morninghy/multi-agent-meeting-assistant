"""SMTP 邮件发送客户端"""

from __future__ import annotations

import asyncio
import os
from email.message import EmailMessage

import aiosmtplib
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class EmailClient:
    """
    SMTP 邮件发送客户端

    职责:
    - 通过 SMTP 发送邮件
    - 发送会议纪要报告邮件

    配置（环境变量）:
    - SMTP_HOST: SMTP 服务器地址（为空则禁用）
    - SMTP_PORT: 端口（587=TLS, 465=SSL）
    - SMTP_USER: 用户名
    - SMTP_PASSWORD: 密码
    - SMTP_FROM: 发件人地址
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        from_addr: str | None = None,
    ):
        self._host = host or os.getenv("SMTP_HOST", "")
        self._port = port or int(os.getenv("SMTP_PORT", "587"))
        self._username = username or os.getenv("SMTP_USER", "")
        self._password = password or os.getenv("SMTP_PASSWORD", "")
        self._from_addr = from_addr or os.getenv("SMTP_FROM", "")
        self._enabled = bool(self._host)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def send_email(
        self,
        to: list[str],
        subject: str,
        body: str,
        html: str | None = None,
    ) -> bool:
        """
        发送邮件

        Args:
            to: 收件人列表
            subject: 邮件主题
            body: 纯文本正文
            html: HTML 正文（可选）

        Returns:
            是否发送成功
        """
        if not self._enabled:
            logger.warning("SMTP not configured, skipping email")
            return False

        if not to:
            logger.warning("No recipients, skipping email")
            return False

        msg = EmailMessage()
        msg["From"] = self._from_addr
        msg["To"] = ", ".join(to)
        msg["Subject"] = subject
        msg.set_content(body)

        if html:
            msg.add_alternative(html, subtype="html")

        try:
            use_tls = self._port == 587
            await aiosmtplib.send(
                msg,
                hostname=self._host,
                port=self._port,
                username=self._username,
                password=self._password,
                start_tls=use_tls,
                use_tls=(not use_tls and self._port == 465),
            )
            logger.info(f"Email sent to {to}: {subject}")
            return True
        except (aiosmtplib.SMTPException, OSError, asyncio.TimeoutError) as e:
            logger.error(f"SMTP error: {e}")
            return False

    async def send_meeting_report(
        self,
        title: str,
        participants: list[str],
        summary_md: str,
        actions_md: str,
        insights_md: str,
    ) -> bool:
        """
        发送会议报告邮件

        Args:
            title: 会议标题
            participants: 参会人邮箱列表
            summary_md: 会议纪要 Markdown
            actions_md: 待办事项 Markdown
            insights_md: 会议洞察 Markdown

        Returns:
            是否发送成功
        """
        subject = f"会议纪要 | {title}"
        body = (
            f"会议主题: {title}\n\n"
            f"---\n\n"
            f"会议纪要\n{summary_md}\n\n"
            f"---\n\n"
            f"待办事项\n{actions_md}\n\n"
            f"---\n\n"
            f"会议洞察\n{insights_md}\n"
        )
        return await self.send_email(to=participants, subject=subject, body=body)

    async def close(self) -> None:
        """关闭连接（SMTP 无持久连接，此方法为接口一致性）"""
        pass
