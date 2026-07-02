"""Jira Cloud 集成客户端 - 自动创建和管理待办事项"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class JiraClient:
    """
    Jira Cloud REST API 客户端

    职责:
    - 创建 Issue（从会议待办自动同步）
    - 查询 Issue 状态（用于跟踪待办完成情况）
    - 更新 Issue（添加评论等）

    API 文档: https://developer.atlassian.com/cloud/jira/platform/rest/v3/
    """

    def __init__(
        self,
        server: str | None = None,
        email: str | None = None,
        api_token: str | None = None,
        project_key: str = "MEET",
    ):
        self.server = server or os.getenv("JIRA_SERVER", "")
        self.email = email or os.getenv("JIRA_EMAIL", "")
        self.api_token = api_token or os.getenv("JIRA_API_TOKEN", "")
        self.project_key = project_key or os.getenv("JIRA_PROJECT_KEY", "MEET")
        self._jira = None
        self._enabled = bool(self.server and self.email and self.api_token)

    def _get_client(self):
        """懒加载 Jira 客户端"""
        if self._jira is None and self._enabled:
            from jira import JIRA
            self._jira = JIRA(
                server=self.server,
                basic_auth=(self.email, self.api_token),
            )
        return self._jira

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def create_issue(
        self,
        summary: str,
        description: str = "",
        assignee: str | None = None,
        due_date: str | None = None,
        priority: str = "Medium",
        issue_type: str = "Task",
        labels: list[str] | None = None,
    ) -> dict[str, str]:
        """
        创建 Jira Issue

        Args:
            summary: 任务标题
            description: 任务描述
            assignee: 负责人（Jira 用户名或邮箱）
            due_date: 截止日期 YYYY-MM-DD
            priority: 优先级 Low/Medium/High/Urgent
            issue_type: Issue 类型 Task/Bug/Story
            labels: 标签列表

        Returns:
            {"key": "MEET-42", "id": "10042", "url": "https://..."}
        """
        if not self._enabled:
            logger.warning("Jira integration not configured, skipping")
            return {"key": "DISABLED", "id": "", "url": ""}

        client = self._get_client()

        fields: dict[str, Any] = {
            "project": {"key": self.project_key},
            "summary": summary,
            "description": description or f"自动创建自会议助手系统\n\n{summary}",
            "issuetype": {"name": issue_type},
            "priority": {"name": priority},
        }

        if assignee:
            fields["assignee"] = {"name": assignee}
        if due_date:
            fields["duedate"] = due_date
        if labels:
            fields["labels"] = labels + ["meeting-auto"]
        else:
            fields["labels"] = ["meeting-auto"]

        issue = client.create_issue(fields=fields)
        result = {
            "key": issue.key,
            "id": str(issue.id),
            "url": f"{self.server}/browse/{issue.key}",
        }
        logger.info(f"Created Jira issue: {result['key']} - {summary}")
        return result

    def get_issue_status(self, issue_key: str) -> str:
        """查询 Issue 当前状态"""
        if not self._enabled:
            return "DISABLED"
        client = self._get_client()
        issue = client.issue(issue_key)
        return str(issue.fields.status)

    def add_comment(self, issue_key: str, comment: str) -> None:
        """为 Issue 添加评论"""
        if not self._enabled:
            return
        client = self._get_client()
        client.add_comment(issue_key, comment)
        logger.info(f"Added comment to {issue_key}")

    # 用户名映射（实际项目中从企业通讯录或配置获取）
    USER_MAPPING: dict[str, str] = {}

    def resolve_user(self, display_name: str) -> str | None:
        """将显示名映射为 Jira 用户名"""
        return self.USER_MAPPING.get(display_name)

    @staticmethod
    def map_priority(priority: str) -> str:
        """将系统优先级映射为 Jira 优先级"""
        mapping = {
            "low": "Low",
            "medium": "Medium",
            "high": "High",
            "urgent": "Highest",
        }
        return mapping.get(priority.lower(), "Medium")
