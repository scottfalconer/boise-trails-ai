import json
from unittest import mock

import pytest

from trail_route_ai import plan_review


def test_build_review_prompt():
    prompt = plan_review.build_review_prompt("hello")
    assert prompt[0]["role"] == "system"
    assert "expert strategy auditor" in prompt[0]["content"]
    assert "PLAN START" in prompt[1]["content"]


def test_token_count():
    messages = plan_review.build_review_prompt("short plan")
    tokens = plan_review.count_tokens(messages)
    assert tokens > 0


def test_review_plan_parsing(monkeypatch):
    fake_resp = mock.Mock()
    fake_resp.choices = [mock.Mock(message=mock.Mock(content='{"errors":[],"risks":[],"opportunities":[]}'))]
    fake_resp.usage = mock.Mock(completion_tokens=10)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    with mock.patch("openai.chat.completions.create", return_value=fake_resp):
        data = plan_review.review_plan("plan", run_id="test", dry_run=False)
    assert data == {"errors": [], "risks": [], "opportunities": []}
