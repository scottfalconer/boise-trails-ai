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

def test_review_plan_dry_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    result = plan_review.review_plan("plan", run_id="run1", dry_run=True)
    assert result == {}
    record_path = tmp_path / "reviews" / "run1.jsonl"
    assert record_path.exists()
    rec = json.loads(record_path.read_text().splitlines()[0])
    assert rec["response"] == {}


def test_review_plan_prompt_too_long(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(plan_review, "MAX_TOKENS_PER_REVIEW", 1)
    with pytest.raises(ValueError):
        plan_review.review_plan("long plan", run_id="x", dry_run=False)


def test_review_plan_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(plan_review, "MAX_TOKENS_PER_REVIEW", 10000)
    def fail(*args, **kwargs):
        raise Exception("fail")

    monkeypatch.setattr(plan_review.openai.chat.completions, "create", fail)
    result = plan_review.review_plan("plan", run_id="err", retries=2, dry_run=False)
    assert result is None
