import json
import os
import time
from typing import List, Dict, Any

import openai
import tiktoken
import logging

logger = logging.getLogger(__name__)

MODEL = "o3-2025-06-13"
MODEL_CONTEXT_LIMIT = 8192
MAX_RESPONSE_TOKENS = 1000
# 75% of context window reserved for prompt minus response tokens
MAX_TOKENS_PER_REVIEW = int(0.75 * MODEL_CONTEXT_LIMIT) - MAX_RESPONSE_TOKENS


def build_review_prompt(plan_text: str) -> List[Dict[str, str]]:
    system = (
        "You are an expert strategy auditor. Review the plan provided by "
        "the user. Identify any factual or logical errors, overlooked risks, "
        "or additional opportunities. Reply ONLY in valid JSON with the "
        "structure {\"errors\":[],\"risks\":[],\"opportunities\":[]}."
    )
    user_msg = f"### PLAN START\n{plan_text}\n### PLAN END"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]


def count_tokens(messages: List[Dict[str, str]], model: str = MODEL) -> int:
    enc = tiktoken.encoding_for_model(model)
    tokens = 0
    for m in messages:
        tokens += 4  # per-message overhead
        tokens += len(enc.encode(m.get("content", "")))
    tokens += 2  # priming
    return tokens


def review_plan(
    plan_text: str,
    run_id: str,
    model: str = MODEL,
    max_response_tokens: int = MAX_RESPONSE_TOKENS,
    timeout: int = 30,
    retries: int = 3,
    dry_run: bool = False,
) -> Dict[str, Any] | None:
    prompt = build_review_prompt(plan_text)
    num_prompt_tokens = count_tokens(prompt, model)
    if num_prompt_tokens > MAX_TOKENS_PER_REVIEW:
        raise ValueError(
            f"Prompt is too long ({num_prompt_tokens} tokens) for review"
        )

    record = {
        "run_id": run_id,
        "timestamp": time.time(),
        "prompt_tokens": num_prompt_tokens,
        "model": model,
    }

    if dry_run:
        record["response"] = {}
        _write_review_record(run_id, record)
        return {}

    last_err = None
    for attempt in range(retries):
        try:
            resp = openai.chat.completions.create(
                model=model,
                messages=prompt,
                max_tokens=max_response_tokens,
                temperature=0.3,
                timeout=timeout,
                request_timeout=timeout,
            )
            content = resp.choices[0].message.content
            record["completion_tokens"] = resp.usage.completion_tokens
            record["response_raw"] = content
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                data = None
            record["response"] = data
            _write_review_record(run_id, record)
            return data
        except openai.OpenAIError as e:  # openai API errors
            logger.error("OpenAI API error: %s", e)
            last_err = e
            sleep = 2 ** attempt
            time.sleep(sleep)
    if last_err:
        record["error"] = str(last_err)
        _write_review_record(run_id, record)
    return None


def _write_review_record(run_id: str, record: Dict[str, Any]) -> None:
    os.makedirs("reviews", exist_ok=True)
    path = os.path.join("reviews", f"{run_id}.jsonl")
    with open(path, "a") as f:
        json.dump(record, f)
        f.write("\n")

