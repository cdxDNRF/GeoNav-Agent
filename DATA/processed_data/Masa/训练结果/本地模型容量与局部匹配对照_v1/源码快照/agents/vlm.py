"""OpenAI-compatible 双图动作策略；不持有环境、任务清单或目标位置。"""
import base64
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
import re
from pathlib import Path
import time
from urllib.parse import urlparse

import httpx

from env.environment import Observation

PROJECT = Path(__file__).resolve().parents[2]
PROMPT_VERSION = "aerial-pair-four-actions-v1"
RANKED_PROMPT_VERSION = "aerial-pair-ranked-actions-v1"
CONFIRMATION_RANKED_PROMPT_VERSION = "aerial-pair-ranked-confirmation-v1"
HIERARCHICAL_PROMPT_VERSION = "aerial-pair-hierarchical-search-v1"
SYSTEM_PROMPT = """You are an aerial-image search policy in a 5 by 5 grid. The first image is the target clue and the second is your CURRENT local observation. Rows increase downward and columns increase rightward. You see no unvisited cells and do not know the target coordinates or true distance. The target may not be directionally identifiable from these two images: choose a reasonable exploratory move using only visible evidence and the provided visited-cell IDs. Boundary moves stay in place and cost one step. Output ONLY a JSON object with one key, action, whose value is up, right, down, or left. Do not output explanations or reasoning. Never output stop."""
RANKED_SYSTEM_PROMPT = """You are an aerial-image search policy in a 5 by 5 grid. The first image is the target clue and the second is your CURRENT local observation. Rows increase downward and columns increase rightward. You see no unvisited cells and do not know the target coordinates or true distance. The target may not be directionally identifiable from these two images: rank exploratory moves using only visible evidence and the provided visited-cell IDs. Boundary moves stay in place and cost one step. Output ONLY a JSON object with exactly one key, ranked_actions. Its value must be a permutation containing each of up, right, down, left exactly once. Put your preferred action first. Do not output explanations, scores, confidence, reasoning, or stop."""""
CONFIRMATION_RANKED_SYSTEM_PROMPT = """You are an aerial-image search policy in a 5 by 5 grid. The first image is the target clue and the second is your CURRENT local observation. Rows increase downward and columns increase rightward. You see no unvisited cells and do not know the target coordinates or true distance. Rank moves using visible evidence and the provided visited-cell IDs. Also assess whether the current view provides evidence that the agent is in the target's local neighborhood: use target_evidence exactly low, medium, or high. High means the current and target views provide strong local confirmation; it does not mean you know target coordinates and it is not a stop action. Boundary moves stay in place and cost one step. Output ONLY a JSON object with exactly two keys, ranked_actions and target_evidence. ranked_actions must be a permutation containing each of up, right, down, left exactly once. Put your preferred action first. Do not output explanations, scores, confidence, reasoning, or stop."""""
HIERARCHICAL_SYSTEM_PROMPT = """You are an aerial-image search policy in a 5 by 5 grid. The first image is the target clue and the second is your CURRENT local observation. Rows increase downward and columns increase rightward. The 5 by 5 grid is abstracted into nine coarse regions using region_id=r(row//2)c(col//2): r0c0, r0c1, r0c2, r1c0, r1c1, r1c2, r2c0, r2c1, r2c2. You see no unvisited cells and do not know target coordinates or true distance. First rank the coarse regions using visible target evidence; then rank the four local moves. Use only the two images and public visited-cell IDs. Boundary moves stay in place and cost one step. Output ONLY a JSON object with exactly two keys, ranked_regions and ranked_actions. ranked_regions must be a permutation of all nine region IDs. ranked_actions must be a permutation of up, right, down, left. Do not output explanations, scores, confidence, reasoning, or stop."""""


@dataclass(frozen=True)
class APIConfig:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    timeout: float = 90.0
    max_tokens: int = 128

    def __post_init__(self):
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("云端API必须使用不含凭据的HTTPS URL")
        if not self.api_key or not self.model or self.timeout <= 0 or self.max_tokens < 1:
            raise ValueError("缺少API配置或参数非法")

    @classmethod
    def load(cls, path: Path | None = None):
        values = {}
        file = Path(path) if path else PROJECT / ".env"
        if file.exists():
            for line in file.read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.lstrip().startswith("#"):
                    key, value = line.split("=", 1)
                    values[key.strip()] = value.strip().strip('\"\'')
        def get(name, default=""):
            return os.environ.get(name, values.get(name, default))
        return cls(get("VLM_BASE_URL", "https://ollama.com/v1"),
                   get("VLM_MODEL", "gemma4:31b"), get("VLM_API_KEY"),
                   max_tokens=int(get("VLM_MAX_TOKENS", "128")))

    def public(self):
        return {"base_url": self.base_url, "model": self.model,
                "timeout_seconds": self.timeout, "max_tokens": self.max_tokens,
                "temperature": 0, "stream": False,
                "response_format": "json_object", "api_key": "[REDACTED]"}


def _observation_request(obs: Observation, config: APIConfig, system_prompt: str, prompt_version: str, output_schema: str):
    state = {"position_row_col": list(obs.position), "grid_size": obs.grid_size,
             "remaining_budget": obs.remaining_budget,
             "visited_cell_ids": list(obs.visited),
             "legal_actions": list(obs.legal_actions)}
    content = [{"type": "text", "text": "Target clue image:"},
               {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(obs.target_image).decode("ascii")}},
               {"type": "text", "text": "Current local observation image:"},
               {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(obs.current_image).decode("ascii")}},
               {"type": "text", "text": json.dumps(state, separators=(",", ":"))}]
    payload = {"model": config.model, "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}],
        "temperature": 0, "max_tokens": config.max_tokens,
        "response_format": {"type": "json_object"}, "stream": False}
    audit = {"public_state": state,
             "current_image_sha256": sha256(obs.current_image).hexdigest(),
             "target_image_sha256": sha256(obs.target_image).hexdigest(),
             "image_count": 2,
             "prompt_version": prompt_version,
             "output_schema": output_schema,
             "system_prompt_sha256": sha256(system_prompt.encode()).hexdigest()}
    return payload, audit


def observation_request(obs: Observation, config: APIConfig):
    return _observation_request(obs, config, SYSTEM_PROMPT, PROMPT_VERSION, "action")


def ranked_observation_request(obs: Observation, config: APIConfig):
    return _observation_request(obs, config, RANKED_SYSTEM_PROMPT, RANKED_PROMPT_VERSION, "ranked_actions")


def confirmation_ranked_observation_request(obs: Observation, config: APIConfig):
    return _observation_request(obs, config, CONFIRMATION_RANKED_SYSTEM_PROMPT,
                                CONFIRMATION_RANKED_PROMPT_VERSION,
                                "ranked_actions_with_target_evidence")


def hierarchical_observation_request(obs: Observation, config: APIConfig):
    return _observation_request(obs, config, HIERARCHICAL_SYSTEM_PROMPT,
                                HIERARCHICAL_PROMPT_VERSION,
                                "hierarchical_ranked_regions_actions")


def _parse_json_object(content, finish_reason):
    if not isinstance(content, str) or finish_reason == "length":
        raise ValueError("missing or truncated response")
    text = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    normalized = match is not None
    if match:
        text = match.group(1).strip()
    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate JSON keys")
            obj[key] = value
        return obj
    return json.loads(text, object_pairs_hook=unique_object), normalized


def parse_action(content, legal_actions, finish_reason):
    parsed, normalized = _parse_json_object(content, finish_reason)
    if not isinstance(parsed, dict) or set(parsed) != {"action"} or parsed["action"] not in legal_actions:
        raise ValueError("invalid action schema")
    return parsed["action"], normalized


def parse_ranked_actions(content, legal_actions, finish_reason):
    parsed, normalized = _parse_json_object(content, finish_reason)
    actions = tuple(legal_actions)
    if not isinstance(parsed, dict) or set(parsed) != {"ranked_actions"}:
        raise ValueError("invalid ranked action schema")
    ranked = parsed["ranked_actions"]
    if not isinstance(ranked, list) or len(ranked) != len(actions):
        raise ValueError("ranked_actions must cover every legal action once")
    if any(not isinstance(action, str) for action in ranked) or len(set(ranked)) != len(ranked) or set(ranked) != set(actions):
        raise ValueError("ranked_actions must be a legal permutation")
    return tuple(ranked), normalized


def parse_ranked_confirmation(content, legal_actions, finish_reason):
    parsed, normalized = _parse_json_object(content, finish_reason)
    actions = tuple(legal_actions)
    if not isinstance(parsed, dict) or set(parsed) != {"ranked_actions", "target_evidence"}:
        raise ValueError("invalid ranked confirmation schema")
    ranked = parsed["ranked_actions"]
    evidence = parsed["target_evidence"]
    if (not isinstance(ranked, list) or len(ranked) != len(actions)
            or any(not isinstance(action, str) for action in ranked)
            or len(set(ranked)) != len(ranked) or set(ranked) != set(actions)):
        raise ValueError("ranked_actions must be a legal permutation")
    if evidence not in ("low", "medium", "high"):
        raise ValueError("target_evidence must be low, medium, or high")
    return tuple(ranked), evidence, normalized


def parse_hierarchical_ranked(content, legal_actions, region_ids, finish_reason):
    parsed, normalized = _parse_json_object(content, finish_reason)
    actions = tuple(legal_actions)
    regions = tuple(region_ids)
    if not isinstance(parsed, dict) or set(parsed) != {"ranked_regions", "ranked_actions"}:
        raise ValueError("invalid hierarchical ranked schema")
    ranked_regions = parsed["ranked_regions"]
    ranked_actions = parsed["ranked_actions"]
    if (not isinstance(ranked_regions, list) or len(ranked_regions) != len(regions)
            or any(not isinstance(region, str) for region in ranked_regions)
            or len(set(ranked_regions)) != len(ranked_regions)
            or set(ranked_regions) != set(regions)):
        raise ValueError("ranked_regions must be a legal permutation")
    if (not isinstance(ranked_actions, list) or len(ranked_actions) != len(actions)
            or any(not isinstance(action, str) for action in ranked_actions)
            or len(set(ranked_actions)) != len(ranked_actions)
            or set(ranked_actions) != set(actions)):
        raise ValueError("ranked_actions must be a legal permutation")
    return tuple(ranked_regions), tuple(ranked_actions), normalized


class APITrialError(RuntimeError):
    def __init__(self, reason: str, record: dict):
        super().__init__(reason)
        self.reason = reason
        self.record = record


class VLMPolicy:
    def __init__(self, config: APIConfig, client=None):
        self.config = config
        self.client = client or httpx.Client(timeout=config.timeout, follow_redirects=False)
        self._owns_client = client is None
        self.calls = []

    def close(self):
        if self._owns_client:
            self.client.close()

    def _redact(self, value):
        if isinstance(value, str):
            return value.replace(self.config.api_key, "[REDACTED]")
        if isinstance(value, dict):
            return {key: self._redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        return value

    def _exchange(self, method, endpoint, payload=None):
        start = time.perf_counter()
        record = {"request_kind": endpoint, "attempt": 1, "http_status": None,
                  "usage": None, "status": "pending", "time_to_first_token": None,
                  "retry_policy": "no_automatic_retries_in_pilot"}
        try:
            response = self.client.request(method, self.config.base_url.rstrip("/") + endpoint,
                headers={"Authorization": "Bearer " + self.config.api_key}, json=payload)
        except httpx.RequestError as exc:
            record.update(status="transport_error", error_type=type(exc).__name__,
                          latency_seconds=time.perf_counter() - start)
            raise APITrialError("transport_error", record) from None
        record.update(http_status=response.status_code,
                      latency_seconds=time.perf_counter() - start)
        if response.status_code >= 300:
            try:
                error = response.json().get("error", {})
                # 只记录错误摘要，避免响应服务意外回显完整图像请求。
                if isinstance(error, dict):
                    record["error_code"] = self._redact(str(error.get("code", "")))[:200]
                    record["error_message"] = self._redact(str(error.get("message", "")))[:1000]
            except (ValueError, AttributeError):
                pass
            record["status"] = "http_error"
            raise APITrialError("http_error", record)
        try:
            data = response.json()
        except ValueError:
            record["status"] = "invalid_response_json"
            raise APITrialError("invalid_response_json", record) from None
        return data, record

    def probe(self):
        data, record = self._exchange("GET", "/models")
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            record["status"] = "invalid_models_schema"
            raise APITrialError("invalid_models_schema", record)
        ids = [entry.get("id") for entry in data["data"] if isinstance(entry, dict)]
        record.update(status="ok", requested_model=self.config.model,
                      requested_model_listed=self.config.model in ids, available_model_ids=ids)
        return record

    def act(self, obs: Observation):
        payload, audit = observation_request(obs, self.config)
        try:
            data, record = self._exchange("POST", "/chat/completions", payload)
        except APITrialError as exc:
            exc.record.update(audit)
            self.calls.append(exc.record)
            raise
        record.update(audit)
        if not isinstance(data, dict):
            record["status"] = "invalid_response_schema"
            self.calls.append(record)
            raise APITrialError("invalid_response_schema", record)
        record.update(returned_model=data.get("model"), response_id=data.get("id"),
                      usage=self._redact(data.get("usage")),
                      system_fingerprint=data.get("system_fingerprint"))
        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message.get("content")
            record.update(raw_content=self._redact(content), finish_reason=choice.get("finish_reason"),
                          reasoning_present=bool(message.get("reasoning") or message.get("reasoning_content")))
            # 不请求、解析或保存隐藏推理；仅严格校验最终动作。
            action, normalized = parse_action(content, obs.legal_actions, choice.get("finish_reason"))
            record["format_normalization"] = "markdown_fence" if normalized else "none"
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            record["status"] = "invalid_action"
            self.calls.append(record)
            raise APITrialError("invalid_action", record) from None
        record.update(status="ok", action=action, output_schema="action")
        self.calls.append(record)
        return action, record

    def rank(self, obs: Observation):
        payload, audit = ranked_observation_request(obs, self.config)
        try:
            data, record = self._exchange("POST", "/chat/completions", payload)
        except APITrialError as exc:
            exc.record.update(audit)
            self.calls.append(exc.record)
            raise
        record.update(audit)
        if not isinstance(data, dict):
            record["status"] = "invalid_response_schema"
            self.calls.append(record)
            raise APITrialError("invalid_response_schema", record)
        record.update(returned_model=data.get("model"), response_id=data.get("id"),
                      usage=self._redact(data.get("usage")),
                      system_fingerprint=data.get("system_fingerprint"))
        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message.get("content")
            record.update(raw_content=self._redact(content), finish_reason=choice.get("finish_reason"),
                          reasoning_present=bool(message.get("reasoning") or message.get("reasoning_content")))
            ranked, normalized = parse_ranked_actions(content, obs.legal_actions, choice.get("finish_reason"))
            record["format_normalization"] = "markdown_fence" if normalized else "none"
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            record["status"] = "invalid_ranked_actions"
            self.calls.append(record)
            raise APITrialError("invalid_ranked_actions", record) from None
        record.update(status="ok", ranked_actions=list(ranked), output_schema="ranked_actions")
        self.calls.append(record)
        return ranked, record

    def rank_with_confirmation(self, obs: Observation):
        payload, audit = confirmation_ranked_observation_request(obs, self.config)
        try:
            data, record = self._exchange("POST", "/chat/completions", payload)
        except APITrialError as exc:
            exc.record.update(audit)
            self.calls.append(exc.record)
            raise
        record.update(audit)
        if not isinstance(data, dict):
            record["status"] = "invalid_response_schema"
            self.calls.append(record)
            raise APITrialError("invalid_response_schema", record)
        record.update(returned_model=data.get("model"), response_id=data.get("id"),
                      usage=self._redact(data.get("usage")),
                      system_fingerprint=data.get("system_fingerprint"))
        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message.get("content")
            record.update(raw_content=self._redact(content), finish_reason=choice.get("finish_reason"),
                          reasoning_present=bool(message.get("reasoning") or message.get("reasoning_content")))
            ranked, evidence, normalized = parse_ranked_confirmation(
                content, obs.legal_actions, choice.get("finish_reason"))
            record["format_normalization"] = "markdown_fence" if normalized else "none"
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            record["status"] = "invalid_ranked_confirmation"
            self.calls.append(record)
            raise APITrialError("invalid_ranked_confirmation", record) from None
        record.update(status="ok", ranked_actions=list(ranked), target_evidence=evidence,
                      output_schema="ranked_actions_with_target_evidence")
        self.calls.append(record)
        return ranked, evidence, record

    def rank_hierarchical(self, obs: Observation):
        from agents.governor import COARSE_REGIONS
        payload, audit = hierarchical_observation_request(obs, self.config)
        try:
            data, record = self._exchange("POST", "/chat/completions", payload)
        except APITrialError as exc:
            exc.record.update(audit)
            self.calls.append(exc.record)
            raise
        record.update(audit)
        if not isinstance(data, dict):
            record["status"] = "invalid_response_schema"
            self.calls.append(record)
            raise APITrialError("invalid_response_schema", record)
        record.update(returned_model=data.get("model"), response_id=data.get("id"),
                      usage=self._redact(data.get("usage")),
                      system_fingerprint=data.get("system_fingerprint"))
        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message.get("content")
            record.update(raw_content=self._redact(content), finish_reason=choice.get("finish_reason"),
                          reasoning_present=bool(message.get("reasoning") or message.get("reasoning_content")))
            regions, ranked, normalized = parse_hierarchical_ranked(
                content, obs.legal_actions, COARSE_REGIONS, choice.get("finish_reason"))
            record["format_normalization"] = "markdown_fence" if normalized else "none"
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            record["status"] = "invalid_hierarchical_ranked"
            self.calls.append(record)
            raise APITrialError("invalid_hierarchical_ranked", record) from None
        record.update(status="ok", ranked_regions=list(regions), ranked_actions=list(ranked),
                      output_schema="hierarchical_ranked_regions_actions")
        self.calls.append(record)
        return regions, ranked, record
