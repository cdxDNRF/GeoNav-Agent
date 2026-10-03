"""Explicit opt-in for the user-supplied loopback cloud API aggregator."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import httpx
from agents.vlm import VLMPolicy

GATEWAY = 'http://127.0.0.1:8790/v1'
MODELS = ('buddy/deepseek-v4.1-flash', 'buddy/glm-5.3-flash', 'qoder/qfmodel')


@dataclass(frozen=True)
class GatewayConfig:
    model: str
    base_url: str = GATEWAY
    api_key: str = field(default='codex-gateway-dummy', repr=False)
    timeout: float = 45.0
    max_tokens: int = 1024

    def __post_init__(self):
        if self.base_url != GATEWAY or self.model not in MODELS:
            raise ValueError('only explicitly registered loopback gateway/models allowed')
        if not self.api_key or not 0 < self.timeout <= 90 or not 1 <= self.max_tokens <= 4096:
            raise ValueError('invalid gateway limits')

    def public(self):
        return dict(base_url=self.base_url, model=self.model, api_key='[REDACTED]',
                    provider_kind='cloud_api_aggregator', timeout_seconds=self.timeout,
                    max_tokens=self.max_tokens, temperature=0, stream=False,
                    response_format='json_object')


class GatewayTransport:
    def __init__(self, config, transport=None):
        # Never send existing HTTPS-cloud credentials to this separate option.
        self.client = httpx.Client(timeout=config.timeout, trust_env=False,
                                   follow_redirects=False, transport=transport)

    def request(self, method, url, headers=None, json=None):
        allowed = {('GET', GATEWAY+'/models'), ('POST', GATEWAY+'/chat/completions')}
        if (method, url) not in allowed:
            raise ValueError('unregistered gateway route; redirects are never followed')
        return self.client.request(method, url, headers=headers, json=json)

    def close(self):
        self.client.close()


def make_gateway_policy(model, *, timeout=45.0, max_tokens=1024):
    """Use as a context manager; all existing VLM schemas remain strict."""
    from contextlib import contextmanager
    @contextmanager
    def opened():
        config = GatewayConfig(model=model, timeout=timeout, max_tokens=max_tokens)
        transport = GatewayTransport(config)
        try:
            yield VLMPolicy(config, client=transport)
        finally:
            transport.close()
    return opened()


def registered_options():
    path = Path(__file__).resolve().parents[2]/'cloud_provider_options.json'
    data = json.loads(path.read_text(encoding='utf8'))
    if data['base_url'] != GATEWAY or tuple(data['models']) != MODELS:
        raise ValueError('registry and adapter differ')
    return data
