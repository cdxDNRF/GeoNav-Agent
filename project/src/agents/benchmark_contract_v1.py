"""Public, immutable inputs for shared benchmarking; no model or evaluator imports."""
from dataclasses import dataclass
import math
from typing import Protocol

from env.environment import Observation, StepInfo


@dataclass(frozen=True)
class FeatureBundle:
    channels: tuple[tuple[str, tuple[float, ...]], ...]

    def __post_init__(self):
        if type(self.channels) is not tuple or not self.channels:
            raise ValueError('immutable nonempty channels required')
        names = []
        for name, values in self.channels:
            if not isinstance(name, str) or not name or type(values) is not tuple or not values:
                raise ValueError('invalid feature channel')
            if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
                raise ValueError('finite numerical features required')
            names.append(name)
        if names != sorted(set(names)):
            raise ValueError('unique sorted channel names required')

    @property
    def schema(self):
        return tuple((name, len(values)) for name, values in self.channels)


@dataclass(frozen=True)
class PolicyRequest:
    observation: Observation
    current_features: FeatureBundle
    target_features: FeatureBundle | None
    target_enabled: bool
    last_step: StepInfo | None

    def __post_init__(self):
        if self.target_enabled != (self.target_features is not None):
            raise ValueError('target enable flag differs from features')
        if not self.target_enabled and self.observation.target_image:
            raise ValueError('disabled target image must be absent')
        if self.target_enabled and self.current_features.schema != self.target_features.schema:
            raise ValueError('feature schemas differ')


class BenchmarkPolicy(Protocol):
    def reset(self) -> None: ...
    def decide(self, request: PolicyRequest) -> str: ...


class ScriptedPolicy:
    """Synthetic-only control; not a learned baseline or a real navigation strategy."""
    def __init__(self, actions: tuple[str, ...]):
        self._actions = tuple(actions)
        self.reset()

    def reset(self):
        self._index = 0

    def decide(self, request):
        action = self._actions[self._index % len(self._actions)]
        self._index += 1
        return action
