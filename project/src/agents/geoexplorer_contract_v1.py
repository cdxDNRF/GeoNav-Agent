"""Own history bridge for the GeoExplorer candidate. No upstream code or model execution.

The injected selector is a fake callback in BENCH-002. Tensor conversion, original
positional encoding, weights and learned inference remain future work.
"""
from dataclasses import dataclass
from agents.benchmark_contract_v1 import FeatureBundle, PolicyRequest


@dataclass(frozen=True)
class HistoryFrame:
    position: tuple[int, int]
    features: FeatureBundle


@dataclass(frozen=True)
class HistoryInput:
    target: FeatureBundle | None
    states: tuple[HistoryFrame, ...]
    actions: tuple[str, ...]
    grid_size: int
    remaining_budget: int
    legal_actions: tuple[str, ...]


class GeoExplorerHistoryBridge:
    def __init__(self, selector):
        self._selector = selector
        self.reset()

    def reset(self):
        self._states = []
        self._actions = []
        self._target = None
        self._started = False

    def decide(self, request: PolicyRequest):
        step = 0 if request.last_step is None else request.last_step.step_count
        if step != len(self._actions):
            raise ValueError('history step discontinuity; reset each episode')
        if self._started and request.target_features != self._target:
            raise ValueError('target changed within episode')
        self._target = request.target_features
        self._started = True
        self._states.append(HistoryFrame(request.observation.position, request.current_features))
        context = HistoryInput(self._target, tuple(self._states), tuple(self._actions),
                               request.observation.grid_size, request.observation.remaining_budget,
                               request.observation.legal_actions)
        action = self._selector(context)
        if action not in ('up', 'right', 'down', 'left'):
            raise ValueError('selector returned invalid action')
        self._actions.append(action)
        return action
