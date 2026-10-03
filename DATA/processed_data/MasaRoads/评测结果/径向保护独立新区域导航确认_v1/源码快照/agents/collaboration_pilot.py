"""Bounded independent-context collaboration; accepts public observations only."""
import base64
from hashlib import sha256
import json
import re

from agents.vlm import VLMPolicy, APITrialError, _parse_json_object
from env.episode import ACTIONS

MODES = ('SingleOnce', 'SingleReflect', 'TwoAgent')
MAX_REVIEWS = 2
MAX_CALLS = 606
MAX_HISTORY_IMAGES = 2
PROMPT_VERSION = 'edge-disagreement-collaboration-v1'
COMMON = """You control aerial-image target search in a 5x5 grid, with rows increasing down and columns right. You know public position, visited cells and remaining moves, but never see unvisited imagery or true target coordinates/distance. Target and current images are separate labelled inputs. Spatial similarity alone does not establish a direction. Inspect shared boundaries when visible; an adjacent target may have boundary continuity. Raw local cue scores are not calibrated confidence. Prefer abstention (target_evidence=unknown) when directional evidence is insufficient. A frozen local explorer/cue system supplies public proposals. You may suggest any boundary-legal action. Output ONLY JSON with exactly action, target_evidence, evidence_refs. action is one listed legal action; target_evidence is supported, opposed or unknown; evidence_refs is a list of zero to three provided image labels. supported means visible target evidence supports YOUR chosen action; opposed means the local cue lacks support. No stop, coordinates of the goal, prose, confidence or reasoning."""
PROMPTS = {
    'planner': COMMON + ' You are the exploration planner. Use your own retained messages and public observations to propose an action.',
    'reflector': COMMON + ' You are the SAME agent reflecting on your current proposal, which is provided. Independently reconsider its visible evidence and return your final recommendation.',
    'verifier': COMMON + ' You are the independent target verifier with a separate context. The planner CURRENT proposal is deliberately withheld. Determine your own action and support/abstention; prior peer messages may inform you.',
}
PREFLIGHT_PROMPT = 'Inspect two labelled aerial images. Output ONLY JSON with exactly one boolean key same_pixels: true if the two images show identical pixel content, false otherwise. No explanation.'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def digest_payload(value):
    return sha256(canonical(value)).hexdigest()


def boundary_legal(obs):
    r, c = obs.position
    return tuple(a for a, (dr, dc) in ACTIONS.items() if 0 <= r+dr < obs.grid_size and 0 <= c+dc < obs.grid_size)


def review_trigger(base, reviews):
    return reviews < MAX_REVIEWS and base['cue_action'] is not None and base['cue_action'] != base['explorer_action']


def parse_recommendation(content, finish, legal, labels):
    obj, normalized = _parse_json_object(content, finish)
    if not isinstance(obj, dict) or set(obj) != {'action', 'target_evidence', 'evidence_refs'}:
        raise ValueError('invalid recommendation keys')
    if obj['action'] not in legal or obj['target_evidence'] not in ('supported', 'opposed', 'unknown'):
        raise ValueError('invalid action/evidence')
    refs = obj['evidence_refs']
    if not isinstance(refs, list) or len(refs) > 3 or any(not isinstance(x, str) or x not in labels for x in refs) or len(refs) != len(set(refs)):
        raise ValueError('invalid evidence references')
    if obj['target_evidence'] == 'supported' and not {'target', 'current'}.issubset(refs):
        raise ValueError('supported target direction must cite target and current')
    return obj, normalized


def coordinate(mode, base_action, recommendations):
    if any(x is None for x in recommendations):
        return base_action, 'cloud_error_fallback'
    first = recommendations[0]
    if mode == 'SingleOnce':
        if first['target_evidence'] == 'supported':
            return first['action'], 'single_supported'
        return base_action, 'single_abstention_fallback'
    second = recommendations[1]
    if first['action'] == second['action'] and all(x['target_evidence'] == 'supported' for x in recommendations):
        return first['action'], 'joint_supported'
    return base_action, 'disagreement_or_abstention_fallback'


def image_part(label, payload):
    return [{'type': 'text', 'text': label + ':'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(payload).decode('ascii')}}]


def request_for(obs, base, frames, memory, role, config, current_proposal=None):
    if role not in PROMPTS or (role != 'reflector' and current_proposal is not None):
        raise ValueError('independent contexts must not receive current proposal')
    if role == 'reflector' and current_proposal is None:
        raise ValueError('reflection requires current proposal')
    images = [('target', obs.target_image), ('current', obs.current_image)]
    images += [(f'history_{i}', f['pixels']) for i, f in enumerate(frames[-MAX_HISTORY_IMAGES:])]
    state = dict(position=list(obs.position), grid_size=obs.grid_size, remaining_budget=obs.remaining_budget,
        visited=list(obs.visited), legal_actions=list(boundary_legal(obs)),
        local_proposals=dict(action=base['action'], explorer_action=base['explorer_action'], cue_action=base['cue_action'],
                             raw_cue_scores=base['probabilities']),
        history_frames=[{k: v for k, v in f.items() if k != 'pixels'} for f in frames[-MAX_HISTORY_IMAGES:]],
        prior_messages=memory, role=role)
    if role == 'reflector':
        state['current_proposal'] = current_proposal
    content = []
    for label, payload in images:
        content.extend(image_part(label, payload))
    content.append({'type': 'text', 'text': canonical(state).decode('utf-8')})
    payload = dict(model=config.model, messages=[{'role': 'system', 'content': PROMPTS[role]},
        {'role': 'user', 'content': content}], temperature=0, max_tokens=config.max_tokens,
        response_format={'type': 'json_object'}, stream=False)
    audit = dict(prompt_version=PROMPT_VERSION, role=role, public_input=state,
        image_labels=[label for label, _ in images], image_sha256=[sha256(p).hexdigest() for _, p in images],
        image_count=len(images), system_prompt_sha256=sha256(PROMPTS[role].encode()).hexdigest(),
        request_sha256=digest_payload(payload), current_proposal_visible=role == 'reflector')
    return payload, audit


class PilotAPI(VLMPolicy):
    def __init__(self, config, client, output, max_calls=MAX_CALLS):
        super().__init__(config, client)
        self.output = output
        self.total_calls = 0
        self.max_calls = max_calls
        self.consecutive_errors = 0
        self.circuit_reason = None

    def _persist(self, name, record):
        with (self.output / name).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')

    def exchange(self, payload, audit, tag, parser):
        if self.circuit_reason or self.total_calls >= self.max_calls:
            raise RuntimeError('frozen cloud circuit/budget closed')
        self.total_calls += 1
        intent = dict(call_id=self.total_calls, tag=tag, **audit)
        self._persist('请求意图.jsonl', intent)
        record = dict(intent)
        result = None
        try:
            data, transport = self._exchange('POST', '/chat/completions', payload)
            record.update(transport)
            record.update(returned_model=data.get('model'), response_id=data.get('id'), usage=self._redact(data.get('usage')))
            choice = data['choices'][0]
            message = choice['message']
            content = message.get('content')
            safe = self._redact(content)
            if isinstance(safe, str):
                safe = re.sub(r'data:image/[^\s"\']+', '[IMAGE_REDACTED]', safe)[:4000]
            record.update(raw_content=safe, finish_reason=choice.get('finish_reason'),
                reasoning_present=bool(message.get('reasoning') or message.get('reasoning_content')))
            result, normalized = parser(content, choice.get('finish_reason'))
            record.update(status='ok', parsed=result, format_normalization='markdown_fence' if normalized else 'none')
        except APITrialError as exc:
            record.update(exc.record)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            record['status'] = 'invalid_recommendation'
        if record['status'] == 'ok':
            self.consecutive_errors = 0
        else:
            self.consecutive_errors += 1
            if record.get('http_status') in (401, 403, 429) or self.consecutive_errors >= 3:
                self.circuit_reason = 'provider_status_or_three_consecutive_errors'
        self._persist('API响应.jsonl', record)
        return result, record

    def recommend(self, obs, base, frames, memory, role, tag, current_proposal=None):
        payload, audit = request_for(obs, base, frames, memory, role, self.config, current_proposal)
        return self.exchange(payload, audit, tag,
            lambda content, finish: parse_recommendation(content, finish, boundary_legal(obs), audit['image_labels']))


class CollaborationLayer:
    """Independent role memories are episode-local; no environment reference exists."""
    def __init__(self, mode, api):
        if mode not in MODES:
            raise ValueError('unknown arm')
        self.mode = mode
        self.api = api
        self.reset()

    def reset(self):
        self.frames = []
        self.memories = {'planner': [], 'verifier': [], 'single': []}
        self.reviews = 0

    def act(self, obs, base, tag):
        triggered = review_trigger(base, self.reviews)
        result = dict(triggered=triggered, review_index=None, recommendations=[], call_ids=[], action=base['action'],
                      coordination_reason='no_review', role_memories_before={k: list(v) for k, v in self.memories.items()})
        if triggered and not self.api.circuit_reason:
            self.reviews += 1
            result['review_index'] = self.reviews
            first_memory = self.memories['planner' if self.mode == 'TwoAgent' else 'single']
            first, record = self.api.recommend(obs, base, self.frames, first_memory, 'planner', tag)
            items = [first]
            result['call_ids'].append(record['call_id'])
            if self.mode != 'SingleOnce' and not self.api.circuit_reason:
                role = 'verifier' if self.mode == 'TwoAgent' else 'reflector'
                memory = self.memories['verifier' if self.mode == 'TwoAgent' else 'single']
                # Invalid first output still consumes its slot; second call uses explicit empty proposal for reflection.
                second, record = self.api.recommend(obs, base, self.frames, memory, role, tag,
                    current_proposal=(first or {'invalid': True}) if role == 'reflector' else None)
                items.append(second)
                result['call_ids'].append(record['call_id'])
            elif self.mode != 'SingleOnce':
                items.append(None)
            result['recommendations'] = items
            result['action'], result['coordination_reason'] = coordinate(self.mode, base['action'], items)
            shared = dict(position=list(obs.position), remaining_budget=obs.remaining_budget, planner=items[0],
                          checker=items[1] if len(items) == 2 else None, executed_action=result['action'])
            if self.mode == 'TwoAgent':
                self.memories['planner'].append(dict(self_role='planner', exchange=shared))
                self.memories['verifier'].append(dict(self_role='verifier', exchange=shared))
            else:
                self.memories['single'].append(dict(self_role='single', exchange=shared))
        elif triggered:
            result['coordination_reason'] = 'cloud_circuit_fallback'
        self.frames.append(dict(pixels=obs.current_image, position=list(obs.position), remaining_budget=obs.remaining_budget))
        self.frames = self.frames[-MAX_HISTORY_IMAGES:]
        return result
