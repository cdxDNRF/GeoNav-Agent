"""Single public-state veto: preserve the original move to a new cell."""
from agents.local_ledger_option import LocalLedgerOption
from env.episode import ACTIONS


class ProtectedLedgerOption(LocalLedgerOption):
    def decide(self, obs, base):
        result = super().decide(obs, base)
        proposed = result['action']
        dr, dc = ACTIONS[base['action']]
        r, c = obs.position[0]+dr, obs.position[1]+dc
        if not (0 <= r < obs.grid_size and 0 <= c < obs.grid_size):
            raise ValueError('original policy must already be legal')
        original_next = r*obs.grid_size+c
        veto = (result['control'] == 'bounded_option' and proposed != base['action']
                and original_next not in obs.visited)
        if veto:
            # Cancel the remainder, retain the original earliest/once-only trigger.
            self.pending = []
            self.previous = base['action']
            result.update(action=base['action'], control='protected_fresh_move',
                          option_interrupted=True, pending_after=[])
        result.update(proposed_action=proposed, protection_veto=veto,
                      original_next_cell=original_next)
        return result
