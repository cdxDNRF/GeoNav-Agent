"""BENCH-012: saved evidence consolidation and coordinate-only arithmetic.

No model import, feature bank, prediction, policy, optimizer or navigation.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BATCH = ROOT / '项目导航/GeoExplorer复现差异与诊断收束_v1'
BASE = 'DATA/processed_data/Masa/'
OFFICIAL = '项目导航/GeoExplorer资源验证_v1/公开来源/'
INPUTS = {
    'training_protocol': BASE+'训练结果/GeoExplorer共同来源训练_v2/元数据/冻结执行协议.json',
    'training_audit': BASE+'训练结果/GeoExplorer共同来源训练_v2/核验/训练复核_完整_v1.json',
    'training_summary': BASE+'训练结果/GeoExplorer共同来源训练_v2/训练结果汇总.json',
    'comparison': BASE+'评测结果/共同学习基线对照_v1/核验/正式独立复核_v1.json',
    'comparison_protocol': BASE+'评测结果/共同学习基线对照_v1/元数据/正式冻结协议_v1.json',
    'comparison_report': BASE+'评测结果/共同学习基线对照_v1/同题对照报告_v2.md',
    'reproduction_note': BASE+'评测结果/共同学习基线对照_v1/复现说明_v1.md',
    'goal': BASE+'评测结果/GeoExplorer目标利用诊断_v1/核验/诊断独立复核_v1.json',
    'gradient': BASE+'评测结果/GeoExplorer梯度贡献诊断_v1/核验/梯度独立复核_v1.json',
    'curve': BASE+'评测结果/GeoExplorer目标学习曲线诊断_v1/核验/曲线汇总_v1.json',
    'curve_audit': BASE+'评测结果/GeoExplorer目标学习曲线诊断_v1/核验/曲线独立复核_v1.json',
    'context': BASE+'评测结果/GeoExplorer监督情境覆盖诊断_v1/核验/情境覆盖汇总_v1.json',
    'context_audit': BASE+'评测结果/GeoExplorer监督情境覆盖诊断_v1/核验/情境独立复核_v1.json',
    'official_model': OFFICIAL+'GeoExplorer__models__model_falcon.py',
    'official_config': OFFICIAL+'GeoExplorer__config.py',
    'official_pretrain': OFFICIAL+'GeoExplorer__pretrain.py',
    'official_losses': OFFICIAL+'GeoExplorer__models__pretrain_model.py',
    'official_train': OFFICIAL+'GeoExplorer__train.py',
    'official_binding': OFFICIAL+'完整来源绑定.json',
    'actual_constructor': 'project/src/train/geoexplorer_resource_probe_v1.py',
    'actual_policy': 'project/src/agents/geoexplorer_policy_v1.py',
    'default_5': 'project/local_policy_default.json',
    'defaults': 'project/local_policy_defaults_v1.json',
}
SEAL_BATCHES = [
    BASE+'评测结果/'+name+'_v1' for name in
    ['共同学习基线对照', 'GeoExplorer目标利用诊断', 'GeoExplorer梯度贡献诊断',
     'GeoExplorer目标学习曲线诊断', 'GeoExplorer监督情境覆盖诊断']
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return digest


def write_new(path, value):
    # Refuse occupied output, including failed earlier attempts.
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        if isinstance(value, str): stream.write(value)
        else: stream.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def verify_binding(protocol):
    for name, expected in protocol['bindings'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('binding changed: '+name)


def extract_evidence(documents):
    for name in ['training_audit', 'comparison', 'goal', 'gradient', 'curve_audit', 'context_audit']:
        if documents[name]['status'] != 'passed': raise ValueError('audit not passed: '+name)
    train = documents['training_protocol']
    audit = documents['training_audit']
    if not audit['training_completed'] or sorted(audit['completed_seeds']) != [0,1,2]:
        raise ValueError('three completed seeds required')
    runs = []
    for row in audit['runs']:
        c = row['counts']
        if (c['microbatches'],c['epochs'],c['environment_steps'],c['pretrain_optimizer_updates'],c['ppo_optimizer_updates']) != (32700,300,163840,300,1024):
            raise ValueError('original budget changed')
        runs.append({'seed': row['seed'], 'seconds': row['result']['elapsed_seconds'],
                     'reserved_mib': c['peak_reserved_mib'], 'checkpoint_seals': len(row['checkpoint_seals'])})
    comparison = documents['comparison']
    metrics = comparison['aggregate_metrics']
    if len(metrics) != 8 or sum(m['planned'] for m in metrics) != 6000:
        raise ValueError('comparison denominator changed')
    if {(m['method'], m['condition']) for m in metrics} != {
        (m,c) for m in ['M0','GeoExplorer'] for c in ['Full','Disabled','Mean','Wrong']}:
        raise ValueError('missing or duplicate arm')
    for m in metrics:
        if m['planned'] != 750 or abs(m['sr']-m['successes']/m['planned']) > 1e-12:
            raise ValueError('planned-denominator SR mismatch')
    goal = [{k:v for k,v in row.items() if k not in ['by_source','by_seed']}
            for row in documents['goal']['headline']]
    grad = next(r for r in documents['gradient']['summary']
                if r['stage']=='epoch300' and r['group']=='backbone')
    curve = [{k:v for k,v in row.items() if k not in ['by_source','by_seed']}
             for row in documents['curve']['headline']]
    context = documents['context']['pooled']
    p = context['regime_positions']
    if sum(p.values()) != context['supervised_positions'] or context['active_positions'] != p['unseen']+p['seen_before']:
        raise ValueError('mutually exclusive position denominators inconsistent')
    if any([documents['gradient']['loss_weight_candidate_screen_passed'],
            documents['curve']['supervision_progress_screen_passed'],
            documents['context']['context_insufficiency_screen_passed']]):
        raise ValueError('closed factor unexpectedly changed')
    return {
        'training': {'runs': runs, 'total_seconds': sum(r['seconds'] for r in runs),
                     'fit_sources': len(train['fit_areas']), 'known_development_sources': len(train['known_development_sources']),
                     'official_commit': train['budget']['official_commit'],
                     'differences': train['differences'], 'dimensions': train['dimensions'],
                     'environment': train['environment'],
                     'previous_failed_cost_separate': documents['training_summary']['extra_failed_compute_reported_separately']},
        'navigation': {'metrics': metrics, 'contrasts': comparison['contrasts'],
                       'target_gates': comparison['target_effect_gates'],
                       'source_file_groups': comparison['source_file_groups'],
                       'geographic_nonoverlap_verified': comparison['geographic_nonoverlap_verified']},
        'diagnostics': {'goal': goal, 'gradient_epoch300_backbone': grad, 'curve': curve,
                        'curve_changes': documents['curve']['changes'], 'context': context,
                        'closed_factors': ['lambda_state_0.1', 'more_pretraining_only', 'unseen_context_shortage']},
    }


def coordinate_vectors(grid, phase_multiplier, dtype):
    if type(grid) is not int or grid < 2: raise ValueError('grid must be integer >=2')
    if phase_multiplier not in [1,2]: raise ValueError('only frozen half/full circle')
    ids = np.arange(grid*grid, dtype=np.int64)
    rows, columns = ids//grid, ids%grid
    # Mathematical/NumPy simulation; NOT a torch CUDA byte-equivalence claim.
    angles = [np.asarray(phase_multiplier*np.pi, dtype=dtype)*np.asarray(q, dtype=dtype)/np.asarray(grid-1,dtype=dtype)
              for q in [columns,rows]]
    return np.stack([np.sin(angles[0]),np.sin(angles[1]),np.cos(angles[0]),np.cos(angles[1])],axis=1)


def progress_actions(cell, goal, grid):
    r,c = divmod(cell,grid); gr,gc = divmod(goal,grid)
    return {i for i,yes in enumerate([gr<r,gc>c,gr>r,gc<c]) if yes}


def geometry_report(grid, phase_multiplier, tolerance):
    vectors = coordinate_vectors(grid,phase_multiplier,np.float32)
    distances = np.sqrt(np.sum((vectors[:,None,:]-vectors[None,:,:])**2,axis=2))
    pairs = [(a,b) for a in range(grid*grid) for b in range(a+1,grid*grid) if distances[a,b] <= tolerance]
    seen = set(); components = []
    for cell in range(grid*grid):
        if cell in seen: continue
        component = {cell}; pending = [cell]
        while pending:
            node = pending.pop()
            for a,b in pairs:
                other = b if a==node else a if b==node else None
                if other is not None and other not in component:
                    component.add(other); pending.append(other)
        seen |= component; components.append(sorted(component))
    conflicts = []
    for a,b in pairs:
        for goal in range(grid*grid):
            pa,pb = progress_actions(a,goal,grid),progress_actions(b,goal,grid)
            if pa and pb and not pa & pb:
                conflicts.append({'current_a':a,'current_b':b,'offline_goal':goal,
                                  'progress_a':sorted(pa),'progress_b':sorted(pb)})
    distinct_distances = [float(distances[a,b]) for a in range(grid*grid) for b in range(a+1,grid*grid)]
    return {'grid':grid, 'phase_multiplier':phase_multiplier, 'dtype':'NumPy float32 simulation',
            'tolerance_l2':tolerance, 'classes':components, 'class_count':len(components),
            'non_singleton_classes':sum(len(c)>1 for c in components),
            'cells_in_non_singletons':sum(len(c) for c in components if len(c)>1),
            'near_pairs':[list(p) for p in pairs], 'near_pair_count':len(pairs),
            'minimum_distinct_cell_l2':min(distinct_distances),
            'maximum_near_pair_l2':max([float(distances[a,b]) for a,b in pairs],default=None),
            'coordinate_only_disjoint_direction_labels':conflicts,
            'complete_visual_input_collision_asserted':False,
            'cuda_byte_equivalence_asserted':False}


def line_anchor(role, needle, span=1):
    lines = (ROOT/INPUTS[role]).read_text(encoding='utf-8-sig').splitlines()
    found = [i+1 for i,line in enumerate(lines) if needle in line]
    if len(found) != 1: raise ValueError('nonunique anchor '+role+' '+needle)
    return {'file':INPUTS[role],'start':found[0], 'end':found[0]+span-1, 'text':needle}


def source_contract():
    tree = ast.parse((ROOT/INPUTS['official_model']).read_text(encoding='utf-8-sig'))
    coords = [n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='get_coord_feats']
    if len(coords)!=1: raise ValueError('coordinate function not unique')
    line = line_anchor('official_model','rel_pos = torch.stack')
    if '2*np.pi' not in line['text']:
        actual = (ROOT/line['file']).read_text(encoding='utf-8-sig').splitlines()[line['start']-1]
        if actual.count('2*np.pi')!=4: raise ValueError('full-circle function changed')
    return {'coordinate_function': {'file': INPUTS['official_model'], 'start':coords[0].lineno,'end':coords[0].end_lineno},
            'coordinate_observations_only':line_anchor('official_model','inputs_embeds[:, 1:] += rel_pos_embeds'),
            'public_position_input':line_anchor('actual_policy','positions=tuple(s.position[0]*5+s.position[1]'),
            'constructor':line_anchor('actual_constructor','model=module.FalconForCausalLM(config)'),
            'original_dimensions':line_anchor('official_model','config.hidden_size = 1152',7),
            'original_no_language_weight_loading':line_anchor('official_model','model = FalconForCausalLM(config)'),
            'losses':line_anchor('official_losses','loss = loss_action + loss_state'),
            'workers':line_anchor('official_losses',"self.num_workers = kwargs.get('num_workers', 16)"),
            'pretrain_checkpoint':line_anchor('official_pretrain',"monitor='val_loss'",4),
            'accumulation':line_anchor('official_pretrain','accumulate_grad_batches=256'),
            'upper_budget':line_anchor('official_config','cfg.train.hparams.max_training_timesteps = int(1e8)'),
            'four_actions':line_anchor('official_config','cfg.train.num_actions = 4')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true'); args=parser.parse_args()
    if not args.run: raise SystemExit('use --run once after preparation')
    protocol = read(BATCH/'元数据/冻结收束协议_v1.json'); verify_binding(protocol)
    documents={k:read(ROOT/p) for k,p in INPUTS.items() if p.endswith('.json')}
    result=extract_evidence(documents)
    result.update({'status':'completed_static_consolidation', 'protocol_sha256':sha(BATCH/'元数据/冻结收束协议_v1.json'),
                   'source_contract':source_contract(),
                   'coordinate_math':[geometry_report(protocol['grid'],m,protocol['near_tolerance_l2']) for m in [2,1]],
                   'new_model_forward':0,'new_training':0,'new_navigation':0,'new_sr':False,
                   'api_calls':0,'reserve_consumption':0,'independent_person_review':False})
    write_new(BATCH/'核验/证据与静态几何汇总_v1.json',result)
    print(json.dumps({'status':result['status'],'coordinate_classes':[r['class_count'] for r in result['coordinate_math']]},ensure_ascii=False))


if __name__=='__main__': main()
