"""Feature-cache execution adapter; each policy receives only its two given image profiles."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import torch
from agents.precomputed_scaled_edge import PrecomputedScaledEdgeNavigator
from agents.edge_cue import EdgeTargetCueHead
from agents.spatial_relation import make_policy
from env.environment import image_payload
from env.scaled_grid import ScaledGridEnv
from eval.scaled_edge_pilot import MEANS,result_rows
from train.curiosity_controlled import write_new
from train.dyncur_tiny import digest


def load_agent(folder,plan,means,device,condition):
    explorer=make_policy('Small256').to(device).eval();head=EdgeTargetCueHead().to(device).eval()
    explorer.load_state_dict(torch.load(folder/'explorer.pt',map_location=device,weights_only=True))
    head.load_state_dict(torch.load(folder/'head.pt',map_location=device,weights_only=True))
    explorer.requires_grad_(False);head.requires_grad_(False)
    return PrecomputedScaledEdgeNavigator(explorer,head,device,*[means[n] for n in MEANS],plan['threshold'],plan['arm'],condition)


def row_run(agent,ep,env,g,l,profiles,wrong):
    agent.reset();obs=env.reset(ep);key=ep.split+'__'+ep.area
    cue=wrong[ep.episode_id]['cue_cell'] if agent.condition=='CueWrong' else ep.goal
    target=image_payload(env.root/'patches'/ep.split/ep.area/f'patch_{cue}.jpg');decisions=[]
    while not env.done:
        cell=obs.position[0]*10+obs.position[1]
        dec=agent.act_with_profiles(replace(obs,target_image=target),g[key][cell],l[key][cell],g[key][cue],l[key][cue],profiles[key][cell],profiles[key][cue])
        decisions.append(dec);obs,_,info=env.step(dec['action'])
        if info.out_of_bounds:raise ValueError('illegal neural move')
    return dict(**env.evaluator_result(),area=ep.area,source=ep.source_tile,distance=ep.dist,condition=agent.condition,arm=agent.arm,decisions=decisions)


def neural_result(folder,plan,c,episodes,g,l,profiles,means,wrong,device,data_root):
    first=load_agent(folder,plan,means,device,c);second=load_agent(folder,plan,means,device,c)
    env=ScaledGridEnv(data_root);rows=[];path=folder/f'导航_{c}_轨迹.jsonl'
    with path.open('x',encoding='utf-8') as f:
        for ep in episodes:
            a=row_run(first,ep,env,g,l,profiles,wrong);b=row_run(second,ep,env,g,l,profiles,wrong)
            if a!=b:raise ValueError('separately loaded checkpoint replay differs')
            rows.append(a);f.write(json.dumps(a,ensure_ascii=False)+'\n')
    result=result_rows(rows);result['audit']=dict(checkpoint_replay=True,trajectory_sha256=digest(path),head_sha256=digest(folder/'head.pt'),explorer_sha256=digest(folder/'explorer.pt'))
    write_new(folder/f'导航_{c}_结果.json',result);return result
