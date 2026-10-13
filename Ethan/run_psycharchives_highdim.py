"""Train-only feature audit and 16D slow/fast manifold input expansion pilot.

Licensed records/IDs are never written. Reserved test people are never scored.
"""
import json
from pathlib import Path
import collections
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from run_psycharchives_manifold import CUES,TRAITS,split_people,sha256
from run_psycharchives_sequence_manifold import (SequenceManifold,batch_view,train_loss,
                                                development_metrics,WEIGHTS)
from run_psycharchives_evaluation_12 import trait_probe,independent_windows,matching_metrics

ROOT=Path(__file__).resolve().parent
SOURCE=Path.home()/'Downloads'


class HighDimManifold(SequenceManifold):
    def __init__(self,d):
        super().__init__(d)
        self.history=torch.nn.GRU(2*d,64,batch_first=True)
        self.slow=torch.nn.Linear(64,32)  # 16 mean + 16 logvariance
        self.fast=torch.nn.Sequential(torch.nn.Linear(2*d+16,64),torch.nn.Tanh(),torch.nn.Linear(64,8))
        self.transition=torch.nn.Sequential(torch.nn.Linear(20,64),torch.nn.Tanh(),torch.nn.Linear(64,4))
        self.decoder=torch.nn.Sequential(torch.nn.Linear(20,64),torch.nn.Tanh(),torch.nn.Linear(64,d))
        self.next_valence=torch.nn.Sequential(torch.nn.Linear(20,64),torch.nn.Tanh(),torch.nn.Linear(64,1))
        self.trait_head=torch.nn.Linear(16,5)

    def decode_person(self,phi):
        return self.decoder(torch.cat([phi,phi.new_zeros(len(phi),4)],-1))


def prepare(rows,people,split,cues):
    training=rows[rows.user_id.isin(split['train'])]
    imputer=SimpleImputer(strategy='median').fit(training[cues])
    scaler=StandardScaler().fit(imputer.transform(training[cues]))
    t=people[people.user_id.isin(split['train']) & people[TRAITS].notna().all(1)]
    ts=StandardScaler().fit(t[TRAITS])
    lookup=people.set_index('user_id')[TRAITS]
    blocks={}
    for name in ('train','dev'):
        groups=list(rows[rows.user_id.isin(split[name])].groupby('user_id',sort=True))
        n,steps,d=len(groups),max(len(g) for _,g in groups),len(cues)
        x=np.zeros((n,steps,d),dtype='float32'); obs=np.zeros_like(x)
        valid=np.zeros((n,steps),dtype=bool); y=np.zeros((n,steps),dtype='float32')
        traits=np.zeros((n,5),dtype='float32'); tv=np.zeros(n,dtype=bool); lengths=[]
        for i,(uid,g) in enumerate(groups):
            l=len(g); lengths.append(l)
            x[i,:l]=scaler.transform(imputer.transform(g[cues]))
            obs[i,:l]=g[cues].notna().to_numpy(dtype='float32')
            valid[i,:l]=True; y[i,:l]=(g.valence.to_numpy()-1)/5
            v=lookup.loc[uid].to_numpy(dtype=float)
            if np.isfinite(v).all():
                tv[i]=True; traits[i]=ts.transform(pd.DataFrame([v],columns=TRAITS))[0]
        blocks[name]={'x':torch.tensor(x),'observed':torch.tensor(obs),'valid':torch.tensor(valid),
                      'y':torch.tensor(y),'traits':torch.tensor(traits),'trait_valid':torch.tensor(tv),
                      'lengths':torch.tensor(lengths),'people':n,'pairs':sum(l-1 for l in lengths)}
    pre={'cues':cues,'medians':imputer.statistics_.tolist(),'mean':scaler.mean_.tolist(),
         'scale':scaler.scale_.tolist(),'trait_mean':ts.mean_.tolist(),'trait_scale':ts.scale_.tolist()}
    return blocks,pre


def path_distance(model,a,b):
    n=len(a); start=a[:,None].expand(n,n,16).reshape(-1,16)
    end=b[None].expand(n,n,16).reshape(-1,16); step=(end-start)/8
    with torch.no_grad():
        previous=model.decode_person(start); length=start.new_zeros(len(start))
        for k in range(1,9):
            current=model.decode_person(start+k*step)
            length+=((current-previous).square().sum(1)+1e-4*step.square().sum(1)+1e-12).sqrt()
            previous=current
    return length.reshape(n,n)


def main():
    output=ROOT/'artifacts/psycharchives/highdim_m1_v2'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing overwrite: {output}')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    start=time.time()
    paths={n:SOURCE/n for n in ['Codebook.csv','person_variables.csv','es_sensing_variables.csv']}
    codebook=pd.read_csv(paths['Codebook.csv'],sep=';')
    assert set(codebook.loc[codebook.Feature.duplicated(),'Feature']) <= {'user_id'}
    codebook=codebook.drop_duplicates('Feature').set_index('Feature')
    people=pd.read_csv(paths['person_variables.csv'])
    rows=pd.read_csv(paths['es_sensing_variables.csv'])
    assert not people.user_id.duplicated().any()
    assert not rows.duplicated(['user_id','es_questionnaire_id']).any()
    assert rows.user_id.isin(people.user_id).all()
    assert rows.valence.dropna().between(1,6).all()
    rows=rows.sort_values(['user_id','es_questionnaire_id']).reset_index(drop=True)
    split=split_people(people)
    assert not(split['train']&split['dev'] or split['train']&split['test'] or split['dev']&split['test'])
    # Cue_Group1 distinguishes phone, situation and environment perspectives;
    # these are all legitimate sensors. Exclude actual self-report/ID variables,
    # not sensing columns merely because their cue group has a different name.
    candidates=[c for c in rows if c in codebook.index and
                c not in ('user_id','es_questionnaire_id','valence') and
                not c.startswith('diamonds_') and
                str(codebook.loc[c,'Data_Type']) not in ('ES','Survey')]
    trainmask=rows.user_id.isin(split['train'])
    transformed=[]; rejected=[]
    for c in candidates:
        values=pd.to_numeric(rows[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
        bounds=str(codebook.loc[c,'Value Range'])
        if bounds=='0-1':
            values=values.where(values.between(0,1))
        elif bounds.startswith('1:'):
            values=values.where(values.isin([0,1]))
        if any(k in c for k in ('totalNum','totalDur','avgDur','maxDur','minDur','varDur')):
            values=np.log1p(values.where(values>=0)); transformed.append(c)
        rows[c]=values
        tr=values[trainmask]
        coverage=tr.notna().mean()
        support=tr.groupby(rows.loc[trainmask,'user_id']).count().gt(0).mean()
        most_common=tr.value_counts(normalize=True).max() if tr.notna().any() else 1
        reason=None
        if coverage<.7: reason='training observation coverage <70%'
        elif support<.7: reason='observed in <70% training people'
        elif tr.nunique()<2 or tr.std()<1e-6: reason='near constant'
        elif most_common>.98: reason='dominant value >98% of observed training values'
        if reason: rejected.append({'feature':c,'reason':reason})
    rejected_set={v['feature'] for v in rejected}
    eligible=[c for c in candidates if c not in rejected_set]
    # Greedy train-only redundancy removal; original dictionary column order fixed.
    temp=rows.loc[trainmask,eligible]
    temp=temp.fillna(temp.median())
    corr=temp.corr().abs().to_numpy()
    keep=[]; redundant=[]
    for i,c in enumerate(eligible):
        if keep and np.any(corr[i,keep]>=.98):
            redundant.append(c)
        else: keep.append(i)
    selected=[eligible[i] for i in keep]
    # Common current-valence input in BOTH conditions, not a trait label.
    legacy=list(CUES)
    high=selected+['valence']
    assert len(high)==len(set(high)) and len(high)>22
    audit={'source_columns':len(rows.columns),'sensing_candidates':len(candidates),
           'excluded_self_reports':'8 DIAMONDS outcomes; valence retained identically in both arms',
           'coverage_variation_eligible':len(eligible),'redundancy_removed':len(redundant),
           'selected_sensing_columns':len(selected),'highdim_input_count':len(high),
           'selected_prefix_counts':dict(collections.Counter(c.split('_')[0] for c in selected)),
           'selected_features':selected,'rejections':rejected,'redundant_features':redundant,
           'selection_fitted_on':'train people only','log1p_features':transformed}
    (output/'feature_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    protocol={'seed':42,'epochs':30,'slow_dim':16,'fast_dim':4,'history_hidden':64,
              'trait_training_loss':0,'contrastive_training_loss':0,
              'weights':dict(WEIGHTS,traits=0),'conditions':{'legacy22':legacy,'highdim':high},
              'same_preprocessing_policy_both_arms':True,
              'input_clipping':'none; train median/standardization, domain validation and log1p count/duration',
              'feature_selection':'train >=70% snapshots and >=70% people; nonconstant; <=98% dominant value; abs correlation <.98',
              'source_sha256':{n:sha256(p) for n,p in paths.items()},
              'script_sha256':sha256(Path(__file__)),
              'cohort':{k:len(v) for k,v in split.items()},
              'test':'no test-based selection/scaling; no test model inference or scoring',
              'global_validation':'codebook-fixed range/log transforms applied in memory, never fitted on test',
              'preprocessing_parameters_fitted_on_train':True,
              'split_sha256':{k:__import__('hashlib').sha256(json.dumps(sorted(v)).encode()).hexdigest() for k,v in split.items()},
              'fixed_epochs_not_dev_selected':True,
              'audit_version':'v2 includes all sensing cue groups; completed v1 with Smartphone_Cue-only screening preserved as restricted-scope pilot',
              'metric':'J_person g(phi,z=0)^T J_person g(phi,z=0)+1e-4 I',
              'metric_ambient_units':'training-standardized transformed cues; differs across feature sets',
              'path':'8-segment straight-coordinate path, not optimized geodesic',
              'order':'questionnaire ID, not fixed-hour timestamps',
              'limitations':['single seed; dimension16 choice from IPIP not validated here',
                             'unequal feature counts mean unequal parameter counts',
                             'missingness masks may encode device/context differences',
                             'many KB columns can dominate feature-wise reconstruction',
                             'Gaussian KL and posterior uncertainty are not calibrated']}
    # Codebook-fixed conversion is applied above to sensing columns; valence remains 1..6.
    for c in CUES:
        assert c in rows
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    (output/'training_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in audit.items() if k not in ('selected_features','rejections','redundant_features','log1p_features')}),flush=True)
    report={'protocol':protocol,'runs':[]}
    for condition,cues in [('legacy22',legacy),('highdim',high)]:
        blocks,pre=prepare(rows,people,split,cues)
        torch.manual_seed(42)
        model=HighDimManifold(len(cues))
        opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001)
        weights=dict(WEIGHTS,traits=0)
        history=[]
        for epoch in range(1,31):
            model.train()
            losses=[]
            for idx in torch.randperm(blocks['train']['people']).split(32):
                opt.zero_grad(set_to_none=True)
                loss=train_loss(model,batch_view(blocks['train'],idx),weights)
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
                losses.append(loss.item())
            history.append(float(np.mean(losses)))
            if epoch%10==0: print(condition,epoch,'loss',history[-1],flush=True)
        model.eval()
        result=development_metrics(model,blocks['dev'])
        result.pop('trait_rmse_training_scaled')  # untrained trait head is not evidence
        result['trait_linear_probe']=trait_probe(model,blocks['train'],blocks['dev'])
        early,late=independent_windows(model,blocks['dev'])
        result['window_euclidean']=matching_metrics(torch.cdist(early,late))
        result['window_decoder_path']=matching_metrics(path_distance(model,early,late))
        eig=[]
        for point in early[:10]:
            jac=torch.autograd.functional.jacobian(lambda p:model.decode_person(p[None])[0],point)
            eig.append(torch.linalg.eigvalsh(jac.T@jac+1e-4*torch.eye(16)).detach().numpy())
        eig=np.stack(eig)
        result['metric']={'minimum_eigenvalue':float(eig.min()),'maximum_eigenvalue':float(eig.max()),
                          'median_condition_number':float(np.median(eig[:,-1]/eig[:,0]))}
        result['parameter_count']=sum(p.numel() for p in model.parameters())
        torch.save({'state_dict':model.state_dict(),'preprocessing':pre,'protocol':protocol,
                    'condition':condition,'input_dim':len(cues)},output/f'{condition}_seed42.pt')
        report['runs'].append({'condition':condition,'input_dim':len(cues),'history':history,**result})
        (output/'development_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(condition,'DEV',result['next_valence_rmse_1_to_6'],'TRAIT',result['trait_linear_probe']['macro_trait_r2'],flush=True)
    report['elapsed_seconds']=time.time()-start
    (output/'development_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('COMPLETE',output,flush=True)


if __name__=='__main__': main()
