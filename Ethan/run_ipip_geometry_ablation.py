"""16D matched decoder-path / Euclidean / no-contrastive comparison."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
CONDITIONS = ('decoder_path','euclidean','none')
SEEDS = (42,43,44)


def main():
    output = ROOT/'artifacts/ipip/geometry_ablation_v1'
    output.mkdir(parents=True,exist_ok=True)
    protocol = {'conditions':CONDITIONS,'seeds':SEEDS,'data_seed':42,'latent_dimension':16,
                'epochs':30,'test':'not scored',
                'reuse':'completed 16D decoder-path runs from dimension_sweep_v1',
                'contrastive_weight':.02,'none_weight':0,
                'constants':'same hidden layers, reconstruction, KL, corruption, forms, split, optimizer, checkpoint rule and gallery',
                'path_readout_is_not_geodesic':True,
                'script_sha256':hashlib.sha256((ROOT/'run_ipip_manifold.py').read_bytes()).hexdigest()}
    (output/'ablation_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    reference = np.load(ROOT/'artifacts/ipip/m1_seed42/prepared_data.npz')
    keys = ['train_source_row_indices','dev_source_row_indices','test_source_row_indices','form_a','form_b','mean','std']
    runs=[]
    for condition in CONDITIONS:
        for seed in SEEDS:
            run = (ROOT/f'artifacts/ipip/dimension_sweep_v1/dim16_seed{seed}' if condition=='decoder_path'
                   else output/f'{condition}_seed{seed}')
            if not (run/'development_results.json').exists():
                if run.exists():
                    raise RuntimeError(f'Incomplete run; refusing overwrite: {run}')
                print(f'START condition={condition} seed={seed}',flush=True)
                with (output/f'{condition}_seed{seed}.log').open('w',encoding='utf-8') as log:
                    subprocess.run([sys.executable,'-u',str(ROOT/'run_ipip_manifold.py'),
                                    '--latent-dim','16','--seed',str(seed),'--data-seed','42',
                                    '--contrastive-distance',condition,'--output',str(run)],
                                   stdout=log,stderr=subprocess.STDOUT,check=True)
            p=json.loads((run/'protocol.json').read_text())
            assert p['latent_dim']==16 and p['seed']==seed and p['epochs']==30
            assert p.get('contrastive_distance','decoder_path')==condition
            assert p['test']=='reserved, not scored'
            assert hashlib.sha256((run/'training_source.py').read_bytes()).hexdigest()==p['script_sha256']
            data=np.load(run/'prepared_data.npz')
            for key in keys:
                assert np.array_equal(reference[key],data[key]),(run,key)
            data.close()
            r=json.loads((run/'development_results.json').read_text())
            pred=r['m1_hidden_b_from_a']
            row={'condition':condition,'seed':seed,'run':str(run),'best_epoch':r['best_epoch'],
                 'rmse':pred['hidden_items']['rmse_1_to_5'], 'item_r2':pred['hidden_items']['macro_item_r2'],
                 'facet_r2':pred['hidden_facet_scores']['macro_r2'],
                 'domain_r2':pred['hidden_domain_scores']['macro_r2'],
                 'match_euclidean':r['split_form_case_matching']['euclidean_rank1'],
                 'match_path':r['split_form_case_matching']['decoder_path_rank1'],
                 'metric_condition':r['geometry']['median_condition_number'],
                 'metric_min_eigenvalue':r['geometry']['min_eigenvalue'],
                 'posterior_variance':float(np.mean(r['posterior']['mean_dev_variance_by_dimension']))}
            runs.append(row)
            (output/'progress.json').write_text(json.dumps(runs,indent=2),encoding='utf-8')
            print(f'DONE {condition} seed={seed} RMSE={row["rmse"]:.5f} path_match={row["match_path"]:.4f}',flush=True)
    reference.close()
    fields=[k for k in runs[0] if k not in ('condition','seed','run')]
    summary={c:{k:{'mean':float(np.mean([r[k] for r in runs if r['condition']==c])),
                    'sd':float(np.std([r[k] for r in runs if r['condition']==c],ddof=1))}
                for k in fields} for c in CONDITIONS}
    paired={}
    for c in ('euclidean','none'):
        paired[c]={k:[next(r[k] for r in runs if r['condition']=='decoder_path' and r['seed']==s)
                       -next(r[k] for r in runs if r['condition']==c and r['seed']==s)
                       for s in SEEDS] for k in ['rmse','facet_r2','match_path','match_euclidean']}
    report={'protocol':protocol,'all_splits_forms_scaling_and_source_snapshots_verified':True,
            'runs':runs,'summary':summary,'decoder_path_minus_comparator_by_seed':paired,
            'uncertainty':'seed SD is not a sampling confidence interval; exploratory development-only results'}
    (output/'comparison_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
