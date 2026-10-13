"""Fixed-split 8/16/32 dimensional M1 comparison with three training seeds."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
DIMS = (8, 16, 32)
SEEDS = (42, 43, 44)


def main():
    output = ROOT/'artifacts/ipip/dimension_sweep_v1'
    output.mkdir(parents=True, exist_ok=True)
    protocol = {'dimensions': DIMS, 'training_seeds': SEEDS, 'data_seed': 42,
                'epochs': 30, 'reserved_test': 'not scored',
                'reuse': 'original completed 8D seed42 run; verified split and item forms',
                'selection': 'exploratory development comparison, not confirmatory test',
                'parameter_capacity': 'same hidden layers; latent dimension changes parameter count',
                'kl_normalization': 'KL mean per coordinate, as in original pilot; not dimension-summed KL',
                'source_sha256': hashlib.sha256((ROOT/'run_ipip_manifold.py').read_bytes()).hexdigest()}
    (output/'sweep_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    original = ROOT/'artifacts/ipip/m1_seed42'
    reference = np.load(original/'prepared_data.npz')
    keys = ['train_source_row_indices','dev_source_row_indices','test_source_row_indices',
            'form_a','form_b','mean','std']
    runs = []
    for dim in DIMS:
        for seed in SEEDS:
            run = original if (dim,seed)==(8,42) else output/f'dim{dim}_seed{seed}'
            result_file = run/'development_results.json'
            if not result_file.exists():
                if run.exists():
                    raise RuntimeError(f'Incomplete prior run requires review, refusing overwrite: {run}')
                print(f'START dim={dim} seed={seed}',flush=True)
                with (output/f'dim{dim}_seed{seed}.log').open('w',encoding='utf-8') as log:
                    subprocess.run([sys.executable,'-u',str(ROOT/'run_ipip_manifold.py'),
                                    '--latent-dim',str(dim),'--seed',str(seed),'--data-seed','42',
                                    '--output',str(run)],stdout=log,stderr=subprocess.STDOUT,check=True)
            p = json.loads((run/'protocol.json').read_text())
            assert p['latent_dim']==dim and p['seed']==seed and p['epochs']==30
            assert p['test']=='reserved, not scored'
            actual = np.load(run/'prepared_data.npz')
            for key in keys:
                assert np.array_equal(reference[key],actual[key]), (run,key)
            actual.close()
            result = json.loads(result_file.read_text())
            neural = result['m1_hidden_b_from_a']
            row = {'dim':dim,'seed':seed,'run':str(run),
                   'best_epoch':result['best_epoch'],
                   'rmse':neural['hidden_items']['rmse_1_to_5'],
                   'item_r2':neural['hidden_items']['macro_item_r2'],
                   'facet_r2':neural['hidden_facet_scores']['macro_r2'],
                   'domain_r2':neural['hidden_domain_scores']['macro_r2'],
                   'match_euclidean':result['split_form_case_matching']['euclidean_rank1'],
                   'match_path':result['split_form_case_matching']['decoder_path_rank1'],
                   'metric_condition':result['geometry']['median_condition_number'],
                   'metric_min_eigenvalue':result['geometry']['min_eigenvalue'],
                   'posterior_mean_variance':float(np.mean(result['posterior']['mean_dev_variance_by_dimension'])),
                   'pca_rmse':result['pca_latent_ridge']['hidden_items']['rmse_1_to_5'],
                   'pca_facet_r2':result['pca_latent_ridge']['hidden_facet_scores']['macro_r2'],
                   'raw_ridge_rmse':result['raw_a_ridge']['hidden_items']['rmse_1_to_5']}
            runs.append(row)
            (output/'progress.json').write_text(json.dumps(runs,indent=2),encoding='utf-8')
            print(f'DONE dim={dim} seed={seed} RMSE={row["rmse"]:.5f} facet_R2={row["facet_r2"]:.4f}',flush=True)
    reference.close()
    measures = [key for key in runs[0] if key not in ('dim','seed','run')]
    summary = {}
    for dim in DIMS:
        subset = [r for r in runs if r['dim']==dim]
        summary[str(dim)] = {key:{'mean':float(np.mean([r[key] for r in subset])),
                                  'sd_across_training_seeds':float(np.std([r[key] for r in subset],ddof=1))}
                             for key in measures}
    report = {'protocol':protocol, 'identical_splits_forms_scaling_verified':True,
              'runs':runs,'summary':summary,
              'sd_is_not_sampling_confidence_interval':True}
    (output/'comparison_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
