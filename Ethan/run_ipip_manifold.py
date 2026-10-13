"""IPIP-NEO-300 M1 pilot: masked nonlinear VAE and decoder-pullback geometry.

Reads the original nested ZIP without modifying it. Test responses are not scored.
The questionnaire is cross-sectional; no temporal or LLM-persona claim is made.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xlrd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

RIDGE = 1e-4
DOMAINS = ('N', 'E', 'O', 'A', 'C')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def load_source(path):
    with zipfile.ZipFile(path) as outer:
        member = next(n for n in outer.namelist() if '(2005)' in n)
        with zipfile.ZipFile(io.BytesIO(outer.read(member))) as inner:
            raw = inner.read('ipip20993.dat')
            doc = inner.read('DAT20993.doc')
        key_member = next(n for n in outer.namelist() if 'Scoring Key' in n)
        with zipfile.ZipFile(io.BytesIO(outer.read(key_member))) as inner:
            key_bytes = inner.read('IPIP-NEO-ItemKey.xls')
    frame = pd.read_csv(io.BytesIO(raw), sep='\t', low_memory=False)
    sheet = xlrd.open_workbook(file_contents=key_bytes).sheet_by_index(0)
    items = sorted([{'number': int(sheet.cell_value(i, 0)),
                     'sign': sheet.cell_value(i, 2), 'facet': sheet.cell_value(i, 3),
                     'facet_name': sheet.cell_value(i, 4)}
                    for i in range(1, sheet.nrows)], key=lambda v: v['number'])
    assert [v['number'] for v in items] == list(range(1, 301))
    # The local DAT20993 codebook explicitly states negative items are ALREADY reversed.
    assert b'Reverse-scored items were recoded' in doc
    array = frame[[f'I{i}' for i in range(1, 301)]].apply(
        pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
    observed = np.isin(array, [1, 2, 3, 4, 5])
    array[~observed] = np.nan
    facets = {key: np.array([i for i, v in enumerate(items) if v['facet'] == key])
              for key in sorted({v['facet'] for v in items})}
    assert len(facets) == 30 and all(len(v) == 10 for v in facets.values())
    coverage = observed.sum(1) >= 270
    facet_coverage = np.all(np.stack([observed[:, v].sum(1) >= 8
                                    for v in facets.values()], 1), 1)
    reliable = ((pd.to_numeric(frame.RNDSPLIT, errors='coerce') >= .5) &
                (pd.to_numeric(frame.RNDCORR, errors='coerce') >= 0)).to_numpy()
    eligible = coverage & facet_coverage & reliable
    rows = np.flatnonzero(eligible)
    # Exact answer-vector duplicates, including missingness, are removed BEFORE splitting.
    _, first = np.unique(np.nan_to_num(array[rows], nan=0), axis=0, return_index=True)
    rows = rows[np.sort(first)]
    audit = {'raw_response_cases': len(frame), 'input_items': 300,
             'eligible_before_deduplication': int(eligible.sum()),
             'retained_response_cases': len(rows),
             'exact_duplicate_answer_vectors_removed': int(eligible.sum() - len(rows)),
             'coverage_below_270': int((~coverage).sum()),
             'any_facet_below_8_observed': int((~facet_coverage).sum()),
             'reliability_screen_failed': int((~reliable).sum()),
             'exclusion_counts_overlap': True,
             'retained_missing_fraction': float((~observed[rows]).mean()),
             'rules': '>=270 valid answers, >=8 per facet, RNDSPLIT>=0.5, RNDCORR>=0; exploratory engineering screens, not official diagnostic cutoffs',
             'reverse_scoring': 'already applied in original DAT file; NOT repeated',
             'cases_are_not_verified_unique_people': True,
             'sha256': {'dat': digest(raw), 'codebook': digest(doc), 'item_key': digest(key_bytes)}}
    return array[rows], observed[rows], rows, items, facets, audit


class PersonalityManifold(torch.nn.Module):
    def __init__(self, dimensions=8):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(600, 256), torch.nn.SiLU(),
                                           torch.nn.Linear(256, 128), torch.nn.SiLU())
        self.mu = torch.nn.Linear(128, dimensions)
        self.logvar = torch.nn.Linear(128, dimensions)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(dimensions, 128), torch.nn.SiLU(),
                                           torch.nn.Linear(128, 256), torch.nn.SiLU(),
                                           torch.nn.Linear(256, 300))

    def encode(self, x, mask):
        h = self.encoder(torch.cat([x * mask, mask], 1))
        return self.mu(h), self.logvar(h).clamp(-8, 4)

    def forward(self, x, mask):
        mu, logvar = self.encode(x, mask)
        phi = mu + torch.randn_like(mu) * (.5 * logvar).exp() if self.training else mu
        return self.decoder(phi), mu, logvar


def path_distances(model, a, b, segments=4):
    """All-pairs straight paths. Includes the same ridge term as local_metric."""
    start = a[:, None].expand(-1, len(b), -1).reshape(-1, a.shape[1])
    end = b[None].expand(len(a), -1, -1).reshape(-1, a.shape[1])
    step = (end - start) / segments
    previous = model.decoder(start)
    distance = torch.zeros(len(start), device=a.device)
    for k in range(1, segments + 1):
        current = model.decoder(start + k * step)
        distance = distance + ((current - previous).square().sum(1) +
                               RIDGE * step.square().sum(1) + 1e-12).sqrt()
        previous = current
    return distance.reshape(len(a), len(b))


def masked_mse(pred, target, observed):
    return ((pred - target).square() * observed).sum() / observed.sum().clamp_min(1)


def split_form_contrastive(model, a, b, condition='decoder_path'):
    """Matched objective; only the distance definition changes across conditions."""
    if condition == 'none' or len(a) < 2:
        return a.sum() * 0
    distances = (torch.cdist(a, b) if condition == 'euclidean'
                 else path_distances(model, a, b))
    off = ~torch.eye(len(a), dtype=torch.bool, device=a.device)
    scale = distances[off].mean().clamp_min(1e-6)
    logits = -distances / (scale * .2)
    labels = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))/2


def item_scores(truth, predicted, observed):
    residual = np.where(observed, truth - predicted, 0)
    n = observed.sum(0)
    means = np.divide(np.where(observed, truth, 0).sum(0), n,
                      out=np.zeros_like(n, dtype=np.float64), where=n > 0)
    ss = np.where(observed, (truth - means)**2, 0).sum(0)
    valid = ss > 0
    r2 = 1 - (residual**2).sum(0)[valid] / ss[valid]
    return {'rmse_1_to_5': float(np.sqrt((residual**2).sum() / observed.sum())),
            'macro_item_r2': float(r2.mean())}


def scale_scores(truth, predicted, observed, groups):
    out = {}
    for name, ids in groups.items():
        counts = observed[:, ids].sum(1)
        valid = counts >= max(2, int(.6 * len(ids)))
        actual = np.where(observed[:, ids], truth[:, ids], 0).sum(1) / counts.clip(1)
        estimate = np.where(observed[:, ids], predicted[:, ids], 0).sum(1) / counts.clip(1)
        a, p = actual[valid], estimate[valid]
        ss = ((a - a.mean())**2).sum()
        out[name] = {'r2': float(1 - ((a-p)**2).sum() / ss),
                     'rmse_mean_item_1_to_5': float(np.sqrt(((a-p)**2).mean())),
                     'valid_dev_cases': int(valid.sum())}
    return {'macro_r2': float(np.mean([v['r2'] for v in out.values()])), 'by_scale': out}


@torch.no_grad()
def infer(model, x, observed, form=None):
    predictions, coordinates, variances = [], [], []
    for start in range(0, len(x), 512):
        mask = observed[start:start+512] if form is None else observed[start:start+512]*form
        p, mu, lv = model(x[start:start+512], mask)
        predictions.append(p.cpu().numpy())
        coordinates.append(mu.cpu().numpy())
        variances.append(lv.exp().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(coordinates), np.concatenate(variances)


def geometry_diagnostics(model, phi, device):
    coords = torch.tensor(phi[:32], device=device)
    eigens = []
    finite_diff_error = []
    for p in coords:
        jac = torch.func.jacfwd(model.decoder)(p)
        g = jac.T @ jac + RIDGE * torch.eye(p.numel(), device=device)
        eigens.append(torch.linalg.eigvalsh(g).detach().cpu().numpy())
        direction = torch.ones_like(p) / np.sqrt(p.numel())
        finite = (model.decoder(p + 1e-3 * direction)-model.decoder(p-1e-3*direction))/2e-3
        finite_diff_error.append(float((finite-jac@direction).norm().detach().cpu()))
    eigens = np.stack(eigens)
    with torch.no_grad():
        dist = path_distances(model, coords[:16], coords[16:32], 32).diagonal()
        euclid = torch.linalg.vector_norm(coords[:16]-coords[16:32], dim=1)
    return {'metric': 'J_g(phi)^T J_g(phi) + 1e-4 I in train-standardized item units',
            'sampled_dev_cases': len(coords), 'min_eigenvalue': float(eigens.min()),
            'max_eigenvalue': float(eigens.max()),
            'median_condition_number': float(np.median(eigens[:, -1] / eigens[:, 0])),
            'mean_jacobian_directional_finite_difference_error': float(np.mean(finite_diff_error)),
            'median_straight_decoder_path_length_16_pairs': float(dist.median().cpu()),
            'median_coordinate_distance_16_pairs': float(euclid.median().cpu()),
            'path_is_not_optimized_geodesic': True,
            'positive_metric_is_not_evidence_of_curvature_or_personality_validity': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path(__file__).parent/'data/Kajonius_Johnson_2019_SUPPL.zip')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'artifacts/ipip/m1_seed42')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-seed', type=int, default=42,
                        help='Fixed response-case split and item forms, independent of training seed')
    parser.add_argument('--latent-dim', type=int, default=8)
    parser.add_argument('--geometry-weight', type=float, default=.02)
    parser.add_argument('--contrastive-distance', choices=['decoder_path','euclidean','none'],
                        default='decoder_path')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite existing run: {args.output}')
    args.output.mkdir(parents=True)
    start = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    raw, obs, row_ids, items, facets, audit = load_source(args.source)
    rng = np.random.default_rng(args.data_seed)
    order = rng.permutation(len(raw))
    ntrain, ndev = int(.7*len(raw)), int(.15*len(raw))
    splits = {'train': order[:ntrain], 'dev': order[ntrain:ntrain+ndev], 'test': order[ntrain+ndev:]}
    form_a = np.zeros(300, dtype=np.float32)
    for ids in facets.values():
        form_a[rng.permutation(ids)[:5]] = 1
    form_b = 1 - form_a
    means = np.nanmean(raw[splits['train']], 0)
    std = np.nanstd(raw[splits['train']], 0).clip(.1)
    normalized = np.nan_to_num((raw - means)/std, nan=0)
    # Test rows are reserved here; never sent into model or metric code.
    blocks = {name: (torch.tensor(normalized[ids], device=args.device),
                     torch.tensor(obs[ids].astype(np.float32), device=args.device))
              for name, ids in splits.items() if name != 'test'}
    a = torch.tensor(form_a, device=args.device)
    b = torch.tensor(form_b, device=args.device)
    protocol = {'model': 'M1 masked denoising nonlinear variational autoencoder with optional split-form contrastive loss',
                'input': '300 already-keyed ordinal items; Gaussian/MSE continuous relaxation',
                'architecture': '600 (values+mask) -> 256 SiLU -> 128 SiLU -> Gaussian phi; phi ->128 SiLU ->256 SiLU ->300',
                'latent_dim': args.latent_dim, 'seed': args.seed, 'data_seed': args.data_seed,
                'epochs': args.epochs,
                'batch_size': 256, 'learning_rate': .001, 'weight_decay': .0001,
                'random_input_dropout': .3, 'kl_weight': .001, 'split_form_reconstruction_weight': .5,
                'geometry_weight': args.geometry_weight if args.contrastive_distance!='none' else 0,
                'contrastive_distance': args.contrastive_distance, 'contrastive_batch_size': 32,
                'temperature': .2, 'training_path_segments': 4, 'ridge': RIDGE,
                'objective': 'masked full reconstruction + .5*(A-view reconstruction+B-view reconstruction)/2 + .001 KL + effective geometry_weight symmetric split-form InfoNCE using contrastive_distance',
                'best_checkpoint_rule': 'lowest dev B-item standardized MSE from A-input',
                'split_counts': {name: len(ids) for name, ids in splits.items()},
                'unit': 'response case, not verified unique person', 'test': 'reserved, not scored',
                'form_a_item_numbers': (np.flatnonzero(form_a)+1).tolist(),
                'form_b_item_numbers': (np.flatnonzero(form_b)+1).tolist(),
                'references': ['https://arxiv.org/abs/1312.6114', 'https://arxiv.org/abs/1710.11379',
                               'https://doi.org/10.23668/psycharchives.2448'],
                'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                            'device': args.device, 'gpu': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None},
                'script_sha256': digest(Path(__file__).read_bytes())}
    write_json(args.output/'protocol.json', protocol)
    # Preserve the exact executable source with every run, even if the live script changes.
    (args.output/'training_source.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.output/'data_audit.json', audit)
    write_json(args.output/'item_metadata.json', items)
    np.savez_compressed(args.output/'prepared_data.npz', x=normalized, observed=obs,
                        mean=means, std=std, **{f'{k}_source_row_indices': row_ids[v] for k,v in splits.items()},
                        **{f'{k}_indices': v for k,v in splits.items()}, form_a=form_a, form_b=form_b)
    print(json.dumps({'data_audit': audit, 'split_counts': protocol['split_counts']}), flush=True)
    model = PersonalityManifold(args.latent_dim).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    train_x, train_obs = blocks['train']
    dev_x, dev_obs = blocks['dev']
    history, best, best_state, best_epoch = [], float('inf'), None, None
    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        for indices in torch.randperm(len(train_x), device=args.device).split(256):
            x, observed = train_x[indices], train_obs[indices]
            mask = observed * (torch.rand_like(observed) > .3)
            p, mu, lv = model(x, mask)
            rec = masked_mse(p, x, observed)
            kl = -.5 * (1 + lv - mu.square()-lv.exp()).mean()
            pa, mua, _ = model(x, observed*a)
            pb, mub, _ = model(x, observed*b)
            cross = (masked_mse(pa,x,observed)+masked_mse(pb,x,observed))/2
            m = min(32, len(x))
            identity = split_form_contrastive(model, mua[:m], mub[:m], args.contrastive_distance)
            loss = rec + .5*cross + .001*kl + args.geometry_weight*identity
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            losses.append([loss.item(), rec.item(), kl.item(), identity.item()])
        model.eval()
        pred, _, _ = infer(model, dev_x, dev_obs, a)
        mask = obs[splits['dev']]*form_b
        mse = float((((pred-normalized[splits['dev']])**2)*mask).sum()/mask.sum())
        entry = dict(zip(['loss', 'reconstruction', 'kl', 'contrastive'], np.mean(losses,0).tolist()))
        entry.update(epoch=epoch, dev_hidden_b_mse_standardized=mse)
        history.append(entry)
        if mse < best:
            best, best_epoch = mse, epoch
            best_state = copy.deepcopy(model.state_dict())
        write_json(args.output/'training_history.json', history)
        print(f'epoch={epoch:02d} loss={entry["loss"]:.4f} dev_hidden_B_MSE={mse:.4f}', flush=True)
    model.load_state_dict(best_state)
    model.eval()
    torch.save({'state_dict': model.state_dict(), 'mean': means, 'std': std, 'protocol': protocol,
                'best_epoch': best_epoch, 'form_a': form_a, 'form_b': form_b}, args.output/'model.pt')
    pred_full, phi, variance = infer(model, dev_x, dev_obs)
    pred_a, phi_a, _ = infer(model, dev_x, dev_obs, a)
    _, phi_b, _ = infer(model, dev_x, dev_obs, b)
    pred_a = pred_a * std + means
    pred_full = pred_full * std + means
    truth = np.nan_to_num(raw[splits['dev']], nan=0)
    hidden_obs = obs[splits['dev']] & form_b.astype(bool)
    facet_b = {key: ids[form_b[ids].astype(bool)] for key,ids in facets.items()}
    domain_b = {key: np.array([i for i,v in enumerate(items) if v['facet'].startswith(key) and form_b[i]])
                for key in DOMAINS}
    def evaluate(p):
        return {'hidden_items': item_scores(truth, p, hidden_obs),
                'hidden_facet_scores': scale_scores(truth,p,hidden_obs,facet_b),
                'hidden_domain_scores': scale_scores(truth,p,hidden_obs,domain_b)}
    results = {'status': 'single-seed exploratory development evaluation; test unscored',
               'best_epoch': best_epoch, 'm1_hidden_b_from_a': evaluate(pred_a),
               'mean_baseline': evaluate(np.broadcast_to(means, truth.shape)),
               'full_input_reconstruction_not_independent_validity': item_scores(truth,pred_full,obs[splits['dev']])}
    # Fair observed-A-only baselines: missing A uses training means; fit targets only on train.
    aa, bb = np.flatnonzero(form_a), np.flatnonzero(form_b)
    xt = normalized[splits['train']][:,aa]
    yt = normalized[splits['train']][:,bb]
    xd = normalized[splits['dev']][:,aa]
    # Missing B labels use training means (zero standardized); documented baseline simplification.
    pca = PCA(n_components=args.latent_dim, random_state=args.seed).fit(xt)
    for name, features, query in [('pca_latent_ridge',pca.transform(xt),pca.transform(xd)),
                                  ('raw_a_ridge',xt,xd)]:
        ridge = Ridge(alpha=10).fit(features,yt)
        p = np.broadcast_to(means,truth.shape).copy()
        p[:,bb] = ridge.predict(query)*std[bb]+means[bb]
        results[name] = evaluate(p)
    results['baseline_missing_target_policy'] = 'Train-mean-imputed B targets for ridge; neural loss masks missing targets; a limitation for strict head-to-head comparison'
    # Fixed gallery of 256 development cases, same questionnaire split-form, NOT test-retest.
    gallery = min(256,len(phi_a))
    ca, cb = torch.tensor(phi_a[:gallery],device=args.device),torch.tensor(phi_b[:gallery],device=args.device)
    with torch.no_grad():
        eu = torch.cdist(ca,cb)
        path = path_distances(model,ca,cb,8)
        results['split_form_case_matching'] = {
            'gallery': gallery, 'chance_rank1': 1/gallery,
            'euclidean_rank1': float((eu.argmin(1)==torch.arange(gallery,device=args.device)).float().mean().cpu()),
            'decoder_path_rank1': float((path.argmin(1)==torch.arange(gallery,device=args.device)).float().mean().cpu()),
            'not_temporal_stability': True}
    results['geometry'] = geometry_diagnostics(model,phi,args.device)
    results['posterior'] = {'mean_dev_variance_by_dimension': variance.mean(0).tolist(),
                            'mean_coordinate_std_by_dimension': phi.std(0).tolist(),
                            'uncertainty_not_calibrated': True}
    results['elapsed_seconds'] = time.time()-start
    np.savez_compressed(args.output/'dev_posteriors.npz', phi=phi,variance=variance,phi_a=phi_a,phi_b=phi_b)
    write_json(args.output/'development_results.json',results)
    print(json.dumps({'output':str(args.output), 'best_epoch':best_epoch,
                      'hidden_items':results['m1_hidden_b_from_a']['hidden_items'],
                      'split_form_matching':results['split_form_case_matching']},indent=2),flush=True)


if __name__ == '__main__':
    main()
