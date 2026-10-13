"""Small CPU implementation checks; does not access questionnaire respondents."""
import unittest
import numpy as np
import torch
from run_ipip_manifold import (PersonalityManifold, path_distances, masked_mse,
                               item_scores, split_form_contrastive)


class ManifoldChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.model = PersonalityManifold(8).eval()

    def test_unobserved_inputs_do_not_leak(self):
        x = torch.randn(4, 300)
        mask = torch.ones_like(x)
        mask[:, 150:] = 0
        changed = x.clone()
        changed[:, 150:] += 100
        p, mu, _ = self.model(x, mask)
        pp, mm, _ = self.model(changed, mask)
        torch.testing.assert_close(p, pp)
        torch.testing.assert_close(mu, mm)

    def test_paths_symmetric_and_finite_gradient(self):
        a = torch.randn(4, 8, requires_grad=True)
        d = path_distances(self.model, a, a)
        torch.testing.assert_close(d, d.T, atol=1e-5, rtol=1e-5)
        d.mean().backward()
        self.assertTrue(torch.isfinite(a.grad).all())

    def test_metric_positive_and_finite_difference(self):
        p = torch.randn(8)
        j = torch.func.jacfwd(self.model.decoder)(p)
        g = j.T @ j + 1e-4*torch.eye(8)
        self.assertGreater(torch.linalg.eigvalsh(g).min().item(), 0)
        direction = torch.ones(8)/np.sqrt(8)
        finite = (self.model.decoder(p+.001*direction)-self.model.decoder(p-.001*direction))/.002
        torch.testing.assert_close(j@direction, finite, atol=1e-4, rtol=1e-3)

    def test_masked_loss_and_score(self):
        pred = torch.tensor([[1.,100.],[2.,100.]])
        truth = torch.tensor([[1.,0.],[2.,0.]])
        observed = torch.tensor([[1.,0.],[1.,0.]])
        self.assertEqual(masked_mse(pred,truth,observed).item(),0)
        score = item_scores(truth.numpy(),pred.numpy(),observed.numpy().astype(bool))
        self.assertEqual(score['rmse_1_to_5'],0)
        self.assertEqual(score['macro_item_r2'],1)

    def test_contrastive_conditions(self):
        for condition in ('decoder_path','euclidean','none'):
            a=torch.randn(5,8,requires_grad=True)
            b=torch.randn(5,8,requires_grad=True)
            loss=split_form_contrastive(self.model,a,b,condition)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(a.grad).all())
            if condition=='none':
                self.assertEqual(loss.item(),0)

    def test_refactor_preserves_original_path_objective(self):
        a=torch.randn(5,8)
        b=torch.randn(5,8)
        d=path_distances(self.model,a,b)
        scale=d[~torch.eye(5,dtype=torch.bool)].mean().clamp_min(1e-6)
        logits=-d/(scale*.2)
        target=torch.arange(5)
        old=(torch.nn.functional.cross_entropy(logits,target)+
             torch.nn.functional.cross_entropy(logits.T,target))/2
        torch.testing.assert_close(old,split_form_contrastive(self.model,a,b),rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
