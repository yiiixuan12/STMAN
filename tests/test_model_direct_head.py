#!/usr/bin/env python
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import STFractalTransformer


class ModelDirectHeadTests(unittest.TestCase):
    def _make_inputs(self):
        x = torch.randn(2, 6, 4, 2)
        A = torch.eye(4)
        sf = torch.randn(4, 3)
        tf = torch.randn(4, 3)
        return x, A, sf, tf

    def test_legacy_direct_head_still_produces_expected_shape(self):
        model = STFractalTransformer(
            n_nodes=4,
            in_dim=2,
            d_model=8,
            n_heads=2,
            seq_len=6,
            pred_len=24,
            sf_dim=3,
            tf_dim=3,
            out_dim=1,
            use_direct_pred=True,
        )
        x, A, sf, tf = self._make_inputs()

        y_hat, _ = model(x, A, sf, tf)

        self.assertEqual(tuple(y_hat.shape), (2, 24, 4, 1))

    def test_chunkwise_direct_head_uses_per_chunk_temporal_projection(self):
        model = STFractalTransformer(
            n_nodes=4,
            in_dim=2,
            d_model=8,
            n_heads=2,
            seq_len=6,
            pred_len=24,
            sf_dim=3,
            tf_dim=3,
            out_dim=1,
            use_direct_pred=True,
            direct_head_mode="chunkwise",
        )
        x, A, sf, tf = self._make_inputs()

        y_hat, _ = model(x, A, sf, tf)

        self.assertEqual(tuple(y_hat.shape), (2, 24, 4, 1))
        self.assertEqual(model.temporal_chunk_proj.out_features, model._dp_n_chunks)

    def test_attn_direct_head_uses_content_adaptive_chunk_queries(self):
        model = STFractalTransformer(
            n_nodes=4,
            in_dim=2,
            d_model=8,
            n_heads=2,
            seq_len=6,
            pred_len=24,
            sf_dim=3,
            tf_dim=3,
            out_dim=1,
            use_direct_pred=True,
            direct_head_mode="attn",
        )
        x, A, sf, tf = self._make_inputs()

        y_hat, _ = model(x, A, sf, tf)

        self.assertEqual(tuple(y_hat.shape), (2, 24, 4, 1))
        self.assertEqual(model.chunk_query.num_embeddings, model._dp_n_chunks)
        self.assertEqual(model.temporal_key.out_features, model.d_model)

    def test_hybrid_spatial_mode_keeps_direct_head_shape(self):
        model = STFractalTransformer(
            n_nodes=4,
            in_dim=2,
            d_model=8,
            n_heads=2,
            seq_len=6,
            pred_len=12,
            sf_dim=3,
            tf_dim=3,
            out_dim=1,
            use_direct_pred=True,
            direct_head_mode="attn",
            spatial_mode="hybrid",
            k_hop=2,
        )
        x, A, sf, tf = self._make_inputs()

        y_hat, _ = model(x, A, sf, tf)

        self.assertEqual(tuple(y_hat.shape), (2, 12, 4, 1))

    def test_dow_features_and_gated_residual_keep_shape(self):
        model = STFractalTransformer(
            n_nodes=4,
            in_dim=3,
            d_model=8,
            n_heads=2,
            seq_len=6,
            pred_len=24,
            sf_dim=3,
            tf_dim=3,
            out_dim=1,
            use_direct_pred=True,
            direct_head_mode="attn",
            direct_step_refine=True,
            spatial_mode="hybrid",
            k_hop=2,
            gated_residual=True,
            use_future_time_features=True,
        )
        x = torch.randn(2, 6, 4, 3)
        x[:, :, :, 1] = torch.linspace(0.0, 5.0 / 288.0, 6).view(1, 6, 1)
        x[:, :, :, 2] = 2.0 / 7.0
        A = torch.eye(4)
        sf = torch.randn(4, 3)
        tf = torch.randn(4, 3)

        y_hat, _ = model(x, A, sf, tf)

        self.assertEqual(tuple(y_hat.shape), (2, 24, 4, 1))

    def test_time_feature_start_keeps_raw_exogenous_channels_out_of_calendar_inference(self):
        model = STFractalTransformer(
            n_nodes=4,
            in_dim=4,
            d_model=8,
            n_heads=2,
            seq_len=6,
            pred_len=12,
            sf_dim=3,
            tf_dim=3,
            out_dim=1,
            use_direct_pred=True,
            direct_head_mode="attn",
            direct_step_refine=True,
            use_future_time_features=True,
            time_feature_start=3,
        )
        x = torch.randn(2, 6, 4, 4)
        x[:, :, :, 3] = torch.linspace(0.0, 5.0 / 288.0, 6).view(1, 6, 1)
        A = torch.eye(4)
        sf = torch.randn(4, 3)
        tf = torch.randn(4, 3)

        y_hat, _ = model(x, A, sf, tf)

        self.assertEqual(tuple(y_hat.shape), (2, 12, 4, 1))
        self.assertEqual(model.time_feature_start, 3)
        self.assertEqual(model.time_feature_dim, 1)


if __name__ == "__main__":
    unittest.main()
