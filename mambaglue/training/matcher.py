"""glue-factory-compatible MambaGlue matcher for training.

This module exposes ``MambaGlueMatcher``, a ``BaseModel`` subclass that wires
the inference-only modules from :mod:`mambaglue.mambaglue` into the data
contract expected by ``gluefactory``'s ``two_view_pipeline`` and training
loop. The mathematical model is unchanged from
:class:`mambaglue.mambaglue.MambaGlue`; only the I/O shape, early-exit logic,
gradient checkpointing, and loss head differ.

Reference for the contract this matcher implements:
``gluefactory/models/matchers/lightglue.py`` in the glue-factory repo.
"""

from __future__ import annotations

import torch
import torch.utils.checkpoint
from torch import nn

from gluefactory.models.base_model import BaseModel
from gluefactory.models.utils.losses import NLLLoss
from gluefactory.models.utils.metrics import matcher_metrics

from ..mambaglue import (
    LearnableFourierPositionalEncoding,
    MatchAssignment,
    TokenConfidence,
    TransformerMambaLayer,
    filter_matches,
    normalize_keypoints,
)


class TrainableTokenConfidence(TokenConfidence):
    """MambaGlue's deep confidence MLP + a LightGlue-style BCE loss head.

    The inference module (``mambaglue.mambaglue.TokenConfidence``) does not ship
    a loss method, so we subclass it and add one that mirrors the training-time
    loss in ``gluefactory.models.matchers.lightglue.TokenConfidence``.
    The pre-sigmoid logit is obtained by running the MLP up to, but not
    including, the final Sigmoid.
    """

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def _logits(self, desc):
        x = desc.detach()
        for layer in list(self.token.children())[:-1]:  # skip the trailing Sigmoid
            x = layer(x)
        return x.squeeze(-1)

    def loss(self, desc0, desc1, la_now, la_final):
        logit0 = self._logits(desc0)
        logit1 = self._logits(desc1)
        la_now, la_final = la_now.detach(), la_final.detach()
        correct0 = (
            la_final[:, :-1, :].max(-1).indices
            == la_now[:, :-1, :].max(-1).indices
        )
        correct1 = (
            la_final[:, :, :-1].max(-2).indices
            == la_now[:, :, :-1].max(-2).indices
        )
        return (
            self.loss_fn(logit0, correct0.float()).mean(-1)
            + self.loss_fn(logit1, correct1.float()).mean(-1)
        ) / 2.0


class MambaGlueMatcher(BaseModel):
    default_conf = {
        "name": "mambaglue_matcher",
        "input_dim": 256,
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": False,
        "depth_confidence": -1,
        "width_confidence": -1,
        "filter_threshold": 0.0,
        "checkpointed": True,
        "weights": None,
        "loss": {
            "gamma": 1.0,
            "fn": "nll",
            "nll_balancing": 0.5,
            "gamma_f": 0.0,
        },
    }

    required_data_keys = [
        "keypoints0",
        "keypoints1",
        "descriptors0",
        "descriptors1",
        "view0",
        "view1",
    ]

    def _init(self, conf):
        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(
                conf.input_dim, conf.descriptor_dim, bias=True
            )
        else:
            self.input_proj = nn.Identity()

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * conf.add_scale_ori, head_dim, head_dim
        )

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim
        self.transformermambas = nn.ModuleList(
            [TransformerMambaLayer(d, h, conf.flash) for _ in range(n)]
        )
        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList(
            [TrainableTokenConfidence(d) for _ in range(n - 1)]
        )

        self.loss_fn = NLLLoss(conf.loss)

    def _forward(self, data: dict) -> dict:
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"

        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        b, m, _ = kpts0.shape
        _, n, _ = kpts1.shape
        device = kpts0.device

        size0 = data["view0"].get("image_size")
        size1 = data["view1"].get("image_size")
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()

        if self.conf.add_scale_ori:
            sc0, o0 = data["scales0"], data["oris0"]
            sc1, o1 = data["scales1"], data["oris1"]
            kpts0 = torch.cat(
                [
                    kpts0,
                    sc0 if sc0.dim() == 3 else sc0[..., None],
                    o0 if o0.dim() == 3 else o0[..., None],
                ],
                -1,
            )
            kpts1 = torch.cat(
                [
                    kpts1,
                    sc1 if sc1.dim() == 3 else sc1[..., None],
                    o1 if o1.dim() == 3 else o1[..., None],
                ],
                -1,
            )

        desc0 = data["descriptors0"].contiguous()
        desc1 = data["descriptors1"].contiguous()
        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim
        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()

        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        do_early_stop = self.conf.depth_confidence > 0 and not self.training
        do_point_pruning = self.conf.width_confidence > 0 and not self.training

        all_desc0, all_desc1 = [], []
        if do_point_pruning:
            ind0 = torch.arange(0, m, device=device)[None]
            ind1 = torch.arange(0, n, device=device)[None]
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)

        token0, token1 = None, None
        for i in range(self.conf.n_layers):
            if self.conf.checkpointed and self.training:
                desc0, desc1 = torch.utils.checkpoint.checkpoint(
                    self.transformermambas[i],
                    desc0,
                    desc1,
                    encoding0,
                    encoding1,
                    use_reentrant=False,
                )
            else:
                desc0, desc1 = self.transformermambas[i](
                    desc0, desc1, encoding0, encoding1
                )
            if self.training or i == self.conf.n_layers - 1:
                all_desc0.append(desc0)
                all_desc1.append(desc1)
                continue

            if do_early_stop:
                assert b == 1
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self._check_if_stop(
                    token0[..., :m, :], token1[..., :n, :], i, m + n
                ):
                    break
            if do_point_pruning:
                assert b == 1
                scores0 = self.log_assignment[i].get_matchability(desc0)
                prunemask0 = self._get_pruning_mask(token0, scores0, i)
                keep0 = torch.where(prunemask0)[1]
                ind0 = ind0.index_select(1, keep0)
                desc0 = desc0.index_select(1, keep0)
                encoding0 = encoding0.index_select(-2, keep0)
                prune0[:, ind0] += 1
                scores1 = self.log_assignment[i].get_matchability(desc1)
                prunemask1 = self._get_pruning_mask(token1, scores1, i)
                keep1 = torch.where(prunemask1)[1]
                ind1 = ind1.index_select(1, keep1)
                desc1 = desc1.index_select(1, keep1)
                encoding1 = encoding1.index_select(-2, keep1)
                prune1[:, ind1] += 1

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)

        if do_point_pruning:
            m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            m0_[:, ind0] = torch.where(
                m0 == -1, -1, ind1.gather(1, m0.clamp(min=0))
            )
            m1_[:, ind1] = torch.where(
                m1 == -1, -1, ind0.gather(1, m1.clamp(min=0))
            )
            mscores0_ = torch.zeros((b, m), device=mscores0.device)
            mscores1_ = torch.zeros((b, n), device=mscores1.device)
            mscores0_[:, ind0] = mscores0
            mscores1_[:, ind1] = mscores1
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
        else:
            prune0 = torch.ones_like(mscores0) * self.conf.n_layers
            prune1 = torch.ones_like(mscores1) * self.conf.n_layers

        pred = {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "ref_descriptors0": torch.stack(all_desc0, 1),
            "ref_descriptors1": torch.stack(all_desc1, 1),
            "log_assignment": scores,
            "prune0": prune0,
            "prune1": prune1,
        }
        return pred

    def loss(self, pred, data):
        def loss_params(pred, i):
            la, _ = self.log_assignment[i](
                pred["ref_descriptors0"][:, i], pred["ref_descriptors1"][:, i]
            )
            return {"log_assignment": la}

        sum_weights = 1.0
        nll, gt_weights, loss_metrics = self.loss_fn(loss_params(pred, -1), data)
        N = pred["ref_descriptors0"].shape[1]
        losses = {"total": nll, "last": nll.clone().detach(), **loss_metrics}

        if self.training:
            losses["confidence"] = 0.0

        losses["row_norm"] = pred["log_assignment"].exp()[:, :-1].sum(2).mean(1)
        for i in range(N - 1):
            params_i = loss_params(pred, i)
            nll, _, _ = self.loss_fn(params_i, data, weights=gt_weights)

            if self.conf.loss.gamma > 0.0:
                weight = self.conf.loss.gamma ** (N - i - 1)
            else:
                weight = i + 1
            sum_weights += weight
            losses["total"] = losses["total"] + nll * weight

            losses["confidence"] += self.token_confidence[i].loss(
                pred["ref_descriptors0"][:, i],
                pred["ref_descriptors1"][:, i],
                params_i["log_assignment"],
                pred["log_assignment"],
            ) / (N - 1)

            del params_i
        losses["total"] /= sum_weights

        if self.training:
            losses["total"] = losses["total"] + losses["confidence"]

        if not self.training:
            metrics = matcher_metrics(pred, data)
        else:
            metrics = {}
        return losses, metrics

    def _confidence_threshold(self, layer_index: int) -> float:
        import numpy as np

        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.conf.n_layers)
        return float(np.clip(threshold, 0, 1))

    def _get_pruning_mask(self, confidences, scores, layer_index):
        keep = scores > (1 - self.conf.width_confidence)
        if confidences is not None:
            keep |= confidences <= self._confidence_threshold(layer_index)
        return keep

    def _check_if_stop(self, confidences0, confidences1, layer_index, num_points):
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self._confidence_threshold(layer_index)
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.conf.depth_confidence


__main_model__ = MambaGlueMatcher
