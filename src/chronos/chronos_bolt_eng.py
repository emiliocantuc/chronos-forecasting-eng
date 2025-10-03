import torch
import torch.nn as nn

from einops import repeat, rearrange, einsum

from typing import Optional
from chronos.chronos_bolt import ChronosBoltModelForForecasting, ChronosBoltOutput


# def energy_score_masked(
#     y: torch.Tensor,
#     preds: torch.Tensor,
#     mask: torch.Tensor | None = None,
#     beta: float = 1.0,
#     p: float = 2.0,
#     lamb: float = 0.5,
#     return_components: bool = False,
#     eps: float = 1e-8,
# ):
#     """
#     Masked generalized energy score (Engression-style).

#     y:     (B, *)
#     preds: (B, M, *)
#     mask:  (B, *)  1.0 for observed, 0.0 for missing. If None, all ones.

#     We weight each dimension by w = mask / sqrt(#observed) per item, to avoid
#     horizon-length bias and keep magnitudes comparable across examples.
#     """
#     assert preds.shape[0] == y.shape[0] and preds.shape[2:] == y.shape[1:], (
#         f"y and preds should only differ in the first dimension: {y.shape} vs {preds.shape}"
#     )

#     B, M, *rest = preds.shape
#     y_flat = rearrange(y, "b ... -> b 1 (...)")  # (B,1,H)
#     preds_flat = rearrange(preds, "b m ... -> b m (...)")  # (B,M,H)
#     H = preds_flat.shape[-1]

#     if mask is None:
#         mask = torch.ones((B, H), device=preds.device, dtype=preds.dtype)
#     else:
#         mask = rearrange(mask, "b ... -> b (...)").to(preds.device, preds.dtype)

#     # Per-item normalization of mask so loss isn’t dominated by longer valid horizons
#     obs = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)  # (B,1)
#     w = mask / obs.sqrt()  # (B,H)

#     # ----- Term 1: E || Y - y ||_p^beta over masked dims -----
#     diff = (preds_flat - y_flat) * w.unsqueeze(1)  # (B,M,H)
#     term1 = torch.linalg.vector_norm(diff, ord=p, dim=2).pow(beta)  # (B,M)
#     term1 = term1.mean()  # scalar

#     # ----- Term 2: - (lambda/2) * E || Y - Y' ||_p^beta over masked dims -----
#     term2 = torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
#     if M > 1:
#         Z = preds_flat * w.unsqueeze(1)  # (B,M,H), weighted samples
#         # pairwise distances (includes zeros on diagonal)
#         pdist = torch.cdist(Z, Z, p=p).clamp_min(0).pow(beta)  # (B,M,M)
#         # exclude diagonal, average per batch item: sum_offdiag / (M*(M-1))
#         sum_all = pdist.sum(dim=(1, 2))  # (B,)
#         sum_diag = torch.diagonal(pdist, dim1=1, dim2=2).sum(dim=1)  # (B,)
#         mean_off = (sum_all - sum_diag) / (M * (M - 1))
#         term2 = -lamb * mean_off.mean()  # scalar

#     total = term1 + term2
#     if return_components:
#         return total, term1, term2
#     return total


def energy_score_w_mask(
    y: torch.Tensor,
    preds: torch.Tensor,
    beta: float = 1.0,
    p: float = 2.0,
    lamb: float = 0.5,
    return_components: bool = False,
    mask=None,
    normalize: bool = False,  # True -> divide by #observed per series
):
    """
    y:     (B, *)
    preds: (B, M, *)
    mask:  (B, *) in {0,1}/bool. If None, treated as all ones.
    """
    assert preds.shape[0] == y.shape[0] and preds.shape[2:] == y.shape[1:], (
        f"y and preds should only differ in the first dimension: {y.shape} vs {preds.shape}"
    )

    # B, M, *_ = preds.shape
    # y_flat     = rearrange(y,     'b ... -> b 1 (...)')      # (B,1,H)
    # preds_flat = rearrange(preds, 'b m ... -> b m (...)')    # (B,M,H)
    # H = preds_flat.shape[-1]

    # w = torch.ones((B, H), device=preds.device, dtype=preds.dtype) if mask is None \
    #     else rearrange(mask, 'b ... -> b (...)').to(preds.device, preds.dtype)

    # # Optionally normalize by #observed to keep scale stable across different masks
    # if normalize:
    #     obs = w.sum(dim=-1, keepdim=True).clamp_min(1.0)
    #     w = w / obs

    # # ---- Term 1 ----
    # diff  = (preds_flat - y_flat) * w.unsqueeze(1)           # (B,M,H)
    # term1 = torch.linalg.vector_norm(diff, ord=p, dim=2).pow(beta).mean()

    # # ---- Term 2 ----
    # term2 = torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
    # if M > 1:
    #     Z = preds_flat * w.unsqueeze(1)                      # (B,M,H)
    #     pairwise = torch.cdist(Z, Z, p=p).clamp_min(0).pow(beta)  # (B,M,M)
    #     # match original scaling (exclude diagonal in expectation): mean * M/(M-1)
    #     term2 = -lamb * (pairwise.mean(dim=(1,2)) * (M / (M - 1.0))).mean()

    # total = term1 + term2
    # if return_components:
    #     return total, term1, term2
    # return total

    b, m, *rest = preds.shape
    y = rearrange(y, "b ... -> b 1 (...)")
    preds = rearrange(preds, "b m ... -> b m (...)")

    if mask is None:
        w = torch.ones_like(y)
    else:
        w = rearrange(mask, "b ... -> b 1 (...)").to(preds.device, preds.dtype)

    if normalize:
        obs = w.sum(dim=-1, keepdim=True).clamp_min(1.0)
        w = w / obs

    # Term 1: the absolute error between the predicted and true values
    term1 = torch.linalg.vector_norm((preds - y) * w, ord=p, dim=2).pow(beta).mean()

    # Term 2: pairwise absolute differences between the predicted values
    term2 = torch.tensor(0.0, device=preds.device, dtype=preds.dtype)

    if m > 1:
        # cdist is convenient. The result shape before sum is (n, m, m).
        Z = preds * w
        pairwise_l1_dists = torch.cdist(Z, Z, p=p).pow(beta).mean() * m / (m - 1)
        term2 = -lamb * pairwise_l1_dists

    if return_components:
        return term1 + term2, term1, term2

    return term1 + term2


# TODO
class EngHead(nn.Module):
    def __init__(self):
        pass


class ChronosBoltWithEngressionModel(ChronosBoltModelForForecasting):
    def __init__(self, config, m: int = 4, **kwargs):
        super().__init__(config=config, **kwargs)
        self.m = m

        # TODO think more about init
        self.o_proj = nn.Parameter(
            torch.ones(self.num_quantiles, 1) / self.num_quantiles
        )

    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        m: Optional[int] = None,
    ) -> ChronosBoltOutput:
        batch_size = context.size(0)

        m = int(m) if m is not None else self.m
        assert m > 1, "m must be greater than 1"

        self.train()  # always force for MC dropout

        # repeat context and mask m times
        # TODO what if m is too large?
        context = repeat(context, "b ... -> (b m) ...", m=m)
        if mask is not None:
            mask = repeat(mask, "b ... -> (b m) ...", m=m)

        hidden_states, loc_scale, input_embeds, attention_mask = self.encode(
            context=context, mask=mask
        )
        sequence_output = self.decode(input_embeds, attention_mask, hidden_states)

        quantile_preds_shape = (
            batch_size * m,
            self.num_quantiles,
            self.chronos_config.prediction_length,
        )
        sample_preds_shape = (
            batch_size,
            m,
            self.chronos_config.prediction_length,
        )

        quantile_preds = self.output_patch_embedding(sequence_output).view(
            *quantile_preds_shape
        )

        # output head: take all predicted quantiles and output a single sample
        sample_preds = einsum(quantile_preds, self.o_proj, "b q l, q o -> b o l")
        sample_preds = rearrange(sample_preds, "(b m) 1 l -> b m l", m=m)

        loss = None
        if target is not None:
            # normalize target
            target_loc_scale = (
                loc_scale[0][::m, :],
                loc_scale[1][::m, :],
            )  # since loc_scale is repeated m times
            target, _ = self.instance_norm(target, target_loc_scale)
            # target = target.unsqueeze(1)  # type: ignore
            assert self.chronos_config.prediction_length >= target.shape[-1]

            target = target.to(sample_preds.device)
            target_mask = (
                target_mask.to(sample_preds.device)  # target_mask.unsqueeze(1)
                if target_mask is not None
                else ~torch.isnan(target)
            )
            target[~target_mask] = 0.0

            # pad target and target_mask if they are shorter than model's prediction_length
            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (
                    *target.shape[:-1],
                    self.chronos_config.prediction_length - target.shape[-1],
                )
                target = torch.cat(
                    [target, torch.zeros(padding_shape).to(target)], dim=-1
                )
                target_mask = torch.cat(
                    [target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1
                )

            loss = energy_score_w_mask(y=target, preds=sample_preds, mask=target_mask)

        # Unscale predictions
        sample_preds = self.instance_norm.inverse(
            sample_preds.view(batch_size * m, -1),
            loc_scale,
        ).view(*sample_preds_shape)

        return ChronosBoltOutput(
            loss=loss,
            quantile_preds=sample_preds,
        )
