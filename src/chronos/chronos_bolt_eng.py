from dataclasses import dataclass
import torch
import torch.nn as nn

from einops import repeat, rearrange, einsum
from einops.layers.torch import Rearrange

from typing import Optional
from chronos.chronos_bolt import ChronosBoltModelForForecasting, ChronosBoltOutput


def energy_score_w_mask(
    y: torch.Tensor,
    preds: torch.Tensor,
    beta: float = 1.0,
    p: float = 2.0,
    lamb: float = 0.5,
    return_components: bool = False,
    mask=None,
    normalize: bool = True,  # True -> divide by #observed per series
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


class EngHead(nn.Module):
    def __init__(
        self, n_quantiles: int, n_pred: int, d_noise: int, d_hidden: int | None = None
    ):
        super().__init__()
        self.d_noise = d_noise

        d_in = (n_quantiles + d_noise) * n_pred
        d_hidden = d_hidden or d_in * 4

        self.ff = nn.Sequential(
            Rearrange("b q l -> b (q l)"),
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, n_pred),
        )

    def forward(self, x: torch.Tensor, m: int) -> torch.Tensor:
        b, q, l = x.shape  # (batch, quantile, time series length)

        x = repeat(x, "b ... -> (b m) ...", m=m)
        eps = torch.randn((b * m, self.d_noise, *x.shape[2:]), device=x.device)
        x = torch.cat([x, eps], dim=1)  # append noise as extra quantile channels

        out = self.ff(x)
        out = rearrange(out, "(b m) ... -> b m ...", m=m)

        return out


@dataclass
class ChronosBoltWithEngressionOutput(ChronosBoltOutput):
    loss_term1: Optional[torch.Tensor] = None
    loss_term2: Optional[torch.Tensor] = None


class ChronosBoltWithEngressionModel(ChronosBoltModelForForecasting):
    def __init__(self, config, m: int = 4, **kwargs):
        super().__init__(config=config, **kwargs)
        self.m = m
        self.out_proj = EngHead(
            n_quantiles=self.num_quantiles,
            n_pred=self.chronos_config.prediction_length,
            # d_hidden=128,
            d_noise=32 - self.num_quantiles,
        )

        # TODO think more about init
        # self.o_proj = nn.Parameter(
        #     torch.ones(self.num_quantiles, 1) / self.num_quantiles
        # )

    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        m: Optional[int] = None,
    ) -> ChronosBoltOutput:
        b, l = context.shape
        q = self.num_quantiles
        m = m or self.m

        hidden_states, loc_scale, input_embeds, attention_mask = self.encode(
            context=context, mask=mask
        )
        sequence_output = self.decode(input_embeds, attention_mask, hidden_states)

        quantile_preds_shape = (
            b,
            q,
            self.chronos_config.prediction_length,
        )
        sample_preds_shape = (
            b,
            m,
            self.chronos_config.prediction_length,
        )

        quantile_preds = self.output_patch_embedding(sequence_output).view(
            *quantile_preds_shape
        )

        # pass through stochastic engression head
        sample_preds = self.out_proj(quantile_preds, m)

        loss = None
        if target is not None:
            # normalize target
            target, _ = self.instance_norm(target, loc_scale)
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

            loss, term1, term2 = energy_score_w_mask(
                y=target, preds=sample_preds, mask=target_mask, return_components=True
            )
            term2 = -term2  # negate to log as positive

        # Unscale predictions
        repeated_loc_scale = (
            repeat(loc_scale[0], "b 1 -> (b m) 1", m=m),
            repeat(loc_scale[1], "b 1 -> (b m) 1", m=m),
        )

        sample_preds = self.instance_norm.inverse(
            sample_preds.view(b * m, -1), repeated_loc_scale
        ).view(*sample_preds_shape)

        # For logging
        self._last_terms = {
            "loss_term1": term1.detach(),
            "loss_term2": term2.detach(),
            "loss_total": loss.detach(),
            "std_across_samples": sample_preds.detach().std(dim=1).mean(),
        }

        return ChronosBoltWithEngressionOutput(
            loss=loss,
            quantile_preds=sample_preds,
            loss_term1=term1,
            loss_term2=term2,
        )
