import logging
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange, einsum

from chronos.chronos_bolt import (
    ChronosBoltModelForForecasting,
    ChronosBoltOutput,
)

from .base import BaseChronosPipeline, ForecastType
from typing import List, Optional, Tuple, Union
import warnings
from transformers import AutoConfig


logger = logging.getLogger(__file__)


def energy_score_w_mask(
    y: torch.Tensor,
    preds: torch.Tensor,
    beta: float = 1.0,
    p: float = 2.0,
    lamb: float = 0.5,
    return_components: bool = False,
    mask=None,
    normalize: bool = True,  # divide by #observed per series
):
    """
    y:     (B, *)
    preds: (B, M, *)
    mask:  (B, *) in {0,1}/bool. If None, treated as all ones.
    """
    assert preds.shape[0] == y.shape[0] and preds.shape[2:] == y.shape[1:], (
        f"y and preds should only differ in the first dimension: {y.shape} vs {preds.shape}"
    )

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


# class EngHead(nn.Module):
#     def __init__(
#         self, n_quantiles: int, n_pred: int, d_noise: int, d_hidden: int | None = None
#     ):
#         super().__init__()
#         self.d_noise = d_noise

#         d_in = (n_quantiles + d_noise) * n_pred
#         d_hidden = d_hidden or d_in * 4

#         self.ff = nn.Sequential(
#             Rearrange("b q l -> b (q l)"),
#             nn.Linear(d_in, d_hidden),
#             nn.ReLU(),
#             nn.Linear(d_hidden, d_hidden),
#             nn.ReLU(),
#             nn.Linear(d_hidden, n_pred),
#         )

#     def forward(self, x: torch.Tensor, m: int) -> torch.Tensor:
#         b, q, l = x.shape  # (batch, quantile, time series length)

#         x = repeat(x, "b ... -> (b m) ...", m=m)
#         eps = torch.randn(
#             (b * m, self.d_noise, *x.shape[2:]), device=x.device, dtype=x.dtype
#         )
#         x = torch.cat([x, eps], dim=1)  # append noise as extra quantile channels

#         out = self.ff(x)
#         out = rearrange(out, "(b m) ... -> b m ...", m=m)

#         return out


# class EngResHead(nn.Module):
#     def __init__(self, features_dim: int, noise_dim: int, out_dim: int, h_dim: int):
#         super().__init__()
#         self.noise_dim = noise_dim
#         self.net = nn.Sequential(
#             nn.Linear(features_dim + noise_dim, h_dim),
#             nn.GELU(),
#             nn.Linear(h_dim, h_dim),
#             nn.GELU(),
#             nn.Linear(h_dim, out_dim),
#         )
#         self.residual_layer = nn.Linear(features_dim, out_dim)

#     def forward(self, x: torch.Tensor, m: int) -> torch.Tensor:
#         b, c, d = x.shape  # (batch, features_dim)

#         x_in = rearrange(x, "b ... -> b 1 ...")

#         x = repeat(x, "b ... -> (b m) ...", m=m)
#         eps = torch.randn((b * m, c, self.noise_dim), device=x.device, dtype=x.dtype)

#         x = torch.cat([x, eps], dim=-1)

#         out = self.net(x)
#         out = rearrange(out, "(b m) ... -> b m ...", m=m)

#         res = self.residual_layer(x_in)

#         return out + res


class NoiseEngResHead(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_quantiles: int,
        noise_dim: int,
        out_dim: int,
        h_dim: int,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(model_dim + num_quantiles + noise_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, out_dim),
        )
        self.residual_layer = nn.Linear(model_dim, out_dim)

    def forward(self, h: torch.Tensor, q: torch.Tensor, m: int) -> torch.Tensor:
        b, c, d = h.shape  # (batch, 1, d_model)
        b, _q, l = q.shape  # (batch, quantiles, pred len)

        res = self.residual_layer(rearrange(h, "b ... -> b 1 ..."))  # b, 1, pred len

        q = rearrange(q.mean(-1), "b q -> b 1 q")
        h = torch.cat([h, q], dim=-1)  # b, 1, d_model + quantiles

        h = repeat(h, "b ... -> b m ...", m=m)  # b, m, 1, d_model + quantiles
        eps = torch.randn((b, m, c, self.noise_dim), device=h.device, dtype=h.dtype)
        h = torch.cat([h, eps], dim=-1)
        out = self.net(h)
        # out = out - out.mean(dim=1, keepdim=True)  # zero-mean across m noise samples

        return out + res


@dataclass
class ChronosBoltWithEngressionOutput(ChronosBoltOutput):
    loss_term1: Optional[torch.Tensor] = None
    loss_term2: Optional[torch.Tensor] = None


class ChronosBoltWithEngressionModel(ChronosBoltModelForForecasting):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)

        self.out_proj_q = nn.Parameter(torch.ones(self.num_quantiles, 1))

        self.out_proj_noise = NoiseEngResHead(
            model_dim=config.d_model,
            num_quantiles=self.num_quantiles,
            noise_dim=55,  # config.d_noise,  # TODO as arg
            h_dim=config.d_ff,  # 1024
            out_dim=self.chronos_config.prediction_length,
        )

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

        hidden_states, loc_scale, input_embeds, attention_mask = self.encode(
            context=context, mask=mask
        )
        sequence_output = self.decode(
            input_embeds, attention_mask, hidden_states
        )  # (b, 1, d_model)

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

        with torch.no_grad():
            quantile_preds = self.output_patch_embedding(sequence_output).view(
                *quantile_preds_shape
            )
        q_preds = einsum(
            quantile_preds,
            F.softmax(self.out_proj_q, dim=0),
            "b q l, q o -> b o l",
        )

        noise_preds = self.out_proj_noise(sequence_output, quantile_preds, m).view(
            *sample_preds_shape
        )

        sample_preds = q_preds + noise_preds

        loss = term1 = term2 = None
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

            # For logging
            self._last_terms = {
                "loss_term1": term1.detach(),
                "loss_term2": term2.detach(),
                "loss_total": loss.detach(),
                "std_across_samples": sample_preds.detach().std(dim=1).mean(),
            }

        # Unscale predictions
        repeated_loc_scale = (
            repeat(loc_scale[0], "b 1 -> (b m) 1", m=m),
            repeat(loc_scale[1], "b 1 -> (b m) 1", m=m),
        )

        sample_preds = self.instance_norm.inverse(
            sample_preds.view(b * m, -1), repeated_loc_scale
        ).view(*sample_preds_shape)

        return ChronosBoltWithEngressionOutput(
            loss=loss,
            quantile_preds=sample_preds,
            loss_term1=term1,
            loss_term2=term2,
        )


class ChronosBoltWithEngressionPipeline(BaseChronosPipeline):
    forecast_type: ForecastType = (
        ForecastType.SAMPLES
    )  # TODO check if this is fair comparison w/bolt
    default_context_length: int = 2048

    def __init__(self, model: ChronosBoltModelForForecasting):
        super().__init__(inner_model=model)  # type: ignore
        self.model = model

    @property
    def quantiles(self) -> List[float]:
        return self.model.config.chronos_config["quantiles"]

    # TODO add embed method

    def predict(  # type: ignore[override]
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        num_samples: int,
        prediction_length: Optional[int] = None,
        limit_prediction_length: bool = False,
        mc_dropout: bool = False,  # TODO
    ) -> torch.Tensor:
        """
        Get forecasts for the given time series.

        Refer to the base method (``BaseChronosPipeline.predict``)
        for details on shared parameters.
        Additional parameters
        ---------------------
        limit_prediction_length
            Force prediction length smaller or equal than the
            built-in prediction length from the model. False by
            default. When true, fail loudly if longer predictions
            are requested, otherwise longer predictions are allowed.

        Returns
        -------
        torch.Tensor
            Forecasts of shape (batch_size, num_samples, prediction_length).

        Raises
        ------
        ValueError
            When limit_prediction_length is True and the prediction_length is
            greater than model's trainig prediction_length.
        """
        context_tensor = self._prepare_and_validate_context(context=context)

        model_context_length = self.model.config.chronos_config["context_length"]
        model_prediction_length = self.model.config.chronos_config["prediction_length"]
        if prediction_length is None:
            prediction_length = model_prediction_length

        if prediction_length > model_prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {model_prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg)

        predictions = []
        remaining = prediction_length

        # We truncate the context here because otherwise batches with very long
        # context could take up large amounts of GPU memory unnecessarily.
        if context_tensor.shape[-1] > model_context_length:
            context_tensor = context_tensor[..., -model_context_length:]

        # TODO: We unroll the forecast of Chronos Bolt greedily with the full forecast
        # horizon that the model was trained with (i.e., 64). This results in variance collapsing
        # every 64 steps.
        context_tensor = context_tensor.to(
            device=self.model.device,
            dtype=torch.float32,
        )
        while remaining > 0:
            with torch.no_grad():
                prediction = self.model(
                    context=context_tensor, m=num_samples
                ).quantile_preds.to(context_tensor)

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            # Central preds are medians across m samples
            central_prediction = prediction.median(dim=1).values

            context_tensor = torch.cat([context_tensor, central_prediction], dim=-1)

        return torch.cat(predictions, dim=-1)[..., :prediction_length].to(
            dtype=torch.float32, device="cpu"
        )

    def predict_quantiles(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        num_samples: int,
        prediction_length: Optional[int] = None,
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **predict_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Refer to the base method (``BaseChronosPipeline.predict_quantiles``).
        """
        # shape (batch_size, num_samples, prediction_length)
        predictions = self.predict(
            context,
            num_samples=num_samples,
            prediction_length=prediction_length,
            **predict_kwargs,
        ).detach()

        quantile_levels = torch.tensor(quantile_levels, dtype=predictions.dtype)
        qs = torch.quantile(predictions, dim=1, q=quantile_levels)

        median = predictions[:, :, quantile_levels.index(0.5)]
        mean = predictions.mean(dim=1)

        # NOTE: the median is returned as the mean here
        return qs, median

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Load the model, either from a local path or from the HuggingFace Hub.
        Supports the same arguments as ``AutoConfig`` and ``AutoModel``
        from ``transformers``.
        """

        print("here from pretrained")
        config = AutoConfig.from_pretrained(*args, **kwargs)
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        architecture = config.architectures[0]
        class_ = globals().get(architecture)

        if class_ is None:
            logger.warning(
                f"Unknown architecture: {architecture}, defaulting to ChronosBoltWithEngressionModel"
            )
            class_ = ChronosBoltWithEngressionModel

        model = class_.from_pretrained(*args, **kwargs)
        return cls(model=model)
