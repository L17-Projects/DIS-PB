from typing import Optional, Tuple

import torch


def bincount(
    input: torch.Tensor, batch: Optional[torch.Tensor] = None, minlength: int = 0
):
    assert input.ndim == 1
    if batch is None:
        return torch.bincount(input, minlength=minlength)
    else:
        assert batch.shape == input.shape

        length = input.max().item() + 1
        if minlength == 0:
            minlength = length
        if length > minlength:
            raise ValueError(
                f"minlength {minlength} too small for input with integers up to and including {length}"
            )

        # Flatten indexes
        # Make each "class" in input into a per-input class.
        input = input + batch * minlength

        num_batch = batch.max() + 1

        return torch.bincount(input, minlength=minlength * num_batch).reshape(
            num_batch, minlength
        )


def nanstd_mean(
    x: torch.Tensor, dim: int, unbiased: bool, keepdim: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.sum().isnan():
        # NaN path:
        mean = torch.nanmean(x, dim=dim, keepdim=True)
        variance = torch.nanmean((x - mean).square(), dim=dim, keepdim=keepdim)
        if unbiased:
            # Bessel's correction:
            variance *= x.shape[dim] / (x.shape[dim] - 1)
        if not keepdim:
            mean.squeeze_(dim=dim)
        return variance.sqrt(), mean
    else:
        return torch.std_mean(x, dim=dim, unbiased=unbiased, keepdim=keepdim)
