# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Traceable, tensor-in/tensor-out wrappers for the diffusion components.

Each wrapper is the unit a QuantSim wraps and the unit exported to ONNX, so its
``forward`` takes and returns plain tensors only -- no ``ModelOutput``, no
``return_dict``, no config lookups at call time. The argument order of
``forward`` is the ONNX input order and matches the component's ``input_names``
on :class:`~.base.DiffusionModel`.
"""

import torch


class TextEncoderWrapper(torch.nn.Module):
    """``input_ids (B, 77) int -> encoder_hidden_states (B, 77, D)``.

    Returns the last hidden state, which is what Stable Diffusion conditions the
    UNet on (not the pooled output).
    """

    def __init__(self, text_encoder: torch.nn.Module):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(input_ids, return_dict=False)[0]


class UNetWrapper(torch.nn.Module):
    """``(latents, timestep, encoder_hidden_states) -> noise_pred``.

    ``latents`` is the scheduler-scaled model input ``(B, 4, h, w)``;
    ``timestep`` is a ``(B,)`` float tensor, so the exported graph takes the
    timestep as data rather than baking it in.
    """

    def __init__(self, unet: torch.nn.Module):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.unet(
            latents,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


class VAEDecoderWrapper(torch.nn.Module):
    """``latents (B, 4, h, w) -> image (B, 3, H, W)`` in ``[-1, 1]``.

    Takes the *raw* denoised latents: the VAE scaling factor is applied inside,
    so callers (and the exported graph) never need to know it. Only the decoder
    half of the VAE is kept; the encoder is unused for text-to-image.
    """

    def __init__(self, vae: torch.nn.Module):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder
        self.scaling_factor = float(vae.config.scaling_factor)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents / self.scaling_factor
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        return self.decoder(z)
