# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Turns an image into a byte string and back. This is what makes an image trainable
#   by the one byte-level trainer: every codebook holds exactly 256 entries, so one
#   code is one byte, vocab stays 256, and an encoded image is indistinguishable from
#   prose to the corpus format, the trainer and the engine.
# - Residual VQ (the SoundStream/EnCodec construction on a 2D grid): plane 0 quantizes
#   the latent, plane 1 quantizes what plane 0 missed, and so on. Planes are emitted
#   coarse to fine, so a prefix of the byte string is already a valid lower-fidelity
#   image -- that is what makes anytime decoding possible.
# - Encoder is a patchify conv (ViT's patch embedding) at exactly the decoder's patch
#   size, so encoder grid and decoder grid are the same object and no resampling sits
#   between them. Decoding reuses image_decode.PatchDecoder; this module owns the
#   analysis half and the codebooks, never a second decoder.
# - Training here is codec fitting only: a caller drives fit_step over image batches.
#   No byte model, no corpus, no run checkpoint.
# veritate_core/plugin/image_codec.py
# ------------------------------------------------------------------------------------
# Imports:

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from veritate_core.plugin.image_decode import RGB, VOCAB_BYTE_LEVEL, PatchDecoder

# ------------------------------------------------------------------------------------
# Constants

DEFAULT_PLANES       = 4
DEFAULT_LATENT_DIM   = 32
DEFAULT_PATCH        = 20
DEFAULT_ENC_BLOCKS   = 2
DEFAULT_DEC_HIDDEN   = 64
DEFAULT_BAND         = 4
DEFAULT_COMMIT_BETA  = 0.25
# The masked objective needs a byte that is never a real code. Reserving the top one
# costs a single codebook entry and keeps vocab at 256, so the engine is untouched.
MASK_BYTE            = VOCAB_BYTE_LEVEL - 1
CODEBOOK_ENTRIES     = VOCAB_BYTE_LEVEL - 1

# ------------------------------------------------------------------------------------
# Functions


class Encoder(nn.Module):
    """Patchify conv at the decoder's patch size, then refinement at grid resolution."""

    def __init__(self, latent_dim, patch, blocks):
        super().__init__()
        self.stem = nn.Conv2d(RGB, latent_dim, kernel_size=patch, stride=patch)
        self.blocks = nn.ModuleList([nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
                                     for _ in range(blocks)])

    def forward(self, images):
        h = self.stem(images)
        for conv in self.blocks:
            h = h + F.gelu(conv(h))
        return h


class ResidualVQ(nn.Module):
    """Stacked 256-entry codebooks. Each plane quantizes what the planes above it left
    behind, so plane count buys fidelity without ever leaving the byte alphabet."""

    def __init__(self, planes, dim):
        super().__init__()
        self.planes = planes
        self.dim = dim
        self.codebooks = nn.Parameter(torch.randn(planes, CODEBOOK_ENTRIES, dim) * 0.1)

    def quantize(self, latent):
        """latent [B, gh, gw, dim] -> (codes [B, planes, gh, gw], quantized, commitment)."""
        residual = latent
        codes, quantized = [], torch.zeros_like(latent)
        for plane in range(self.planes):
            book = self.codebooks[plane]
            flat = residual.reshape(-1, self.dim)
            index = torch.cdist(flat, book).argmin(dim=1)
            picked = book[index].reshape(residual.shape)
            codes.append(index.reshape(residual.shape[:-1]))
            quantized = quantized + picked
            residual = residual - picked
        commitment = F.mse_loss(quantized.detach(), latent)
        # Straight-through: the decoder sees the quantized grid, the encoder is
        # differentiated as if it had passed through unchanged.
        quantized = latent + (quantized - latent).detach()
        return torch.stack(codes, dim=1), quantized, commitment

    def features(self, codes):
        """codes [B, planes, gh, gw] -> summed codebook vectors [B, gh, gw, dim]."""
        total = 0
        for plane in range(self.planes):
            total = total + F.embedding(codes[:, plane], self.codebooks[plane])
        return total


class ImageCodec(nn.Module):
    """Encoder + residual VQ + the shared patch decoder. Owns the image <-> bytes
    contract and nothing else."""

    def __init__(self, planes=DEFAULT_PLANES, latent_dim=DEFAULT_LATENT_DIM, patch=DEFAULT_PATCH,
                 enc_blocks=DEFAULT_ENC_BLOCKS, dec_hidden=DEFAULT_DEC_HIDDEN, band=DEFAULT_BAND,
                 commit_beta=DEFAULT_COMMIT_BETA):
        super().__init__()
        self.config = {"planes": planes, "latent_dim": latent_dim, "patch": patch,
                       "enc_blocks": enc_blocks, "dec_hidden": dec_hidden, "band": band,
                       "commit_beta": commit_beta}
        self.patch = patch
        self.planes = planes
        self.commit_beta = commit_beta
        self.encoder = Encoder(latent_dim, patch, enc_blocks)
        self.vq = ResidualVQ(planes, latent_dim)
        self.decoder = PatchDecoder(latent_dim, patch, dec_hidden, band)

    def code_bytes(self, height, width):
        """Length in bytes of one encoded frame at this geometry."""
        if height % self.patch or width % self.patch:
            raise ValueError("patch " + str(self.patch) + " does not divide "
                             + str(height) + "x" + str(width))
        return self.planes * (height // self.patch) * (width // self.patch)

    def encode(self, images):
        """images [B, 3, H, W] in [0, 1] -> codes [B, planes, gh, gw]."""
        latent = self.encoder(images).permute(0, 2, 3, 1)
        codes, _, _ = self.vq.quantize(latent)
        return codes

    def decode(self, codes):
        """codes [planes, gh, gw] -> uint8 frame [H, W, 3]. Plane-major, so a prefix of
        the planes decodes to a valid coarser image."""
        features = self.vq.features(codes.unsqueeze(0))[0]
        return self.decoder.render(features)

    def forward(self, images):
        """Training path: reconstruction in [0, 1] plus the losses fit_step sums."""
        latent = self.encoder(images).permute(0, 2, 3, 1)
        codes, quantized, commitment = self.vq.quantize(latent)
        recon = self.decoder(quantized)
        target = images.permute(0, 2, 3, 1)
        return recon, {"recon": F.l1_loss(recon, target), "commit": commitment, "codes": codes}

    def fit_step(self, images, optimizer):
        """One optimizer step on a batch. Returns the scalar losses, detached."""
        optimizer.zero_grad(set_to_none=True)
        _, parts = self(images)
        loss = parts["recon"] + self.commit_beta * parts["commit"]
        loss.backward()
        optimizer.step()
        return {k: float(v.detach()) for k, v in
                (("loss", loss), ("recon", parts["recon"]), ("commit", parts["commit"]))}

    def to_bytes(self, codes):
        """codes [planes, gh, gw] -> the plane-major byte string that enters a corpus."""
        return bytes(codes.reshape(-1).to(torch.uint8).tolist())

    def from_bytes(self, blob, height, width):
        """Inverse of to_bytes at a known geometry."""
        expected = self.code_bytes(height, width)
        if len(blob) != expected:
            raise ValueError("expected " + str(expected) + " code bytes, got " + str(len(blob)))
        flat = torch.frombuffer(bytearray(blob), dtype=torch.uint8).long()
        return flat.reshape(self.planes, height // self.patch, width // self.patch)


def save(codec, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"config": codec.config, "state": codec.state_dict()}, path)
    return path


def load(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    codec = ImageCodec(**blob["config"])
    codec.load_state_dict(blob["state"])
    codec.eval()
    return codec
