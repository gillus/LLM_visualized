"""Shared definition of the tuned lens so the trainer and the generator agree
on its shape. A tuned lens is one small affine "translator" per pre-final layer
that maps that layer's hidden state into the final residual space; the frozen
final norm + unembedding then decode it. Identity-initialised, so an untrained
tuned lens == the plain logit lens.

Reference: Belrose et al., "Eliciting Latent Predictions with the Tuned Lens".
"""
import torch
import torch.nn as nn


class TunedLens(nn.Module):
    def __init__(self, n_layers, d_model):
        """n_layers = number of PRE-final hidden states to correct, i.e. indices
        0..n_layers-1 of `hidden_states` (index 0 = embeddings). The final
        hidden state needs no translator (it is the model's real output)."""
        super().__init__()
        self.translators = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(n_layers)]
        )
        for lin in self.translators:
            nn.init.eye_(lin.weight)   # identity init: starts equal to the plain lens
            nn.init.zeros_(lin.bias)

    def forward(self, layer_idx, h):
        return self.translators[layer_idx](h)

    @property
    def n_layers(self):
        return len(self.translators)


def save_tuned_lens(lens, model_name, path):
    torch.save({
        "model": model_name,
        "n_layers": lens.n_layers,
        "d_model": lens.translators[0].in_features,
        "state_dict": lens.state_dict(),
    }, path)


def load_tuned_lens(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=True)
    lens = TunedLens(ck["n_layers"], ck["d_model"])
    lens.load_state_dict(ck["state_dict"])
    lens.eval()
    return lens, ck
