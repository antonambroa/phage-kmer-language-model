import torch

from phage_kmer_lm.model import CausalKmerTransformer, ModelConfig


def tiny_model():
    torch.manual_seed(1)
    model = CausalKmerTransformer(
        ModelConfig(vocab_size=32, pad_id=0, embed_dim=16, num_heads=4, hidden_dim=32, num_layers=2, dropout=0.0, max_len=16)
    )
    model.eval()
    return model


def test_causal_mask_blocks_future_information():
    model = tiny_model()
    a = torch.tensor([[1, 5, 6, 7, 8]])
    b = torch.tensor([[1, 5, 6, 20, 21]])
    with torch.no_grad():
        logits_a = model(a)
        logits_b = model(b)
    # Changing positions 3+ must not affect logits produced at positions 0..2.
    assert torch.allclose(logits_a[:, :3], logits_b[:, :3], atol=1e-6)


def test_padding_shape():
    model = tiny_model()
    x = torch.tensor([[1, 2, 3, 0, 0], [1, 4, 5, 6, 7]])
    assert model(x).shape == (2, 5, 32)
