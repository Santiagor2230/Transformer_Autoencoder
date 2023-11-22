import torch.nn as nn
from multi_head_attention import MultiHeadAttention

# 2 multi head attentions and 1 forward network
class DecoderBlock(nn.Module):
  def __init__(self, d_k, d_model, n_heads, max_len, dropout_prob=0.1):
    super().__init__()

    self.ln1 = nn.LayerNorm(d_model)
    self.ln2 = nn.LayerNorm(d_model)
    self.ln3 = nn.LayerNorm(d_model)

    #causal masking allows

    #first multihead attention in decoder with causal masking
    self.mha1 = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal=True)
    #second multihead attention in decoder with no causal masking
    self.mha2 = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal=False)
    self.ann = nn.Sequential(
        nn.Linear(d_model, d_model * 4),
        nn.GELU(),
        nn.Linear(d_model * 4, d_model),
        nn. Dropout(dropout_prob),
    )
    self.dropout =nn.Dropout(p=dropout_prob)

  def forward(self, enc_output, dec_input, enc_mask=None, dec_mask=None):
    #selt-attention on decoder input
    x = self.ln1(
        dec_input + self.mha1(dec_input, dec_input, dec_input, dec_mask))
    # multi-head attention including encoder output
    x = self.ln2(x + self.mha2(x, enc_output, enc_output, enc_mask))

    x = self.ln3(x + self.ann(x))
    x = self.dropout(x)
    return x