import torch.nn as nn
from multi_head_attention import MultiHeadAttention

class EncoderBlock(nn.Module):
  def  __init__(self, d_k, d_model, n_heads, max_len, dropout_prob=0.1):
    super().__init__()


    self.ln1 = nn.LayerNorm(d_model)
    self.ln2 = nn.LayerNorm(d_model)
    self.mha = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal=False)
    self.ann = nn.Sequential(
        nn.Linear(d_model, d_model * 4),
        nn.GELU(),
        nn.Linear(d_model *4, d_model),
        nn.Dropout(dropout_prob),
    )
    self.dropout = nn.Dropout(p=dropout_prob)

  def forward(self, x, pad_mask=None):
    x = self.ln1(x + self.mha(x, x, x, pad_mask))
    x = self.ln2(x+self.ann(x))
    x = self.dropout(x)
    return x