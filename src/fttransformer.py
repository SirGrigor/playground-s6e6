"""FT-Transformer leg — attention-based tabular model (Gorishniy et al.). A STRONG and DECORRELATED
paradigm: attention learns sample-specific feature interactions, a completely different error profile
from our MLPs (RealMLP/TabM) and trees (LGBM/ExtraTrees). The "strong AND decorrelated" leg the weak
CPU paradigms couldn't be — both our analysis and Gemini independently pointed here.

Architecture drafted with brother Gemini; Claude verified + hardened:
- stable transformer init (std=0.02, not randn std=1 → avoids attention blow-up),
- outputs (batch, 1, n_classes) SOFTMAX so it drops straight into realmlp._fit_one (n_ens=1) and reuses
  our balanced-softmax metric-aware loss + EMA + schedule,
- signature matches RealMLPNet: (output_dim, cat_dims, n_numerical, cfg).
Selected via cfg["arch"]=="ft".
"""
from __future__ import annotations


def build_fttransformer():
    """Lazy builder (torch imported only when used)."""
    import torch
    import torch.nn as nn

    class _Block(nn.Module):
        def __init__(self, d, heads, ffn, attn_drop, ffn_drop):
            super().__init__()
            self.n1 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, heads, dropout=attn_drop, batch_first=True)
            self.n2 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(ffn_drop), nn.Linear(ffn, d))

        def forward(self, x):
            h = self.n1(x)
            x = x + self.attn(h, h, h, need_weights=False)[0]
            return x + self.ffn(self.n2(x))

    class FTTransformerNet(nn.Module):
        def __init__(self, output_dim, cat_dims, n_numerical, cfg):
            super().__init__()
            d = cfg.get("d_token", 192); nb = cfg.get("n_blocks", 3); heads = cfg.get("n_heads", 8)
            ad = cfg.get("attention_dropout", 0.2); fd = cfg.get("ffn_dropout", 0.1)
            ffn = int(d * cfg.get("ffn_factor", 4 / 3))
            self.n_numerical = n_numerical
            self.cls = nn.Parameter(torch.empty(1, 1, d)); nn.init.normal_(self.cls, std=0.02)
            self.num_w = nn.Parameter(torch.empty(n_numerical, d)); nn.init.normal_(self.num_w, std=0.02)
            self.num_b = nn.Parameter(torch.zeros(n_numerical, d))
            self.cat_emb = nn.ModuleList([nn.Embedding(c, d) for c in cat_dims])
            for e in self.cat_emb:
                nn.init.normal_(e.weight, std=0.02)
            self.blocks = nn.Sequential(*[_Block(d, heads, ffn, ad, fd) for _ in range(nb)])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, output_dim)
            self._drops = []   # FT owns its dropout (attn/ffn, fixed); no-op for _fit_one's drop schedule

        def forward(self, x_num, x_cat):
            b = x_num.shape[0]
            toks = [self.cls.expand(b, -1, -1)]
            if self.n_numerical:
                toks.append(x_num.unsqueeze(-1) * self.num_w + self.num_b)   # (b, n_num, d)
            if len(self.cat_emb):
                toks.append(torch.stack([e(x_cat[:, i]) for i, e in enumerate(self.cat_emb)], dim=1))
            x = self.blocks(torch.cat(toks, dim=1))
            logits = self.head(self.norm(x[:, 0]))
            return torch.softmax(logits, dim=1).unsqueeze(1)   # (b, 1, output_dim) → _fit_one (n_ens=1)

    return FTTransformerNet
