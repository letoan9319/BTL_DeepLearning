"""Small encoder-decoder Transformer. Boolean masks are True on visible keys."""
import math
from dataclasses import dataclass

import torch
from torch import nn

PAD, UNK, BOS, EOS = range(4)


@dataclass
class Config:
    width: int = 256
    heads: int = 4
    layers: int = 3
    ff: int = 1024
    dropout: float = 0.1
    limit: int = 128


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        assert c.width % c.heads == 0
        self.heads = c.heads
        self.q, self.k, self.v, self.out = [nn.Linear(c.width, c.width) for _ in range(4)]
        self.dropout = nn.Dropout(c.dropout)

    def split(self, x):
        return x.reshape(x.size(0), x.size(1), self.heads, -1).transpose(1, 2)

    def forward(self, query, memory, mask):
        return self.attend(query, *self.project_memory(memory), mask)

    def project_memory(self, memory):
        return self.split(self.k(memory)), self.split(self.v(memory))

    def attend(self, query, k, v, mask=None):
        q = self.split(self.q(query))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = scores.softmax(-1)
        context = (self.dropout(weights) @ v).transpose(1, 2).reshape_as(query)
        return self.out(context), weights


class Block(nn.Module):
    def __init__(self, c, decoder=False):
        super().__init__()
        self.attention = Attention(c)
        self.norm1 = nn.LayerNorm(c.width)
        self.cross = Attention(c) if decoder else None
        if decoder:
            self.norm_cross = nn.LayerNorm(c.width)
        self.ff = nn.Sequential(nn.Linear(c.width, c.ff), nn.ReLU(), nn.Dropout(c.dropout),
                                nn.Linear(c.ff, c.width))
        self.norm2 = nn.LayerNorm(c.width)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x, mask, memory=None, source_mask=None):
        update, _ = self.attention(x, x, mask)
        x = self.norm1(x + self.dropout(update))
        weights = None
        if self.cross is not None:
            update, weights = self.cross(x, memory, source_mask)
            x = self.norm_cross(x + self.dropout(update))
        return self.norm2(x + self.dropout(self.ff(x))), weights


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, c):
        super().__init__()
        self.source = nn.Embedding(src_vocab, c.width, padding_idx=PAD)
        self.target = nn.Embedding(tgt_vocab, c.width, padding_idx=PAD)
        self.encoder = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.decoder = nn.ModuleList(Block(c, decoder=True) for _ in range(c.layers))
        self.output = nn.Linear(c.width, tgt_vocab)
        self.dropout = nn.Dropout(c.dropout)
        angles = torch.arange(c.limit)[:, None] * torch.exp(
            torch.arange(0, c.width, 2) * (-math.log(10000) / c.width))
        pos = torch.zeros(c.limit, c.width)
        pos[:, 0::2] = angles.sin()
        pos[:, 1::2] = angles[:, :c.width // 2].cos()
        self.register_buffer("position", pos)

    def embed(self, ids, embedding):
        return self.dropout(embedding(ids) * math.sqrt(embedding.embedding_dim)
                            + self.position[:ids.size(1)])

    def encode(self, source):
        mask = source.ne(PAD)[:, None, None, :]
        x = self.embed(source, self.source)
        for layer in self.encoder:
            x, _ = layer(x, mask)
        return x, mask

    def decode(self, target, memory, source_mask):
        causal = torch.ones((target.size(1), target.size(1)), device=target.device,
                            dtype=torch.bool).tril()
        mask = causal & target.ne(PAD)[:, None, None, :]
        x = self.embed(target, self.target)
        attention = []
        for layer in self.decoder:
            x, weights = layer(x, mask, memory, source_mask)
            attention.append(weights)
        return self.output(x), attention

    def decode_step(self, token, position, cache, cross_keys, source_mask):
        # Earlier states are immutable under the causal mask. Extend each layer
        # by one token and reuse the encoder projections across decoding steps.
        x = self.target(token) * math.sqrt(self.target.embedding_dim) + self.position[position]
        updated = []
        for layer, previous, cross in zip(self.decoder, cache, cross_keys, strict=True):
            k, v = layer.attention.project_memory(x)
            if previous is not None:
                k = torch.cat((previous[0], k), dim=2)
                v = torch.cat((previous[1], v), dim=2)
            updated.append((k, v))
            update, _ = layer.attention.attend(x, k, v)
            x = layer.norm1(x + update)
            update, _ = layer.cross.attend(x, *cross, source_mask)
            x = layer.norm_cross(x + update)
            x = layer.norm2(x + layer.ff(x))
        return self.output(x[:, 0]), updated

    def forward(self, source, target):
        return self.decode(target, *self.encode(source))


@torch.inference_mode()
def translate_batch(model, source, width=1, limit=128, alpha=1.0):
    """Beam search per sentence with a separate best completed hypothesis."""
    memory, mask = model.encode(source)
    batch = len(source)
    memory = memory.repeat_interleave(width, 0)
    mask = mask.repeat_interleave(width, 0)
    cross_keys = [layer.cross.project_memory(memory) for layer in model.decoder]
    cache = [None] * len(model.decoder)
    active = source.new_full((batch, width, 1), BOS)
    scores = memory.new_full((batch, width), -torch.inf)
    scores[:, 0] = 0
    best_score = memory.new_full((batch,), -torch.inf)
    best = source.new_full((batch, limit), PAD)
    done = torch.zeros(batch, dtype=torch.bool, device=source.device)
    rows = torch.arange(batch, device=source.device)
    for step in range(1, limit + 1):
        logits, cache = model.decode_step(active[:, :, -1].reshape(-1, 1), step - 1,
                                          cache, cross_keys, mask)
        probs = logits.float().log_softmax(-1).reshape(batch, width, -1)
        probs[:, :, [PAD, BOS]] = -torch.inf
        if width == 1:
            tokens = probs.argmax(-1)
            best[:, step-1] = torch.where(done, PAD, tokens[:, 0])
            done |= tokens[:, 0].eq(EOS)
            active = torch.cat((active, tokens.unsqueeze(-1)), -1)
        else:
            candidates = probs + scores.unsqueeze(-1)
            values, indices = candidates.flatten(1).topk(2 * width, dim=1)
            parents, tokens = indices // probs.size(-1), indices % probs.size(-1)
            ends = values[:, :width].masked_fill(tokens[:, :width] != EOS, -torch.inf)
            end_score, end_rank = ends.max(1)
            end_score = end_score / step**alpha
            improved = (end_score > best_score) & ~done
            completed = active[rows, parents[rows, end_rank], 1:]
            best[improved] = PAD
            best[improved, :step-1] = completed[improved]
            best[improved, step-1] = EOS
            best_score = torch.maximum(best_score, end_score)
            scores, keep = values.masked_fill(tokens == EOS, -torch.inf).topk(width, dim=1)
            kept_parents = parents.gather(1, keep)
            flat_parents = (rows[:, None] * width + kept_parents).flatten()
            cache = [(k.index_select(0, flat_parents), v.index_select(0, flat_parents)) for k, v in cache]
            prefix = active[rows[:, None], kept_parents]
            active = torch.cat((prefix, tokens.gather(1, keep).unsqueeze(-1)), -1)
            done |= best_score >= scores[:, 0] / limit**alpha
        if done.all():
            break
    if width > 1:
        fallback = ~best_score.isfinite()
        best[fallback, :step] = active[fallback, 0, 1:]
    return [[i for i in row if i != PAD] for row in best.tolist()]


def translate(model, source, width=1, limit=128, alpha=1.0):
    return translate_batch(model, source, width, limit, alpha)[0]
