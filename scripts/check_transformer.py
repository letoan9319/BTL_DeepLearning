"""Fast acceptance check on real training pairs, including a checkpoint round trip."""
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from scripts.corpus import records
from scripts.transformer import BOS, EOS, Config, Transformer, PAD, translate, translate_batch
from scripts.tv3_data import Tokenizer, collate, encode_rows


def check():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    rows = list(islice(records("train"), 4))
    c = Config(width=32, heads=4, layers=2, ff=64, limit=32, dropout=0.)
    with TemporaryDirectory() as tmp:
        toks = [Tokenizer(tmp, lang, "bpe", rows) for lang in ("en", "vi")]
        pairs, _ = encode_rows(rows, toks, c.limit)
        src, tgt = collate(pairs)
        model = Transformer(*(len(t.vocab) for t in toks), c)
        before, attention = model(src, tgt[:, :-1])
        changed = tgt[:, :-1].clone()
        changed[:, -1] = 1
        torch.testing.assert_close(model(src, changed)[0][:, :-1], before[:, :-1])
        for weights in attention:
            assert weights.shape == (len(src), c.heads, tgt.size(1)-1, src.size(1))
            assert (weights.masked_select(src.eq(PAD)[:, None, None, :]) == 0).all()
            torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]))
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        for step in range(100):
            logits, _ = model(src, tgt[:, :-1])
            loss = F.cross_entropy(logits.flatten(0, 1), tgt[:, 1:].reshape(-1), ignore_index=PAD)
            if step == 0:
                initial = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        assert loss.item() < initial * 0.2
        model.eval()
        path = Path(tmp) / "mini.pt"
        torch.save(model.state_dict(), path)
        restored = Transformer(*(len(t.vocab) for t in toks), c).eval()
        restored.load_state_dict(torch.load(path, weights_only=True))
        torch.testing.assert_close(model(src, tgt[:, :-1])[0], restored(src, tgt[:, :-1])[0])
        with torch.inference_mode():
            memory, mask = restored.encode(src)
            keys = [layer.cross.project_memory(memory) for layer in restored.decoder]
            cache = [None] * len(keys)
            full = restored(src, tgt[:, :-1])[0]
            for j in range(min(5, tgt.size(1)-1)):
                step_logits, cache = restored.decode_step(tgt[:, j:j+1], j, cache, keys, mask)
                torch.testing.assert_close(step_logits, full[:, j], rtol=1e-4, atol=1e-5)
        for width in (1, 4):
            ids = translate(restored, src[:1], width, c.limit)
            assert ids == translate(model, src[:1], width, c.limit)
            assert 0 < len(ids) <= c.limit
            assert translate_batch(restored, src, width, c.limit) == [
                translate(restored, row[None], width, c.limit) for row in src]
        prefix = [BOS]
        with torch.inference_mode():
            for _ in range(c.limit):
                logits = restored(src[:1], torch.tensor([prefix]))[0][0, -1]
                logits[[PAD, BOS]] = -torch.inf
                prefix.append(logits.argmax().item())
                if prefix[-1] == EOS:
                    break
        assert translate(restored, src[:1], 1, c.limit) == prefix[1:]
        return {"initial_loss": initial, "final_loss": loss.item(), "steps": 100}


if __name__ == "__main__":
    print(check())
