"""Train-only tokenizers compatible with notebook 01 and aligned tensor batches."""
import json
from collections import Counter
from pathlib import Path

import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence

from scripts.transformer import BOS, EOS, PAD, UNK


def words(text, lang):
    if lang == "en":
        return text.split()
    from underthesea import word_tokenize
    return word_tokenize(text, format="text").split()


class Tokenizer:
    def __init__(self, folder, lang, kind, train=None):
        self.lang = lang
        self.sp = None
        prefix = Path(folder) / f"{lang}-{kind}"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        if kind == "word":
            path = prefix.with_suffix(".json")
            if train is not None and not path.exists():
                freq = Counter(w for row in train for w in words(row[lang], lang))
                vocab = ["<pad>", "<unk>", "<s>", "</s>"]
                for special in vocab:
                    freq.pop(special, None)
                vocab.extend(w for w, _ in freq.most_common(7996))
                path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf8")
            self.vocab = json.loads(path.read_text(encoding="utf8"))
            self.lookup = {w: i for i, w in enumerate(self.vocab)}
        else:
            path = prefix.with_suffix(".model")
            if train is not None and not path.exists():
                spm.SentencePieceTrainer.train(
                    sentence_iterator=(r[lang] for r in train), model_prefix=str(prefix),
                    model_type=kind, vocab_size=8000, character_coverage=1.0,
                    normalization_rule_name="identity", pad_id=PAD, unk_id=UNK,
                    bos_id=BOS, eos_id=EOS, hard_vocab_limit=False, minloglevel=2)
            self.sp = spm.SentencePieceProcessor(model_file=str(path))
            self.vocab = [self.sp.id_to_piece(i) for i in range(len(self.sp))]

    def encode(self, text):
        if self.sp is not None:
            return self.sp.encode(text)
        return [self.lookup.get(w, UNK) for w in words(text, self.lang)]

    def decode(self, ids):
        ids = [i for i in ids if i not in (PAD, BOS, EOS)]
        if self.sp is not None:
            return self.sp.decode(ids)
        return " ".join(self.vocab[i].replace("_", " ") for i in ids)


def encode_rows(rows, tokenizers, limit):
    pairs, clipped = [], [0, 0]
    for row in rows:
        pair = []
        for i, tok in enumerate(tokenizers):
            ids = tok.encode(row[tok.lang])
            clipped[i] += len(ids) + 2 > limit
            pair.append(torch.tensor([BOS, *ids[:limit - 2], EOS]))
        pairs.append(pair)
    return pairs, clipped


def collate(pairs):
    return tuple(pad_sequence(group, batch_first=True, padding_value=PAD)
                 for group in zip(*pairs))
