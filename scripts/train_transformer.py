"""From the repository root: python -m scripts.train_transformer --kind bpe"""
import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from scripts.corpus import ROOT, records
from scripts.transformer import BOS, Config, Transformer, EOS, PAD, translate_batch
from scripts.tv3_data import Tokenizer, collate, encode_rows


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")


def batches(pairs, size, shuffle, seed):
    # Shuffle examples, then bucket by length locally to reduce padding.
    indices = list(range(len(pairs)))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(indices)
    groups = []
    for start in range(0, len(indices), size * 32):
        bucket = sorted(indices[start:start + size * 32],
                        key=lambda i: max(map(len, pairs[i])))
        groups.extend(bucket[j:j + size] for j in range(0, len(bucket), size))
    if shuffle:
        rng.shuffle(groups)
    for group in groups:
        yield collate([pairs[i] for i in group])


def epoch(model, pairs, device, batch, seed, optimizer=None):
    model.train(optimizer is not None)
    total, tokens = 0., 0
    for idx, (src, tgt) in enumerate(batches(pairs, batch, model.training, seed), 1):
        src, tgt = src.to(device), tgt.to(device)
        labels = tgt[:, 1:].reshape(-1)
        count = labels.ne(PAD).sum().item()
        with torch.set_grad_enabled(model.training), torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()):
            logits, _ = model(src, tgt[:, :-1])
            loss = F.cross_entropy(logits.flatten(0, 1), labels, ignore_index=PAD)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        total += loss.item() * count
        tokens += count
        if model.training and idx % 100 == 0:
            print(f"batch {idx}/{math.ceil(len(pairs)/batch)} loss {total/tokens:.4f}", flush=True)
    return total / tokens


def load_best(folder, device):
    state = torch.load(folder / "best.pt", map_location=device, weights_only=True)
    c = Config(**state["config"])
    toks = [Tokenizer(folder / "tokenizers", lang, state["kind"]) for lang in ("en", "vi")]
    model = Transformer(*(len(tok.vocab) for tok in toks), c).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), toks, c


def fit(kind, folder, device, epochs, batch):
    if (folder / "best.pt").exists():
        raise FileExistsError(f"Existing checkpoint in {folder}. Use --export-only or another --output.")
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    c = Config()
    train_rows = list(records("train"))
    toks = [Tokenizer(folder / "tokenizers", lang, kind, train_rows) for lang in ("en", "vi")]
    train, train_clip = encode_rows(train_rows, toks, c.limit)
    valid, valid_clip = encode_rows(records("validation"), toks, c.limit)
    model = Transformer(*(len(t.vocab) for t in toks), c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, fused=device.type == "cuda")
    details = dict(config=asdict(c), kind=kind, seed=42, epochs=epochs, batch=batch,
                   lr=0.0003, optimizer="AdamW", parameters=sum(p.numel() for p in model.parameters()),
                   train=len(train), validation=len(valid),
                   clipped_en_vi={"train": train_clip, "validation": valid_clip},
                   vocab_en_vi=[len(tok.vocab) for tok in toks],
                   libraries={pkg: version(pkg) for pkg in ("sentencepiece", "sacrebleu", "underthesea")},
                   pytorch=str(torch.__version__), device=torch.cuda.get_device_name() if device.type=="cuda" else "cpu")
    save_json(folder / "config.json", details)
    best, history = math.inf, []
    with SummaryWriter(str(folder / "tensorboard")) as writer:
        for n in range(1, epochs + 1):
            start = time.perf_counter()
            loss = epoch(model, train, device, batch, 42+n, optimizer)
            val = epoch(model, valid, device, batch, 42)
            history.append(dict(epoch=n, train_loss=loss, validation_loss=val, seconds=time.perf_counter()-start))
            print(history[-1], flush=True)
            for name in ("train_loss", "validation_loss"):
                writer.add_scalar(name, history[-1][name], n)
            if val < best:
                best = val
                torch.save(dict(config=asdict(c), kind=kind, epoch=n, model=model.state_dict()), folder / "best.tmp")
                os.replace(folder / "best.tmp", folder / "best.pt")
            save_json(folder / "history.json", history)


def attention_example(model, toks, src, ids, row, folder, decoder):
    from matplotlib import pyplot as plt
    prefix = src.new_tensor([[BOS, *ids[:-1]]])
    with torch.inference_mode():
        _, attn = model(src, prefix)
    weights = attn[-1][0, 0].cpu()
    source = [toks[0].vocab[i] for i in src[0].tolist()]
    target = [toks[1].vocab[i] for i in ids]
    stem = folder / f"attention-{row['id']}-{decoder}"
    save_json(stem.with_suffix(".json"), dict(**row, prediction=toks[1].decode(ids),
              source_tokens=source, target_tokens=target, layer=len(attn), head=1, weights=weights.tolist()))
    fig, ax = plt.subplots(figsize=(max(6,len(source)*.32), max(3,len(target)*.28)))
    ax.imshow(weights, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(source)), labels=source, rotation=90)
    ax.set_yticks(range(len(target)), labels=target)
    ax.set(title=f"{row['id']} | {decoder} | layer {len(attn)}, head 1", xlabel="English", ylabel="Vietnamese")
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)


def export(folder, device, kind, predictions):
    from sacrebleu.metrics import BLEU
    torch.set_float32_matmul_precision("high")
    model, toks, c = load_best(folder, device)
    for split in ("validation", "test"):
        rows = list(records(split))
        sources = [torch.tensor([BOS, *toks[0].encode(row["en"])[:c.limit-2], EOS]) for row in rows]
        destination = predictions / split
        destination.mkdir(parents=True, exist_ok=True)
        for decoder, width in (("greedy", 1), ("beam4", 4)):
            outputs, capped = [], 0
            examples = []
            start = time.perf_counter()
            batch = 32
            for offset in range(0, len(sources), batch):
                src = torch.nn.utils.rnn.pad_sequence(sources[offset:offset + batch], batch_first=True)
                decoded = translate_batch(model, src.to(device), width, c.limit)
                outputs.extend(toks[1].decode(ids) for ids in decoded)
                capped += sum(ids[-1] != EOS for ids in decoded)
                if split == "validation":
                    examples.extend((i, ids) for i, ids in enumerate(decoded, offset)
                                    if rows[i]["id"] in ("tst2012-1", "tst2012-2", "tst2012-7"))
            elapsed = time.perf_counter() - start
            stem = f"transformer-{kind}-{decoder}-seed42"
            path = destination / (stem + ".jsonl")
            with path.with_suffix(".tmp").open("w", encoding="utf8") as stream:
                for row, pred in zip(rows, outputs, strict=True):
                    print(json.dumps(dict(id=row["id"], prediction=pred), ensure_ascii=False), file=stream)
            os.replace(path.with_suffix(".tmp"), path)
            metric = BLEU(tokenize="13a", lowercase=False)
            bleu = metric.corpus_score(outputs, [[r["vi"] for r in rows]]).score
            save_json(folder / f"{split}-{decoder}.json", dict(bleu=bleu, signature=str(metric.get_signature()),
                      checkpoint="best.pt", count=len(rows), capped=capped, capped_fraction=capped/len(rows),
                      sentences_per_second=len(rows)/elapsed, inference_batch=batch))
            print(f"{kind} {split} {decoder}: {len(rows)} rows, BLEU {bleu:.2f}", flush=True)
            if split == "validation":
                with SummaryWriter(str(folder / "tensorboard")) as writer:
                    writer.add_scalar(f"validation_bleu/{decoder}", bleu)
            for i, ids in examples:
                attention_example(model, toks, sources[i][None].to(device), ids, rows[i], folder, decoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("word", "bpe", "unigram"), required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = args.output / "transformer" / args.kind
    if not args.export_only:
        fit(args.kind, run, device, args.epochs, args.batch)
    if not args.train_only:
        export(run, device, args.kind, args.output / "predictions")
