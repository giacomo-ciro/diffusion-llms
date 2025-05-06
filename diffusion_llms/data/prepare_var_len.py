"""
Stream the FineWeb dataset from HF, find samples with context_length
tokens and save them to two contiguous np.memmap, one for train and
the other for test, with appropriate padding so they all have context_length
total tokens (of which some are true text tokens and others are eos+pad...).
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from diffusion_llms.tokenizers.custom_gpt_w_pad import CustomGPT2TokenizerWithPad


@dataclass
class PreprocConfig:
    eos_id: int
    pad_id: int
    ctx_len: int
    train_docs: int
    test_docs: int
    buffer_size: int
    shuffle_seed: int
    output_root: Path


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config() -> PreprocConfig:
    parser = argparse.ArgumentParser(
        description="Single-pass preprocess FineWeb dataset"
    )
    parser.add_argument("config", type=Path, help="Path to JSON config file")
    parser.add_argument(
        "-tr", "--train", type=int, default=10, help="Number of sequences for train set"
    )
    parser.add_argument(
        "-te", "--test", type=int, default=1, help="Number of sequences for test set"
    )
    parser.add_argument(
        "--buffer_size", type=int, default=10000, help="Shuffle buffer size"
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    if not args.config.is_file():
        logging.error(f"Config file not found: {args.config}")
        sys.exit(1)
    with args.config.open() as f:
        cfg = json.load(f)

    return PreprocConfig(
        eos_id=cfg["eos_token_id"],
        pad_id=cfg["pad_token_id"],
        ctx_len=cfg.get("context_length", 256),
        train_docs=args.train,
        test_docs=args.test,
        buffer_size=args.buffer_size,
        shuffle_seed=args.seed,
        output_root=Path(cfg.get("output_dir", Path.cwd())),
    )


def prepare_dataset(cfg: PreprocConfig):
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    return ds.shuffle(buffer_size=cfg.buffer_size, seed=cfg.shuffle_seed)


def tokenize_stream(raw_ds, tokenizer, ctx_len: int):
    def _proc(ex):
        ids = tokenizer.apply_template(
            ex["text"], template_type=1, max_length=ctx_len, padding=True
        )
        return {"ids": ids}

    return raw_ds.map(_proc, remove_columns=["text"])


def collect_valid_sequences(token_stream, cfg: PreprocConfig):
    valid = []
    total_needed = cfg.train_docs + cfg.test_docs
    checked = 0
    for sample in tqdm(token_stream, desc="Filtering sequences", unit="seq"):
        ids = sample["ids"]
        checked += 1
        # if ends with PAD => original text shorter than ctx_len
        if ids[-1] == cfg.pad_id:
            valid.append(ids)
            if len(valid) >= total_needed:
                break
    
    if len(valid) < total_needed:
        logging.error(
            f"Only found {len(valid)} valid sequences, but need {total_needed}"
        )
        sys.exit(1)
    logging.info(f"Collected {len(valid)} valid sequences. Checked {checked} samples.")
    return valid, checked


def write_memmaps(valid_seqs, cfg: PreprocConfig, puid: str):
    outdir = cfg.output_root / puid
    outdir.mkdir(parents=True, exist_ok=False)

    ctx = cfg.ctx_len
    dt = np.uint16
    train_n = cfg.train_docs
    test_n = cfg.test_docs

    train_path = outdir / "train.bin"
    test_path = outdir / "test.bin"

    arr_train = np.memmap(train_path, dtype=dt, mode="w+", shape=(train_n * ctx,))
    arr_test = np.memmap(test_path, dtype=dt, mode="w+", shape=(test_n * ctx,))

    train_text_tokens = 0
    test_text_tokens = 0

    for idx, ids in enumerate(valid_seqs):
        pad_count = ids.count(cfg.pad_id)
        text_len = ctx - pad_count

        if idx < train_n:
            arr_train[idx * ctx : (idx + 1) * ctx] = ids
            train_text_tokens += text_len
        else:
            t_idx = idx - train_n
            arr_test[t_idx * ctx : (t_idx + 1) * ctx] = ids
            test_text_tokens += text_len

    arr_train.flush()
    arr_test.flush()
    return outdir, train_text_tokens, test_text_tokens


def write_metadata(
    outdir: Path, puid: str, cfg: PreprocConfig, train_tokens: int, test_tokens: int, checked: int, cmd: str, config_path: str
):
    meta = outdir / f"{puid}.txt"
    ctx = cfg.ctx_len
    train_n = cfg.train_docs
    test_n = cfg.test_docs
    eos_id = cfg.eos_id
    pad_id = cfg.pad_id
    # Formatting numbers with commas
    def fmt(n):
        return f"{n:,}"
    def avg(a, b):
        return f"{a / b:.2f}" if b else "0.00"
    with meta.open("w") as f:
        f.write(
            f"Metadata for Test / Train Datasets {puid}\n\n"
            f"Generated on: {puid}\n"
            f"Using: $ {cmd}\n\n"
            f"Total Checked Samples: {fmt(checked)}\n\n"
            f"== Hyper-params ==\n"
            f"EOS token id: {eos_id}\n"
            f"PAD token id: {pad_id}\n"
            f"Format: [text tokens] [{eos_id}] [{pad_id}, ..., {pad_id}]\n"
            f"Context Length: {ctx}\n\n"
            f"== Train ==\n"
            f"Valid (target): {fmt(train_n)} ({fmt(train_n)})\n"
            f"Text Tokens: {fmt(train_tokens)}\n"
            f"Tot Tokens: {fmt(train_n * ctx)}\n"
            f"Average Text Tokens per Sample: {avg(train_tokens, train_n)}\n\n"
            f"== Test ==\n"
            f"Valid (target): {fmt(test_n)} ({fmt(test_n)})\n"
            f"Text Tokens: {fmt(test_tokens)}\n"
            f"Tot Tokens: {fmt(test_n * ctx)}\n"
            f"Average Text Tokens per Sample: {avg(test_tokens, test_n)}\n"
        )
    logging.info(f"Metadata written to {meta}")


def main():
    setup_logging()
    logging.info("Loading configuration...")
    cfg = load_config()
    puid = time.strftime("%d%m%Y%H%M%S")

    logging.info("Loading tokenizer...")
    tokenizer = CustomGPT2TokenizerWithPad()

    raw_ds = prepare_dataset(cfg)
    tok_ds = tokenize_stream(raw_ds, tokenizer, cfg.ctx_len)

    # Get command line for metadata
    cmd = f"python prepare_var_len.py --config {sys.argv[1]} --train {cfg.train_docs} --test {cfg.test_docs}"
    config_path = sys.argv[1]

    valid, checked = collect_valid_sequences(tok_ds, cfg)
    outdir, tr_toks, te_toks = write_memmaps(valid, cfg, puid)
    write_metadata(outdir, puid, cfg, tr_toks, te_toks, checked, cmd, config_path)

    logging.info(f"Train memmap: {outdir / 'train.bin'}")
    logging.info(f"Test  memmap: {outdir / 'test.bin'}")
    logging.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
    sys.exit(0)
