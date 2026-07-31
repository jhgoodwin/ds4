#!/usr/bin/python3
"""
DSpark Markov probe — standalone.
Loads Markov weights (Q8_0, ~70 MB on disk → ~530 MB F32) + vocab (header only).
No base model weights. CPU only.

The Markov head is a rank-256 bilinear n-gram (W1@W2) over 129280 vocab.
It captures bigram token statistics but is NOT the full draft model.
Full draft predictions require HC from base model + 3 stage blocks (not loaded here).

Usage:
  ./draft_probe.py -n 20 --temp 0.8
  ./draft_probe.py -n 50 --temp 1.0 --seed 0
  ./draft_probe.py --entropy 1000  # dump top-k entropy analysis
"""
import struct, sys, argparse, os

try:
    import numpy as np
except:
    print("numpy required: pip install numpy")
    sys.exit(1)

def gstr(f):
    slen = struct.unpack('<Q', f.read(8))[0]
    return f.read(slen)

def parse_gguf(path, load_vocab=False):
    """Parse GGUF. Returns (info_dict, vocab_list, bos_id)."""
    TYPES = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 8:-1, 9:-1, 10:8, 11:8, 12:2}
    with open(path, 'rb') as f:
        magic = f.read(4); ver = struct.unpack('<I', f.read(4))[0]
        n_t = struct.unpack('<Q', f.read(8))[0]
        n_m = struct.unpack('<Q', f.read(8))[0]
        meta = {}
        for _ in range(n_m):
            key = gstr(f).decode()
            tc = struct.unpack('<I', f.read(4))[0]
            if tc == 8:
                meta[key] = gstr(f)
            elif tc == 9:
                at = struct.unpack('<I', f.read(4))[0]
                al = struct.unpack('<Q', f.read(8))[0]
                if load_vocab and key == 'tokenizer.ggml.tokens':
                    arr = [gstr(f) if at == 8 else f.read(4) for _ in range(al)]
                elif load_vocab and key == 'tokenizer.ggml.scores':
                    arr = [struct.unpack('<f', f.read(4))[0] for _ in range(al)]
                else:
                    arr = None
                    for _ in range(al):
                        if at == 8: gstr(f)
                        elif at in (5, 4, 6): f.read(4)
                        elif at == 10: f.read(8)
                        else: f.read(4)
                meta[key] = arr
            elif tc == 0: meta[key] = struct.unpack('<B', f.read(1))[0]
            elif tc == 1: meta[key] = struct.unpack('<b', f.read(1))[0]
            elif tc == 2: meta[key] = struct.unpack('<H', f.read(2))[0]
            elif tc == 3: meta[key] = struct.unpack('<h', f.read(2))[0]
            elif tc == 4: meta[key] = struct.unpack('<I', f.read(4))[0]
            elif tc == 5: meta[key] = struct.unpack('<i', f.read(4))[0]
            elif tc == 6: meta[key] = struct.unpack('<f', f.read(4))[0]
            elif tc == 7: meta[key] = bool(struct.unpack('<B', f.read(1))[0])
            elif tc == 10: meta[key] = struct.unpack('<q', f.read(8))[0]
            elif tc == 11: meta[key] = struct.unpack('<d', f.read(8))[0]
            elif tc == 12: meta[key] = struct.unpack('<e', f.read(2))[0]
            else: f.read(8); meta[key] = None

        info = {}
        for _ in range(n_t):
            name = gstr(f).decode()
            nd = struct.unpack('<I', f.read(4))[0]
            dims = struct.unpack(f'<{nd}Q', f.read(8 * nd))
            dt = struct.unpack('<I', f.read(4))[0]
            off = struct.unpack('<Q', f.read(8))[0]
            info[name] = (dims, dt, off)

    vocab = meta.get('tokenizer.ggml.tokens', []) if load_vocab else []
    bos_id = meta.get('tokenizer.ggml.bos_token_id', 0)
    return info, vocab, bos_id


def load_q8_0(path, off, n):
    """Load Q8_0 quantized tensor, return F32 numpy array. Blocks of 32."""
    nb = (n + 31) // 32
    out = np.empty(n, dtype=np.float32)
    with open(path, 'rb') as f:
        f.seek(off)
        for i in range(nb):
            scale = struct.unpack('<e', f.read(2))[0]
            vals = struct.unpack('<32b', f.read(32))
            start = i * 32
            end = min(start + 32, n)
            out[start:end] = np.array(vals[:end - start], dtype=np.float32) * scale
    return out


def decode_tok(vocab, tid):
    """Decode token ID to display string."""
    if tid < len(vocab):
        raw = vocab[tid]
        if isinstance(raw, bytes):
            try:
                return raw.decode('utf-8', errors='replace')
            except:
                return repr(raw)
        return str(raw)
    return f"[ID:{tid}]"


def top_k_tokens(vocab, probs, k=10):
    """Return (token_ids, probabilities, decoded strings) for top-k."""
    idx = np.argsort(probs)[-k:][::-1]
    tokens = []
    for i in idx:
        tokens.append((int(i), float(probs[i]), decode_tok(vocab, int(i))))
    return tokens


def markov_forward(w1, w2, token_id):
    """Compute Markov logits for next token given current token ID."""
    n_vocab = w1.shape[1]
    oh = np.zeros(n_vocab, dtype=np.float32)
    if token_id < n_vocab:
        oh[token_id] = 1.0
    h = oh @ w1.T  # [n_vocab] @ [n_vocab, rank] = [rank]
    logits = h @ w2  # [rank] @ [rank, n_vocab] = [n_vocab]
    return logits


def main():
    ap = argparse.ArgumentParser(description="DSpark Markov probe")
    ap.add_argument("--support", default="/opt/ds4/gguf/DeepSeek-V4-Flash-DSpark-support.gguf",
                    help="DSpark support GGUF")
    ap.add_argument("--base", default="/opt/ds4/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf",
                    help="Base model GGUF (vocab only)")
    ap.add_argument("-n", type=int, default=20, help="tokens to generate")
    ap.add_argument("--temp", type=float, default=0.8, help="sampling temperature")
    ap.add_argument("--seed", type=int, default=None, help="seed token ID (default: BOS)")
    ap.add_argument("--entropy", type=int, default=0,
                    help="analyze entropy for first N seed tokens")
    ap.add_argument("--topk", type=int, default=5, help="top-k to display in entropy mode")
    args = ap.parse_args()

    # ── Load Markov weights ──
    print(f"Loading {args.support}...", file=sys.stderr)
    info, _, _ = parse_gguf(args.support)
    sp = args.support

    for tn in ["mtp.2.markov_head.markov_w1.weight", "mtp.2.markov_head.markov_w2.weight"]:
        dims, dt, off = info[tn]
        n = 1
        for d in dims:
            n *= d
        w = load_q8_0(sp, off, n).reshape(dims)  # GGUF dims = (rank, n_vocab)
        if 'w1' in tn:
            w1 = w
        else:
            w2 = w

    n_vocab = w1.shape[1]
    rank = w1.shape[0]
    print(f"  Markov: rank={rank} vocab={n_vocab}  "
          f"(weights: {w1.nbytes/1e6:.0f} MB F32)", file=sys.stderr)

    # ── Load vocabulary ──
    print(f"Loading vocab from {args.base}...", file=sys.stderr)
    _, vocab, bos_id = parse_gguf(args.base, load_vocab=True)
    print(f"  Vocab: {len(vocab)} tokens, BOS={bos_id}", file=sys.stderr)

    # ── Entropy analysis mode ──
    if args.entropy > 0:
        print(f"\n{'Seed':>7}  {'H(bits)':>8}  {'Top-1':>8}  p_top1  Decoded")
        print("-" * 65)
        for seed in range(min(args.entropy, n_vocab)):
            logits = markov_forward(w1, w2, seed)
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            H = -np.sum(probs * np.log2(probs + 1e-30))
            top = int(np.argmax(probs))
            print(f"{seed:7d}  {H:8.2f}  {top:8d}  {probs[top]:.4f}  {decode_tok(vocab, top)[:40]}")
        return

    # ── Chain generation ──
    seed = args.seed if args.seed is not None else bos_id
    print(f"\nSeed: {seed} ('{decode_tok(vocab, seed)[:40]}')  temp={args.temp}", file=sys.stderr)

    tok = seed
    total_lp = 0.0
    n_tok = 0
    unique = set()

    for i in range(args.n):
        logits = markov_forward(w1, w2, tok)
        logits = logits / args.temp
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()

        top5 = top_k_tokens(vocab, probs, 5)

        # Sample
        tok = int(np.random.choice(n_vocab, p=probs))
        lp = float(np.log(probs[tok]))
        unique.add(tok)
        total_lp += lp

        display = decode_tok(vocab, tok).replace('\n', '\\n')[:50]
        top_display = ' | '.join(f"{t}({s[:15]})" for t, p, s in top5[:3])
        print(f"  [{i:2d}] {tok:6d}  lp={lp:+.2f}  '{display}'  [{top_display}]")
        n_tok += 1

    avg_lp = total_lp / n_tok if n_tok else 0
    print(f"\n  {n_tok} tokens, {len(unique)} unique, "
          f"avg lp={avg_lp:.3f}, ppl={np.exp(-avg_lp):.1f}", file=sys.stderr)
    print(f"  Note: Markov-only (rank-256 bigram). No HC / stage blocks.", file=sys.stderr)


if __name__ == "__main__":
    main()
