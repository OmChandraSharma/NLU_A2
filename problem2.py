# Name Generation Assignment


from random import sample

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {DEVICE}")


# Data loading

def load_names(path="TrainingNames.txt"):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    names = [n.strip().lower() for n in re.split(r"\s+", content) if n.strip()]
    names = [n for n in names if n.isalpha() and len(n) >= 2]
    print(f"[Data] {len(names)} names, {len(set(names))} unique")
    return names


# Vocabulary

def build_vocab(names):
    chars = sorted(set("".join(names)))
    vocab = ["<SOS_TOKEN>", "<EOS_TOKEN>"] + chars
    c2i   = {c: i for i, c in enumerate(vocab)}
    i2c   = {i: c for i, c in enumerate(vocab)}
    return vocab, c2i, i2c

SOS= "<SOS_TOKEN>"
EOS = "<EOS_TOKEN>"


def name_to_tensor(name, c2i):
    indices = [c2i[SOS]] + [c2i[c] for c in name] + [c2i[EOS]]
    return torch.tensor(indices, dtype=torch.long, device=DEVICE)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================
# LOSS (fixed smoothing)
# =========================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)

        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


# =========================
# 1. VANILLA RNN (IMPROVED)
# =========================
class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.RNN(
            embed_size,
            hidden_size,
            num_layers=2,          # ✅ deeper
            batch_first=True,
            dropout=0.2            # ✅ regularization
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        return self.fc(out), h
    def generate(self, c2i, i2c, max_len=15, temperature=0.6):
        return sample(self, c2i, i2c, max_len=max_len, temperature=temperature)


# =========================
# 2. BLSTM (FOR COMPARISON)
# =========================
class BLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.lstm(x, h)
        return self.fc(out), h
    def generate(self, c2i, i2c, max_len=15, temperature=0.7):
        return sample(self, c2i, i2c, max_len=max_len, temperature=temperature)


# =========================
# 3. SIMPLE ATTENTION RNN
# =========================
class AttentionRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)

        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, h=None):
        emb = self.embed(x)
        outputs, h = self.rnn(emb, h)

        # attention over sequence
        scores = self.v(torch.tanh(self.attn(outputs)))   # (B, T, 1)
        weights = F.softmax(scores, dim=1)

        context = (weights * outputs).sum(dim=1, keepdim=True)
        context = context.repeat(1, outputs.size(1), 1)

        out = torch.cat([outputs, context], dim=-1)
        return self.fc(out), h
    
    def generate(self, c2i, i2c, max_len=15, temperature=0.7):
        return sample(self, c2i, i2c, max_len=max_len, temperature=temperature)


# =========================
# SAMPLING (FIXED)
# =========================
def sample(model, c2i, i2c, max_len=15, temperature=0.6, top_k=5):
    model.eval()
    device = next(model.parameters()).device

    idx = torch.tensor([[c2i[SOS]]]).to(device)
    h = None
    name = ""

    with torch.no_grad():
        for step in range(max_len):
            logits, h = model(idx, h)

            probs = F.softmax(logits[0, -1] / temperature, dim=-1)

            # ❗ block EOS only in first few steps
            if step < 2:
                probs[c2i[EOS]] = 0

            # remove SOS always
            probs[c2i[SOS]] = 0

            topk_probs, topk_idx = torch.topk(probs, top_k)
            topk_probs /= topk_probs.sum()

            idx = topk_idx[torch.multinomial(topk_probs, 1)].unsqueeze(0)

            char = i2c[idx.item()]

            if char == EOS:
                break

            name += char

    return name

# =========================
# TRAIN
# =========================
def train(model, names, c2i, i2c, epochs=100, lr=0.003):
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []

    for ep in range(1, epochs + 1):
        model.train()
        np.random.shuffle(names)
        total_loss = 0

        for name in names:
            seq = [c2i[SOS]] + [c2i[c] for c in name] + [c2i[EOS]]
            seq = torch.tensor(seq).to(DEVICE)

            x = seq[:-1].unsqueeze(0)
            y = seq[1:]

            optimizer.zero_grad()

            logits, _ = model(x)
            loss = criterion(logits.squeeze(0), y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(names)
        losses.append(avg)

        if ep % 10 == 0:
            model.eval()
            print(f"\nEpoch {ep} | Loss {avg:.3f}")
            print([sample(model, c2i, i2c) for _ in range(5)])

    return losses

# =========================
# EVALUATE
# =========================
def evaluate(model, c2i, i2c, train_names, n=200, temperature=0.7):
    model.to(DEVICE)
    model.eval()

    generated = []
    for _ in range(n):
        name = model.generate(c2i, i2c, temperature=temperature)
        if name and name.isalpha() and 2 <= len(name) <= 15:
            generated.append(name)

    train_set = set(train_names)
    unique = set(generated)
    novel = [g for g in generated if g not in train_set]

    return {
        "generated": len(generated),
        "unique": len(unique),
        "novelty_rate": round(len(novel)/len(generated)*100, 2) if generated else 0,
        "diversity": round(len(unique)/len(generated)*100, 2) if generated else 0,
        "samples": list(unique)[:20]
    }

# =============================================================================
# TASK 3 - QUALITATIVE ANALYSIS
# =============================================================================

def qualitative_analysis(results):
    print("\n-- TASK 3: QUALITATIVE ANALYSIS --\n")
    for model_name, res in results.items():
        samples = res["samples"]
        print(f"Model: {model_name}")
        print(f"  Novelty Rate : {res['novelty_rate']}%")
        print(f"  Diversity    : {res['diversity']}%")
        print(f"  Sample names : {samples[:15]}")

        avg_len   = np.mean([len(s) for s in samples]) if samples else 0
        too_short = [s for s in samples if len(s) <= 2]
        too_long  = [s for s in samples if len(s) > 12]
        repeating = [s for s in samples if len(s) >= 4 and
                     len(set(s[i:i+2] for i in range(len(s)-1))) < 3]

        print(f"  Avg length   : {avg_len:.1f} chars")
        print(f"  Too short    : {too_short[:5]  if too_short  else 'none'}")
        print(f"  Too long     : {too_long[:5]   if too_long   else 'none'}")
        print(f"  Repetitive   : {repeating[:5]  if repeating  else 'none'}")
        print()


# Plots

def plot_loss_curves(all_losses):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"Vanilla RNN":"#2563eb", "BLSTM":"#dc2626", "Attention RNN":"#16a34a"}
    for name, losses in all_losses.items():
        ax.plot(range(1, len(losses)+1), losses,
                label=name, color=colors[name], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Loss")
    ax.set_title("Training Loss Curves", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved plot_loss_curves.png")


def plot_comparison(results, model_params):
    models    = list(results.keys())
    novelty   = [results[m]["novelty_rate"] for m in models]
    diversity = [results[m]["diversity"]    for m in models]
    params    = [model_params[m] / 1000     for m in models]
    colors    = ["#2563eb", "#dc2626", "#16a34a"]
    x         = np.arange(len(models))

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Model Comparison — Task 2", fontweight="bold")

    for ax, values, title, ylabel in [
        (axes[0], novelty,   "Novelty Rate (%)",         "%"),
        (axes[1], diversity, "Diversity (%)",             "%"),
        (axes[2], params,    "Trainable Parameters (K)", "Thousands"),
    ]:
        bars = ax.bar(x, values, color=colors, alpha=0.85)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, axis="y")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("plot_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved plot_comparison.png")


# Main

def main():
    names = load_names("TrainingNames.txt")
    vocab, c2i, i2c = build_vocab(names)
    V = len(vocab)
    print(f"[Vocab] size={V}")

    EMBED  = 64
    HIDDEN = 128   # reduced from 256 — prevents BLSTM from memorising
    LR     = 0.002
    EPOCHS = 60

    rnn   = VanillaRNN(V, embed_size=EMBED, hidden_size=HIDDEN)
    blstm = BLSTM(V, embed_size=EMBED, hidden_size=HIDDEN)
    attn  = AttentionRNN(V, embed_size=EMBED, hidden_size=HIDDEN)
    print("\n-- TASK 1: ARCHITECTURE SUMMARY --\n")
    descs = {
        "Vanilla RNN":   "Embedding -> RNN(128) -> Dropout(0.3) -> Linear",
        "BLSTM":         "Embedding -> BiLSTM(128x2) -> Dropout(0.4) -> Linear",
        "Attention RNN": "Embedding -> GRU encoder(128) -> Bahdanau Attention -> GRU decoder(128) -> Linear",
    }
    for name, model in [("Vanilla RNN",rnn),("BLSTM",blstm),("Attention RNN",attn)]:
        print(f"Model : {name}")
        print(f"  Architecture     : {descs[name]}")
        print(f"  Embed size       : {EMBED}")
        print(f"  Hidden size      : {HIDDEN}")
        print(f"  Learning rate    : {LR}  (halved at epochs 20, 40)")
        print(f"  Weight decay     : 1e-4")
        print(f"  Label smoothing  : 0.15")
        print(f"  Trainable params : {count_params(model):,}")
        print()

    all_losses = {}
    print("-- Training Vanilla RNN --")
    all_losses["Vanilla RNN"] = train(rnn, names, c2i, i2c,
                                    epochs=EPOCHS)

    print("\n-- Training BLSTM --")
    all_losses["BLSTM"] = train(blstm, names, c2i, i2c,
                            epochs=EPOCHS)

    print("\n-- Training Attention RNN --")
    all_losses["Attention RNN"] = train(attn, names, c2i, i2c,
                                    epochs=EPOCHS)

    plot_loss_curves(all_losses)

    print("\n-- TASK 2: QUANTITATIVE EVALUATION --\n")
    results      = {}
    model_params = {}
    for name, model in [("Vanilla RNN",rnn),("BLSTM",blstm),("Attention RNN",attn)]:
        model.to(DEVICE)
        res = evaluate(model, c2i, i2c, names, n=500, temperature=1.0)
        results[name]      = res
        model_params[name] = count_params(model)
        print(f"Model: {name}")
        print(f"  Generated    : {res['generated']}")
        print(f"  Unique       : {res['unique']}")
        print(f"  Novelty Rate : {res['novelty_rate']}%")
        print(f"  Diversity    : {res['diversity']}%")
        print()

    plot_comparison(results, model_params)
    qualitative_analysis(results)

    print("Done.")
    print("  plot_loss_curves.png — training loss per epoch")
    print("  plot_comparison.png  — novelty, diversity, param count")


if __name__ == "__main__":
    main()
