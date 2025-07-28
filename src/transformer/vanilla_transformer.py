import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import re
import numpy as np

class HyperParams:
    def __init__(self):
        self.d_x = 512
        self.n_layers = 6
        self.n_heads = 8
        self.d_k = self.d_x // self.n_heads
        self.d_v = self.d_x // self.n_heads
        self.dropout = 0.1
        self.max_length = 50

class QEDTokenizer:
    def __init__(self):
        self.special_tokens = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.vocab = self.special_tokens.copy()
        self.next_id = len(self.vocab)
        self.operators = ["+", "-", "*", "/", "^"]
        self.variables = ["m_d", "m_u", "s_11", "s_12"]

    def add_token(self, token):
        if token not in self.vocab:
            self.vocab[token] = self.next_id
            self.next_id += 1

    def build_vocab(self, expressions):
        for expr in expressions:
            tokens = self.tokenize(expr)
            for token in tokens:
                self.add_token(token)

    def tokenize(self, expr):
        tokens = []
        expr = expr.replace(" ", "")
        i = 0
        while i < len(expr):
            matched = False
            for var in self.variables:
                if expr[i:].startswith(var):
                    tokens.append(var)
                    i += len(var)
                    matched = True
                    break
            if matched:
                continue
            if expr[i] in self.operators:
                tokens.append(expr[i])
                i += 1
                continue
            num = ""
            while i < len(expr) and (expr[i].isdigit() or expr[i] == "."):
                num += expr[i]
                i += 1
            if num:
                tokens.append(num)
                continue
            i += 1
        return tokens

    def encode(self, expr, max_length):
        tokens = self.tokenize(expr)
        ids = [self.vocab["<BOS>"]] + [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens] + [self.vocab["<EOS>"]]
        ids = ids[:max_length]
        ids += [self.vocab["<PAD>"]] * (max_length - len(ids))
        return ids

    def decode(self, ids):
        tokens = [k for id in ids for k, v in self.vocab.items() if v == id and k not in ["<PAD>", "<BOS>", "<EOS>"]]
        return "".join(tokens)

    def get_vocab(self):
        return self.vocab

class QEDDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = str(self.data.iloc[idx]["amp"])
        trg = str(self.data.iloc[idx]["sqamp"])
        src_ids = self.tokenizer.encode(src, self.max_length)
        trg_ids = self.tokenizer.encode(trg, self.max_length)
        return {
            "input_ids": torch.tensor(src_ids, dtype=torch.long),
            "labels": torch.tensor(trg_ids, dtype=torch.long)
        }

class StandardEmbedding(nn.Module):
    def __init__(self, d_vocab, d_x, dropout, max_length):
        super().__init__()
        self.d_x = d_x
        self.dropout = nn.Dropout(dropout)
        self.tok_embedding = nn.Embedding(d_vocab, d_x)
        self.scale = torch.sqrt(torch.FloatTensor([d_x]))
        pe = torch.zeros(max_length, d_x)
        position = torch.arange(0., max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_x, 2) * -(math.log(10000.0) / d_x))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, src):
        if src.max().item() >= self.tok_embedding.num_embeddings:
            raise ValueError(f"Input indices {src.max().item()} exceed vocab size {self.tok_embedding.num_embeddings}")
        tok_emb = self.tok_embedding(src) * self.scale.to(src.device)
        pos_emb = self.pe[:, :src.size(1)]
        x = tok_emb + pos_emb
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_x, n_heads, dropout):
        super().__init__()
        self.d_x = d_x
        self.n_heads = n_heads
        self.d_k = d_x // n_heads
        self.W_q = nn.Linear(d_x, d_x)
        self.W_k = nn.Linear(d_x, d_x)
        self.W_v = nn.Linear(d_x, d_x)
        self.W_o = nn.Linear(d_x, d_x)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        bsz = query.size(0)
        Q = self.W_q(query).view(bsz, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(bsz, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(bsz, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(bsz, -1, self.d_x)
        return self.W_o(context)

class EncoderLayer(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.attn = MultiHeadAttention(p.d_x, p.n_heads, p.dropout)
        self.ff = nn.Linear(p.d_x, p.d_x)
        self.norm1 = nn.LayerNorm(p.d_x)
        self.norm2 = nn.LayerNorm(p.d_x)
        self.dropout = nn.Dropout(p.dropout)

    def forward(self, src, src_mask):
        z = self.norm1(src)
        z = self.attn(z, z, z, src_mask)
        src = src + self.dropout(z)
        z = self.norm2(src)
        z = F.relu(self.ff(z))
        src = src + self.dropout(z)
        return src

class Encoder(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(p) for _ in range(p.n_layers)])

    def forward(self, src, src_mask):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

class DecoderLayer(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.self_attn = MultiHeadAttention(p.d_x, p.n_heads, p.dropout)
        self.enc_attn = MultiHeadAttention(p.d_x, p.n_heads, p.dropout)
        self.ff = nn.Linear(p.d_x, p.d_x)
        self.norm1 = nn.LayerNorm(p.d_x)
        self.norm2 = nn.LayerNorm(p.d_x)
        self.norm3 = nn.LayerNorm(p.d_x)
        self.dropout = nn.Dropout(p.dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        z = self.norm1(trg)
        z = self.self_attn(z, z, z, trg_mask)
        trg = trg + self.dropout(z)
        z = self.norm2(trg)
        z = self.enc_attn(z, enc_src, enc_src, src_mask)
        trg = trg + self.dropout(z)
        z = self.norm3(trg)
        z = F.relu(self.ff(z))
        trg = trg + self.dropout(z)
        return trg

class Decoder(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(p) for _ in range(p.n_layers)])

    def forward(self, trg, enc_src, trg_mask, src_mask):
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        return trg

class Transformer(nn.Module):
    def __init__(self, d_vocab, d_x, n_layers, n_heads, dropout, max_length, pad_idx):
        super().__init__()
        self.p = HyperParams()
        self.p.d_x = d_x
        self.p.n_layers = n_layers
        self.p.n_heads = n_heads
        self.p.dropout = dropout
        self.p.max_length = max_length
        self.pad_idx = pad_idx
        self.embedding = StandardEmbedding(d_vocab, d_x, dropout, max_length)
        self.encoder = Encoder(self.p)
        self.decoder = Decoder(self.p)
        self.out = nn.Linear(d_x, d_vocab)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def make_masks(self, src, trg):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len, device=trg.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return src_mask, trg_mask

    def forward(self, src, trg):
        src_mask, trg_mask = self.make_masks(src, trg)
        src_emb = self.embedding(src)
        trg_emb = self.embedding(trg)
        enc_src = self.encoder(src_emb, src_mask)
        dec_out = self.decoder(trg_emb, enc_src, trg_mask, src_mask)
        logits = self.out(dec_out)
        return logits

    def greedy_inference(self, src, sos_idx, eos_idx, max_length):
        self.eval()
        src = src.to(self.device)
        batch_size = src.size(0)
        src_mask = self.make_masks(src, src)[0]
        enc_src = self.encoder(self.embedding(src), src_mask)
        trg = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=self.device)
        for _ in range(max_length):
            trg_mask = self.make_masks(trg, trg)[1]
            out = self.decoder(self.embedding(trg), enc_src, trg_mask, src_mask)
            logits = self.out(out[:, -1])
            pred = torch.argmax(logits, dim=-1).unsqueeze(1)
            trg = torch.cat([trg, pred], dim=1)
            if torch.all(pred == eos_idx):
                break
        return trg

def train_and_evaluate(model, train_loader, val_loader, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_idx)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            src = batch["input_ids"].to(device)
            trg = batch["labels"].to(device)
            optimizer.zero_grad()
            output = model(src, trg[:, :-1])
            output = output.view(-1, output.size(-1))
            target = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            preds = torch.argmax(output, dim=-1)
            mask = target != model.pad_idx
            train_correct += (preds[mask] == target[mask]).sum().item()
            train_total += mask.sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch["input_ids"].to(device)
                trg = batch["labels"].to(device)
                output = model(src, trg[:, :-1])
                output = output.view(-1, output.size(-1))
                target = trg[:, 1:].contiguous().view(-1)
                loss = criterion(output, target)
                val_loss += loss.item()
                preds = torch.argmax(output, dim=-1)
                mask = target != model.pad_idx
                val_correct += (preds[mask] == target[mask]).sum().item()
                val_total += mask.sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")

