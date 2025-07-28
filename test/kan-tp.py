import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import warnings
from typing import OrderedDict, Tuple, List
import time

# KANLinear (updated)
class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size  # Changed from grid_size to self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        output = output.view(*original_shape[:-1], self.out_features)
        return output

# Hyperparameters
class HyperParams:
    def __init__(self):
        self.d_x = 512
        self.n_layers = 6
        self.n_heads = 8
        self.d_k = self.d_x // self.n_heads
        self.d_v = self.d_x // self.n_heads
        self.dropout = 0.1
        self.max_length = 50

# SymbolicQEDTokenizer (unchanged)
class SymbolicQEDTokenizer:
    def __init__(self, df=None, index_token_pool_size=100, special_symbols=None, unk_idx=1, to_replace=True):
        self.amps = df.amp.tolist() if df is not None else None
        self.sqamps = df.sqamp.tolist() if df is not None else None
        if index_token_pool_size < 50:
            warnings.warn(f"Index token pool size ({index_token_pool_size}) may be insufficient. Consider using at least 50-100 tokens for symbolic tasks.", UserWarning)
        self.index_pool = [f"INDEX_{i}" for i in range(index_token_pool_size)]
        self.particle_index_pool = [f"PINDEX_{i}" for i in range(index_token_pool_size)]
        self.special_symbols = special_symbols or ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"]
        self.unk_idx = unk_idx
        self.to_replace = to_replace
        self.pattern_underscore_curly = re.compile(r'\b[\w]+(?:_[\w]+)*_{')
        self.pattern_mass = re.compile(r'\bm_([a-z]+)\b')
        self.pattern_mandelstam = re.compile(r'\bs_(\d{2,})\b')
        self.pattern_momentum = re.compile(r'\bp_(\d+)\b')
        self.pattern_single_s = re.compile(r'\bs_(\d+)\b(?!\d)')
        self.pattern_exponent = re.compile(r'\^(\w+|\([^)]+\))')
        self.pattern_special = re.compile(r'_([uv])|\\(\w+_\d+|\w+\b)')
        self.pattern_num_123 = re.compile(r'\b(?![psijkl]_)(?!MOMENTUM_)(?!MASS_)(?!P_)(?!S_)(?!MANDELSTAM_)\w+_\d+\b')
        self.pattern_particle = re.compile(r'(?P<prefix>\b(?:\w+_)?)?(?P<target>[ijkl]_\d+\b)')

    def preprocess_expression(self, expr):
        expr = expr.replace(' * ', '*').replace(' / ', '/').replace(' ^ ', '^')
        expr = expr.replace(' + ', '+').replace(' - ', '-')
        expr = expr.replace("+-", "-")
        expr = expr.replace("-+", "-")
        expr = ' '.join(expr.split())
        expr = expr.replace('me', 'm_e')  # Adjust based on dataset
        return expr

    @staticmethod
    def remove_whitespace(expression: str) -> str:
        return re.sub(r'\s+', '', expression)

    def protect_structures(self, ampl: str) -> Tuple[str, List[str]]:
        protected = []
        return ampl, protected

    def physics_aware_replace(self, ampl: str, is_source: bool = True) -> str:
        ampl = self.remove_whitespace(ampl)
        ampl = re.sub(r'\bi\b(?!\w)', 'I_UNIT', ampl)
        ampl = re.sub(r'\be\b(?=\^|[+\-*/()| ])', 'E_CHARGE', ampl)
        ampl = ampl.replace('reg_prop', 'REG_PROP')
        ampl = self.pattern_mandelstam.sub(r'MANDELSTAM_\1', ampl)
        ampl = self.pattern_momentum.sub(r'P_\1', ampl)
        ampl = self.pattern_single_s.sub(r'S_\1', ampl)
        ampl = ampl.replace('(*)', 'CONJ')
        return ampl

    def replace_indices(self, ampl: str, is_source: bool = True) -> str:
        if not self.to_replace:
            return ampl
        index_pool = iter(self.index_pool)
        particle_index_pool = iter(self.particle_index_pool)
        index_pool_set = set(self.index_pool) if is_source else set()

        ampl = self.pattern_mandelstam.sub(lambda m: f'MANDELSTAM_{m.group(1)}', ampl)

        def get_unique_matches(pattern):
            matches = list(OrderedDict.fromkeys(pattern.findall(ampl)))
            return [m for m in matches if m not in index_pool_set]

        def replace_particle_tokens():
            nonlocal ampl
            matches = list(OrderedDict.fromkeys(
                m.group('target') for m in sorted(self.pattern_particle.finditer(ampl), key=lambda m: m.start())
            ))
            try:
                mapping = {m: next(particle_index_pool) for m in matches}
            except StopIteration:
                raise RuntimeError("particle_index_pool exhausted. Increase the size of the particle_index_pool.")
            for key in sorted(mapping.keys(), key=len, reverse=True):
                ampl = ampl.replace(key, mapping[key])

        matches = get_unique_matches(self.pattern_num_123)
        try:
            for match in matches:
                ampl = ampl.replace(match, next(index_pool))
        except StopIteration:
            raise RuntimeError("index_pool exhausted. Increase pool size.")
        replace_particle_tokens()
        return ampl

    def tokenize_expression(self, ampl: str, protected: List[str], is_source: bool = True) -> List[str]:
        ampl = ampl.replace('\\\\', '\\')
        def replace_special(match):
            if match.group(1):  # Spinor suffix (_u or _v)
                return f' _ {match.group(1)} '
            elif match.group(2):  # LaTeX index (\INDEX_0 or \+)
                return f' \\ {match.group(2)} '
        ampl = self.pattern_special.sub(replace_special, ampl)
        if is_source:
            ampl = self.pattern_underscore_curly.sub(lambda match: f' {match.group(0)} ', ampl)
            for symbol in ['{', '}', ',']:
                ampl = ampl.replace(symbol, f' {symbol} ')
        for symbol in ['/', '+', '-', '*', '(', ')', '^']:
            ampl = ampl.replace(symbol, f' {symbol} ')
        ampl = self.pattern_exponent.sub(r' ^ \1 ', ampl)
        ampl = ampl.replace('_PINDEX', '_ PINDEX').replace('_INDEX', '_ INDEX')
        ampl = ampl.replace('REG_PROP', ' reg_prop ')
        ampl = re.sub(r' +', ' ', ampl).strip()
        tokens = [token for token in ampl.split(' ') if token]
        final_tokens = []
        for token in tokens:
            if token.startswith('PROTECTED_'):
                try:
                    idx = int(token.split('_')[1])
                    final_tokens.append(protected[idx])
                except (IndexError, ValueError):
                    final_tokens.append(token)
            else:
                final_tokens.append(token)
        return final_tokens

    def src_tokenize(self, ampl: str) -> List[str]:
        try:
            ampl = self.preprocess_expression(ampl)
            ampl, protected = self.protect_structures(ampl)
            ampl = self.physics_aware_replace(ampl, is_source=True)
            ampl = self.replace_indices(ampl, is_source=True)
            return self.tokenize_expression(ampl, protected, is_source=True)
        except Exception as e:
            warnings.warn(f"Source tokenization failed for '{ampl}': {e}")
            return [self.special_symbols[self.unk_idx]]

    def tgt_tokenize(self, sqampl: str) -> List[str]:
        try:
            sqampl = self.preprocess_expression(sqampl)
            sqampl, protected = self.protect_structures(sqampl)
            sqampl = self.physics_aware_replace(sqampl, is_source=False)
            sqampl = self.replace_indices(sqampl, is_source=False)
            return self.tokenize_expression(sqampl, protected, is_source=False)
        except Exception as e:
            warnings.warn(f"Target tokenization failed for '{sqampl}': {e}")
            return [self.special_symbols[self.unk_idx]]

    def build_src_vocab(self) -> set:
        if self.amps is None:
            return set()
        vocab_set = set()
        start_time = time.time()
        for expr in tqdm(self.amps, desc="Processing source vocab"):
            vocab_set.update(self.src_tokenize(expr))
        end_time = time.time()
        print(f"Source vocab built in {end_time - start_time:.2f} seconds, size: {len(vocab_set)}")
        return vocab_set

    def build_tgt_vocab(self) -> set:
        if self.sqamps is None:
            return set()
        vocab_set = set()
        start_time = time.time()
        for expr in tqdm(self.sqamps, desc="Processing target vocab"):
            vocab_set.update(self.tgt_tokenize(expr))
        end_time = time.time()
        print(f"Target vocab built in {end_time - start_time:.2f} seconds, size: {len(vocab_set)}")
        return vocab_set

class SymbolicVocab:
    def __init__(self, tokens: set, special_symbols: list, bos_idx: int, pad_idx: int, eos_idx: int, unk_idx: int, sep_idx: int):
        self.token_list = special_symbols + sorted(list(tokens))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.token_list)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.unk_idx = unk_idx
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.sep_idx = sep_idx
        self.unk_tok = special_symbols[unk_idx]
        self.pad_tok = special_symbols[pad_idx]
        self.bos_tok = special_symbols[bos_idx]
        self.eos_tok = special_symbols[eos_idx]
        self.sep_tok = special_symbols[sep_idx]

    def encode(self, tokens: list) -> list:
        return [self.token_to_idx.get(token, self.unk_idx) for token in tokens]

    def decode(self, indices: list, include_special_tokens: bool = True) -> list:
        if include_special_tokens:
            return [self.idx_to_token.get(idx, self.unk_tok) for idx in indices]
        return [self.idx_to_token.get(idx, self.unk_tok) for idx in indices 
                if idx not in {self.pad_idx, self.bos_idx, self.eos_idx, self.sep_idx}]

    def __len__(self) -> int:
        return len(self.token_list)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.idx_to_token.get(item, self.unk_tok)
        return self.token_to_idx.get(item, self.unk_idx)

# QED Dataset (unchanged)
class QEDDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        start_time = time.time()
        self.src_vocab = SymbolicVocab(
            tokens=tokenizer.build_src_vocab(),
            special_symbols=tokenizer.special_symbols,
            bos_idx=2,
            pad_idx=0,
            eos_idx=3,
            unk_idx=1,
            sep_idx=4
        )
        self.tgt_vocab = SymbolicVocab(
            tokens=tokenizer.build_tgt_vocab(),
            special_symbols=tokenizer.special_symbols,
            bos_idx=2,
            pad_idx=0,
            eos_idx=3,
            unk_idx=1,
            sep_idx=4
        )
        end_time = time.time()
        print(f"Dataset initialized in {end_time - start_time:.2f} seconds, src_vocab_size: {len(self.src_vocab)}, tgt_vocab_size: {len(self.tgt_vocab)}")
        if len(self.src_vocab) == 5 or len(self.tgt_vocab) == 5:  # Only special tokens
            warnings.warn("Vocabulary size is minimal (only special tokens). Check dataset or tokenization.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = str(self.data.iloc[idx]["amp"])
        trg = str(self.data.iloc[idx]["sqamp"])
        src_tokens = self.tokenizer.src_tokenize(src)
        trg_tokens = self.tokenizer.tgt_tokenize(trg)
        src_ids = self.src_vocab.encode(src_tokens)
        trg_ids = self.tgt_vocab.encode(trg_tokens)
        src_ids = src_ids[:self.max_length] + [self.src_vocab.pad_idx] * (self.max_length - len(src_ids))
        trg_ids = trg_ids[:self.max_length] + [self.tgt_vocab.pad_idx] * (self.max_length - len(trg_ids))
        return {
            "input_ids": torch.tensor(src_ids, dtype=torch.long),
            "labels": torch.tensor(trg_ids, dtype=torch.long)
        }

# Role-Filler Embedding (unchanged)
class RoleFillerEmbedding(nn.Module):
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
        self.linear = nn.Linear(d_x, d_x)
        nn.init.normal_(self.linear.weight, mean=0, std=1./math.sqrt(d_x))
        nn.init.zeros_(self.linear.bias)

    def forward(self, src):
        if src.max().item() >= self.tok_embedding.num_embeddings:
            raise ValueError(f"Input indices {src.max().item()} exceed vocab size {self.tok_embedding.num_embeddings}")
        tok_emb = self.tok_embedding(src) * self.scale.to(src.device)
        seq_length = src.size(1)
        pos_emb = self.pe[:, :seq_length, :]  # Slice to match input length
        x = tok_emb + pos_emb
        r = self.linear(x) + 1
        z = x * r
        return self.dropout(z)

# Multi-Head Attention (unchanged)
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

# Encoder Layer (unchanged)
class EncoderLayer(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.attn = MultiHeadAttention(p.d_x, p.n_heads, p.dropout)
        self.kan = KANLinear(p.d_x, p.d_x, grid_size=5, spline_order=3)
        self.norm1 = nn.LayerNorm(p.d_x)
        self.norm2 = nn.LayerNorm(p.d_x)
        self.dropout = nn.Dropout(p.dropout)

    def forward(self, src, src_mask):
        z = self.norm1(src)
        z = self.attn(z, z, z, src_mask)
        src = src + self.dropout(z)
        z = self.norm2(src)
        z = self.kan(z)
        src = src + self.dropout(z)
        return src

# Encoder (unchanged)
class Encoder(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(p) for _ in range(p.n_layers)])

    def forward(self, src, src_mask):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

# Decoder Layer (unchanged)
class DecoderLayer(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.self_attn = MultiHeadAttention(p.d_x, p.n_heads, p.dropout)
        self.enc_attn = MultiHeadAttention(p.d_x, p.n_heads, p.dropout)
        self.kan = KANLinear(p.d_x, p.d_x, grid_size=5, spline_order=3)
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
        z = self.kan(z)
        trg = trg + self.dropout(z)
        return trg

# Decoder (unchanged)
class Decoder(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(p) for _ in range(p.n_layers)])

    def forward(self, trg, enc_src, trg_mask, src_mask):
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        return trg

# Transformer (unchanged)
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
        self.embedding = RoleFillerEmbedding(d_vocab, d_x, dropout, max_length)
        self.encoder = Encoder(self.p)
        self.decoder = Decoder(self.p)
        self.out = KANLinear(d_x, d_vocab, grid_size=5, spline_order=3)
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

# Training and Evaluation (unchanged)
def train_and_evaluate(model, train_loader, val_loader, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_idx)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        start_time = time.time()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
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
        end_time = time.time()
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)"):
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
        print(f"Epoch {epoch+1}/{epochs} completed in {end_time - start_time:.2f} seconds: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")

# Main (unchanged)
def main():
    # Hyperparameters
    d_x = 512
    n_layers = 6
    n_heads = 8
    dropout = 0.1
    max_length = 300  # Increased to handle longer sequences like the test case
    pad_idx = 0
    sos_idx = 2
    eos_idx = 3
    epochs = 10
    lr = 1e-4
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sample QED data
    start_time = time.time()
    data_df = pd.read_csv(r'D:\DecoderKAN\QED_data\test-flow.csv')
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")

    # Tokenizer
    start_time = time.time()
    tokenizer = SymbolicQEDTokenizer(df=data_df, index_token_pool_size=100, special_symbols=["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"], to_replace=True)
    d_vocab = max(len(tokenizer.build_src_vocab()) + 5, len(tokenizer.build_tgt_vocab()) + 5)  # +5 for special tokens
    print(f"Tokenizer initialized in {time.time() - start_time:.2f} seconds")

    # Dataset
    start_time = time.time()
    dataset = QEDDataset(data_df, tokenizer, max_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = Subset(dataset, range(train_size)), Subset(dataset, range(train_size, len(dataset)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    print(f"Data loaders prepared in {time.time() - start_time:.2f} seconds")

    # Model
    start_time = time.time()
    model = Transformer(d_vocab, d_x, n_layers, n_heads, dropout, max_length, pad_idx)
    model.to(device)
    print(f"Model initialized in {time.time() - start_time:.2f} seconds")

    # Train
    start_time = time.time()
    train_and_evaluate(model, train_loader, val_loader, epochs, lr, device)
    print(f"Training completed in {time.time() - start_time:.2f} seconds")

    # Test Inference
    model.eval()
    start_time = time.time()
    test_expr = r"1/9*i*e^2*(p_2_\INDEX_0*gamma_{+\INDEX_0,INDEX_1,INDEX_2}*gamma_{\INDEX_3,INDEX_4,INDEX_1}*gamma_{\INDEX_5,INDEX_2,INDEX_6}*A_{MOMENTUM_0,+\INDEX_5}(p_3)^(*)*A_{MOMENTUM_1,+\INDEX_3}(p_4)^(*)*b_{MOMENTUM_2,INDEX_6}(p_2)_u*b_{MOMENTUM_3,INDEX_4}(p_1)_v^(*)+-p_3_\INDEX_0*gamma_{+\INDEX_0,INDEX_7,INDEX_8}*gamma_{\INDEX_3,INDEX_9,INDEX_7}*gamma_{\INDEX_5,INDEX_8,INDEX_10}*A_{MOMENTUM_0,+\INDEX_5}(p_3)^(*)*A_{MOMENTUM_1,+\INDEX_3}(p_4)^(*)*b_{MOMENTUM_2,INDEX_10}(p_2)_u*b_{MOMENTUM_3,INDEX_9}(p_1)_v^(*)+m_b*gamma_{\INDEX_3,INDEX_11,INDEX_12}*gamma_{\INDEX_5,INDEX_12,INDEX_13}*A_{MOMENTUM_0,+\INDEX_5}(p_3)^(*)*A_{MOMENTUM_1,+\INDEX_3}(p_4)^(*)*b_{MOMENTUM_2,INDEX_13}(p_2)_u*b_{MOMENTUM_3,INDEX_11}(p_1)_v^(*))/(m_b^2+-s_22+2*s_23+-s_33+-reg_prop)"
    src_tokens = tokenizer.src_tokenize(test_expr)
    src_ids = torch.tensor([dataset.src_vocab.encode(src_tokens)], device=device)
    output = model.greedy_inference(src_ids, sos_idx, eos_idx, max_length)
    decoded = dataset.tgt_vocab.decode(output[0].tolist())
    print(f"Inference completed in {time.time() - start_time:.2f} seconds")
    print(f"Input: {test_expr}")
    print(f"Output IDs: {output.tolist()}")
    print(f"Output: {''.join(decoded)}")

if __name__ == "__main__":
    main()