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

# SineKANLayer
def forward_step(i_n, grid_size, A, K, C):
    ratio = A * grid_size**(-K) + C
    i_n1 = ratio * i_n
    return i_n1

class SineKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, device='cpu', grid_size=5, is_first=False, add_bias=True, norm_freq=True):
        super(SineKANLayer, self).__init__()
        self.grid_size = grid_size
        self.device = device
        self.is_first = is_first
        self.add_bias = add_bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A, self.K, self.C = 0.9724108095811765, 0.9884401790754128, 0.999449553483052
        
        self.grid_norm_factor = (torch.arange(grid_size) + 1)
        self.grid_norm_factor = self.grid_norm_factor.reshape(1, 1, grid_size)
            
        if is_first:
            self.amplitudes = nn.Parameter(torch.empty(output_dim, input_dim, 1).normal_(0, .4) / output_dim / self.grid_norm_factor)
        else:
            self.amplitudes = nn.Parameter(torch.empty(output_dim, input_dim, 1).uniform_(-1, 1) / output_dim / self.grid_norm_factor)

        grid_phase = torch.arange(1, grid_size + 1).reshape(1, 1, 1, grid_size) / (grid_size + 1)
        self.input_phase = torch.linspace(0, math.pi, input_dim).reshape(1, 1, input_dim, 1).to(device)
        phase = grid_phase.to(device) + self.input_phase

        if norm_freq:
            self.freq = nn.Parameter(torch.arange(1, grid_size + 1).float().reshape(1, 1, 1, grid_size) / (grid_size + 1)**(1 - is_first))
        else:
            self.freq = nn.Parameter(torch.arange(1, grid_size + 1).float().reshape(1, 1, 1, grid_size))

        for i in range(1, self.grid_size):
            phase = forward_step(phase, i, self.A, self.K, self.C)
        self.register_buffer('phase', phase)
        
        if self.add_bias:
            self.bias = nn.Parameter(torch.ones(1, output_dim) / output_dim)

    def forward(self, x):
        x_shape = x.shape
        output_shape = x_shape[0:-1] + (self.output_dim,)
        x = torch.reshape(x, (-1, self.input_dim))
        x_reshaped = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        s = torch.sin(x_reshaped * self.freq + self.phase)
        y = torch.einsum('ijkl,jkl->ij', s, self.amplitudes)
        if self.add_bias:
            y += self.bias
        y = torch.reshape(y, output_shape)
        return y

# Hyperparameters
class HyperParams:
    def __init__(self):
        self.d_x = 512
        self.n_layers = 6
        self.n_heads = 8
        self.d_k = self.d_x // self.n_heads
        self.d_v = self.d_x // self.n_heads
        self.dropout = 0.1
        self.attention_dropout = 0.1  # Added to fix the error
        self.max_length = 300
        self.enc_emb_dim = self.d_x
        self.dec_emb_dim = self.d_x
        self.n_enc_layers = self.n_layers
        self.n_dec_layers = self.n_layers
        self.n_enc_heads = self.n_heads
        self.n_dec_heads = self.n_heads
        self.n_enc_hidden_layers = 1
        self.n_dec_hidden_layers = 1
        self.enc_has_pos_emb = True
        self.dec_has_pos_emb = True
        self.sinusoidal_embeddings = True
        self.norm_attention = False
        self.xav_init = False
        self.gelu_activation = False
        self.enc_gated = False
        self.dec_gated = False
        self.gated = False
        self.scalar_gate = False
        self.biased_gates = False
        self.gate_bias = 0.0
        self.enc_act = False
        self.dec_act = False
        self.enc_loop_idx = -1
        self.dec_loop_idx = -1
        self.enc_loops = 1
        self.dec_loops = 1
        self.act_threshold = 0.0
        self.act_biased = False
        self.act_bias = 0.0
        self.act_ponder_coupling = 0.0
        self.max_src_len = 0
        self.architecture = "encoder-decoder"
        self.share_inout_emb = False
        self.fp16 = False
        self.eos_index = 3
        self.pad_index = 0
        self.sep_index = 4

# SymbolicQEDTokenizer
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
        expr = expr.replace('me', 'm_e')
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
            matches = list(OrderedDict.fromkeys(m.group('target') for m in sorted(self.pattern_particle.finditer(ampl), key=lambda m: m.start())))
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
            if match.group(1):
                return f' _ {match.group(1)} '
            elif match.group(2):
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

# QED Dataset
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
        if len(self.src_vocab) == 5 or len(self.tgt_vocab) == 5:
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
        trg_ids = trg_ids[:self.max_length] + [self.src_vocab.pad_idx] * (self.max_length - len(trg_ids))
        return {
            "input_ids": torch.tensor(src_ids, dtype=torch.long),
            "labels": torch.tensor(trg_ids, dtype=torch.long)
        }

# Role-Filler Embedding
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

# MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, src_dim, dropout, normalized_attention, xav_init=False):
        super().__init__()
        self.dim = dim
        self.src_dim = src_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.normalized_attention = normalized_attention
        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(src_dim, dim)
        self.v_lin = nn.Linear(src_dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        if self.normalized_attention:
            self.attention_scale = nn.Parameter(
                torch.tensor(1.0 / math.sqrt(dim // n_heads))
            )
        if xav_init:
            gain = (1 / math.sqrt(2)) if self.src_dim == self.dim else 1.0
            nn.init.xavier_uniform_(self.q_lin.weight, gain=gain)
            nn.init.xavier_uniform_(self.k_lin.weight, gain=gain)
            nn.init.xavier_uniform_(self.v_lin.weight, gain=gain)
            nn.init.xavier_uniform_(self.out_lin.weight)
            nn.init.constant_(self.out_lin.bias, 0.0)

    def forward(self, input, mask, kv=None, use_cache=False, first_loop=True):
        assert not (use_cache and self.cache is None)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if not use_cache else self.cache["slen"] + qlen
        else:
            klen = kv.size(1)
        assert dim == self.dim
        n_heads = self.n_heads
        dim_per_head = dim // n_heads

        # Adjust mask reshape based on its dimensionality
        if mask.dim() == 4:  # Causal mask from make_masks [bs, 1, qlen, qlen]
            mask_reshape = mask.size()  # [bs, 1, qlen, qlen]
        elif mask.dim() == 3:  # Non-causal mask [bs, qlen, qlen]
            mask_reshape = (bs, 1, qlen, klen)  # Expand for heads
        else:
            raise ValueError(f"Unexpected mask dimension: {mask.dim()}")

        def shape(x):
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(input))
        if kv is None:
            k = shape(self.k_lin(input))
            v = shape(self.v_lin(input))
        elif not use_cache or self.layer_id not in self.cache:
            k = v = kv
            k = shape(self.k_lin(k))
            v = shape(self.v_lin(v))

        if use_cache:
            if self.layer_id in self.cache:
                if kv is None and first_loop:
                    k_, v_ = self.cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)
                    v = torch.cat([v_, v], dim=2)
                else:
                    k, v = self.cache[self.layer_id]
            self.cache[self.layer_id] = (k, v)
        if self.normalized_attention:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            q = q * self.attention_scale
        else:
            q = q / math.sqrt(dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))
        mask = (mask == 0).view(mask_reshape).expand(-1, n_heads, -1, -1)  # Expand for n_heads
        scores.masked_fill_(mask, -float("inf"))
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        context = torch.matmul(weights, v)
        context = unshape(context)
        return self.out_lin(context)
# TransformerFFN
class TransformerFFN(nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, hidden_layers, dropout, gelu_activation=False, xav_init=False):
        super().__init__()
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.act = F.gelu if gelu_activation else F.relu
        self.midlin = nn.ModuleList()
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        for i in range(1, self.hidden_layers):
            self.midlin.append(nn.Linear(dim_hidden, dim_hidden))
        self.lin2 = nn.Linear(dim_hidden, out_dim)
        if xav_init:
            nn.init.xavier_uniform_(self.lin1.weight)
            nn.init.constant_(self.lin1.bias, 0.0)
            for mlin in self.midlin:
                nn.init.xavier_uniform_(mlin.weight)
                nn.init.constant_(mlin.bias, 0.0)
            nn.init.xavier_uniform_(self.lin2.weight)
            nn.init.constant_(self.lin2.bias, 0.0)

    def forward(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for mlin in self.midlin:
            x = mlin(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x

# TransformerLayer
class TransformerLayer(nn.Module):
    def __init__(self, params, is_encoder, gated=False, is_last=False):
        super().__init__()
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.is_last = is_last
        self.dim = params.enc_emb_dim if is_encoder else params.dec_emb_dim
        self.src_dim = params.enc_emb_dim
        self.hidden_dim = self.dim * 4
        self.n_hidden_layers = params.n_enc_hidden_layers if is_encoder else params.n_dec_hidden_layers
        self.n_heads = params.n_enc_heads if is_encoder else params.n_dec_heads
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        self.gated = gated
        assert self.dim % self.n_heads == 0
        self.self_attention = MultiHeadAttention(
            self.n_heads, self.dim, self.dim, dropout=self.attention_dropout, normalized_attention=params.norm_attention
        )
        self.layer_norm1 = nn.LayerNorm(self.dim, eps=1e-12)
        if self.is_decoder:
            self.layer_norm15 = nn.LayerNorm(self.dim, eps=1e-12)
            self.cross_attention = MultiHeadAttention(
                self.n_heads, self.dim, self.src_dim, dropout=self.attention_dropout, normalized_attention=params.norm_attention
            )
        if self.is_last and self.is_decoder:
            self.ffn = SineKANLayer(self.dim, self.dim)
        else:
            self.ffn = TransformerFFN(self.dim, self.hidden_dim, self.dim, self.n_hidden_layers, dropout=self.dropout)
        self.layer_norm2 = nn.LayerNorm(self.dim, eps=1e-12)

    def forward(self, x, attn_mask, src_mask, src_enc, use_cache=False, cache=None, loop_count=1):
        tensor = x
        for i in range(loop_count):
            self.self_attention.cache = cache
            attn = self.self_attention(tensor, attn_mask, use_cache=use_cache, first_loop=(i==0))
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            output = tensor + attn
            output = self.layer_norm1(output)
            if self.is_decoder and src_enc is not None:
                self.cross_attention.cache = cache
                attn = self.cross_attention(output, src_mask, kv=src_enc, use_cache=use_cache, first_loop=(i==0))
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                output = output + attn
                output = self.layer_norm15(output)
            output = output + self.ffn(output)
            output = self.layer_norm2(output)
            tensor = output
        return tensor

# TransformerModel
# TransformerModel
class TransformerModel(nn.Module):
    def __init__(self, params, id2word, d_vocab, max_length, pad_idx):
        super().__init__()
        self.params = params
        self.d_vocab = d_vocab
        self.max_length = max_length
        self.pad_idx = pad_idx
        self.embedding = RoleFillerEmbedding(d_vocab, params.enc_emb_dim, params.dropout, max_length)
        self.layers = nn.ModuleList()
        for layer_id in range(params.n_enc_layers):
            self.layers.append(TransformerLayer(params, is_encoder=True))
        for layer_id in range(params.n_dec_layers):
            is_last = (layer_id == params.n_dec_layers - 1)
            self.layers.append(TransformerLayer(params, is_encoder=False, is_last=is_last))
        self.proj = nn.Linear(params.dec_emb_dim, d_vocab, bias=True)
        self.cache = None
        self.to(params.device)

    def make_masks(self, src, trg):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len, device=trg.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return src_mask, trg_mask

    def forward(self, src, trg):
        src_mask, trg_mask = self.make_masks(src, trg)
        enc_out = self.embedding(src)  # Embed src
        dec_in = self.embedding(trg)   # Embed trg
        for i, layer in enumerate(self.layers):
            if i < self.params.n_enc_layers:
                enc_out = layer(enc_out, src_mask, None, None)
            else:
                dec_out = layer(dec_in, trg_mask, src_mask, enc_out)
                dec_in = dec_out  # Update dec_in for subsequent layers
        logits = self.proj(dec_out)
        return logits

    def greedy_inference(self, src, sos_idx, eos_idx, max_length):
        self.eval()
        src = src.to(self.params.device)
        if src.size(1) > max_length:
            src = src[:, :max_length]
        batch_size = src.size(0)
        src_mask = self.make_masks(src, src)[0]
        enc_out = self.embedding(src)  # Embed src for encoder
        for i, layer in enumerate(self.layers):
            if i < self.params.n_enc_layers:
                enc_out = layer(enc_out, src_mask, None, None)
        trg = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=self.params.device)
        for i in range(max_length):
            trg_mask = self.make_masks(trg, trg)[1]
            dec_in = self.embedding(trg)  # Embed trg for each step
            for j, layer in enumerate(self.layers[self.params.n_enc_layers:], start=self.params.n_enc_layers):
                dec_in = layer(dec_in, trg_mask, src_mask, enc_out)
            logits = self.proj(dec_in[:, -1])
            pred = torch.argmax(logits, dim=-1).unsqueeze(1)
            trg = torch.cat([trg, pred], dim=1)
            if torch.all(pred == eos_idx):
                break
        return trg

# Training and Evaluation
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
            if torch.isnan(loss).any():
                print(f"NaN loss detected for batch. Skipping. Output stats: min={output.min():.4f}, max={output.max():.4f}")
                continue
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
                if torch.isnan(loss).any():
                    print(f"NaN loss detected for batch. Skipping. Output stats: min={output.min():.4f}, max={output.max():.4f}")
                    continue
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
# Main
def main():
    d_x = 512
    n_layers = 6
    n_heads = 8
    dropout = 0.1
    max_length = 300
    pad_idx = 0
    sos_idx = 2
    eos_idx = 3
    epochs = 1
    lr = 1e-4
    batch_size = 16
    device = torch.device("cpu")

    start_time = time.time()
    data_df = pd.read_csv(r'D:\DecoderKAN\QED_data\test-flow.csv')
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    tokenizer = SymbolicQEDTokenizer(df=data_df, index_token_pool_size=100, special_symbols=["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"], to_replace=True)
    d_vocab = max(len(tokenizer.build_src_vocab()) + 5, len(tokenizer.build_tgt_vocab()) + 5)
    print(f"Tokenizer initialized in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    dataset = QEDDataset(data_df, tokenizer, max_length)
    train_size = 50
    val_size = 25
    train_dataset, val_dataset = Subset(dataset, range(train_size)), Subset(dataset, range(train_size, train_size + val_size))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    print(f"Data loaders prepared in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    params = HyperParams()
    params.device = device
    params.n_words = d_vocab
    model = TransformerModel(params, dataset.tgt_vocab.idx_to_token, d_vocab, max_length, pad_idx)
    model.to(device)
    print(f"Model initialized in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    train_and_evaluate(model, train_loader, val_loader, epochs, lr, device)
    print(f"Training completed in {time.time() - start_time:.2f} seconds")

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