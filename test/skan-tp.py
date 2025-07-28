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
from typing import List, Union, OrderedDict, Tuple
import time 

# S-KANformer Components
def forward_step(i_n, grid_size, A, K, C):
    ratio = A * grid_size**(-K) + C
    i_n1 = ratio * i_n
    return i_n1

class SineKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, device='cuda', grid_size=8, is_first=False, add_bias=True, norm_freq=True):
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
            self.amplitudes = torch.nn.Parameter(torch.empty(output_dim, input_dim, 1).normal_(0, .4) / output_dim / self.grid_norm_factor)
        else:
            self.amplitudes = torch.nn.Parameter(torch.empty(output_dim, input_dim, 1).uniform_(-1, 1) / output_dim / self.grid_norm_factor)

        grid_phase = torch.arange(1, grid_size + 1).reshape(1, 1, 1, grid_size) / (grid_size + 1)
        self.input_phase = torch.linspace(0, math.pi, input_dim).reshape(1, 1, input_dim, 1).to(device)
        phase = grid_phase.to(device) + self.input_phase

        if norm_freq:
            self.freq = torch.nn.Parameter(torch.arange(1, grid_size + 1).float().reshape(1, 1, 1, grid_size) / (grid_size + 1)**(1 - is_first))
        else:
            self.freq = torch.nn.Parameter(torch.arange(1, grid_size + 1).float().reshape(1, 1, 1, grid_size))

        for i in range(1, self.grid_size):
            phase = forward_step(phase, i, self.A, self.K, self.C)
        self.register_buffer('phase', phase)
        
        if self.add_bias:
            self.bias = torch.nn.Parameter(torch.ones(1, output_dim) / output_dim)

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

class KANFeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, grid_size: int = 8, device: Union[str, int] = 'cuda') -> None:
        super().__init__()
        self.kan = SineKANLayer(d_model, d_model, device=device, grid_size=grid_size, is_first=False, add_bias=True)

    def forward(self, x):
        return self.kan(x)

class KANProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, grid_size: int = 8, device: Union[str, int] = 'cuda') -> None:
        super().__init__()
        self.kan = SineKANLayer(d_model, vocab_size, device=device, grid_size=grid_size, is_first=False, add_bias=True)

    def forward(self, x):
        return self.kan(x)

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)

class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, ff_block: KANFeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.ff_block = ff_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.ff_block)
        return x

class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: KANFeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.ff_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.ff_block)
        return x

class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: KANProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)

def build_kanformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, 
                    N: int=3, h: int=8, dropout: float=0.1, d_ff: int=4096, ff_dims: List[int]=[8192], device: Union[str, int] = 'cuda') -> Transformer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        ff_block = KANFeedForwardBlock(d_model, grid_size=8, device=device)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, ff_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        ff_block = KANFeedForwardBlock(d_model, grid_size=8, device=device)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, ff_block, dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    projection_layer = KANProjectionLayer(d_model, tgt_vocab_size, grid_size=8, device=device)
    
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    for _, p in transformer.named_parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

# Your Tokenization and Dataset
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
        trg_ids = trg_ids[:self.max_length] + [self.tgt_vocab.pad_idx] * (self.max_length - len(trg_ids))
        return {
            "input_ids": torch.tensor(src_ids, dtype=torch.long),
            "labels": torch.tensor(trg_ids, dtype=torch.long)
        }

def main():
    # Hyperparameters
    d_model = 512
    N = 3
    h = 8
    dropout = 0.1
    d_ff = 4096
    ff_dims = [8192]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_length = 300  # Match your dataset max_length
    epochs = 10
    lr = 1e-4
    batch_size = 16

    # Load Data
    start_time = time.time()
    data_df = pd.read_csv(r'D:\DecoderKAN\QED_data\test-flow.csv')
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")

    # Tokenizer
    start_time = time.time()
    tokenizer = SymbolicQEDTokenizer(df=data_df, index_token_pool_size=100, special_symbols=["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"], to_replace=True)
    src_vocab_size = len(tokenizer.build_src_vocab()) + 5  # +5 for special tokens
    tgt_vocab_size = len(tokenizer.build_tgt_vocab()) + 5  # +5 for special tokens
    print(f"Tokenizer initialized in {time.time() - start_time:.2f} seconds")

    # Dataset and DataLoader
    start_time = time.time()
    dataset = QEDDataset(data_df, tokenizer, max_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = Subset(dataset, range(train_size)), Subset(dataset, range(train_size, len(dataset)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    print(f"Data loaders prepared in {time.time() - start_time:.2f} seconds")

    # Build Model
    start_time = time.time()
    model = build_kanformer(src_vocab_size, tgt_vocab_size, max_length, max_length, d_model, N, h, dropout, d_ff, ff_dims, device)
    model.to(device)
    print(f"Model initialized in {time.time() - start_time:.2f} seconds")

    # Training Loop
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # pad_idx = 0
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
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, src_len, 1]
            tgt_len = trg[:, :-1].size(1)  # 299
            tgt_pad_mask = (trg[:, :-1] != 0).unsqueeze(1).unsqueeze(3)  # [batch_size, 1, 299, 1]
            tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()  # [299, 299]
            tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0).unsqueeze(0)  # [batch_size, 1, 299, 299]
            enc_output = model.encode(src, src_mask)
            dec_output = model.decode(enc_output, src_mask, trg[:, :-1], tgt_mask)
            logits = model.project(dec_output)
            output = logits.view(-1, logits.size(-1))
            target = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            preds = torch.argmax(output, dim=-1)
            mask = target != 0
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
                src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
                tgt_len = trg[:, :-1].size(1)
                tgt_pad_mask = (trg[:, :-1] != 0).unsqueeze(1).unsqueeze(3)
                tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()
                tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0).unsqueeze(0)
                enc_output = model.encode(src, src_mask)
                dec_output = model.decode(enc_output, src_mask, trg[:, :-1], tgt_mask)
                logits = model.project(dec_output)
                output = logits.view(-1, logits.size(-1))
                target = trg[:, 1:].contiguous().view(-1)
                loss = criterion(output, target)
                val_loss += loss.item()
                preds = torch.argmax(output, dim=-1)
                mask = target != 0
                val_correct += (preds[mask] == target[mask]).sum().item()
                val_total += mask.sum().item()
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} completed in {end_time - start_time:.2f} seconds: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")
    # Inference
    model.eval()
    start_time = time.time()
    test_expr = r"1/9*i*e^2*(p_2_\INDEX_0*gamma_{+\INDEX_0,INDEX_1,INDEX_2}*gamma_{\INDEX_3,INDEX_4,INDEX_1}*gamma_{\INDEX_5,INDEX_2,INDEX_6}*A_{MOMENTUM_0,+\INDEX_5}(p_3)^(*)*A_{MOMENTUM_1,+\INDEX_3}(p_4)^(*)*b_{MOMENTUM_2,INDEX_6}(p_2)_u*b_{MOMENTUM_3,INDEX_4}(p_1)_v^(*)+-p_3_\INDEX_0*gamma_{+\INDEX_0,INDEX_7,INDEX_8}*gamma_{\INDEX_3,INDEX_9,INDEX_7}*gamma_{\INDEX_5,INDEX_8,INDEX_10}*A_{MOMENTUM_0,+\INDEX_5}(p_3)^(*)*A_{MOMENTUM_1,+\INDEX_3}(p_4)^(*)*b_{MOMENTUM_2,INDEX_10}(p_2)_u*b_{MOMENTUM_3,INDEX_9}(p_1)_v^(*)+m_b*gamma_{\INDEX_3,INDEX_11,INDEX_12}*gamma_{\INDEX_5,INDEX_12,INDEX_13}*A_{MOMENTUM_0,+\INDEX_5}(p_3)^(*)*A_{MOMENTUM_1,+\INDEX_3}(p_4)^(*)*b_{MOMENTUM_2,INDEX_13}(p_2)_u*b_{MOMENTUM_3,INDEX_11}(p_1)_v^(*))/(m_b^2+-s_22+2*s_23+-s_33+-reg_prop)"
    src_tokens = tokenizer.src_tokenize(test_expr)
    src_ids = torch.tensor([dataset.src_vocab.encode(src_tokens)], device=device)
    src_mask = (src_ids != 0).unsqueeze(1).unsqueeze(2)
    enc_output = model.encode(src_ids, src_mask)
    trg = torch.full((1, 1), 2, dtype=torch.long, device=device)  # Start with BOS token
    for _ in range(max_length):
        tgt_mask = (trg != 0).unsqueeze(1).unsqueeze(3) & torch.tril(torch.ones(trg.size(1), trg.size(1), device=device)).bool()
        dec_output = model.decode(enc_output, src_mask, trg, tgt_mask)
        logits = model.project(dec_output[:, -1])
        pred = torch.argmax(logits, dim=-1).unsqueeze(1)
        trg = torch.cat([trg, pred], dim=1)
        if pred.item() == 3:  # EOS token
            break
    decoded = dataset.tgt_vocab.decode(trg[0].tolist())
    print(f"Inference completed in {time.time() - start_time:.2f} seconds")
    print(f"Input: {test_expr}")
    print(f"Output IDs: {trg.tolist()}")
    print(f"Output: {''.join(decoded)}")

if __name__ == "__main__":
    main()