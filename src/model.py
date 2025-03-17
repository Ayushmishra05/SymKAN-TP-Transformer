import torch
import torch.nn as nn
import math
from einops import repeat

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model 
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) 
    
    def forward(self, x):
        print(f"Input shape: {x.shape}")
        return self.embedding(x) * math.sqrt(self.d_model) 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=44, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].requires_grad_(False)
        return self.dropout(x)

class LayerNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias 

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = nn.Dropout(p=dropout)
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q_seq_len = q.size(1)
        kv_seq_len = k.size(1)
        
        q = self.q_linear(q).view(bs, q_seq_len, self.num_heads, self.d_k)
        k = self.k_linear(k).view(bs, kv_seq_len, self.num_heads, self.d_k)
        v = self.v_linear(v).view(bs, kv_seq_len, self.num_heads, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = torch.nn.functional.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(bs, q_seq_len, self.d_model)
        output = self.out(output)
        
        return output

class Residual(nn.Module):
    def __init__(self, sublayer, d_model, dropout=0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = LayerNorm()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, *args):
        sublayer_output = self.sublayer(self.norm(x), *args)
        assert x.shape == sublayer_output.shape, f"Shape mismatch: {x.shape} vs {sublayer_output.shape}"
        return x + self.dropout(sublayer_output)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.residual1 = Residual(self.mha, d_model, dropout)
        self.residual2 = Residual(self.ff, d_model, dropout)

    def forward(self, x, mask=None):
        x = self.residual1(x, x, x, mask)
        x = self.residual2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout=0.1):  # Fixed parameter order
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) 
                                   for _ in range(num_layers)])
        self.norm = LayerNorm()

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.residual1 = Residual(self.self_mha, d_model, dropout)
        self.residual2 = Residual(self.cross_mha, d_model, dropout)
        self.residual3 = Residual(self.ff, d_model, dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask, cross_mask):
        x = self.residual1(x, x, x, tgt_mask)
        x = self.residual2(x, encoder_output, encoder_output, cross_mask)
        x = self.residual3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, tgt_mask, cross_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask, cross_mask)
        return self.norm(x)

class Projection(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(d_model, num_heads, num_layers, d_ff, dropout)
        self.decoder = Decoder(d_model, num_heads, num_layers, d_ff, dropout)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask, cross_mask=None):
        src_embed = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src_embed = self.pos_encoding(src_embed)
        encoder_output = self.encoder(src_embed, src_mask)
        
        tgt_embed = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        tgt_embed = self.pos_encoding(tgt_embed)
        decoder_output = self.decoder(tgt_embed, encoder_output, src_mask, tgt_mask, cross_mask)
        
        return self.output_layer(decoder_output)

def build_transformer(vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
    return Transformer(vocab_size, d_model, num_heads, num_layers, d_ff, dropout)