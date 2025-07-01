
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.lib import HyperParams

LOG_TERMINAL = False  # slow down if enabled

def is_nan_string(x):
  if not LOG_TERMINAL:
    return "skip"
  return "NaN" if torch.isnan(x).any() else "ok"

def is_nan(d):
  if LOG_TERMINAL:
    for t in d:
      if torch.isnan(t).any():
        return True
  return False

def debug(*args):
  if LOG_TERMINAL:
    print(*args)

def build_transformer(params, pad_idx):
    p = HyperParams()
    p.d_vocab = params.input_dim
    p.d_pos = 509  # max input size

    p.d_f = params.filter

    p.n_L = params.n_layers
    p.n_I = params.n_heads

    p.d_x = params.hidden  # token embedding dimension
    p.d_p = params.hidden  # position embedding dimension

    p.d_v = p.d_x // p.n_I  # value dimension
    p.d_r = p.d_x // p.n_I  # role dimension

    p.d_k = p.d_x // p.n_I  # key dimension
    p.d_q = p.d_x // p.n_I  # query dimension

    p.dropout = params.dropout

    embedding = EmbeddingMultilinearSinusoidal(d_vocab=params.input_dim,
                                               d_x=p.d_x,
                                               dropout=params.dropout,
                                               max_length=509)
    encoder = Encoder(p=p)
    decoder = Decoder(p=p)
    model = Seq2Seq(p=p,
                    embedding=embedding,
                    encoder=encoder,
                    decoder=decoder,
                    pad_idx=pad_idx)

    return model


class Encoder(nn.Module):
  def __init__(self, p):
    super().__init__()

    layers = [EncoderLayer(p)]
    for _ in range(p.n_L - 1):
      layers.append(EncoderLayer(p))
    self.layers = nn.ModuleList(layers)

  def forward(self, src, src_mask):
    # src = [batch_size, src_seq_len]
    # src_mask = [batch_size, 1, attn_size]
    debug("\nencoder")
    debug("src: ", src.shape, is_nan_string(src))
    debug("src_mask: ", src_mask.shape, is_nan_string(src_mask))

    for layer in self.layers:
      src = layer(src, src_mask)

    return src


class EncoderLayer(nn.Module):
  def __init__(self, p):
    super().__init__()
    # d_in list of input dimension of z e.g. [p.d_x, p.d_p] or [p.d_v, p.d_r]
    d_h = p.d_x

    # sublayer 1
    self.layernorm1 = nn.LayerNorm(d_h)
    self.MHA = SelfAttention(p)
    self.dropout1 = nn.Dropout(p.dropout)
    # sublayer 2
    self.layernorm2 = nn.LayerNorm(d_h)
    # Replace PositionwiseFeedforward with KANLayer
    self.densefilter = KANLayer(in_dim=d_h, out_dim=d_h, num_knots=5, spline_order=3)
    self.dropout2 = nn.Dropout(p.dropout)
    # output
    self.layernorm3 = nn.LayerNorm(d_h)

  def forward(self, src, src_mask):
    # src = [batch_size, src_seq_size, hid_dim]
    # src_mask = [batch_size, src_seq_size]

    # sublayer 1
    z = self.layernorm1(src)
    z = self.MHA(z, z, z, src_mask)
    z = self.dropout1(z)
    src = src + z

    # sublayer 2
    z = self.layernorm2(src)
    z = self.densefilter(z)
    z = self.dropout2(z)
    src = src + z

    return self.layernorm3(src)


class EmbeddingMultilinearSinusoidal(nn.Module):
  def __init__(self, d_vocab, d_x, dropout, max_length):
    super(EmbeddingMultilinearSinusoidal, self).__init__()
    self.dropout = nn.Dropout(dropout)
    self.max_length = max_length
    self.d_x = d_x

    # token encodings
    self.tok_embedding = nn.Embedding(d_vocab, d_x)
    self.scale = torch.sqrt(torch.FloatTensor([d_x]))

    # sinusoidal encoding
    pe = torch.zeros(max_length, d_x)
    position = torch.arange(0., max_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0., d_x, 2) *
                         -(math.log(10000.0) / d_x))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)
    # pe = [1, seq_len, d_p]

    # x -> r
    self.linear = nn.Linear(d_x, d_x)
    self.mul_scale = torch.FloatTensor([1. / math.sqrt(math.sqrt(2) - 1)])

    self.reset_parameters()

  def forward(self, src):
    # src = [batch_size, src_seq_len]
    tok_emb = self.tok_embedding(src) * self.scale.to(src.device)

    # sinusoidal pos embedding
    pos_sin_emb = torch.autograd.Variable(self.pe[:, :src.size(1)],
                                          requires_grad=False)
    x = tok_emb + pos_sin_emb
    # x = [batch_size, src_seq_len, d_x]

    r = self.linear(x) + 1  # such that initially ~N(1,1)
    # r = [batch_size, src_seq_len, d_r]

    z = x * r
    z = self.dropout(z)
    # src = [batch_size, src_seq_len, d_x*d_r]
    return z

  def transpose_forward(self, trg):

    logits = torch.matmul(trg, torch.transpose(self.tok_embedding.weight, 0, 1))
    return logits

  def reset_parameters(self):
    nn.init.normal_(self.tok_embedding.weight,
                    mean=0,
                    std=1./math.sqrt(self.d_x))
    nn.init.normal_(self.linear.weight,
                    mean=0,
                    std=1./math.sqrt(self.d_x))


class SelfAttention(nn.Module):
  def __init__(self, p):
    super().__init__()
    self.p = p
    self.d_h = p.d_x
    self.n_I = p.n_I

    self.W_q = nn.Linear(self.d_h, p.d_q * p.n_I)
    self.W_k = nn.Linear(self.d_h, p.d_k * p.n_I)
    self.W_v = nn.Linear(self.d_h, p.d_v * p.n_I)
    self.W_r = nn.Linear(self.d_h, p.d_r * p.n_I)

    self.W_o = nn.Linear(p.d_v * p.n_I, p.d_x)

    self.dropout = nn.Dropout(p.dropout)
    self.dot_scale = torch.FloatTensor([math.sqrt(p.d_k)])
    self.mul_scale = torch.FloatTensor([1./math.sqrt(math.sqrt(2) - 1)])

  def forward(self, query, key, value, mask=None):
    # query = key = value = [batch_size, seq_len, hid_dim]
    # src_mask = [batch_size, 1, 1, pad_seq]
    # trg_mask = [batch_size, 1, pad_seq, past_seq]

    bsz = query.shape[0]

    Q = self.W_q(query)
    K = self.W_k(key)
    V = self.W_v(value)
    R = self.W_r(query)
    # Q, K, V, R = [batch_size, seq_len, n_heads * d_*]

    Q = Q.view(bsz, -1, self.n_I, self.p.d_q).permute(0,2,1,3)
    K = K.view(bsz, -1, self.n_I, self.p.d_k).permute(0,2,1,3)
    V = V.view(bsz, -1, self.n_I, self.p.d_v).permute(0,2,1,3)
    R = R.view(bsz, -1, self.n_I, self.p.d_r).permute(0,2,1,3)
    # Q, K, V, R = [batch_size, n_heads, seq_size, d_*]

    dot = torch.einsum('bhid,bhjd->bhij', Q, K) / self.dot_scale.to(key.device)
    # energy   = [batch_size, n_heads, query_pos     , key_pos]
    # src_mask = [batch_size, 1      , 1             , attn]
    # trg_mask = [batch_size, 1      , query_specific, attn]

    if mask is not None:
      dot = dot.masked_fill(mask == 0, -1e10)

    attention = self.dropout(F.softmax(dot, dim=-1))
    # attention = [batch_size, n_heads, seq_size, seq_size]

    v_bar = torch.einsum('bhjd,bhij->bhid', V, attention)
    # v_bar = [batch_size, n_heads, seq_size, d_v]

    # bind
    new_v = v_bar * R
    new_v = new_v.permute(0,2,1,3).contiguous()
    # v_bar = [batch_size, seq_size, n_heads, d_v]

    new_v = new_v.view(bsz, -1, self.n_I * self.p.d_v)
    # new_v = [batch_size, src_seq_size, n_heads * d_v]

    x = self.W_o(new_v)
    # x = [batch_size, seq_size, d_x]

    return x

  def reset_parameters(self):
    nn.init.xavier_uniform_(self.W_q.weight)
    nn.init.xavier_uniform_(self.W_k.weight)
    nn.init.xavier_uniform_(self.W_v.weight)
    nn.init.xavier_uniform_(self.W_o.weight)

    nn.init.normal_(self.W_r.weight,
                    mean=0,
                    std=1./math.sqrt(self.p.d_r))


class Decoder(nn.Module):
  def __init__(self, p):
    super().__init__()

    self.layers = nn.ModuleList([DecoderLayer(p) for _ in range(p.n_L)])

  def forward(self, trg, src, trg_mask, src_mask):
    # trg = [batch_size, trg_seq_size, hid_dim]
    # src = [batch_size, src_seq_size, hid_dim]
    # trg_mask = [batch_size, trg_seq_size]
    # src_mask = [batch_size, src_seq_size]

    for layer in self.layers:
      trg = layer(trg, src, trg_mask, src_mask)

    return trg


class DecoderLayer(nn.Module):
  def __init__(self, p):
    super().__init__()
    d_h = p.d_x
    # sublayer 1
    self.layernorm1 = nn.LayerNorm(d_h)
    self.selfAttn = SelfAttention(p)
    self.dropout1 = nn.Dropout(p.dropout)
    # sublayer 2
    self.layernorm2 = nn.LayerNorm(d_h)
    self.encAttn = SelfAttention(p)
    self.dropout2 = nn.Dropout(p.dropout)
    # sublayer 3
    self.layernorm3 = nn.LayerNorm(d_h)
    # Replace PositionwiseFeedforward with KANLayer
    self.densefilter = KANLayer(in_dim=d_h, out_dim=d_h, num_knots=5, spline_order=3)
    self.dropout3 = nn.Dropout(p.dropout)

    # output
    self.layernorm4 = nn.LayerNorm(d_h)

  def forward(self, trg, src, trg_mask, src_mask):
    # trg = [batch_size, trg_seq_size, hid_dim]
    # src = [batch_size, src_seq_size, hid_dim]
    # trg_mask = [batch_size, trg_seq_size]
    # src_mask = [batch_size, src_seq_size]

    # self attention
    z = self.layernorm1(trg)
    z = self.selfAttn(z, z, z, trg_mask)
    z = self.dropout1(z)
    trg = trg + z

    # encoder attention
    z = self.layernorm2(trg)
    z = self.encAttn(z, src, src, src_mask)
    z = self.dropout2(z)
    trg = trg + z

    # dense filter
    z = self.layernorm3(trg)
    z = self.densefilter(z)
    z = self.dropout3(z)
    trg = trg + z

    return self.layernorm4(trg)


class Seq2Seq(nn.Module):
    def __init__(self, p, embedding, encoder, decoder, pad_idx):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.p = p
        self.kan_layer = KANLayer(in_dim=p.d_x, out_dim=p.d_vocab, num_knots=5, spline_order=3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def make_masks(self, src, trg):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(
            torch.ones((trg_len, trg_len), dtype=torch.uint8, device=trg.device))
        trg_mask = trg_pad_mask.type(torch.ByteTensor) & trg_sub_mask.type(torch.ByteTensor)
        return src_mask.to(src.device), trg_mask.to(src.device)

    def forward(self, src, trg):
        src_mask, trg_mask = self.make_masks(src, trg)
        src = self.embedding(src)
        trg = self.embedding(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, trg_mask, src_mask)
        logits = self.kan_layer(out)  
        return logits

    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(src.device)

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(
            torch.ones((trg_len, trg_len), dtype=torch.uint8, device=trg.device))
        trg_mask = trg_pad_mask.type(torch.ByteTensor) & trg_sub_mask.type(torch.ByteTensor)
        return trg_mask.to(trg.device)

    def greedy_inference(self, model, src, sos_idx, eos_idx, max_length):
        model.eval()
        src = src.to(model.device)
        src_mask = model.make_src_mask(src)
        src_emb = model.embedding(src)
        enc_src = model.encoder(src_emb, src_mask)
        trg = torch.ones(src.shape[0], 1).fill_(sos_idx).type_as(src).to(model.device)
        done = torch.zeros(src.shape[0], dtype=torch.uint8).to(model.device)
        for _ in range(max_length):
            trg_emb = model.embedding(trg)
            trg_mask = model.make_trg_mask(trg)
            output = model.decoder(src=enc_src, trg=trg_emb, src_mask=src_mask, trg_mask=trg_mask)
            logits = model.kan_layer(output)  # [batch_size, trg_seq_size, d_vocab]
            pred = torch.argmax(logits[:, [-1], :], dim=-1)
            trg = torch.cat([trg, pred], dim=1)
            eos_match = (pred.squeeze(1) == eos_idx)
            done = done | eos_match
            if done.sum() == src.shape[0]:
                break
        return trg

    def get_interpretability(self, src, trg):
        self.eval()
        with torch.no_grad():
            src_mask, trg_mask = self.make_masks(src, trg)
            src = self.embedding(src)
            trg = self.embedding(trg)
            enc_src = self.encoder(src, src_mask)
            out = self.decoder(trg, enc_src, trg_mask, src_mask)
            interpretability_info = self.kan_layer.get_interpretability(out)
        return interpretability_info


class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_knots=5, spline_order=3):
        super(KANLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_knots = num_knots
        self.spline_order = spline_order

        total_knots = num_knots + spline_order
        self.knots = torch.linspace(-1, 1, total_knots)

        self.inner_coeffs = nn.Parameter(
            torch.randn(in_dim, num_knots) * 0.1
        )

        self.outer_coeffs = nn.Parameter(
            torch.randn(out_dim, in_dim) * 0.1
        )

    def bspline_basis(self, x, knot_idx, order):
        if order == 0:
            if knot_idx + 1 >= len(self.knots):
                return torch.zeros_like(x)
            return ((self.knots[knot_idx] <= x) & (x < self.knots[knot_idx + 1])).float()
        
        left_term = 0
        right_term = 0
        if knot_idx + order < len(self.knots) and self.knots[knot_idx + order] != self.knots[knot_idx]:
            left_term = ((x - self.knots[knot_idx]) / (self.knots[knot_idx + order] - self.knots[knot_idx])) * \
                        self.bspline_basis(x, knot_idx, order - 1)
        if knot_idx + order + 1 < len(self.knots) and self.knots[knot_idx + order + 1] != self.knots[knot_idx + 1]:
            right_term = ((self.knots[knot_idx + order + 1] - x) / (self.knots[knot_idx + order + 1] - self.knots[knot_idx + 1])) * \
                         self.bspline_basis(x, knot_idx + 1, order - 1)
        return left_term + right_term

    def forward(self, x):
        batch_size, seq_len, in_dim = x.shape
        assert in_dim == self.in_dim, f"Input dimension mismatch: expected {self.in_dim}, got {in_dim}"

        x = torch.tanh(x)

        inner_outputs = torch.zeros(batch_size, seq_len, self.in_dim, device=x.device)
        for i in range(self.in_dim):
            x_i = x[:, :, i]
            basis_values = torch.zeros(batch_size, seq_len, self.num_knots, device=x.device)
            for k in range(self.num_knots - self.spline_order):
                basis_values[:, :, k] = self.bspline_basis(x_i, k, self.spline_order)
            inner_outputs[:, :, i] = (basis_values * self.inner_coeffs[i]).sum(dim=-1)

        outputs = torch.zeros(batch_size, seq_len, self.out_dim, device=x.device)
        for j in range(self.out_dim):
            outputs[:, :, j] = (inner_outputs * self.outer_coeffs[j]).sum(dim=-1)

        return outputs 

    def get_interpretability(self, x):
        """Extract interpretability information."""
        batch_size, seq_len, in_dim = x.shape
        x = torch.tanh(x)

        inner_outputs = torch.zeros(batch_size, seq_len, self.in_dim, device=x.device)
        for i in range(self.in_dim):
            x_i = x[:, :, i]
            basis_values = torch.zeros(batch_size, seq_len, self.num_knots, device=x.device)
            for k in range(self.num_knots - self.spline_order):
                basis_values[:, :, k] = self.bspline_basis(x_i, k, self.spline_order)
            inner_outputs[:, :, i] = (basis_values * self.inner_coeffs[i]).sum(dim=-1)

        contributions = torch.zeros(batch_size, seq_len, self.out_dim, self.in_dim, device=x.device)
        for j in range(self.out_dim):
            contributions[:, :, j, :] = inner_outputs * self.outer_coeffs[j]

        return {
            "inner_outputs": inner_outputs,  
            "contributions": contributions   
        }

    def get_l2_regularization(self):
        l2_inner = torch.norm(self.inner_coeffs, p=2) ** 2
        l2_outer = torch.norm(self.outer_coeffs, p=2) ** 2
        return l2_inner + l2_outer