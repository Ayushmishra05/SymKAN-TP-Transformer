import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
                / self.grid_size
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

        grid: torch.Tensor = self.grid
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

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        
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

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class RoleFillerEmbedding(nn.Module):
    """
    Role-Filler embedding with sinusoidal positional encoding and multiplicative binding
    """
    def __init__(self, vocab_size, d_model, max_length=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        
        # Token embeddings (fillers)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)
        
        # Sinusoidal positional encoding (roles)
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0., max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        # Role transformation for binding
        self.role_transform = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.token_embedding.weight, mean=0, std=1./math.sqrt(self.d_model))
        nn.init.normal_(self.role_transform.weight, mean=0, std=1./math.sqrt(self.d_model))
        nn.init.zeros_(self.role_transform.bias)
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        seq_len = x.size(1)
        
        # Get token embeddings (fillers)
        token_emb = self.token_embedding(x) * self.scale
        
        # Get positional embeddings (roles)
        pos_emb = self.pe[:, :seq_len]
        
        # Transform roles for binding
        roles = self.role_transform(pos_emb) + 1  # Add 1 to avoid zero multiplication
        
        # Multiplicative binding: filler * role
        bound_representation = token_emb * roles
        
        return self.dropout(bound_representation)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(context)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, kan_grid_size=5):
        super().__init__()
        
        # Multi-head attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # KAN feedforward
        self.kan_layer = KANLinear(d_model, d_model, grid_size=kan_grid_size)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # KAN feedforward with residual connection
        kan_out = self.kan_layer(x)
        x = self.norm2(x + self.dropout2(kan_out))
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, kan_grid_size=5):
        super().__init__()
        
        # Self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # KAN feedforward
        self.kan_layer = KANLinear(d_model, d_model, grid_size=kan_grid_size)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # Self-attention with causal mask
        self_attn_out = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))
        
        # Cross-attention with encoder output
        cross_attn_out = self.cross_attention(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))
        
        # KAN feedforward
        kan_out = self.kan_layer(x)
        x = self.norm3(x + self.dropout3(kan_out))
        
        return x


class SymKANTPTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=6,
        max_length=512,
        dropout=0.1,
        kan_grid_size=5,
        pad_idx=0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Role-filler embeddings
        self.embedding = RoleFillerEmbedding(vocab_size, d_model, max_length, dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, dropout, kan_grid_size)
            for _ in range(n_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, dropout, kan_grid_size)
            for _ in range(n_decoder_layers)
        ])
        
        # Output projection with KAN
        self.output_projection = KANLinear(d_model, vocab_size, grid_size=kan_grid_size)
        
    def create_padding_mask(self, x):
        """Create mask for padding tokens"""
        return (x != self.pad_idx).unsqueeze(1).unsqueeze(2)
    
    def create_causal_mask(self, size):
        """Create causal mask for decoder self-attention"""
        mask = torch.tril(torch.ones(size, size))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def encode(self, src, src_mask=None):
        """Encode source sequence"""
        x = self.embedding(src)
        
        if src_mask is None:
            src_mask = self.create_padding_mask(src)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        """Decode target sequence"""
        x = self.embedding(tgt)
        
        if tgt_mask is None:
            tgt_padding_mask = self.create_padding_mask(tgt)
            tgt_causal_mask = self.create_causal_mask(tgt.size(1)).to(tgt.device)
            tgt_mask = tgt_padding_mask & tgt_causal_mask
        
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        
        return x
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """Forward pass for training"""
        # Encode
        memory = self.encode(src, src_mask)
        
        # Decode
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits
    
    def generate(self, src, max_length=50, start_token=1, end_token=2):
        """Generate sequence using greedy decoding"""
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        # Encode source
        memory = self.encode(src)
        
        # Initialize target with start token
        tgt = torch.full((batch_size, 1), start_token, device=device)
        
        for _ in range(max_length):
            # Decode current sequence
            output = self.decode(tgt, memory)
            
            # Get next token probabilities
            logits = self.output_projection(output[:, -1:])
            next_token = torch.argmax(logits, dim=-1)
            
            # Append to target sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Check if all sequences have generated end token
            if (next_token == end_token).all():
                break
        
        return tgt[:, 1:]  # Remove start token
    
    def get_kan_regularization(self):
        """Get regularization loss from all KAN layers"""
        reg_loss = 0
        for layer in self.encoder_layers:
            reg_loss += layer.kan_layer.regularization_loss()
        for layer in self.decoder_layers:
            reg_loss += layer.kan_layer.regularization_loss()
        reg_loss += self.output_projection.regularization_loss()
        return reg_loss


def create_model(vocab_size, **kwargs):
    """Factory function to create the model"""
    return SymKANTPTransformer(vocab_size, **kwargs)


# Example usage
if __name__ == "__main__":
    # Model parameters
    vocab_size = 10000
    d_model = 512
    n_heads = 8
    n_encoder_layers = 6
    n_decoder_layers = 6
    
    # Create model
    model = create_model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        kan_grid_size=5
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example forward pass
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    src = torch.randint(1, vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_seq_len))
    
    # Forward pass
    logits = model(src, tgt)
    print(f"Output shape: {logits.shape}")
    
    # Generate example
    generated = model.generate(src, max_length=20)
    print(f"Generated shape: {generated.shape}")
    
    # KAN regularization
    kan_reg = model.get_kan_regularization()
    print(f"KAN regularization loss: {kan_reg.item()}")