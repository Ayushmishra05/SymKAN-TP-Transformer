import torch.nn as nn 
from src.sym_kan_transformer.model import * 



def build_kanformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, 
                    N: int=3, h: int=8, dropout: float=0.1, d_ff: int=4096, ff_dims: List[int]=[8192], device: Union[str, int] = 'cuda') -> Transformer:
    src_embed = RoleFillerEmbedding(src_vocab_size, d_model, dropout, src_seq_len)
    tgt_embed = RoleFillerEmbedding(tgt_vocab_size, d_model, dropout, tgt_seq_len)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        ff_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, ff_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for i in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        ff_block = FeedForwardBlock(d_model, d_ff, dropout)
        kan_block = KANFeedForwardBlock(d_model, ff_dims, device=device)
        if i == N-1:
            decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, kan_block, dropout, is_kan=True)
        else:
            decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, ff_block, dropout, is_kan=False)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(ff_dims[-1], nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(ff_dims[-1], tgt_vocab_size)
    
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    for _, p in transformer.named_parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
