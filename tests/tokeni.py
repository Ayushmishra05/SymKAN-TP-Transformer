# D:\DecoderKAN\tests\tokeni.py
from transformers import PreTrainedTokenizer
from preprocess.tokenizersplit import SymbolicQEDTokenizer, SymbolicVocab, reconstruct_expression
import pandas as pd
import os
import json

class QEDHuggingFaceTokenizer(PreTrainedTokenizer):
    def __init__(self, df, max_length=509, **kwargs):
        # Initialize vocabs first
        self.qed_tokenizer = SymbolicQEDTokenizer(
            df=df,
            index_token_pool_size=100,
            special_symbols=['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<SEP>'],
            unk_idx=1,
            to_replace=True
        )
        self.src_vocab = SymbolicVocab(
            self.qed_tokenizer.build_src_vocab(),
            special_symbols=['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<SEP>'],
            bos_idx=2, pad_idx=0, eos_idx=3, unk_idx=1, sep_idx=4
        )
        self.tgt_vocab = SymbolicVocab(
            self.qed_tokenizer.build_tgt_vocab(),
            special_symbols=['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<SEP>'],
            bos_idx=2, pad_idx=0, eos_idx=3, unk_idx=1, sep_idx=4
        )
        self._vocab = self.tgt_vocab.token_to_idx  # Cache vocab for get_vocab()

        # Call super().__init__ with special tokens
        super().__init__(
            pad_token='<PAD>',
            unk_token='<UNK>',
            bos_token='<BOS>',
            eos_token='<EOS>',
            sep_token='<SEP>',
            max_length=max_length,
            **kwargs
        )
        self.max_length = max_length

    @property
    def vocab_size(self):
        return len(self._vocab)  # Override vocab_size

    def _tokenize(self, text, is_source=True):
        return self.qed_tokenizer.src_tokenize(text) if is_source else ['<BOS>'] + self.qed_tokenizer.tgt_tokenize(text) + ['<EOS>']

    def _convert_token_to_id(self, token):
        return self.tgt_vocab.token_to_idx.get(token, self.tgt_vocab.unk_idx)  # Use token_to_idx.get

    def _convert_id_to_token(self, index):
        return self.tgt_vocab.idx_to_token.get(index, self.tgt_vocab.unk_tok)

    def encode(self, text, is_source=True, **kwargs):
        tokens = self._tokenize(text, is_source)
        ids = [self._convert_token_to_id(token) for token in tokens]
        return ids[:self.max_length] + [self.pad_token_id] * (self.max_length - len(ids)) if len(ids) < self.max_length else ids[:self.max_length]

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        tokens = [self._convert_id_to_token(id) for id in token_ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<SEP>']]
        return reconstruct_expression(tokens)

    def get_vocab(self):
        return self._vocab

    def save_vocabulary(self, save_directory, filename_prefix=None):
        vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + 'vocab.json')
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)
        return (vocab_file,)