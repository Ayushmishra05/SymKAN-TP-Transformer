from typing import List, Set, Optional, Tuple
from collections import OrderedDict
from functools import lru_cache
from tqdm import tqdm
import pandas as pd
import sympy as sp
import warnings
import json
import re

class SymbolicQEDTokenizer:
    def __init__(self, df: Optional[pd.DataFrame] = None, 
                 index_token_pool_size: int = 100,
                 special_symbols: List[str] = None,
                 unk_idx: int = 1,
                 to_replace: bool = False):
        self.amps = df.amp.tolist() if df is not None else None
        self.sqamps = df.sqamp.tolist() if df is not None else None
        
        if index_token_pool_size < 50:
            warnings.warn(
                f"Index token pool size ({index_token_pool_size}) may be insufficient. "
                "Consider using at least 50-100 tokens for symbolic tasks.",
                UserWarning
            )

        self.index_pool = [f"INDEX_{i}" for i in range(index_token_pool_size)]
        self.particle_index_pool = [f"PINDEX_{i}" for i in range(index_token_pool_size)]
        self.special_symbols = special_symbols or ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"]
        self.unk_idx = unk_idx
        self.to_replace = to_replace
        self.pattern_underscore_curly = re.compile(r'\b[\w]+(?:_[\w]+)*_{')  # e.g., gamma_{, b_{MOMENTUM_0
        self.pattern_mass = re.compile(r'\bm_([a-z]+)\b')  # e.g., m_c
        self.pattern_mandelstam = re.compile(r'\bs_(\d{2,})\b')  # e.g., s_12
        self.pattern_momentum = re.compile(r'\bp_(\d+)\b')  # e.g., p_1
        self.pattern_single_s = re.compile(r'\bs_(\d+)\b(?!\d)')  # e.g., s_1
        self.pattern_exponent = re.compile(r'\^(\w+|\([^)]+\))')  # e.g., ^2, ^(*)
        self.pattern_special = re.compile(r'_([uv])|\\(\w+_\d+|\w+\b)')  # Spinor suffixes and LaTeX indices
        self.pattern_num_123 = re.compile(r'\b(?![psijkl]_)(?!MOMENTUM_)(?!MASS_)(?!P_)(?!S_)(?!MANDELSTAM_)\w+_\d+\b')  # Non-physics indices
        self.pattern_particle = re.compile(r'(?P<prefix>\b(?:\w+_)?)?(?P<target>[ijkl]_\d+\b)')  # Particle indices

    @staticmethod
    def remove_whitespace(expression: str) -> str:
        return re.sub(r'\s+', '', expression)

    def protect_structures(self, ampl: str) -> Tuple[str, List[str]]:
        protected = []
        return ampl, protected

    def physics_aware_replace(self, ampl: str, is_source: bool = True) -> str:
        ampl = self.remove_whitespace(ampl)
        ampl = re.sub(r'\bi\b(?!\w)', 'I_UNIT', ampl)
        ampl = re.sub(r'\be\b(?=\^|[+\-*/()| ])', 'E_CHARGE', ampl)  # Handle e^X or standalone e
        ampl = ampl.replace('reg_prop', 'REG_PROP')
        # ampl = self.pattern_mass.sub(r'MASS_\1', ampl)
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

        # Apply Mandelstam replacement first to prevent override by pattern_num_123
        ampl = self.pattern_mandelstam.sub(lambda m: f'MANDELSTAM_{m.group(1)}', ampl)

        def get_unique_matches(pattern):
            matches = list(OrderedDict.fromkeys(pattern.findall(ampl)))
            # Filter out tokens that are already in index_pool (e.g., INDEX_4) only for source
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
        # Handle spinor suffixes (_u, _v) and LaTeX indices (\INDEX_0, \+)
        def replace_special(match):
            if match.group(1):  # Spinor suffix (_u or _v)
                return f' _ {match.group(1)} '
            elif match.group(2):  # LaTeX index (\INDEX_0 or \+)
                return f' \\ {match.group(2)} '
        ampl = self.pattern_special.sub(replace_special, ampl)
        # Insert spaces for field terms in source expressions
        if is_source:
            ampl = self.pattern_underscore_curly.sub(lambda match: f' {match.group(0)} ', ampl)
            for symbol in ['{', '}', ',']:
                ampl = ampl.replace(symbol, f' {symbol} ')
        # Standard operator splitting
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
            ampl, protected = self.protect_structures(ampl)
            ampl = self.physics_aware_replace(ampl, is_source=True)
            ampl = self.replace_indices(ampl, is_source=True)
            return self.tokenize_expression(ampl, protected, is_source=True)
        except Exception as e:
            warnings.warn(f"Source tokenization failed for '{ampl}': {e}")
            return [self.special_symbols[self.unk_idx]]

    def tgt_tokenize(self, sqampl: str) -> List[str]:
        try:
            sqampl, protected = self.protect_structures(sqampl)
            sqampl = self.physics_aware_replace(sqampl, is_source=False)
            sqampl = self.replace_indices(sqampl, is_source=False)
            return self.tokenize_expression(sqampl, protected, is_source=False)
        except Exception as e:
            warnings.warn(f"Target tokenization failed for '{sqampl}': {e}")
            return [self.special_symbols[self.unk_idx]]

    def build_src_vocab(self) -> Set[str]:
        if self.amps is None:
            return set()
        vocab_set = set()
        for expr in tqdm(self.amps, desc="Processing source vocab"):
            vocab_set.update(self.src_tokenize(expr))
        return vocab_set

    def build_tgt_vocab(self) -> Set[str]:
        if self.sqamps is None:
            return set()
        vocab_set = set()
        for expr in tqdm(self.sqamps, desc="Processing target vocab"):
            vocab_set.update(self.tgt_tokenize(expr))
        return vocab_set

class SymbolicVocab:
    def __init__(self, tokens: Set[str], special_symbols: List[str], 
                 bos_idx: int, pad_idx: int, eos_idx: int, unk_idx: int, sep_idx: int):
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

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_idx.get(token, self.unk_idx) for token in tokens]

    def decode(self, indices: List[int], include_special_tokens: bool = True) -> List[str]:
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

    def tokens(self) -> List[str]:
        return self.token_list

    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            for token in self.token_list:
                f.write(f"{token}\n")
        with open(filepath.replace('.txt', '.json'), 'w', encoding='utf-8') as f:
            json.dump(self.token_to_idx, f, ensure_ascii=False, indent=2)

def reconstruct_expression(tokens: List[str]) -> str:
    expr = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # Handle negative numbers
        if token == '-' and i + 1 < len(tokens) and tokens[i + 1].isdigit():
            expr.append(token)
        elif token.startswith('MASS_'):
            expr.append(f'm_{token[5:]}')
        elif token.startswith('MANDELSTAM_'):
            expr.append(f's_{token[11:]}')
        elif token.startswith('P_'):
            expr.append(f'p_{token[2:]}')
        elif token == 'I_UNIT':
            expr.append('i')
        elif token == 'E_CHARGE':
            expr.append('e')
        elif token == 'REG_PROP':
            expr.append('reg_prop')
        elif token == 'CONJ':
            expr.append('(*)')
        elif token.startswith('S_'):
            expr.append(f's_{token[2:]}')
        # Handle field term prefixes (e.g., gamma_{, +)
        elif token.endswith('_{') and i + 1 < len(tokens) and tokens[i + 1] in ['+', '-', '\\']:
            expr.append(token[:-1])
            i += 1
            continue
        # Handle spinor suffixes (e.g., _, u)
        elif token == '_' and i + 1 < len(tokens) and tokens[i + 1] in ['u', 'v']:
            expr.append(f'_{tokens[i + 1]}')
            i += 1
        # Handle LaTeX backslashes (e.g., \, INDEX_0)
        elif token == '\\' and i + 1 < len(tokens) and (tokens[i + 1].startswith(('INDEX_', 'MOMENTUM_')) or tokens[i + 1] in ['+', '-']):
            expr.append(f'\\{tokens[i + 1]}')
            i += 1
        else:
            expr.append(token)
        i += 1
    return ''.join(expr)

@lru_cache(maxsize=1000)
def parse_qed_expression(expr: str) -> Optional[sp.Expr]:
    i, e, m_b, m_c = sp.symbols('i e m_b m_c')
    s_12, s_14, s_23, s_13, s_24, s_34 = sp.symbols('s_12 s_14 s_23 s_13 s_24 s_34')
    reg_prop = sp.symbols('reg_prop')
    
    expr = expr.replace('^', '**')
    
    try:
        parsed = sp.sympify(expr, locals={
            'i': i,
            'e': e,
            'm_b': m_b,
            'm_c': m_c,
            's_12': s_12,
            's_14': s_14,
            's_23': s_23,
            's_13': s_13,
            's_24': s_24,
            's_34': s_34,
            'reg_prop': reg_prop
        })
        return parsed
    except sp.SympifyError as e:
        print(f"SymPy parsing error for expression '{expr}': {e}")
        return None

def validate_expression(original: str, tokens: List[str], is_source: bool = True) -> Tuple[bool, bool]:
    reconstructed = reconstruct_expression(tokens)
    string_match = original == reconstructed
    
    symbolic_match = False
    if not is_source:
        orig_parsed = parse_qed_expression(original)
        recon_parsed = parse_qed_expression(reconstructed)
        if orig_parsed and recon_parsed:
            symbolic_match = sp.simplify(orig_parsed - recon_parsed) == 0
        else:
            print(f"Symbolic parsing failed for {'original' if not orig_parsed else 'reconstructed'} target expression.")
    
    return string_match, symbolic_match

def main():
    special_symbols = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"]
    
    csv_file = r"QED_data/train_data.csv"
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded dataset with {len(df)} records.")
    except FileNotFoundError:
        print("Dataset not found. Please update the path to QED_data/train_data.csv.")
        return

    tokenizer = SymbolicQEDTokenizer(
        df=df,
        index_token_pool_size=100,
        special_symbols=special_symbols,
        unk_idx=1,
        to_replace=True  # Enable index replacement
    )

    src_expr = "1/9*i*e^2*gamma_{+\\INDEX_0,INDEX_1,INDEX_2}*gamma_{\\INDEX_0,INDEX_3,INDEX_4}*b_{MOMENTUM_0,INDEX_4}(p_3)_v*b_{MOMENTUM_1,INDEX_3}(p_4)_u^(*)*c_{MOMENTUM_2,INDEX_2}(p_1)_u*c_{MOMENTUM_3,INDEX_1}(p_2)_v^(*)/(m_c^2+s_12+1/2*reg_prop)"
    tgt_expr = "1/81*e^4*(16*m_b^2*m_c^2 + 8*m_b^2*s_12 + 8*s_14*s_23 + - 8*s_13*s_24 + 8*m_c^2*s_34)*(m_c^2 + s_12 + 1/2*reg_prop)^(-2)"
    
    print("\n=== Testing Source Expression ===")
    src_tokens = tokenizer.src_tokenize(src_expr)
    src_string_match, _ = validate_expression(src_expr, src_tokens, is_source=True)
    print(f"Source Original: {src_expr}")
    print(f"Source Tokens: {src_tokens}")
    print(f"Source Reconstructed: {reconstruct_expression(src_tokens)}")
    print(f"Source String Match: {src_string_match}")

    print("\n=== Testing Target Expression ===")
    tgt_tokens = tokenizer.tgt_tokenize(tgt_expr)
    tgt_string_match, tgt_symbolic_match = validate_expression(tgt_expr, tgt_tokens, is_source=False)
    print(f"Target Original: {tgt_expr}")
    print(f"Target Tokens: {tgt_tokens}")
    print(f"Target Reconstructed: {reconstruct_expression(tgt_tokens)}")
    print(f"Target String Match: {tgt_string_match}")
    print(f"Target Symbolic Equivalence: {tgt_symbolic_match}")

    target_edge_cases = [
        "(1/2)/(3/4)",
        "10/100",
        "(m_c^2 + s_12)^2",
        "1/2",
        "-1/2"
    ]
    source_edge_cases = [
        "gamma_{+\\INDEX_0,INDEX_1,INDEX_2}",
        "b_{MOMENTUM_0,INDEX_4}(p_3)_v"
    ]
    print("\n=== Testing Target Edge Cases ===")
    for expr in target_edge_cases:
        try:
            tokens = tokenizer.tgt_tokenize(expr)
            string_match, symbolic_match = validate_expression(expr, tokens, is_source=False)
            reconstructed = reconstruct_expression(tokens)
            print(f"\nEdge Case: {expr}")
            print(f"Tokens: {tokens}")
            print(f"Reconstructed: {reconstructed}")
            print(f"String Match: {string_match}")
            print(f"Symbolic Equivalence: {symbolic_match}")
        except Exception as e:
            print(f"\nEdge Case: {expr}")
            print(f"Error: {e}")

    print("\n=== Testing Source Edge Cases ===")
    for expr in source_edge_cases:
        try:
            tokens = tokenizer.src_tokenize(expr)
            string_match, _ = validate_expression(expr, tokens, is_source=True)
            reconstructed = reconstruct_expression(tokens)
            print(f"\nEdge Case: {expr}")
            print(f"Tokens: {tokens}")
            print(f"Reconstructed: {reconstructed}")
            print(f"String Match: {string_match}")
        except Exception as e:
            print(f"\nEdge Case: {expr}")
            print(f"Error: {e}")

    print("\n=== Building Vocabularies ===")
    src_vocab_set = tokenizer.build_src_vocab()
    tgt_vocab_set = tokenizer.build_tgt_vocab()
    print(f"Source Vocabulary Size: {len(src_vocab_set)} tokens")
    print(f"Target Vocabulary Total Size (with special tokens): {len(tgt_vocab_set)}")

    src_vocab = SymbolicVocab(
        tokens=src_vocab_set,
        special_symbols=special_symbols,
        bos_idx=2,
        pad_idx=0,
        eos_idx=3,
        unk_idx=1,
        sep_idx=4
    )
    tgt_vocab = SymbolicVocab(
        tokens=tgt_vocab_set,
        special_symbols=special_symbols,
        bos_idx=2,
        pad_idx=0,
        eos_idx=3,
        unk_idx=1,
        sep_idx=4
    )

    print(f"Source Vocabulary Total Size (with special tokens): {len(src_vocab)}")
    print(f"Target Vocabulary Total Size (with special tokens): {len(tgt_vocab)}")

    src_vocab.save('src_vocab.txt')
    tgt_vocab.save('tgt_vocab.txt')
    print("\nVocabularies saved to 'src_vocab.txt' and 'tgt_vocab.txt'.")
    print("Token-to-index mappings saved to 'src_vocab.json' and 'tgt_vocab.json'.")

    print("\n=== Example Encoding/Decoding ===")
    sample_tgt_expr = df['sqamp'].iloc[0]
    sample_tgt_tokens = tokenizer.tgt_tokenize(sample_tgt_expr)
    encoded = tgt_vocab.encode(sample_tgt_tokens)
    decoded = tgt_vocab.decode(encoded)
    print(f"Sample Target Expression: {sample_tgt_expr}")
    print(f"Tokens: {sample_tgt_tokens}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    print("\n=== Validating Full Dataset ===")
    count = 0
    failed_expressions = []
    for i in tqdm(range(len(df)), desc="Validating dataset"):
        src_expr = df['amp'].iloc[i]
        tgt_expr = df['sqamp'].iloc[i]
        src_tokens = tokenizer.src_tokenize(src_expr)
        tgt_tokens = tokenizer.tgt_tokenize(tgt_expr)
        src_string_match, _ = validate_expression(src_expr, src_tokens, is_source=True)
        tgt_string_match, tgt_symbolic_match = validate_expression(tgt_expr, tgt_tokens, is_source=False)
        if tgt_symbolic_match:
            count += 1
        else:
            print((i, tgt_expr, tgt_tokens, reconstruct_expression(tgt_tokens)))
            failed_expressions.append((i, src_expr, tgt_expr, tgt_tokens, reconstruct_expression(tgt_tokens)))
    
    print(f"\nSymbolic Equivalence True for {count}/{len(df)} target expressions.")
    if failed_expressions:
        with open('failed_expressions.txt', 'w', encoding='utf-8') as f:
            for idx, src, tgt, tokens, recon in failed_expressions:
                f.write(f"Expression {idx+1}:\n")
                f.write(f"Source: {src}\n")
                f.write(f"Target: {tgt}\n")
                f.write(f"Target Tokens: {tokens}\n")
                f.write(f"Reconstructed: {recon}\n\n")
        print(f"Logged {len(failed_expressions)} failed expressions to 'failed_expressions.txt'.")

if __name__ == "__main__":
    main()