import pandas as pd
from preprocess.tokenizersplit import SymbolicQEDTokenizer, reconstruct_expression
df = pd.read_csv("QED_data/train_data.csv")
tokenizer = SymbolicQEDTokenizer(df=df, index_token_pool_size=100, 
                                special_symbols=["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"], 
                                unk_idx=1, to_replace=True)
true_expr = "1/81*e^4*s_14*s_24*(s_23 + -1/2*reg_prop)^(-2) + 2/81*i*e^2*(i*e^2*m_d^2*(m_d^2 + 1/2*s_12)/(s_23 + -1/2*reg_prop) + -1/16*i*e^2*m_d^2*(16*s_14 + 8*s_24)/(s_23 + -1/2*reg_prop))/(s_23 + -1/2*reg_prop)"
tokens = tokenizer.tgt_tokenize(true_expr)
print(tokens)
recon = reconstruct_expression(tokens)
string_match, symbolic_match = tokenizer.validate_expression(true_expr, tokens, is_source=False)
print(f"Reconstructed: {recon}")
print(f"String Match: {string_match}")
print(f"Symbolic Equivalence: {symbolic_match}")