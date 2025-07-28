import torch 
import pandas as pd 
from vanilla_transformer import * 

def train():
    ## Hyperparams for transformer training, can be tuned
    d_x = 512
    n_layers = 6
    n_heads = 8
    dropout = 0.1
    max_length = 50
    pad_idx = 0
    sos_idx = 2
    eos_idx = 3
    epochs = 10
    lr = 1e-4
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_df = pd.read_csv(r'D:\DecoderKAN\QED_data\test-flow.csv')

    tokenizer = QEDTokenizer()
    expressions = pd.concat([data_df["amp"], data_df["sqamp"]]).tolist()
    tokenizer.build_vocab(expressions)
    d_vocab = len(tokenizer.get_vocab())

    dataset = QEDDataset(data_df, tokenizer, max_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = Subset(dataset, range(train_size)), Subset(dataset, range(train_size, len(dataset)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = Transformer(d_vocab, d_x, n_layers, n_heads, dropout, max_length, pad_idx)
    model.to(device)

    train_and_evaluate(model, train_loader, val_loader, epochs, lr, device)

    model.eval()
    test_expr = r"1/9*i*e^2*(p_2_\INDEX_0*gamma_{+\INDEX_0,INDEX_1,INDEX_2}*gamma_{\INDEX_3,INDEX_4,INDEX_1}*gamma_{\INDEX_5,INDEX_2,INDEX_6}*A_{MOMENTUM_0,+\INDEX_5}(p_3)^(*)*A_{MOMENTUM_1,+\INDEX_3}(p_4)^(*)*b_{MOMENTUM_2,INDEX_6}(p_2)_u*b_{MOMENTUM_3,INDEX_4}(p_1)_v^(*)+-p_3_\INDEX_0*gamma_{+\INDEX_0,INDEX_7,INDEX_8}*gamma_{\INDEX_3,INDEX_9,INDEX_7}*gamma_{\INDEX_5,INDEX_8,INDEX_10}*A_{MOMENTUM_0,+\INDEX_5}(p_3)^(*)*A_{MOMENTUM_1,+\INDEX_3}(p_4)^(*)*b_{MOMENTUM_2,INDEX_10}(p_2)_u*b_{MOMENTUM_3,INDEX_9}(p_1)_v^(*)+m_b*gamma_{\INDEX_3,INDEX_11,INDEX_12}*gamma_{\INDEX_5,INDEX_12,INDEX_13}*A_{MOMENTUM_0,+\INDEX_5}(p_3)^(*)*A_{MOMENTUM_1,+\INDEX_3}(p_4)^(*)*b_{MOMENTUM_2,INDEX_13}(p_2)_u*b_{MOMENTUM_3,INDEX_11}(p_1)_v^(*))/(m_b^2+-s_22+2*s_23+-s_33+-reg_prop)"
    src_ids = torch.tensor([tokenizer.encode(test_expr, max_length)], device=device)
    output = model.greedy_inference(src_ids, sos_idx, eos_idx, max_length)
    decoded = tokenizer.decode(output[0].tolist())
    print(f"Input: {test_expr}")
    print(f"Output IDs: {output.tolist()}")
    print(f"Output: {decoded}")


train()
