import os 
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.sym_kan_transformer.tokenizer import SymbolicQEDTokenizer, SymbolicVocab
from src.sym_kan_transformer.build_model import build_kanformer
from src.sym_kan_transformer.config import Config  
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import torch
import time
import warnings

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

def train_and_evaluate(model, train_loader, val_loader, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  
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

def main(config):
    start_time = time.time()
    data_df = pd.read_csv(config.DATA_PATH)
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    tokenizer = SymbolicQEDTokenizer(df=data_df, index_token_pool_size=config.INDEX_TOKEN_POOL_SIZE,
                                    special_symbols=config.SPECIAL_SYMBOLS, unk_idx=config.UNK_IDX,
                                    to_replace=config.TO_REPLACE)
    src_vocab_size = len(tokenizer.build_src_vocab()) + len(config.SPECIAL_SYMBOLS)
    tgt_vocab_size = len(tokenizer.build_tgt_vocab()) + len(config.SPECIAL_SYMBOLS)
    print(f"Tokenizer initialized in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    dataset = QEDDataset(data_df, tokenizer, config.MAX_LENGTH)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = Subset(dataset, range(train_size)), Subset(dataset, range(train_size, len(dataset)))
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    print(f"Data loaders prepared in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    model = build_kanformer(src_vocab_size, tgt_vocab_size, config.MAX_LENGTH, config.MAX_LENGTH,
                           config.D_MODEL, config.N_LAYERS, config.N_HEADS, config.DROPOUT,
                           config.D_FF, config.FF_DIMS, config.DEVICE)
    model.to(config.DEVICE)
    print(f"Model initialized in {time.time() - start_time:.2f} seconds")

    train_and_evaluate(model, train_loader, val_loader, config.EPOCHS, config.LR, config.DEVICE)

    model.eval()
    start_time = time.time()
    test_expr = r"1/9*i*e^2*(p_2_\INDEX_0*gamma_{+\INDEX_0,INDEX_1,INDEX_2}*gamma_{\INDEX_3,INDEX_4,INDEX_1}*gamma_{\INDEX_5,INDEX_2,INDEX_6}*A_{MOMENTUM_0,+\INDEX_5}(p_3)^(*)*A_{MOMENTUM_1,+\INDEX_3}(p_4)^(*)*b_{MOMENTUM_2,INDEX_6}(p_2)_u*b_{MOMENTUM_3,INDEX_4}(p_1)_v^(*)+-p_3_\INDEX_0*gamma_{+\INDEX_0,INDEX_7,INDEX_8}*gamma_{\INDEX_3,INDEX_9,INDEX_7}*gamma_{\INDEX_5,INDEX_8,INDEX_10}*A_{MOMENTUM_0,+\INDEX_5}(p_3)^(*)*A_{MOMENTUM_1,+\INDEX_3}(p_4)^(*)*b_{MOMENTUM_2,INDEX_10}(p_2)_u*b_{MOMENTUM_3,INDEX_9}(p_1)_v^(*)+m_b*gamma_{\INDEX_3,INDEX_11,INDEX_12}*gamma_{\INDEX_5,INDEX_12,INDEX_13}*A_{MOMENTUM_0,+\INDEX_5}(p_3)^(*)*A_{MOMENTUM_1,+\INDEX_3}(p_4)^(*)*b_{MOMENTUM_2,INDEX_13}(p_2)_u*b_{MOMENTUM_3,INDEX_11}(p_1)_v^(*))/(m_b^2+-s_22+2*s_23+-s_33+-reg_prop)"
    src_tokens = tokenizer.src_tokenize(test_expr)
    src_ids = torch.tensor([dataset.src_vocab.encode(src_tokens)], device=config.DEVICE)
    src_mask = (src_ids != 0).unsqueeze(1).unsqueeze(2)
    enc_output = model.encode(src_ids, src_mask)
    trg = torch.full((1, 1), 2, dtype=torch.long, device=config.DEVICE)  # Start with BOS token
    for _ in range(config.MAX_LENGTH):
        tgt_mask = (trg != 0).unsqueeze(1).unsqueeze(3) & torch.tril(torch.ones(trg.size(1), trg.size(1), device=config.DEVICE)).bool()
        dec_output = model.decode(enc_output, src_mask, trg, tgt_mask)
        logits = model.project(dec_output[:, -1])
        pred = torch.argmax(logits, dim=-1).unsqueeze(1)
        trg = torch.cat([trg, pred], dim=1)
        if pred.item() == 3:  
            break
    decoded = dataset.tgt_vocab.decode(trg[0].tolist())
    print(f"Inference completed in {time.time() - start_time:.2f} seconds")
    print(f"Input: {test_expr}")
    print(f"Output IDs: {trg.tolist()}")
    print(f"Output: {''.join(decoded)}")

if __name__ == "__main__":
    config = Config()  
    main(config)