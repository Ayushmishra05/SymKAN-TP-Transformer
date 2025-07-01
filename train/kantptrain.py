import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np
from preprocess.tokenizersplit import SymbolicQEDTokenizer, SymbolicVocab, reconstruct_expression  # Current tokenizer
from src.KAN_TP_Transformer import build_transformer, Seq2Seq, KANLayer  # Current model
from src.parse_and_compare import parse_and_compare  # Current validation function
from torch.cuda.amp import GradScaler, autocast
import time
import utils
import sys
import json


# Hyperparameters
EPOCHS = 1
LEARNING_RATE = 5e-5  # From old pipeline
BATCH_SIZE = 64 # Adjusted for 500 records (smaller than new pipeline's 16 to balance memory with gradient accumulation)
GRADIENT_ACCUMULATION_STEPS = 2  # From old pipeline
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS  # 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 509  # Unified for src and tgt (from new pipeline)
PAD_IDX = 0
DROPOUT = 0.1
FILTER_DIM = 512  # From new pipeline
N_LAYERS = 3
N_HEADS = 4
HIDDEN_DIM = 256
LAMBDA_L2 = 1e-5  # From old pipeline

class HyperParams:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.filter = FILTER_DIM
        self.n_layers = N_LAYERS
        self.n_heads = N_HEADS
        self.hidden = HIDDEN_DIM
        self.dropout = DROPOUT

# Custom QED dataset
class QEDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: SymbolicQEDTokenizer, src_vocab: SymbolicVocab, 
                 tgt_vocab: SymbolicVocab, max_len: int = MAX_LENGTH):
        self.df = df
        self.tokenizer = tokenizer
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        src_expr = self.df['amp'].iloc[idx]
        tgt_expr = self.df['sqamp'].iloc[idx]
        
        # Tokenize and encode
        src_tokens = self.tokenizer.src_tokenize(src_expr)
        tgt_tokens = ['<BOS>'] + self.tokenizer.tgt_tokenize(tgt_expr) + ['<EOS>']
        
        src_ids = self.src_vocab.encode(src_tokens)
        tgt_ids = self.tgt_vocab.encode(tgt_tokens)
        
        # Pad sequences
        src_ids = src_ids + [self.src_vocab['<PAD>']] * (self.max_len - len(src_ids))
        tgt_ids = tgt_ids + [self.tgt_vocab['<PAD>']] * (self.max_len - len(tgt_ids))
        
        return {
            'input_ids': torch.tensor(src_ids, dtype=torch.long),
            'labels': torch.tensor(tgt_ids, dtype=torch.long),
            'tgt_expr': tgt_expr
        }

def train_epoch(model, dataloader, optimizer, criterion, scaler, device, clip=1.0):
    model.train()
    total_loss = 0
    step = 0
    with tqdm(total=len(dataloader), desc="Training", file=sys.stdout) as progress_bar:
        for batch_idx, batch in enumerate(dataloader):
            start_time = time.time()
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast():
                logits = model(input_ids, labels[:, :-1])  # Exclude <EOS>
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))  # Exclude <BOS>
                
                # L2 regularization for KAN layers
                l2_reg = 0
                l2_reg += model.kan_layer.get_l2_regularization()
                for layer in model.encoder.layers:
                    l2_reg += layer.densefilter.get_l2_regularization()
                for layer in model.decoder.layers:
                    l2_reg += layer.densefilter.get_l2_regularization()
                total_loss_batch = loss + LAMBDA_L2 * l2_reg
            
            total_loss_batch = total_loss_batch / GRADIENT_ACCUMULATION_STEPS
            if device.type == 'cuda':
                scaler.scale(total_loss_batch).backward()
            else:
                total_loss_batch.backward()
            
            step += 1
            if step % GRADIENT_ACCUMULATION_STEPS == 0:
                if device.type == 'cuda':
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
                optimizer.zero_grad()
                step = 0
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'l2_reg': f'{l2_reg.item():.4f}'})
            progress_bar.update(1)
            
            batch_time = time.time() - start_time
            if (batch_idx + 1) % 10 == 0:  # Reduced logging frequency for 500 records
                print(f"Batch {batch_idx+1}/{len(dataloader)} completed in {batch_time:.2f} seconds")
    
    if step > 0:
        if device.type == 'cuda':
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, tokenizer, tgt_vocab, device):
    model.eval()
    symbolic_accuracies = []
    val_loss = 0
    with torch.no_grad(), tqdm(total=len(dataloader), desc="Validating", file=sys.stdout) as val_bar:
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            tgt_exprs = batch['tgt_expr']
            
            with autocast():
                logits = model(input_ids, labels[:, :-1])
                loss = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<PAD>'])(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
                val_loss += loss.item()
                
                # Generate predictions
                preds = model.greedy_inference(
                    model, 
                    src=input_ids,
                    sos_idx=tgt_vocab['<BOS>'],
                    eos_idx=tgt_vocab['<EOS>'],
                    max_length=MAX_LENGTH
                )
                
                # Compute symbolic accuracy
                for pred_ids, gt_expr in zip(preds, tgt_exprs):
                    symbolic_match = parse_and_compare(gt_expr, pred_ids.tolist(), tgt_vocab, tokenizer)
                    symbolic_accuracies.append(int(symbolic_match))
            
            val_bar.set_postfix({'val_loss': f'{val_loss / (val_bar.n + 1):.4f}'})
            val_bar.update(1)
    
    return np.mean(symbolic_accuracies) * 100, val_loss / len(dataloader)

def main_training():
    # Config
    device = DEVICE
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    clip = 1.0
    max_len = MAX_LENGTH
    
    # Load dataset (500 records)
    df = pd.read_csv("QED_data/train_data.csv").iloc[:100]
    
    # Initialize tokenizer and vocab
    tokenizer = SymbolicQEDTokenizer(
        df=df,
        index_token_pool_size=100,
        special_symbols=["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"],
        unk_idx=1,
        to_replace=True
    )
    src_vocab_set = tokenizer.build_src_vocab()
    tgt_vocab_set = tokenizer.build_tgt_vocab()
    
    src_vocab = SymbolicVocab(
        tokens=src_vocab_set,
        special_symbols=["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"],
        bos_idx=2,
        pad_idx=0,
        eos_idx=3,
        unk_idx=1,
        sep_idx=4
    )
    tgt_vocab = SymbolicVocab(
        tokens=tgt_vocab_set,
        special_symbols=["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"],
        bos_idx=2,
        pad_idx=0,
        eos_idx=3,
        unk_idx=1,
        sep_idx=4
    )
    
    # Save vocabularies
    src_vocab.save('src_vocab.txt')
    tgt_vocab.save('tgt_vocab.txt')
    with open('tgt_vocab.json', 'w') as f:
        json.dump(tgt_vocab.token_to_idx, f, ensure_ascii=False, indent=2)
    
    # Initialize dataset and dataloader
    dataset = QEDDataset(df, tokenizer, src_vocab, tgt_vocab, max_len)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    params = HyperParams(input_dim=len(src_vocab))
    model = build_transformer(params, pad_idx=src_vocab['<PAD>']).to(device)
    # Update KAN layer to match old pipeline
    model.kan_layer = KANLayer(in_dim=HIDDEN_DIM, out_dim=len(tgt_vocab), num_knots=3, spline_order=2)
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab['<PAD>'])
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Initial forward pass for verification
    print("\nPerforming a basic forward pass to verify the flow...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        tgt_expr = batch['tgt_expr'][0]
        
        with autocast():
            logits = model(input_ids, labels[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Forward pass completed successfully! Loss: {loss.item():.4f}")
        print(f"Output shape: {logits.shape}")
        
        predicted_ids = torch.argmax(logits, dim=-1)
        decoded_tokens = tgt_vocab.decode(predicted_ids[0].tolist(), include_special_tokens=False)
        predicted_expr = reconstruct_expression(decoded_tokens)
        symbolic_match = parse_and_compare(tgt_expr, predicted_ids[0].tolist(), tgt_vocab, tokenizer)
        print(f"True Expression: {tgt_expr}")
        print(f"Predicted Expression: {predicted_expr}")
        print(f"Symbolic Equivalence: {symbolic_match}")
        
        interpretability_info = model.get_interpretability(input_ids, labels[:, :-1])
        print("Interpretability Info:")
        print(f"Inner Outputs Shape: {interpretability_info['inner_outputs'].shape}")
        print(f"Contributions Shape: {interpretability_info['contributions'].shape}")
        
        seq_idx, pos_idx = 0, 0
        predicted_token = predicted_ids[seq_idx, pos_idx].item()
        contributions = interpretability_info['contributions'][seq_idx, pos_idx, predicted_token]
        top_features = torch.argsort(contributions, descending=True)[:5]
        print(f"\nTop contributing features for predicted token {predicted_token} at position {pos_idx}:")
        for feature_idx in top_features:
            print(f"Feature {feature_idx.item()}: Contribution {contributions[feature_idx].item():.4f}")
    
    # Training loop
    best_val_acc = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device, clip)
        val_acc, val_loss = validate_epoch(model, val_loader, tokenizer, tgt_vocab, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Symbolic Accuracy: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'val_acc': val_acc
            }, 'symkan_tp_transformer.pt')
    
    # Log final results
    with open('training_results.txt', 'w') as f:
        f.write(f"Final Symbolic Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Target: ~48.67% (Proposal Page 9)\n")
    
    return model, tokenizer, src_vocab, tgt_vocab

if __name__ == "__main__":
    model, tokenizer, src_vocab, tgt_vocab = main_training()