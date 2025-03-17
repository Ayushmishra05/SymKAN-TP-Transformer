import torch
from torch.optim import AdamW
from model import build_transformer
import pickle
import sys
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training

# Hyperparameters
EPOCHS = 5
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 512
MAX_SEQ_LEN = 44

def create_causal_mask(size, device):
    """Create a causal mask once and reuse it for the same sequence length."""
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool().to(device)
    return ~mask

def train_model():
    try:
        # Initialize model and move to device
        model = build_transformer(vocab_size=VOCAB_SIZE, d_model=512, num_heads=8)
        model.to(DEVICE)
        print(f"Model initialized on {DEVICE}")

        # Optimizer and loss function
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

        # Mixed precision training setup
        scaler = GradScaler() if DEVICE.type == "cuda" else None

        # Load data
        with open(r'src/Dataloaders/train_loader.pkl', 'rb') as f:
            train_loader = pickle.load(f)
        with open(r'src/Dataloaders/val_loader.pkl', 'rb') as f:
            val_loader = pickle.load(f)

        # Precompute causal mask for MAX_SEQ_LEN
        causal_mask = create_causal_mask(MAX_SEQ_LEN, DEVICE)

        # Training loop
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}", file=sys.stdout) as progress_bar:
                for batch in train_loader:
                    # Determine batch structure (tuple or dictionary)
                    if isinstance(batch, (tuple, list)):
                        # Tuple format: (input_ids, attention_mask, labels)
                        input_ids = torch.clamp(batch[0], 0, VOCAB_SIZE - 1).to(DEVICE)
                        labels = torch.clamp(batch[2], 0, VOCAB_SIZE - 1).to(DEVICE)
                    elif isinstance(batch, dict):
                        # Dictionary format: {'input_ids': ..., 'attention_mask': ..., 'labels': ...}
                        input_ids = torch.clamp(batch['input_ids'], 0, VOCAB_SIZE - 1).to(DEVICE)
                        labels = torch.clamp(batch['labels'], 0, VOCAB_SIZE - 1).to(DEVICE)
                    else:
                        raise ValueError(f"Unsupported batch format: {type(batch)}. Expected tuple/list or dict.")

                    src = input_ids
                    tgt = labels

                    src_seq_len = src.size(1)
                    tgt_seq_len = tgt.size(1)
                    batch_size = src.size(0)

                    # Create masks
                    src_mask = (src != 0).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, src_seq_len, src_seq_len)
                    tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
                    tgt_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, tgt_seq_len, tgt_seq_len)
                    tgt_mask = tgt_mask & tgt_padding_mask.expand(batch_size, 1, tgt_seq_len, tgt_seq_len)
                    cross_mask = (src != 0).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, tgt_seq_len - 1, src_seq_len)

                    # Zero gradients
                    optimizer.zero_grad()

                    # Mixed precision training
                    if DEVICE.type == "cuda":
                        with autocast():
                            output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1], cross_mask)
                            loss = loss_fn(output.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1], cross_mask)
                        loss = loss_fn(output.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                    total_loss += loss.item()
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                    progress_bar.update(1)

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

            # Validation
            model.eval()
            val_loss = 0
            if val_loader is not None and len(val_loader) > 0:
                with torch.no_grad(), tqdm(total=len(val_loader), desc="Validation", file=sys.stdout) as val_bar:
                    for val_batch in val_loader:
                        # Determine batch structure
                        if isinstance(val_batch, (tuple, list)):
                            val_input_ids = torch.clamp(val_batch[0], 0, VOCAB_SIZE - 1).to(DEVICE)
                            val_labels = torch.clamp(val_batch[2], 0, VOCAB_SIZE - 1).to(DEVICE)
                        elif isinstance(val_batch, dict):
                            val_input_ids = torch.clamp(val_batch['input_ids'], 0, VOCAB_SIZE - 1).to(DEVICE)
                            val_labels = torch.clamp(val_batch['labels'], 0, VOCAB_SIZE - 1).to(DEVICE)
                        else:
                            raise ValueError(f"Unsupported batch format: {type(val_batch)}. Expected tuple/list or dict.")

                        src = val_input_ids
                        tgt = val_labels

                        src_seq_len = src.size(1)
                        tgt_seq_len = tgt.size(1)
                        batch_size = src.size(0)

                        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, src_seq_len, src_seq_len)
                        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
                        tgt_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, tgt_seq_len, tgt_seq_len)
                        tgt_mask = tgt_mask & tgt_padding_mask.expand(batch_size, 1, tgt_seq_len, tgt_seq_len)
                        cross_mask = (src != 0).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, tgt_seq_len - 1, src_seq_len)

                        if DEVICE.type == "cuda":
                            with autocast():
                                output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1], cross_mask)
                                loss = loss_fn(output.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
                        else:
                            output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1], cross_mask)
                            loss = loss_fn(output.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))

                        val_loss += loss.item()
                        val_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
                        val_bar.update(1)

                avg_val_loss = val_loss / len(val_loader)
                print(f"Validation Loss: {avg_val_loss:.4f}")
            else:
                print("Validation skipped: val_loader is None or empty")

        # Save the model
        torch.save({
            'epoch': EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, "transformer_qed_sequence_full.pth")
        print("Model saved successfully as transformer_qed_sequence_full.pth")

    except Exception as e:
        print(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    train_model()