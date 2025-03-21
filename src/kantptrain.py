import torch
from torch.optim import AdamW
from kantptransformer import build_transformer  # Updated with KANLayer
import pickle
import sys
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training on GPU
from torch.amp import autocast as amp_autocast  # Updated for both CPU and GPU
from transformers import PreTrainedTokenizerFast  # For decoding the output

# Hyperparameters (aligned with TP-Transformer and your previous setup)
EPOCHS = 5
LEARNING_RATE = 5e-5
BATCH_SIZE = 16  # Using gradient accumulation as before
GRADIENT_ACCUMULATION_STEPS = 4  # Process 4 samples per step (16 / 4 = 4)
EFFECTIVE_BATCH_SIZE = BATCH_SIZE  # Effective batch size after accumulation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 512  # Matches your tokenizer's vocab_size
MAX_LENGTH_TEXT = 509  # 99th percentile for text
MAX_LENGTH_LABEL = 412  # Max length for label
PAD_IDX = 0  # Matches the tokenizer's [PAD] token ID
DROPOUT = 0.1  # Default dropout from TP-Transformer
FILTER_DIM = 2048
N_LAYERS = 6
N_HEADS = 8
HIDDEN_DIM = 512

class HyperParams:
    def __init__(self):
        self.input_dim = VOCAB_SIZE
        self.filter = FILTER_DIM
        self.n_layers = N_LAYERS
        self.n_heads = N_HEADS
        self.hidden = HIDDEN_DIM
        self.dropout = DROPOUT

params = HyperParams()

def train_model():
    try:
        # Load the tokenizer for decoding (needed for the forward pass)
        tokenizer = PreTrainedTokenizerFast.from_pretrained("src/tokenizer/QED_tokenizer")
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "[PAD]"

        # Initialize model and move to device
        model = build_transformer(params, pad_idx=PAD_IDX)  # max_seq_len not needed as per updated build_transformer
        model.to(DEVICE)
        print(f"Model initialized on {DEVICE}")

        # Warn if running on CPU
        if DEVICE.type == "cpu":
            print("WARNING: Running on CPU. Training may be very slow due to large sequence lengths (509 and 421). Consider using a GPU for significant speedup.")

        # Optimizer and loss function (cross-entropy for token classification)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        # Mixed precision training setup
        scaler = GradScaler() if DEVICE.type == "cuda" else None

        # Load data
        with open(r'src/Dataloaders/train_loader.pkl', 'rb') as f:
            train_loader = pickle.load(f)
        with open(r'src/Dataloaders/val_loader.pkl', 'rb') as f:
            val_loader = pickle.load(f)

        # Basic Forward Pass to Verify the Flow
        print("\nPerforming a basic forward pass to verify the flow...")
        model.eval()  # Set to evaluation mode for the forward pass
        with torch.no_grad():
            # Get a single batch from the train_loader
            batch = next(iter(train_loader))
            
            # Determine batch structure (tuple or dictionary)
            if isinstance(batch, (tuple, list)):
                # Tuple format: (input_ids, attention_mask, labels)
                input_ids = torch.clamp(batch[0], 0, VOCAB_SIZE - 1).to(DEVICE)
                labels = torch.clamp(batch[2], 0, VOCAB_SIZE - 1).to(DEVICE)
            elif isinstance(batch, dict):
                # Dictionary format: {'input_ids': ..., 'labels': ...}
                input_ids = torch.clamp(batch['input_ids'], 0, VOCAB_SIZE - 1).to(DEVICE)
                labels = torch.clamp(batch['labels'], 0, VOCAB_SIZE - 1).to(DEVICE)
            else:
                raise ValueError(f"Unsupported batch format: {type(batch)}. Expected tuple/list or dict.")

            # Verify input shapes
            print(f"Input IDs shape: {input_ids.shape}")  # Should be [BATCH_SIZE, MAX_LENGTH_TEXT]
            print(f"Labels shape: {labels.shape}")  # Should be [BATCH_SIZE, MAX_LENGTH_LABEL]

            # Forward pass with mixed precision
            with amp_autocast(device_type=DEVICE.type):
                logits = model(input_ids, labels[:, :-1])  # Shape: [batch_size, seq_len, vocab_size]
                loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), labels[:, 1:].reshape(-1))

            print(f"Forward pass completed successfully!")
            print(f"Loss: {loss.item()}")
            print(f"Output shape: {logits.shape}")  # Should be [BATCH_SIZE, MAX_LENGTH_LABEL-1, VOCAB_SIZE]

            # Decode the predicted output (argmax over logits)
            predicted_ids = torch.argmax(logits, dim=-1)  # Shape: [BATCH_SIZE, MAX_LENGTH_LABEL-1]
            predicted_label = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            true_label = tokenizer.decode(labels[0], skip_special_tokens=True)

            print(f"Predicted label (decoded): {predicted_label}")
            print(f"True label (decoded): {true_label}")

            # Extract interpretability information
            interpretability_info = model.get_interpretability(input_ids, labels[:, :-1])
            print("Interpretability Info:")
            print(f"Inner Outputs Shape: {interpretability_info['inner_outputs'].shape}")  # [batch_size, seq_len, hidden_dim]
            print(f"Contributions Shape: {interpretability_info['contributions'].shape}")  # [batch_size, seq_len, vocab_size, hidden_dim]

            # Analyze contributions for the first sequence, first position
            seq_idx = 0
            pos_idx = 0
            predicted_token = predicted_ids[seq_idx, pos_idx].item()
            contributions = interpretability_info['contributions'][seq_idx, pos_idx, predicted_token]  # Shape: [hidden_dim]
            top_features = torch.argsort(contributions, descending=True)[:5]
            print(f"\nTop contributing features for predicted token {predicted_token} at position {pos_idx}:")
            for feature_idx in top_features:
                print(f"Feature {feature_idx.item()}: Contribution {contributions[feature_idx].item():.4f}")

        # Training loop
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            step = 0  # Counter for gradient accumulation steps
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}", file=sys.stdout) as progress_bar:
                for batch in train_loader:
                    # Determine batch structure (tuple or dictionary)
                    if isinstance(batch, (tuple, list)):
                        # Tuple format: (input_ids, attention_mask, labels)
                        input_ids = torch.clamp(batch[0], 0, VOCAB_SIZE - 1).to(DEVICE)
                        labels = torch.clamp(batch[2], 0, VOCAB_SIZE - 1).to(DEVICE)
                    elif isinstance(batch, dict):
                        # Dictionary format: {'input_ids': ..., 'labels': ...}
                        input_ids = torch.clamp(batch['input_ids'], 0, VOCAB_SIZE - 1).to(DEVICE)
                        labels = torch.clamp(batch['labels'], 0, VOCAB_SIZE - 1).to(DEVICE)
                    else:
                        raise ValueError(f"Unsupported batch format: {type(batch)}. Expected tuple/list or dict.")

                    # Forward pass with mixed precision
                    with amp_autocast(device_type=DEVICE.type):
                        logits = model(input_ids, labels[:, :-1])  # Shape: [batch_size, seq_len, vocab_size]
                        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), labels[:, 1:].reshape(-1))

                    loss = loss / GRADIENT_ACCUMULATION_STEPS  # Scale the loss for accumulation

                    # Backward pass
                    if DEVICE.type == "cuda":
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    step += 1
                    if step % GRADIENT_ACCUMULATION_STEPS == 0:
                        if DEVICE.type == "cuda":
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                        optimizer.zero_grad()  # Reset gradients after optimization step
                        step = 0  # Reset step counter

                    total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS  # Unscale the loss for logging
                    progress_bar.set_postfix({'loss': f'{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}'})
                    progress_bar.update(1)

            # Perform final optimization step if there are remaining gradients
            if step > 0:
                if DEVICE.type == "cuda":
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

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

                        # Forward pass (validation) with mixed precision
                        with amp_autocast(device_type=DEVICE.type):
                            val_logits = model(val_input_ids, val_labels[:, :-1])
                            val_loss += loss_fn(val_logits.reshape(-1, VOCAB_SIZE), val_labels[:, 1:].reshape(-1)).item()
                        val_bar.set_postfix({'val_loss': f'{val_loss / (val_bar.n + 1):.4f}'})
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