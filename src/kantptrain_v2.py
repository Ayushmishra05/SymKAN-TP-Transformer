import torch
from torch.optim import AdamW
from kantptransformer import build_transformer, KANLayer
import pickle
import sys
import time  # For timing each batch
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.amp import autocast as amp_autocast  # Updated for both CPU and GPU
from transformers import PreTrainedTokenizerFast  # For decoding the output
import sympy as sp
import re

# SymPyLayer definition
class SymPyLayer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.symbols = {
            'e': sp.Symbol('e'),
            'alpha': sp.Symbol('alpha'),
            'hbar': sp.Symbol('hbar'),
            'c': sp.Symbol('c'),
            'G': sp.Symbol('G'),
            'pi': sp.pi,
            'm_e': sp.Symbol('m_e'),
            'm_mu': sp.Symbol('m_mu'),
            'm_tau': sp.Symbol('m_tau'),
            'm_c': sp.Symbol('m_c'),
            'm_b': sp.Symbol('m_b'),
            'm_t': sp.Symbol('m_t'),
            'm_1': sp.Symbol('m_1'),
            'm_2': sp.Symbol('m_2'),
            's': sp.Symbol('s'),
            't': sp.Symbol('t'),
            'u': sp.Symbol('u'),
            's_11': sp.Symbol('s_11'),
            's_12': sp.Symbol('s_12'),
            's_13': sp.Symbol('s_13'),
            's_14': sp.Symbol('s_14'),
            's_21': sp.Symbol('s_21'),
            's_22': sp.Symbol('s_22'),
            's_23': sp.Symbol('s_23'),
            's_24': sp.Symbol('s_24'),
            's_31': sp.Symbol('s_31'),
            's_32': sp.Symbol('s_32'),
            's_33': sp.Symbol('s_33'),
            's_34': sp.Symbol('s_34'),
            's_41': sp.Symbol('s_41'),
            's_42': sp.Symbol('s_42'),
            's_43': sp.Symbol('s_43'),
            's_44': sp.Symbol('s_44'),
            'p_1': sp.Symbol('p_1'),
            'p_2': sp.Symbol('p_2'),
            'p_3': sp.Symbol('p_3'),
            'p_4': sp.Symbol('p_4'),
            'k_1': sp.Symbol('k_1'),
            'k_2': sp.Symbol('k_2'),
            'q': sp.Symbol('q'),
            'reg_prop': sp.Symbol('reg_prop'),
            'Delta': sp.Symbol('Delta'),
            'eps': sp.Symbol('eps'),
            'g': sp.Symbol('g'),
            'Z': sp.Symbol('Z'),
            'lambda': sp.Symbol('lambda'),
            'beta': sp.Symbol('beta'),
            'i': sp.I,
            'gamma': sp.Symbol('gamma'),
            'epsilon': sp.Symbol('epsilon'),
            'delta': sp.Symbol('delta'),
            'sigma': sp.Symbol('sigma'),
            'theta': sp.Symbol('theta'),
            'phi': sp.Symbol('phi'),
            'omega': sp.Symbol('omega'),
        }

    def parse_expression(self, token_ids):
        """
        Convert a sequence of token IDs into a SymPy expression, dynamically handling unknown symbols.
        Args:
            token_ids (torch.Tensor): Tensor of shape [seq_len] containing token IDs.
        Returns:
            sympy.Expr: Parsed SymPy expression, or None if parsing fails.
        """
        try:
            # Decode token IDs into a string
            expr_str = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            # Preprocess the string
            expr_str = expr_str.replace('Ġ', '')  # Remove tokenizer space marker
            expr_str = expr_str.replace(' _ ', '_')  # Fix indexed variables (e.g., s _ 33 -> s_33)
            expr_str = expr_str.replace('^', '**')  # Replace ^ with ** for SymPy compatibility
            expr_str = expr_str.strip()  # Remove leading/trailing whitespace

            # Remove trailing operators (e.g., "+", "-", "*", "/") to handle incomplete expressions
            expr_str = re.sub(r'[\+\-\*/]\s*$', '', expr_str)

            # Balance parentheses
            open_parens = expr_str.count('(')
            close_parens = expr_str.count(')')
            if open_parens > close_parens:
                expr_str += ')' * (open_parens - close_parens)
            elif close_parens > open_parens:
                expr_str = '(' * (close_parens - open_parens) + expr_str

            # Tokenize the expression string to find potential symbols
            tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*(?:_[a-zA-Z0-9]+)?\b', expr_str)
            local_symbols = self.symbols.copy()
            for token in tokens:
                if token not in local_symbols:
                    if token in dir(sp) or token in ['sin', 'cos', 'tan', 'exp', 'log']:
                        continue
                    print(f"Dynamically adding symbol: {token}")
                    local_symbols[token] = sp.Symbol(token)

            # Parse the string into a SymPy expression
            expr = sp.sympify(expr_str, locals=local_symbols)
            return expr
        except (sp.SympifyError, ValueError, TypeError, SyntaxError) as e:
            print(f"Failed to parse expression: {expr_str}. Error: {e}")
            return None

    def simplify_expression(self, token_ids):
        expr = self.parse_expression(token_ids)
        if expr is None:
            return None
        try:
            simplified_expr = sp.simplify(expr)
            return simplified_expr
        except Exception as e:
            print(f"Failed to simplify expression: {expr}. Error: {e}")
            return expr

    def compare_expressions(self, pred_token_ids, true_token_ids):
        pred_expr = self.parse_expression(pred_token_ids)
        true_expr = self.parse_expression(true_token_ids)
        if pred_expr is None or true_expr is None:
            return False
        try:
            diff = sp.simplify(pred_expr - true_expr)
            return diff == 0
        except Exception as e:
            print(f"Failed to compare expressions. Error: {e}")
            return False

# Hyperparameters (aligned with TP-Transformer and your previous setup)
EPOCHS = 5
LEARNING_RATE = 5e-5
BATCH_SIZE = 2  # Reduced to speed up training on CPU
GRADIENT_ACCUMULATION_STEPS = 2  # Adjusted to maintain effective batch size
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS  # 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 512  # Matches your tokenizer's vocab_size
MAX_LENGTH_TEXT = 509  # Updated to full sequence length
MAX_LENGTH_LABEL = 412  # Updated to full sequence length
PAD_IDX = 0  # Matches the tokenizer's [PAD] token ID
DROPOUT = 0.1  # Default dropout from TP-Transformer
FILTER_DIM = 2048
N_LAYERS = 6
N_HEADS = 8
HIDDEN_DIM = 512
LAMBDA_L2 = 1e-5  # Added: L2 regularization strength for KAN layers

class HyperParams:
    def __init__(self):
        self.input_dim = VOCAB_SIZE
        self.filter = FILTER_DIM
        self.n_layers = N_LAYERS
        self.n_heads = N_HEADS
        self.hidden = HIDDEN_DIM
        self.dropout = DROPOUT

params = HyperParams()

def augment_expression(expr_str):
    """
    Augment a HEP expression by reordering commutative terms (e.g., a + b -> b + a).
    This is a simple example; you can expand it based on HEP-specific rules.
    Args:
        expr_str (str): The expression string to augment.
    Returns:
        str: Augmented expression string.
    """
    try:
        # Split the expression into terms based on '+' (assuming addition is commutative)
        terms = expr_str.split('+')
        if len(terms) > 1:
            # Reverse the order of terms (simple augmentation)
            augmented_terms = terms[::-1]
            augmented_expr = '+'.join(augmented_terms)
            return augmented_expr
        return expr_str
    except Exception as e:
        print(f"Failed to augment expression {expr_str}: {e}")
        return expr_str

def train_model():
    try:
        # Load the tokenizer for decoding (needed for the forward pass)
        tokenizer = PreTrainedTokenizerFast.from_pretrained("src/tokenizer/QED_tokenizer")
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "[PAD]"

        # Initialize SymPyLayer
        sympy_layer = SymPyLayer(tokenizer)

        # Initialize model and move to device
        model = build_transformer(params, pad_idx=PAD_IDX)
        # Optimize KANLayer for faster training
        model.kan_layer = KANLayer(in_dim=HIDDEN_DIM, out_dim=VOCAB_SIZE, num_knots=3, spline_order=2)
        model.to(DEVICE)
        print(f"Model initialized on {DEVICE}")

        # Warn if running on CPU
        if DEVICE.type == "cpu":
            print("WARNING: Running on CPU. Training may be slow. Consider using a GPU for full sequence length (509).")

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
        model.eval()
        with torch.no_grad():
            # Get a single batch from the train_loader
            batch = next(iter(train_loader))
            
            # Determine batch structure (tuple or dictionary)
            if isinstance(batch, (tuple, list)):
                input_ids = torch.clamp(batch[0], 0, VOCAB_SIZE - 1).to(DEVICE)  # Removed clamping to 200
                labels = torch.clamp(batch[2], 0, VOCAB_SIZE - 1).to(DEVICE)  # Removed clamping to 200
            elif isinstance(batch, dict):
                input_ids = torch.clamp(batch['input_ids'], 0, VOCAB_SIZE - 1).to(DEVICE)  # Removed clamping to 200
                labels = torch.clamp(batch['labels'], 0, VOCAB_SIZE - 1).to(DEVICE)  # Removed clamping to 200
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

            # SymPy formalization
            simplified_pred = sympy_layer.simplify_expression(predicted_ids[0])
            simplified_true = sympy_layer.simplify_expression(labels[0])
            print(f"Simplified Predicted Expression: {simplified_pred}")
            print(f"Simplified True Expression: {simplified_true}")
            is_equivalent = sympy_layer.compare_expressions(predicted_ids[0], labels[0])
            print(f"Predicted and True Expressions Equivalent: {is_equivalent}")

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
                for batch_idx, batch in enumerate(train_loader):
                    start_time = time.time()  # Start timing the batch

                    # Determine batch structure (tuple or dictionary)
                    if isinstance(batch, (tuple, list)):
                        input_ids = torch.clamp(batch[0], 0, VOCAB_SIZE - 1).to(DEVICE)  # Removed clamping to 200
                        labels = torch.clamp(batch[2], 0, VOCAB_SIZE - 1).to(DEVICE)  # Removed clamping to 200
                    elif isinstance(batch, dict):
                        input_ids = torch.clamp(batch['input_ids'], 0, VOCAB_SIZE - 1).to(DEVICE)  # Removed clamping to 200
                        labels = torch.clamp(batch['labels'], 0, VOCAB_SIZE - 1).to(DEVICE)  # Removed clamping to 200
                    else:
                        raise ValueError(f"Unsupported batch format: {type(batch)}. Expected tuple/list or dict.")

                    # Forward pass with mixed precision
                    with amp_autocast(device_type=DEVICE.type):
                        logits = model(input_ids, labels[:, :-1])  # Shape: [batch_size, seq_len, vocab_size]
                        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), labels[:, 1:].reshape(-1))

                        # Add L2 regularization from all KAN layers
                        l2_reg = 0
                        # KAN layer at the output stage
                        l2_reg += model.kan_layer.get_l2_regularization()
                        # KAN layers in the encoder
                        for layer in model.encoder.layers:
                            l2_reg += layer.densefilter.get_l2_regularization()
                        # KAN layers in the decoder
                        for layer in model.decoder.layers:
                            l2_reg += layer.densefilter.get_l2_regularization()
                        # Add L2 regularization to the loss
                        total_loss_batch = loss + LAMBDA_L2 * l2_reg

                    total_loss_batch = total_loss_batch / GRADIENT_ACCUMULATION_STEPS  # Scale the loss for accumulation

                    # Backward pass
                    if DEVICE.type == "cuda":
                        scaler.scale(total_loss_batch).backward()
                    else:
                        total_loss_batch.backward()

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
                    progress_bar.set_postfix({'loss': f'{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}', 'l2_reg': f'{l2_reg.item():.4f}'})

                    # SymPy formalization (every 100 batches to avoid slowdown)
                    if (batch_idx + 1) % 100 == 0:
                        with torch.no_grad():
                            predicted_ids = torch.argmax(logits, dim=-1)
                            simplified_pred = sympy_layer.simplify_expression(predicted_ids[0])
                            simplified_true = sympy_layer.simplify_expression(labels[0])
                            is_equivalent = sympy_layer.compare_expressions(predicted_ids[0], labels[0])
                            print(f"\nBatch {batch_idx+1}:")
                            print(f"Simplified Predicted Expression: {simplified_pred}")
                            print(f"Simplified True Expression: {simplified_true}")
                            print(f"Predicted and True Expressions Equivalent: {is_equivalent}")

                    # Print time taken for this batch
                    batch_time = time.time() - start_time
                    print(f"Batch {batch_idx+1}/{len(train_loader)} completed in {batch_time:.2f} seconds")
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

            # Validation with symbolic accuracy
            model.eval()
            val_loss = 0
            correct_symbolic = 0
            total_symbolic = 0
            if val_loader is not None and len(val_loader) > 0:
                with torch.no_grad(), tqdm(total=len(val_loader), desc="Validation", file=sys.stdout) as val_bar:
                    for val_batch in val_loader:
                        # Determine batch structure
                        if isinstance(val_batch, (tuple, list)):
                            val_input_ids = torch.clamp(val_batch[0], 0, VOCAB_SIZE - 1).to(DEVICE)  # Removed clamping to 200
                            val_labels = torch.clamp(val_batch[2], 0, VOCAB_SIZE - 1).to(DEVICE)  # Removed clamping to 200
                        elif isinstance(val_batch, dict):
                            val_input_ids = torch.clamp(val_batch['input_ids'], 0, VOCAB_SIZE - 1).to(DEVICE)  # Removed clamping to 200
                            val_labels = torch.clamp(val_batch['labels'], 0, VOCAB_SIZE - 1).to(DEVICE)  # Removed clamping to 200
                        else:
                            raise ValueError(f"Unsupported batch format: {type(val_batch)}. Expected tuple/list or dict.")

                        # Forward pass (validation) with mixed precision
                        with amp_autocast(device_type=DEVICE.type):
                            val_logits = model(val_input_ids, val_labels[:, :-1])
                            val_loss += loss_fn(val_logits.reshape(-1, VOCAB_SIZE), val_labels[:, 1:].reshape(-1)).item()

                        # Compute symbolic accuracy
                        predicted_ids = torch.argmax(val_logits, dim=-1)
                        for pred, true in zip(predicted_ids, val_labels):
                            if sympy_layer.compare_expressions(pred, true):
                                correct_symbolic += 1
                            total_symbolic += 1

                        val_bar.set_postfix({'val_loss': f'{val_loss / (val_bar.n + 1):.4f}'})
                        val_bar.update(1)

                avg_val_loss = val_loss / len(val_loader)
                symbolic_accuracy = correct_symbolic / total_symbolic * 100 if total_symbolic > 0 else 0
                print(f"Validation Loss: {avg_val_loss:.4f}")
                print(f"Symbolic Accuracy: {symbolic_accuracy:.2f}%")
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