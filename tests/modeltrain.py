# D:\DecoderKAN\tests\modeltrain.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, TrainingArguments, Trainer
from tests.tokeni import QEDHuggingFaceTokenizer
from src.parse_and_compare import parse_and_compare
from preprocess.tokenizersplit import reconstruct_expression  # Import the standalone function
from datasets import Dataset as HFDataset
import numpy as np
import json
from nltk.translate.bleu_score import sentence_bleu
from Levenshtein import distance as levenshtein_distance

# Hyperparameters
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
MAX_LENGTH = 509
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "hf_transformer_output"

class QEDDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LENGTH):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src_expr = self.df['amp'].iloc[idx]
        tgt_expr = self.df['sqamp'].iloc[idx]
        src_ids = self.tokenizer.encode(src_expr, is_source=True, max_length=self.max_len)
        tgt_ids = self.tokenizer.encode(tgt_expr, is_source=False, max_length=self.max_len)
        src_attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in src_ids]
        tgt_attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in tgt_ids]
        return {
            'input_ids': torch.tensor(src_ids, dtype=torch.long),
            'labels': torch.tensor(tgt_ids, dtype=torch.long),
            'attention_mask': torch.tensor(src_attention_mask, dtype=torch.long),
            'decoder_attention_mask': torch.tensor(tgt_attention_mask, dtype=torch.long),
            'tgt_expr': tgt_expr,
            'tgt_ids': torch.tensor(tgt_ids, dtype=torch.long)  # Add for token-level accuracy
        }

class CustomTrainer(Trainer):
    def __init__(self, *args, val_dataset_with_expr=None, custom_tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_dataset_with_expr = val_dataset_with_expr
        self.custom_tokenizer = custom_tokenizer
        self.processing_class = custom_tokenizer
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        if self.val_dataset_with_expr is not None:
            metrics = self.compute_symbolic_accuracy()
            output.update({
                'eval_symbolic_accuracy': metrics['symbolic_accuracy'],
                'eval_token_accuracy': metrics['token_accuracy'],
                'eval_bleu_score': metrics['bleu_score'],
                'eval_edit_distance': metrics['edit_distance']
            })
        return output
    
    def compute_symbolic_accuracy(self):
        self.model.eval()
        symbolic_accuracies = []
        token_accuracies = []
        bleu_scores = []
        edit_distances = []
        max_samples = min(20, len(self.val_dataset_with_expr))
        
        print(f"Evaluating metrics on {max_samples} samples...")
        
        with torch.no_grad():
            for i in range(max_samples):
                try:
                    sample = self.val_dataset_with_expr[i]
                    src = sample['input_ids'].unsqueeze(0).to(DEVICE)
                    attention_mask = sample['attention_mask'].unsqueeze(0).to(DEVICE)
                    gt_expr = sample['tgt_expr']
                    gt_ids = sample['tgt_ids'].tolist()
                    
                    outputs = self.model.generate(
                        src,
                        attention_mask=attention_mask,
                        max_length=min(MAX_LENGTH, 128),
                        min_length=5,
                        num_beams=3,
                        early_stopping=True,
                        no_repeat_ngram_size=2,
                        decoder_start_token_id=self.custom_tokenizer.bos_token_id,
                        eos_token_id=self.custom_tokenizer.eos_token_id,
                        pad_token_id=self.custom_tokenizer.pad_token_id,
                        do_sample=False,
                        temperature=1.0
                    )
                    
                    pred_ids = outputs[0].tolist()
                    try:
                        # Symbolic accuracy
                        symbolic_match = parse_and_compare(
                            gt_expr, pred_ids, self.custom_tokenizer.tgt_vocab, self.custom_tokenizer.qed_tokenizer
                        )
                        symbolic_accuracies.append(int(symbolic_match))
                        
                        # Token-level accuracy
                        min_len = min(len(pred_ids), len(gt_ids))
                        correct_tokens = sum(1 for p, g in zip(pred_ids[:min_len], gt_ids[:min_len]) if p == g)
                        token_accuracy = correct_tokens / max(len(gt_ids), 1)
                        token_accuracies.append(token_accuracy)
                        
                        # BLEU score
                        pred_tokens = self.custom_tokenizer.decode(pred_ids, skip_special_tokens=True)
                        gt_tokens = self.custom_tokenizer.decode(gt_ids, skip_special_tokens=True)
                        bleu = sentence_bleu([gt_tokens], pred_tokens, weights=(0.5, 0.5))
                        bleu_scores.append(bleu)
                        
                        # Edit distance
                        edit_dist = levenshtein_distance(''.join(pred_tokens), ''.join(gt_tokens))
                        edit_distances.append(edit_dist)
                        
                        # Debug: Print first few examples
                        if i < 3:
                            decoded_tokens = self.custom_tokenizer.decode(pred_ids, skip_special_tokens=True)
                            predicted_expr = reconstruct_expression(decoded_tokens)  # Use standalone function
                            print(f"Sample {i} - GT: {gt_expr}")
                            print(f"Sample {i} - Predicted: {predicted_expr}")
                            print(f"Sample {i} - Symbolic Match: {symbolic_match}")
                            print(f"Sample {i} - Token Accuracy: {token_accuracy:.2%}")
                            print(f"Sample {i} - BLEU Score: {bleu:.4f}")
                            print(f"Sample {i} - Edit Distance: {edit_dist}")
                            
                    except Exception as e:
                        print(f"Sample {i} - Parsing failed: {str(e)[:100]}")
                        symbolic_accuracies.append(0)
                        token_accuracies.append(0)
                        bleu_scores.append(0)
                        edit_distances.append(float('inf'))
                        
                except Exception as e:
                    print(f"Error processing sample {i}: {str(e)[:100]}")
                    symbolic_accuracies.append(0)
                    token_accuracies.append(0)
                    bleu_scores.append(0)
                    edit_distances.append(float('inf'))
        
        metrics = {
            'symbolic_accuracy': np.mean(symbolic_accuracies) * 100 if symbolic_accuracies else 0.0,
            'token_accuracy': np.mean(token_accuracies) * 100 if token_accuracies else 0.0,
            'bleu_score': np.mean(bleu_scores) if bleu_scores else 0.0,
            'edit_distance': np.mean([d for d in edit_distances if d != float('inf')]) if any(d != float('inf') for d in edit_distances) else float('inf')
        }
        print(f"Metrics: Symbolic Accuracy: {metrics['symbolic_accuracy']:.2f}%, "
              f"Token Accuracy: {metrics['token_accuracy']:.2f}%, "
              f"BLEU Score: {metrics['bleu_score']:.4f}, "
              f"Edit Distance: {metrics['edit_distance']:.2f}")
        return metrics

def compute_metrics(eval_pred):
    return {"placeholder_metric": 0.0}

def main():
    print("Starting training...")
    
    data_path = os.path.join("QED_data", "train_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_csv(data_path).iloc[:100]
    print(f"Loaded {len(df)} samples")

    tokenizer = QEDHuggingFaceTokenizer(df=df, max_length=MAX_LENGTH)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Vocabulary size: {len(tokenizer.get_vocab())}")

    dataset = QEDDataset(df, tokenizer, MAX_LENGTH)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"Train samples: {train_size}, Val samples: {val_size}")

    val_dataset_with_expr = val_dataset

    def convert_to_hf_dataset(torch_dataset):
        data = {'input_ids': [], 'labels': [], 'attention_mask': [], 'decoder_attention_mask': [], 'tgt_expr': []}
        for item in torch_dataset:
            data['input_ids'].append(item['input_ids'].tolist())
            data['labels'].append(item['labels'].tolist())
            data['attention_mask'].append(item['attention_mask'].tolist())
            data['decoder_attention_mask'].append(item['decoder_attention_mask'].tolist())
            data['tgt_expr'].append(item['tgt_expr'])
        return HFDataset.from_dict(data)

    hf_train_dataset = convert_to_hf_dataset(train_dataset)
    hf_val_dataset = convert_to_hf_dataset(val_dataset)

    vocab_size = len(tokenizer.get_vocab())
    encoder_config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=MAX_LENGTH,
        pad_token_id=tokenizer.pad_token_id,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    decoder_config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=MAX_LENGTH,
        pad_token_id=tokenizer.pad_token_id,
        is_decoder=True,
        add_cross_attention=True,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    
    config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    model = EncoderDecoderModel(config=config)
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_symbolic_accuracy",
        greater_is_better=True,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        fp16=DEVICE.type == "cuda",
        report_to="none",
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        dataloader_num_workers=0,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        compute_metrics=compute_metrics,
        val_dataset_with_expr=val_dataset_with_expr,
        custom_tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print("Training completed. Saving results...")
    with open(os.path.join(OUTPUT_DIR, "training_results.txt"), "w") as f:
        eval_results = trainer.evaluate()
        f.write(f"Final Symbolic Accuracy: {eval_results['eval_symbolic_accuracy']:.2f}%\n")
        f.write(f"Final Token Accuracy: {eval_results['eval_token_accuracy']:.2f}%\n")
        f.write(f"Final BLEU Score: {eval_results['eval_bleu_score']:.4f}\n")
        f.write(f"Final Edit Distance: {eval_results['eval_edit_distance']:.2f}\n")
        f.write(f"Final Loss: {eval_results.get('eval_loss', 'N/A')}\n")
        f.write(f"Target: ~48.67% (Proposal Page 9)\n")

    print("\nTesting individual samples...")
    model.eval()
    for sample_idx in [0, 1, 2]:
        if sample_idx < len(dataset):
            sample_data = dataset[sample_idx]
            src = sample_data['input_ids'].unsqueeze(0).to(DEVICE)
            attention_mask = sample_data['attention_mask'].unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model.generate(
                    src,
                    attention_mask=attention_mask,
                    max_length=128,
                    num_beams=3,
                    early_stopping=True,
                    decoder_start_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False
                )
            
            try:
                decoded_tokens = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
                recon_expr = reconstruct_expression(decoded_tokens)
                symbolic_match = parse_and_compare(sample_data['tgt_expr'], outputs[0].tolist(), tokenizer.tgt_vocab, tokenizer.qed_tokenizer)
                print(f"Index {sample_idx} - Ground Truth: {sample_data['tgt_expr']}")
                print(f"Index {sample_idx} - Predicted: {recon_expr}")
                print(f"Index {sample_idx} - Symbolic Equivalence: {symbolic_match}")
            except Exception as e:
                print(f"Index {sample_idx} - Error: {str(e)[:100]}")
            print()

    with open(os.path.join(OUTPUT_DIR, "tgt_vocab.json"), "w") as f:
        json.dump(tokenizer.get_vocab(), f, ensure_ascii=False, indent=2)
    
    print("Training and evaluation complete!")

if __name__ == "__main__":
    main()