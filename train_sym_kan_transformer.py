import argparse
import sys
from src.sym_kan_transformer.train import main
from src.sym_kan_transformer.config import Config

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the KANformer model with customizable parameters.")
    
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension (default: 512)")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of layers (default: 3)")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads (default: 8)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")
    parser.add_argument("--d_ff", type=int, default=4096, help="Feedforward dimension (default: 4096)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (default: cuda if available, else cpu)")
    parser.add_argument("--max_length", type=int, default=300, help="Maximum sequence length (default: 300)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--token_pool_size", type=int, default=100, help="Token pool size (default: 100)")
    parser.add_argument("--unk_idx", type=int, default=1, help="Unknown index (default: 1)")
    parser.add_argument("--to_replace", type=bool, default=True, help="Whether to replace indices (default: True)")
    parser.add_argument("--data_path", type=str, default=r'D:\DecoderKAN\QED_data\test-flow.csv', help="Path to data CSV (default: D:\\DecoderKAN\\QED_data\\test-flow.csv)")

    args = parser.parse_args()
    return args

def train_from_args():
    args = parse_arguments()

    config = Config(
        D_MODEL=args.d_model,
        N_LAYERS=args.n_layers,
        N_HEADS=args.n_heads,
        DROPOUT=args.dropout,
        D_FF=args.d_ff,
        DEVICE=args.device,
        MAX_LENGTH=args.max_length,
        EPOCHS=args.epochs,
        LR=args.lr,
        BATCH_SIZE=args.batch_size,
        INDEX_TOKEN_POOL_SIZE=args.token_pool_size,
        UNK_IDX=args.unk_idx,
        TO_REPLACE=args.to_replace,
        DATA_PATH=args.data_path
    )

    print("Using configuration:")
    print(config)

    main(config)

if __name__ == "__main__":
    train_from_args()