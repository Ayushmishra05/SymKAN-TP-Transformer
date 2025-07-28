import torch

class Config:
    def __init__(self, D_MODEL: int = 512, N_LAYERS: int = 3, N_HEADS: int = 8, DROPOUT: float = 0.1,
                 D_FF: int = 4096,  DEVICE: str = None,
                 MAX_LENGTH: int = 300, EPOCHS: int = 10, LR: float = 1e-4, BATCH_SIZE: int = 16,
                 INDEX_TOKEN_POOL_SIZE: int = 100,
                 UNK_IDX: int = 1, TO_REPLACE: bool = True, DATA_PATH: str = r'D:\DecoderKAN\QED_data\test-flow.csv'):
        
        self.D_MODEL = D_MODEL
        self.N_LAYERS = N_LAYERS
        self.N_HEADS = N_HEADS
        self.DROPOUT = DROPOUT
        self.D_FF = D_FF
        self.FF_DIMS = [8192] 
        self.DEVICE = DEVICE if DEVICE is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.MAX_LENGTH = MAX_LENGTH
        self.EPOCHS = EPOCHS
        self.LR = LR
        self.BATCH_SIZE = BATCH_SIZE
        self.INDEX_TOKEN_POOL_SIZE = INDEX_TOKEN_POOL_SIZE
        self.SPECIAL_SYMBOLS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"]
        self.UNK_IDX = UNK_IDX
        self.TO_REPLACE = TO_REPLACE
        self.DATA_PATH = DATA_PATH

    @classmethod
    def update(cls, **kwargs):
        instance = cls()
        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance

    def __str__(self):
        return "\n".join(f"{key}: {value}" for key, value in self.__dict__.items())