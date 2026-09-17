from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 512
    batch_size: int = 12
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    vocab_size: int = 50257
    learning_rate: float = 3e-4
    max_epochs: int = 2

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head
