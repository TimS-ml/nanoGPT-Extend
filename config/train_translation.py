# Train an Encoder-Decoder Transformer for translation
# Uses the Multi30k German-English dataset
#
# Example usage:
#   python src/train_translation.py config/train_translation.py
#
# For CPU-only training (e.g., MacBook):
#   python src/train_translation.py config/train_translation.py --device=cpu --compile=False

out_dir = 'out-translation'
eval_interval = 500
eval_iters = 100
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'translation'
wandb_run_name = 'transformer-de-en'

# Dataset
dataset = 'multi30k'
language_pair = ('de', 'en')  # German to English

# Training
gradient_accumulation_steps = 10  # Accumulate gradients
batch_size = 32
max_padding = 72  # Maximum sequence length

# Model architecture (Annotated Transformer defaults)
# Using Annotated Transformer naming with nanoGPT equivalents noted
N = 6           # n_layer: Number of encoder/decoder layers
d_model = 512   # n_embd: Model dimension
d_ff = 2048     # Feed-forward hidden dimension (4 * d_model by default)
h = 8           # n_head: Number of attention heads
dropout = 0.1

# Optimizer (Noam scheduler as in original paper)
learning_rate = 1.0  # Will be scaled by Noam scheduler
warmup = 3000        # Warmup steps for Noam scheduler
beta1 = 0.9
beta2 = 0.98
eps = 1e-9

# Training iterations
num_epochs = 8
max_iters = None  # If set, overrides num_epochs

# Label smoothing (improves BLEU score)
label_smoothing = 0.1

# Distributed training
distributed = False

# Checkpoint
file_prefix = 'multi30k_model_'

# Device settings (uncomment for CPU-only)
# device = 'cpu'
# compile = False


# =============================================================================
# Alternative: Smaller model for quick testing / debugging
# =============================================================================
# Uncomment below for a smaller model that trains faster

# N = 2           # Fewer layers
# d_model = 256   # Smaller model dimension
# d_ff = 1024     # Smaller FF dimension
# h = 4           # Fewer heads
# batch_size = 80
# warmup = 400
# num_epochs = 20
