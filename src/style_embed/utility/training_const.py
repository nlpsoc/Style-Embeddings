from transformers import BertTokenizer
from global_const import set_torch_device

# binary vs. constrastive AV task
TRIPLET_EVALUATOR = "triplet"
BINARY_EVALUATOR = "binary"

# Loss keywords
TRIPLET_LOSS = "triplet"
CONTRASTIVE_ONLINE_LOSS = "contrastive-online"
CONTRASTIVE_LOSS = "contrastive"
COSINE_LOSS = "cosine"

# considered models keywords
BERT_CASED_BASE_MODEL = "bert-base-cased"
BERT_UNCASED_BASE_MODEL = "bert-base-uncased"
ROBERTA_BASE = 'roberta-base'

# tokenizers
UNCASED_TOKENIZER = BertTokenizer.from_pretrained(BERT_UNCASED_BASE_MODEL)
CASED_TOKENIZER = BertTokenizer.from_pretrained(BERT_CASED_BASE_MODEL)

device = set_torch_device()

# Default parameters for training the BERT-like models
BATCH_SIZE = 8  # 16  # 64  # 256 # 16
EVAL_BATCH_SIZE = 4
EPOCHS = 4
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
LEARNING_RATE = 0.00002  # default was 5e-5
EVALUATION_STEPS = 0  # 2100  # 500
MARGIN = 0.5
EVALUATION_STRATEGY = "epoch"
OUTPUT_LABELS = 2  # The number of output labels--2 for binary classification.; 2 if not regression
EPS = 1e-8
CORRECT_BIAS = True  # default: false
