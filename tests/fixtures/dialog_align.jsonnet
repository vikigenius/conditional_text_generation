local SEED = 0;
local READER = "seq2seq";
local CUDA = 0;

local EMBEDDING_DIM = 300;
local HIDDEN_DIM = 512;
local LATENT_DIM = 128;
local BATCH_SIZE = 128;
local NUM_LAYERS = 1;
local BIDIRECTIONAL = true;

local NUM_EPOCHS = 30;
local PATIENCE = 5;
local SUMMARY_INTERVAL = 10;
local GRAD_CLIPPING = 5;
local GRAD_NORM = 5;
local SHOULD_LOG_PARAMETER_STATISTICS = false;
local SHOULD_LOG_LEARNING_RATE = true;
local OPTIMIZER = "adam";
local LEARNING_RATE = 0.001;
local INIT_UNIFORM_RANGE_AROUND_ZERO = 0.1;

{
  "random_seed": SEED,
  "numpy_seed": SEED,
  "pytorch_seed": SEED,
  "dataset_reader": {
    "type": READER
  },
  "vocabulary": {
    "directory_path": "models/dialog_vae/vocabulary"
  },
  "train_data_path": "tests/fixtures/dialog_samples.tsv",
  "validation_data_path": "tests/fixtures/dialog_samples.tsv",
  "model": {
    "type": "dialog_aligner",
    "variational_encoder": {
      "_pretrained": {
        "archive_file": "models/dialog_vae/model.tar.gz",
        "module_path": '_encoder',
        "freeze": true
      },
    },
    "decoder": {
      "_pretrained": {
        "archive_file": "models/dialog_vae/model.tar.gz",
        "module_path": '_decoder',
        "freeze": true,
      },
    },
    "aligner": {
      "type": "gaussian_aligner",
      "latent_dim": LATENT_DIM,
      "hidden_dim": LATENT_DIM,
      "activation": "relu"
    },
    "temperature": 1.0,
  },
  "iterator": {
    "type": "bucket",
    "batch_size" : BATCH_SIZE,
    "sorting_keys": [["source_tokens", "num_tokens"]]
  },
  "trainer": {
    "type": 'callback',
    "num_epochs": NUM_EPOCHS,
    "cuda_device": CUDA,
    "optimizer": {
      "type": OPTIMIZER,
      "lr": LEARNING_RATE
    },
    "callbacks": [
      "generate_training_batches",
      {"type": "train_supervised", "grad_norm": GRAD_NORM, "grad_clipping": GRAD_CLIPPING},
      "checkpoint",
      {"type": "track_metrics", "patience": PATIENCE, "validation_metric": "+F-BLEU"},
      "validate",
      "generate_dialog_samples",
      "log_to_tensorboard"
    ]
  }
}
