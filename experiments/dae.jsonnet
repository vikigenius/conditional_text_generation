local SEED = 0;
local READER = "autoencoder";
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

local ANNEAL_MIN_WEIGHT = 0.0;
local ANNEAL_MAX_WEIGHT = 0.5;
local ANNEAL_WARMUP = 500;
local ANNEAL_NUM_ITER_TO_MAX = 3500;
local ANNEAL_SLOPE = 0.5;

{
  "random_seed": SEED,
  "numpy_seed": SEED,
  "pytorch_seed": SEED,
  "dataset_reader": {
    "type": READER
  },
  "train_data_path": "data/interim/dialog/train_sentences.tsv",
  "validation_data_path": "data/interim/dialog/valid_sentences.tsv",
  "model": {
    "type": "dae",
    "deterministic_encoder": {
      "type": "deterministic",
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "vocab_namespace": "tokens",
            "embedding_dim": EMBEDDING_DIM,
          }
        }
      },
      "encoder": {
        "type": "lstm",
        "input_size": EMBEDDING_DIM,
        "hidden_size": HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        "bidirectional": BIDIRECTIONAL,
      },
      "latent_dim": LATENT_DIM,
    },
    "decoder": {
      "type": "variational_decoder",
      "target_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "vocab_namespace": "tokens",
            "embedding_dim": EMBEDDING_DIM,
          }
        }
      },
      "rnn": {
        "type": "lstm",
        "input_size": EMBEDDING_DIM + LATENT_DIM,
        'num_layers': NUM_LAYERS,
        "hidden_size": HIDDEN_DIM,
      },
      "latent_dim": LATENT_DIM
    },
    "kl_weight": {
      "type": "sigmoid_annealed",
      "min_weight": ANNEAL_MIN_WEIGHT,
      "max_weight": ANNEAL_MAX_WEIGHT,
      "warmup": ANNEAL_WARMUP,
      "num_iter_to_max": ANNEAL_NUM_ITER_TO_MAX,
      "slope": ANNEAL_SLOPE,
    },
    "temperature": 1e-5,
    "initializer": [
      [".*", {"type": "uniform", "a": -INIT_UNIFORM_RANGE_AROUND_ZERO, "b": INIT_UNIFORM_RANGE_AROUND_ZERO}]
    ]
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
      {"type": "track_metrics", "patience": PATIENCE, "validation_metric": "+BLEU"},
      "validate",
      "generate_samples",
      "log_to_tensorboard"
    ]
  }
}
