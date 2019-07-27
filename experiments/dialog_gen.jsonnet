local SEED = 0;
local CUDA = 1;
local READER = "dialog-gan";
local PREDICTOR = 'dialog-gen';

local LATENT_DIM = 128;
local BATCH_SIZE = 32;
local ACTIVATION = 'relu';

local NUM_EPOCHS = 50;
local PATIENCE = 5;
local GRAD_NORM = 5;
local SUMMARY_INTERVAL = 10;
local GEN_OPTIMIZER = "adam";
local DISC_OPTIMIZER = "adam";
local DISC_LEARNING_RATE = 0.00001;
local GEN_LEARNING_RATE = 0.001;

{
  "random_seed": SEED,
  "numpy_seed": SEED,
  "pytorch_seed": SEED,
  "dataset_reader": {
    "type": READER,
  },
  "vocabulary": {
    "directory_path": "models/dialog_dae/vocabulary"
  },
  "train_data_path": "data/interim/dialog/train_dialog.tsv",
  "validation_data_path": "data/interim/dialog/valid_dialog.tsv",
  "model": {
    "type": "dialog_gan",
    "encoder": {
      "_pretrained": {
        "archive_file": "models/dialog_dae/model.tar.gz",
        "module_path": '_encoder',
        "freeze": true
      },
    },
    "decoder": {
      "_pretrained": {
        "archive_file": "models/dialog_dae/model.tar.gz",
        "module_path": '_decoder',
        "freeze": true,
      },
    },
    "generator": {
      "type": "dialog-generator",
      "latent_dim": LATENT_DIM,
      'activation': ACTIVATION
    },
    "discriminator": {
      "type": "dialog-discriminator",
      "input_dim": 2*LATENT_DIM,
      "hidden_dim": LATENT_DIM,
    },
  },
  "iterator": {
    "type": "homogeneous_bucket",
    "batch_size" : BATCH_SIZE,
    "sorting_keys": [["source_tokens", "num_tokens"]],
    "partition_key": "stage"
  },
  "trainer": {
    "type": 'callback',
    "num_epochs": NUM_EPOCHS,
    "cuda_device": CUDA,
    "optimizer": {
      "type": "gan",
      "generator_optimizer": {
        "type": GEN_OPTIMIZER,
        "lr": GEN_LEARNING_RATE
      },
      "discriminator_optimizer": {
        "type": DISC_OPTIMIZER,
        "lr": DISC_LEARNING_RATE
      }
    },
    "callbacks": [
      "gan-callback",
      "checkpoint",
      {"type": "track_metrics", "patience": PATIENCE, "validation_metric": "+_S_BLEU4F"},
      "validate",
      "generate_dialog_samples",
      "log_to_tensorboard"
    ]
  }
}
