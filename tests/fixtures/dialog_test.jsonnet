local SEED = 0;
local CUDA = 0;
local READER = "seq2seq";
local PREDICTOR = 'dialog-gen';

local LATENT_DIM = 128;
local BATCH_SIZE = 2;

local NUM_EPOCHS = 1;
local PATIENCE = 3;
local SUMMARY_INTERVAL = 10;
local OPTIMIZER = "rmsprop";
local DISC_LEARNING_RATE = 0.0001;
local GEN_LEARNING_RATE = 0.0002;

{
  "random_seed": SEED,
  "numpy_seed": SEED,
  "pytorch_seed": SEED,
  "dataset_reader": {
    "type": READER,
  },
  "vocabulary": {
    "directory_path": "models/dialog_vae/vocabulary"
  },
  "train_data_path": "tests/fixtures/dialog_samples.tsv",
  "validation_data_path": "tests/fixtures/dialog_samples.tsv",
  "discriminator": {
    "type": "dialog-discriminator",
    "input_dim": 2*LATENT_DIM,
    "hidden_dim": LATENT_DIM,
    "initializer": [
      [".*", {"type": "normal", "mean": 0, "std": 0.02}],
    ]
  },
  "model": {
    "type": "dialog-generator",
    "latent_dim": LATENT_DIM,
    "latent_encoder": {
      "_pretrained": {
        "archive_file": "models/dialog_vae/model.tar.gz",
        "module_path": '_encoder',
        "freeze": true,
      },
    },
    "latent_decoder": {
      "_pretrained": {
        "archive_file": "models/dialog_vae/model.tar.gz",
        "module_path": '_decoder',
        "freeze": true,
      },
    },
    "temperature": 1e-5,
    "initializer": [
      ["^(?!_latent_encoder)(?!_latent_decoder).*", {"type": "normal", "mean": 0, "std": 0.02}],
    ]
  },
  "iterator": {
    "type": "bucket",
    "batch_size" : BATCH_SIZE,
    "sorting_keys": [["source_tokens", "num_tokens"]]
  },
  "trainer": {
    "type": 'dialog-trainer',
    "num_epochs": NUM_EPOCHS,
    "cuda_device": CUDA,
    "summary_interval": SUMMARY_INTERVAL,
    "generator_optimizer": {
      "type": OPTIMIZER,
      "lr": GEN_LEARNING_RATE
    },
    "discriminator_optimizer": {
      "type": OPTIMIZER,
      "lr": DISC_LEARNING_RATE
    },
  }
}
