# Adversarial Learning on the Latent Space for Diverse Dialog Generation

PyTorch implementation of our paper [Adversarial Learning on the Latent Space for Diverse Dialog Generation,](https://www.aclweb.org/anthology/2020.coling-main.441/) accepted to COLING 2020, Barcelona, Spain (Online).

## Installation
   Install allennlp 0.9.0 using conda or pip
   pip install pyro-ppl
## Running Experiments
   - VAE
     `allennlp train -s models/dialog_vae --include-package src experiments/vae.jsonnet`
   - GAN (run after VAE)
     `allennlp train -s models/dialog_gan_mse --include-package src experiments/dialog_gen.jsonnet`
## Important Code
   - experiments
     - vae.jsonnet (Configuration for VAE)
     - dialog_gen.jsonnet (Configuration for GAN with MSE)
   - src
     - data
       - dataset.py (DatasetReaders)
       - iterator.py (Special iterator for GAN)
     - modules
       - encoders
         - variational_encoder.py (Variational Encoder for VAE)
       - decoders
         - decoder.py (Decoder base class)
         - variational_decoder.py (Variational Decoder for VAE)
       - metrics.py (LMPPL and NLTKSentenceBLEU)
       - annealer.py (Different Annealing Schedules)
     - models
       - vae.py (Variational Autoencoder, SampleGenerator)
       - dialog_generator.py (Generator Module for GAN)
       - dialog_discriminator.py (Discriminator Module for GAN)
       - dialog_gan.py (Dialog Gan, GanOptimizer and SampleGenerator)
       

If you find the code useful, please cite:
```
@inproceedings{khan-etal-2020-adversarial,
title = "Adversarial Learning on the Latent Space for Diverse Dialog Generation",
author = "Khan, Kashif  and
Sahu, Gaurav  and
Balasubramanian, Vikash  and
Mou, Lili  and
Vechtomova, Olga",
booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
month = dec,
year = "2020",
address = "Barcelona, Spain (Online)",
publisher = "International Committee on Computational Linguistics",
url = "https://www.aclweb.org/anthology/2020.coling-main.441",
pages = "5026--5034",
abstract = "Generating relevant responses in a dialog is challenging, and requires not only proper modeling of context in the conversation, but also being able to generate fluent sentences during inference. In this paper, we propose a two-step framework based on generative adversarial nets for generating conditioned responses. Our model first learns a meaningful representation of sentences by autoencoding, and then learns to map an input query to the response representation, which is in turn decoded as a response sentence. Both quantitative and qualitative evaluations show that our model generates more fluent, relevant, and diverse responses than existing state-of-the-art methods.",
}
```
