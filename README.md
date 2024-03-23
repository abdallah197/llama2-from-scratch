# LLaMA 2 from scratch

This repository contains a custom implementation of the LLaMA 2 model, as described in the paper "LLaMA 2: Open
Foundation and Fine-Tuned Chat Models" ([ArXiv](https://arxiv.org/abs/2307.09288)). This implementation focuses on
reproducing and extending some of the key features that distinguish LLaMA 2, including RMS-Normalization, the SwiGLU
activation function, Rotary Positional Embeddings (RoPE), increased context length with Grouped-Query Attention (GQA),
and the KV-caching technique.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage) #todo
- [Training Results](#training results) #todo
- [Configuration](#configuration)
- [Examples](#examples) #todo
- [Acknowledgments](#acknowledgments)

# todo: add emojies

## Introduction

This project aims to build the LLaMA 2 architecture from scratch, incorporating essential advancements in transformer
models. Key enhancements include RMS-Normalization, SwiGLU activation, Rotary Positional Embeddings, and advanced
attention mechanisms like Grouped-Query Attention, all designed to improve model performance, particularly in handling
longer context windows and enhancing the model's positional understanding.

## Features

- **RMS-Normalization**: A simplified version of layer normalization that stabilizes layer activations and aids in model
  convergence.
- **SwiGLU Activation Function**: Replaces ReLU to improve training performance through more efficient activation.
- **Rotary Positional Embeddings (RoPE)**: Enhances positional awareness at each token by adding distance between
  tokens, featured in RoFormer: Enhanced Transformer with Rotary Position Embedding
  ([ArXiv](https://arxiv.org/abs/2104.09864)).
- **Increased Context Length with GQA**: Expands the context window to 4096 tokens and employs grouped-query attention
  for better long document processing.
- **KV-Cache**: A caching technique to improve decoding efficiency and speed.

### Additional Features Implemented:

- **Inference with Top-P Sampling**: Introduces a more dynamic sampling method that adjusts the number of tokens based
  on their cumulative probability.
- **Data & Training Utilities**: The project adds torch wrappers to onboard with pretriaing on any `.txt` file.

## Installation

To install the necessary dependencies, clone this repository and run:

```bash
git clone https://github.com/your-repo/LLaMA-2-Custom.git
cd LLaMA-2-Custom
pip install -r requirements.txt
```

## Usage

To use the repo for inference:

1. Download llama2 SentencePiece tokenizer model [here](https://llama.meta.com/llama-downloads/).
2. Either download llama weights by following steps [here](https://llama.meta.com/llama-downloads/) / or use the repo to
   train your own llama2.
3. Set your inference configuration in config.py
4. pass your list of prompts to the terminal by running `python inference.py prompt1 prompt2 `
## Training Results

After the training and evaluation phases, we can see a consistent drop in both training and evaluation losses,
indicating the model's learning effectiveness. Below is a plot demonstrating this trend over the training steps.

![losses.png](losses.png)

## Configuration

The configuration for the model and training is defined using data classes in Python. You can adjust these
configurations to suit your dataset and training needs.
We have three main config dataclasses:

- ModelArgs.
- DataArgs and
- TrainArgs.

To adjust these configurations, modify the respective fields in the data class instances before initializing your model
or training process. For instance, to increase the number of layers and attention heads, you might do:

```
model_args = ModelArgs(n_layers=48, n_heads=48)
train_args = TrainArgs(lr=5e-4, n_epochs=20)
data_args = DataArgs(filepath='new_dataset.txt')
```

I adjustd the model original HP to fit my compute. Here's a summary of the main configuration settings:

- Model Dimensionality: 2048
- Number of Transformer Layers: 32
- Number of Query Attention Heads: 32
- Optional Number of Heads for Key and Value (n_kv_heads): Can be set for specific requirements
- Vocabulary Size: Set dynamically upon loading the llama2 Sentence Piece tokenizer.
- Operating Mode: 'train/inference', when choosing inference, we apply KV-Cache.

## Acknowledgments

This project has been inspired and informed by various resources and individuals in the AI and machine learning
community. We'd like to extend our gratitude to the following:

- [Andrej Karpathy for his tutorial on training a GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1058s&ab_channel=AndrejKarpathy).
  His insights into neural network architectures and training methodologies have been invaluable.
- [Umar Jamil's guide on Training LLama2 from scratch](https://www.youtube.com/watch?v=oM4VmoabDAI&ab_channel=UmarJamil).
  This resource provided practical insights and a foundational understanding necessary for this implementation.
- The [Meta LLaMA GitHub repository](https://github.com/meta-llama/llama) has been an essential resource for
  understanding the intricacies of the LLaMA 2 model and its implementation.

I am grateful for the knowledge shared by these individuals and communities, which has significantly contributed to the
development of this project.
