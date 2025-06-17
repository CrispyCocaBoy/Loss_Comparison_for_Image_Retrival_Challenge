# CLIP ViT-B/32 for Image Retrieval (CLIPLoss Focus)

This project details the methodology and implementation for training and evaluating a fine-tuned CLIP Vision Transformer (ViT-B/32) model adapted for the task of image retrieval. This iteration focuses specifically on the application and optimization of the Contrastive Language-Image Pre-training (CLIP) Loss. The objective is to leverage the robust pre-trained capabilities of CLIP while specializing its feature representations for a retrieval dataset.
## 1. Overview and Architecture

* We fine-tune a CLIP-based Vision Transformer (ViT-B/32) for image retrieval. The model setup includes:

    * Backbone: CLIP ViT-B/32 (visual encoder)

    * Head: Linear projection with Dropout

    * Loss: CLIP Loss (Contrastive Symmetric Cross-Entropy)

* The projection head maps features into a 512-dimensional embedding space. We support both:

    * Frozen Backbone: Only the projection head is trained.

    * Unfrozen Backbone: The entire encoder and projection head are fine-tuned.

## 2. Key Components and Files

The project is organized modularly. Key files:

```graphql
clip_ClipLoss/
│
├── config.py                  # Centralized configuration (hyperparameters, paths, settings)
├── main.py                    # Entry point for training and evaluation pipeline
├── submit_all.py              # Script to generate and submit retrieval results for all models
│
├── src/                       # Core source code modules
│   ├── finetuned_clip.py              # Wraps the OpenAI CLIP model with a trainable projection head
│   ├── image_text_dataset.py          # Custom Dataset class for image-text pairs
│   ├── training_loop.py              # Training logic and custom CLIP-based loss function
│   ├── evaluation.py                 # Evaluation loop using cosine similarity and FAISS
│   ├── extract_embeddings_clip.py    # Extracts image and text embeddings from CLIP
│   ├── results.py                    # Functions for ranking and saving retrieval results
│   └── submit.py                     # Script for formatting and writing submission files
```


## 3. Training Process

The training pipeline, started via main.py, performs:

- Reproducibility Setup: Sets global seed.

- Configuration Loading: From config.py.

- Model Initialization: Loads CLIP ViT-B/32 and wraps it in FineTunedCLIP.

- Dataset Preparation: Loads and splits data using ImageTextDataset and PyTorch DataLoader.

- Discriminative Learning Rates: Higher LR for projection head; scaled LR for encoder (if unfrozen).

- Epochal Training: Uses train_clip_hybrid to train and validate over multiple epochs.

Checkpointing:

```graphql
clip_ClipLoss/repository/
├── metrics/training_metrics.csv
├── checkpoints/weights_epoch_X.pth
└── all_weights/model_epoch_X.pt
```

## 4. Evaluation and Results

After training, main.py handles evaluation:

* Epoch-wise Evaluation:

    * Loads each model_epoch_X.pt.

    * Extracts embeddings for query and gallery sets.

    * Computes similarities via get_top_k (cosine or Euclidean).

    * Saves results to clip_ClipLoss/repository/results/retrieval_results_epoch_X.json.

* Submission:

    * Use submit_all.py to:

        - Submit results to the competition API.

        - Generate submission_accuracy.csv.

## 5. Installation

To set up the project environment, run from the root of the project

```bash
pip install -r requirements.txt
```

### Note for macOS users

If you encounter an OpenMP runtime error like:

```vbnet
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

you can bypass it by setting the following environment variable before running your scripts:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

This is a temporary workaround. For a proper fix, ensure all dependencies use the same OpenMP runtime (e.g., via Homebrew's libomp).


