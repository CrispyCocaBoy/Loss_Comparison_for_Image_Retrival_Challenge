# Loss Comparison for Image Retrieval Challenge

## 📖 Description

This project was developed for a facial image retrieval challenge. The goal is to evaluate the impact of different loss functions (**ArcFace**, **Triplet Loss**, **CLIP Loss**, **Contrastive Loss**) in face retrieval scenarios. All experiments use the **CLIP ViT-B/32** backbone, enhanced with a projection head. The analysis aims to identify the most effective loss for optimizing retrieval performance and robustness to data variation.  
For more details, see the [paper.pdf](paper.pdf).

---

## 📊 Loss Comparison Table

The following table summarizes the main training settings and results for each loss function:

```latex
\begin{table}[!htbp]
\centering
\small
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Loss} & \textbf{Frz} & \textbf{B\_s} & \textbf{Ep} & \textbf{W\_d} & \textbf{Drp} & \textbf{Score} \\
\midrule
Triplet & T & 64  & 45 & 1e-4 & 0.0 & 586.53 \\
CLIP    & T & 64  & 21 & 1e-4 & 0.3 & 546.53 \\
ArcFace & F & 128 & 38 & 1e-4 & 0.3 & \textbf{616.98} \\
\bottomrule
\end{tabular}%
}
\caption{Compact comparison of loss functions and training settings.}
\label{tab:loss_comparison_small}
\end{table}
```

**Legend:**  
- **Frz**: Whether the backbone was frozen (T=True, F=False)  
- **B\_s**: Batch size  
- **Ep**: Number of epochs  
- **W\_d**: Weight decay  
- **Drp**: Dropout  
- **Score**: Final leaderboard score (higher is better)

---

## 🧠 Model Overview

- **CLIP + Projection Head:**  
  The backbone is CLIP ViT-B/32, followed by a projection head (MLP) to learn discriminative embeddings.
- **Siamese Setup:**  
  The `siamese_network/` folder contains the Siamese architecture, in which two images are passed through the same backbone and the distance between embeddings is used for the loss (Triplet/Contrastive).
- **Per-Loss Implementations:**  
  - `clip_ArcFace/`: ArcFace loss experiments
  - `clip_ClipLoss/`: CLIP loss experiments
  - `clip_Triplet/`: Triplet loss experiments
- **Key Differences:**  
  - ArcFace and CLIP Loss leverage direct classification/contrast among classes.
  - Triplet and Contrastive Loss focus on distances between embedding pairs/triplets.
  - All models use the CLIP ViT-B/32 backbone.

---

## 🏋️‍♂️ Training Details

- **Data Augmentation:**  
  Random crop, horizontal flip, color jitter, resize.
- **Early Stopping:**  
  Validation metric (Top-5) monitoring with a patience of 10 epochs.
- **Evaluation Metrics:**  
  - **Top-1 Accuracy:** Percentage of queries with the correct match at rank 1.
  - **Top-5 / Top-10 Accuracy:** Percentage of queries with the correct match in the top 5/10 results.

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/CrispyCocaBoy/Loss_Comparison_for_Image_Retrival_Challenge.git
cd Loss_Comparison_for_Image_Retrival_Challenge
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate   # or 'venv\Scripts\activate' on Windows
pip install -r requirements.txt
```

### 3. Run training
Each loss has a dedicated folder with its own training script. Example (ArcFace):
```bash
cd clip_ArcFace
python train.py
```
Or for Triplet Loss:
```bash
cd ../clip_Triplet
python train.py
```
Check each folder for specific configuration options.

---

## 📁 Folder Structure

```
Loss_Comparison_for_Image_Retrival_Challenge/
├── clip_ArcFace/        # ArcFace loss implementation & experiments
├── clip_ClipLoss/       # CLIP loss implementation & experiments
├── clip_Triplet/        # Triplet loss implementation & experiments
├── siamese_network/     # Siamese and contrastive loss architectures
├── data/                # Processed datasets
├── data_original/       # Original datasets
├── data_analysis/       # Data analysis scripts/notebooks
├── utilities/           # Utility functions and scripts
├── paper.pdf            # Project paper/report
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🚀 Main Commands

- **Training with ArcFace:**  
  `cd clip_ArcFace && python train.py`
- **Training with Triplet Loss:**  
  `cd ../clip_Triplet && python train.py`
- **Training with CLIP Loss:**  
  `cd ../clip_ClipLoss && python train.py`
- **Siamese/Contrastive:**  
  See `siamese_network/` for scripts and usage.
- **Evaluation:**  
  Check each folder for evaluation scripts and usage examples.

---

## 🤝 Contributing

Contributions are welcome!  
To participate:

1. Fork the project.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes.
4. Open a pull request describing your changes.

---

## 📄 License

This project is released under the [MIT](LICENSE) license.

---

*Note: The directory listing above may be incomplete. See the [full repository on GitHub](https://github.com/CrispyCocaBoy/Loss_Comparison_for_Image_Retrival_Challenge/tree/master/) for all files and folders.*