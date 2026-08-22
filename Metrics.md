# Evaluation Metrics Summary (from Report.pdf)

## Perplexity Scores

Perplexity was calculated on the validation set at the end of each epoch. Lower values indicate better performance.

### Casual Chat Model

| Epoch | Perplexity |
|-------|-----------|
| 1     | 13.87     |
| 2     | 10.83     |
| 3     | 8.82      |
| 4     | 7.33      |
| 5     | 6.95      |

### Tiny Chat Model

| Epoch | Perplexity |
|-------|-----------|
| 1     | 7.70      |

Note: Only trained for 1 epoch due to compute constraints.

### Story Generator Model

| Epoch | Perplexity |
|-------|-----------|
| 1     | 8.09      |
| 2     | 6.86      |
| 3     | 6.26      |

---

## Training & Validation Loss

### Casual Chat Model
- **Training Loss:** 3.01 → 1.70
- **Validation Loss:** 2.62 → 1.93
- **Epochs:** 5

### Tiny Chat Model
- **Training Loss:** 2.39 (final)
- **Validation Loss:** 2.04 (final)
- **Epochs:** 1

### Story Generator Model
- **Training Loss:** 2.44 → 1.87
- **Validation Loss:** 2.09 → 1.83
- **Epochs:** 3

---

## Dataset Sizes

| Model | Dataset | Total Samples | Train Split | Validation Split |
|-------|---------|--------------|-------------|-----------------|
| Casual Chat | Combined Casual + Tiny Chat | 196,000 | 192,838 (98%) | 3,740 (2%) |
| Tiny Chat | TinyChat | 1,000,000 | 980,000 (98%) | 20,000 (2%) |
| Story Generator | TinyStories | 500,000 | 490,000 (98%) | 10,000 (2%) |

---

## Model Configuration (All Models)

| Parameter | Casual Chat | Tiny Chat | Story Gen |
|-----------|------------|-----------|-----------|
| Parameters | ~41M | ~41M | ~41M |
| Embedding Dimension | 420 | 420 | 420 |
| Attention Heads | 6 | 6 | 6 |
| Transformer Layers | 6 | 6 | 6 |
| Feed Forward Size | 2,000 | 2,000 | 2,000 |
| Vocab Size | 32,105 | 32,105 | 32,105 |
| Context Window | 80 tokens | 200 tokens | 285 tokens |
| Batch Size | 4 | 16 | 16 |
| Learning Rate | 0.001 | 0.001 | 0.001 |
| LR Scheduler | Cosine Decay | Cosine Decay | Cosine Decay |
| Warmup | 30% | 30% | 30% |
| Dropout | 0.001 | 0.05 | 0.05 |
| Weight Decay | 0.001 | 0.05 | 0.05 |
| Label Smoothing | 0.001 | 0.05 | 0.05 |
| Gradient Clipping | 1.0 | 1.0 | 1.0 |
| Precision | bfloat16 | bfloat16 | Full precision |

---

## Key Takeaways

1. **All three models** showed consistent perplexity decrease across epochs, indicating stable learning.
2. **Story Generator** achieved the lowest final perplexity (6.26), likely due to pre-trained weight initialization.
3. **Casual Chat** showed strong perplexity improvement from 13.87 → 6.95 over 5 epochs.
4. **Tiny Chat** started at 7.70 perplexity after only 1 epoch, showing promise for further training.
5. **No overfitting** was observed in any model — training and validation loss tracked closely together.
