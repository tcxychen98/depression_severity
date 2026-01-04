# Suicidal Ideation Severity Scale Prediction Model

## Overview

This repository presents a **zero-shot natural language processing (NLP) system** designed to estimate the **severity level of suicidal ideation** from free-text inputs. The project is positioned as a **research and portfolio piece**, demonstrating system design choices, large-scale inference handling, and hardware-aware optimization rather than a clinical solution.

The core objective is to show how modern transformer-based models can be used to perform **semantic severity reasoning** without task-specific fine-tuning, while remaining scalable, fault-tolerant, and computationally efficient on consumer hardware.

A live demonstration of the model logic (Streamlit) is available here:
[https://depressionseverity-dyg8butjocf8ph5qth9ycx.streamlit.app/](https://depressionseverity-dyg8butjocf8ph5qth9ycx.streamlit.app/)

---

## Important Disclaimers

### Mental Health & Safety Disclaimer

This project processes text related to suicidal ideation. It is **not a medical tool**, **not a diagnostic system**, and **must not** be used for clinical decision-making, self-diagnosis, or intervention planning. All outputs are probabilistic model inferences and may be incorrect or misleading in real-world contexts.

If you or someone else is experiencing suicidal thoughts, please seek help from qualified mental health professionals or local emergency services.

### AI Usage Disclosure

The development of this project—including architecture decisions, documentation refinement, and code structuring—was **assisted by AI tools**. All implementation choices were reviewed, validated, and intentionally designed by the author. The final responsibility for correctness, ethics, and usage lies with the human developer.

---

## Problem Framing

Traditional supervised classifiers for suicidal ideation require:

* Carefully labeled datasets
* Rigid category definitions
* Retraining when labels or severity scales change

This project explores an alternative approach: **zero-shot severity inference**, where severity labels are treated as natural language hypotheses and evaluated against user-generated text. This enables flexible classification without retraining while preserving semantic nuance.

---

## Model Selection

### `valhalla/distilbart-mnli-12-1`

The system is built on `valhalla/distilbart-mnli-12-1`, a distilled BART model trained on the **Multi-Genre Natural Language Inference (MNLI)** task.

#### Rationale

1. **Zero-Shot Inference via NLI**
   Rather than predicting labels directly, the model evaluates whether a given text *entails* a severity description. This allows it to reason about categories it has never explicitly seen during training.

2. **Distilled Architecture**
   Compared to full-sized BART models, this variant is:

   * Approximately 50% smaller
   * Roughly 2× faster at inference
   * Retains the majority of semantic reasoning capability

   This makes it suitable for large-scale inference on local GPUs.

3. **Contextual Sensitivity**
   BART-based models perform well on nuanced language understanding, which is critical when differentiating between passive ideation, active ideation, and high-risk expressions.

---

## System Architecture & Logic Flow

The pipeline is designed around **fault-tolerant streaming inference**, enabling long-running jobs over large datasets without excessive memory usage or catastrophic failure.

### 1. Data Partitioning

Input data is split into:

* **Non-suicidal text**: directly assigned a severity score of `0`
* **Potentially suicidal text**: passed through the zero-shot classification pipeline

This design avoids unnecessary GPU computation on irrelevant samples.

---

### 2. Checkpointing & Resume Strategy

To ensure robustness:

* The system checks for an existing `severity_progress.csv`
* If present, it calculates the last completed row and resumes processing automatically

This approach prevents loss of progress due to crashes, power interruptions, or manual termination.

---

### 3. Streaming Batch Inference

Instead of loading the full dataset into memory:

* Text samples are streamed in batches
* Default batch size: **16**, chosen to balance throughput and memory constraints

This ensures stable execution on consumer-grade GPUs.

---

### 4. Periodic Persistence

Inference results are written to disk every **100 rows**, enabling:

* Partial result inspection
* Minimal data loss in failure scenarios

---

## Hardware-Aware Optimization (AMD GPUs)

This project explicitly targets **AMD GPUs on Windows** using **ROCm and FP16 inference**.

### ROCm / HIP Backend

ROCm allows PyTorch to communicate directly with AMD GPUs without relying on DirectML or DirectX translation layers.

Key benefits:

* Reduced inference latency
* Improved kernel efficiency
* Native PyTorch compatibility

---

### FP16 (Half-Precision) Inference

The model is executed in FP16 mode rather than standard FP32.

Advantages:

* Approximately 50% reduction in VRAM usage
* Significantly higher throughput on RDNA 2 / RDNA 3 architectures
* Enables larger batch sizes without memory exhaustion

FP16 is particularly effective for inference-only workloads where numerical precision requirements are lower.

---

## End-to-End Execution Summary

1. Detect available AMD GPU via ROCm
2. Recover processing state from disk if a checkpoint exists
3. Load the DistilBART MNLI model in FP16 mode
4. Stream text samples in fixed-size batches
5. Periodically persist intermediate results
6. Merge inferred severity scores with the non-suicidal baseline

---

## Environment Setup (ROCm on Windows)

ROCm on Windows remains experimental and may require manual troubleshooting.

### Installation References

* ROCm 7 build:
  [https://d2awnip2yjpvqn.cloudfront.net/v2/gfx120X-all/](https://d2awnip2yjpvqn.cloudfront.net/v2/gfx120X-all/)

* Community installation guide:
  [https://www.reddit.com/r/ROCm/comments/1n1jwh3/installation_guide_windows_11_rocm_7_rc_with/](https://www.reddit.com/r/ROCm/comments/1n1jwh3/installation_guide_windows_11_rocm_7_rc_with/)

### PyTorch Installation (ROCm)

```bash
pip install --upgrade --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx120X-all/ torch torchvision torchaudio
```

---

## Application Demo & Usage

This repository also includes a lightweight **Streamlit application** to demonstrate how the zero-shot severity inference works in practice.

### How to Use the App

1. Open the web app link:
   [https://depressionseverity-dyg8butjocf8ph5qth9ycx.streamlit.app/](https://depressionseverity-dyg8butjocf8ph5qth9ycx.streamlit.app/)

2. Copy any Reddit post or free-form text related to mental health or personal distress.

3. Paste the text into the input text box.

4. Click **"Analyze Severity"** to run the zero-shot inference.

5. The application will return:

   * A **severity score ranging from 0 to 5**
   * Higher values indicate stronger semantic alignment with severe suicidal ideation

6. If the predicted severity score is **greater than 3**, the app displays a **warning message** indicating elevated risk.

---

### Example Screenshots



Suggested images:

* App landing page
* Example Reddit text pasted into the input field
* Severity score output (0–5)
* Warning message displayed for scores > 3

You can place images in a `/assets` or `/images` folder and reference them here, for example:

```markdown
![App Input Example](images/app_input.png)
![Severity Output Example](images/app_output.png)
```

---

## Limitations & Ethical Considerations

* Outputs reflect model inference, not ground truth
* Language ambiguity and cultural context can affect predictions
* Biases present in pretraining data may influence results
* The system should never be used for automated decision-making involving individuals

This project is intended to demonstrate engineering and modeling techniques, not to replace human judgment.

---

## License

Released for research and educational purposes. Please review the repository license for full terms.

---

## Author’s Note

This repository is part of a broader exploration into applied NLP system design, focusing on scalability, robustness, and responsible use of modern language models in sensitive domains.
