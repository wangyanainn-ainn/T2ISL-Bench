# Text2SceneLeakageBench (T2ISL-Bench)

Text2SceneLeakageBench is a controlled diagnostic benchmark for studying **semantic leakage** in text-to-image generation. It evaluates whether target text, once rendered correctly on a visual subject, further alters **subject identity** or **scene semantics** to make the text appear more semantically plausible.

Unlike conventional text rendering benchmarks, this benchmark focuses on the **interaction between text, subject, and scene**, rather than local text accuracy alone.

---

## Example: Semantic Leakage under Conflicting Text

Below shows three cases built from the same seed subject.

**Seed:** fire_extinguisher
**Text anchor:** front of the extinguisher cylinder

| Condition               | Description                                            |
| ----------------------- | ------------------------------------------------------ |
| aligned                 | target text matches subject semantics                  |
| conflict                | target text conflicts with subject semantics           |
| conflict + anti_leakage | explicit constraint discouraging semantic compensation |


## Example: Semantic Leakage

<p align="center">
  <img src="assets/figure1.png" width="90%">
</p>

In the conflicting condition, the model may alter the subject identity or introduce additional scene cues to make the target text appear more plausible. This effect is referred to as **semantic leakage**.

---

## Dataset Statistics

The current merged release contains:

* 30 seed subjects
* 360 structured cases
* Each case provides both Chinese and English prompts (`prompt_zh`, `prompt_en`)

Therefore, the benchmark corresponds to **720 language-specific prompt instances** when both languages are counted separately.

---

## What is Semantic Leakage?

Semantic leakage refers to the failure mode in which the target text, after being rendered on the designated text-bearing region, influences the **global interpretation** of the image.

Specifically, it includes:

* altering the subject identity
* introducing scene elements that support the target text
* shifting the overall scene semantics

even when the text is already correctly rendered locally.

---

## Controlled Factors

Each case is constructed from three controlled dimensions:

* **Text relation**

  * `aligned`
  * `conflict_1`
  * `conflict_2`

* **Scene openness**

  * `closed`
  * `open`

* **Prompt mode**

  * `natural`
  * `anti_leakage`

This factorization allows the same subject to be expanded into comparable instances under different semantic tensions, contextual richness, and prompt constraints.

---

## Data Schema

### Seed structure

Each seed defines a subject prior and its text configuration:

* `seed_id`
* `subject_type`
* `leakage_type`
* `subject_name_zh`, `subject_name_en`
* `subject_description_zh`, `subject_description_en`
* `text_anchor_zh`, `text_anchor_en`
* `text_role_zh`, `text_role_en`
* `default_aligned_text_zh`, `default_aligned_text_en`
* `recommended_conflict_1_text_zh`, `recommended_conflict_1_text_en`
* `recommended_conflict_2_text_zh`, `recommended_conflict_2_text_en`

---

### Case structure

Each case instantiates a controlled evaluation condition:

* `case_id`
* `pair_id`
* `subject_type`
* `leakage_type`
* `relation`
* `scene_openness`
* `prompt_mode`
* `target_text_zh`, `target_text_en`
* `subject_name_zh`, `subject_name_en`
* `subject_description_zh`, `subject_description_en`
* `text_anchor_zh`, `text_anchor_en`
* `text_role_zh`, `text_role_en`
* `expected_failure_modes`
* `prompt_zh`, `prompt_en`

---

## Repository Structure

```
.
T2ISL-Bench/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── benchmark_cases/
│   ├── text2sceneleakagebench_seeds_merged_v1.json
│   └── text2sceneleakagebench_cases_merged_v1.json
│
├── benchmark/
│   └── image_generator.py
│
├── mm_auto_eval/
│   ├── __init__.py
│   ├── main.py
│   ├── ocr_eval.py
│   ├── prompts.py
│   ├── utils.py
│   └── vlm_eval.py
│
├── assets/
│   └── fire_extinguisher/
│       ├── aligned.png
│       ├── conflict.png
│       └── anti_leakage.png
│
├── run_pipeline.py
└── run_image_generation.py
```

Only the **merged benchmark files** are released. Intermediate construction splits are not included.

---

## Usage

### 1. Validate dataset consistency

```bash
python run_pipeline.py
```

---

### 2. Image generation

```bash
python run_image_generation.py
```

Environment variables are configured via `.env` (see `.env.example`).

---

### 3. Multimodal evaluation

```bash
python -m mm_auto_eval.main
```

Evaluation pipeline:

1. OCR (text accuracy)
2. VLM-based semantic evaluation
3. Metric aggregation (SSP, SLR, CLR)

---

## Evaluation Protocol

The benchmark evaluates three aspects:

* **Text Accuracy (TAA)**
  Whether the target text is correctly rendered.

* **Subject Preservation (SSP)**
  Whether the subject identity remains unchanged.

* **Scene Leakage (SLR)**
  Whether additional scene cues are introduced to support the target text beyond natural semantics.

* **CLR (Composite Leakage Result)**
  Derived from TAA, SSP, and SLR.

---

## Reproducibility

* Image generation is controlled via environment variables (`IMAGE_MODEL`, etc.)
* Evaluation uses a fixed OCR + VLM pipeline
* Dataset construction is deterministic via seed + factor expansion

---

## Notes

* This repository focuses on **controlled evaluation**, not large-scale open-world coverage.
* The design prioritizes **interpretability, controllability, and failure attribution**.

---

## License

For academic research use only.
