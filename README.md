# Room for Doubt

**Can uncertainty be estimated from dynamics rather than from one final prediction?**

This project studies a simple but important hypothesis:

> A model's doubt is partly encoded in trajectories: how it learns an example during training, or how its internal computation evolves while making a decision.

Most uncertainty baselines use a static signal: final softmax confidence, entropy, energy, calibration temperature, or distance in embedding space. These signals are useful, but often fail under label noise, hard examples, distribution shift, and hallucination-like errors.

**Room for Doubt** asks whether we can do better by recording, compressing, predicting, and eventually generating **doubt trajectories**.

The project is designed as a 6-week student research repository with three levels:

1. **Beginner:** detect doubt from simple training-dynamics statistics.
2. **Intermediate:** distill training dynamics into an inference-time uncertainty head.
3. **Expert:** model distributions of doubt trajectories with conditional flow matching.

The safest publication-oriented direction is the intermediate project. The most novel direction is the expert flow-matching project, but it should only be attempted after the deterministic trajectory-prediction baseline works.

---

## 1. Scientific motivation

Standard classifiers trained with cross-entropy often become overconfident, including on noisy labels and hard examples. A single final confidence score loses much of the learning process: whether the example was learned early or late, forgotten repeatedly, remained ambiguous across epochs, or became confidently wrong.

This project follows the view that uncertainty is not only a property of the final output. It can also be a **dynamic trace**.

Two sources of dynamics are considered:

### Training dynamics

For every training example, we record its behavior across epochs:

- probability assigned to the noisy label;
- probability assigned to the clean label, for evaluation only;
- max softmax confidence;
- margin between the target-label logit and the strongest alternative;
- predicted class;
- loss;
- correctness across epochs;
- forgetting count;
- AUM-like score;
- mean confidence and variability.

This connects directly to Dataset Cartography, forgetting events, and AUM.

### Decision-making dynamics

For language models and generative models, uncertainty may also be reflected in internal generation traces:

- token probabilities;
- token-level entropy;
- selected attention patterns;
- recurrent aggregation of uncertainty signals;
- hidden-state or residual-stream trajectory statistics;
- disagreement between generated answers.

This direction is newer and riskier. RAUQ and semantic entropy are useful references, but the first implementation should focus on classification and CIFAR-10N before moving to LLMs.

---

## 2. Project variants

### Project 1. Beginner: Can Training Dynamics Detect Doubt?

**Level:** beginner  
**Main question:** Can simple training-dynamics descriptors detect noisy labels and hard examples better than final softmax confidence?

This is the most reliable entry point. The student reproduces a compact version of Dataset Cartography, forgetting statistics, and AUM-style analysis on CIFAR-10N.

#### Dataset

Primary dataset:

- **CIFAR-10N**
  - CIFAR-10 images with human-annotated noisy labels.
  - Contains several noisy-label variants such as `aggregate`, `worse`, and random annotator labels.
  - Useful because noise is human-made, not only synthetic random flips.

Stretch dataset:

- **CIFAR-100N**
  - More classes and harder label structure.
  - Use only after CIFAR-10N is fully working.

Optional evaluation datasets:

- **SVHN**
  - Simple OOD dataset for CIFAR-trained classifiers.
- **CIFAR-10-C**
  - Corruption benchmark for robustness and selective prediction.
- **Clean CIFAR-10 test set**
  - Needed for standard accuracy and risk-coverage evaluation.

#### Model

Start with:

- ResNet-18;
- small WideResNet if available.

Avoid ViT at the beginning. The first goal is reliable logging, not model novelty.

#### What to record

For each training sample and epoch, save:

```text
sample_id
epoch
noisy_label
clean_label
predicted_label
loss
max_confidence
prob_noisy_label
prob_clean_label
target_margin
is_correct_noisy
is_correct_clean
```

From these logs, compute:

```text
mean_confidence
confidence_variability
correctness
forgetting_count
AUM-like score
final_loss
final_confidence
```

#### Baselines

- final max softmax confidence;
- final loss;
- entropy;
- energy score;
- AUM;
- forgetting count;
- mean confidence;
- confidence variability;
- MC Dropout if easy to add.

#### Evaluation

Main tasks:

1. **Noisy-label detection**
   - positive class: `noisy_label != clean_label`;
   - metrics: AUROC, AUPRC, FPR@95TPR if meaningful.

2. **Selective classification**
   - reject the most doubtful samples;
   - report risk-coverage and AURC.

3. **Calibration**
   - ECE before and after removing high-doubt samples.

4. **Visualization**
   - Dataset Cartography map: confidence vs variability;
   - AUM histogram for clean vs noisy examples;
   - forgetting count distribution;
   - risk-coverage curve.

#### Expected result

A good result is not a new method yet. A good result is a clean, reproducible diagnostic report showing when training-dynamics descriptors outperform final softmax confidence.

#### Deliverable

```text
experiments/01_cifar10n_cartography/
  config.yaml
  metrics.csv
  report.md
  figures/
    data_map.png
    aum_histogram.png
    forgetting_histogram.png
    risk_coverage.png
```

---

### Project 2. Intermediate: From Training History to Inference-Time Doubt

**Level:** intermediate  
**Main question:** Can we train a lightweight model to predict training-dynamics uncertainty descriptors from the final embedding?

This is the core project. It transforms training dynamics from a post-hoc analysis tool into an inference-time uncertainty estimator.

#### Core idea

Stage 1 trains a classifier and records per-sample trajectories.

Stage 2 freezes the encoder and trains a small head that predicts trajectory descriptors from the final embedding:

```math
z_i = f_\theta(x_i)
```

```math
\hat{s}_i = g_\phi(z_i)
```

where `s_i` is a vector of training-dynamics descriptors.

At inference time, a new sample has no training history. The descriptor head estimates whether it resembles examples that were unstable, ambiguous, frequently forgotten, or likely mislabeled during training.

#### Dataset

Primary:

- **CIFAR-10N**
  - train on one noisy split, such as `aggregate`;
  - evaluate descriptor generalization on `worse` or random annotator labels.

Stretch:

- **CIFAR-100N**
  - harder and more realistic but slower.

OOD/selective evaluation:

- **SVHN**
  - OOD detection from CIFAR-10N-trained encoder;
- **CIFAR-10-C**
  - corruption robustness;
- **CIFAR-10.1**
  - optional natural distribution shift if convenient.

#### Encoders

Start:

- ResNet-18 trained on CIFAR-10N.

Add if time allows:

- frozen ImageNet ResNet features;
- DINOv2 frozen features;
- CLIP image encoder.

The useful ablation is whether trajectory descriptors are easier to predict from supervised classifier features or from generic pretrained features.

#### Descriptor targets

Minimal scalar targets:

```text
AUM
mean_confidence
confidence_variability
forgetting_count
final_loss
```

Recommended vector target:

```text
[AUM, mean_confidence, confidence_variability, forgetting_count, final_loss]
```

Optional targets:

```text
early_late_JS_divergence
probability_of_clean_label
label_agreement_uncertainty
final_correctness_probability
```

Do not use `noisy_label != clean_label` as the main training target. It can be used as an upper-bound diagnostic, but the project should remain about distilling dynamics, not supervised noise detection.

#### Descriptor head

Start simple:

```text
embedding -> MLP -> descriptor vector
```

Recommended losses:

- MSE or Huber for continuous descriptors;
- Poisson or log-MSE for forgetting count;
- multi-task weighted regression;
- optional uncertainty-aware heteroscedastic regression.

#### Baselines

Compare against:

- final softmax confidence;
- entropy;
- energy score;
- Mahalanobis distance in embedding space;
- final loss;
- direct AUM, as an oracle requiring training history;
- deterministic descriptor MLP;
- MC Dropout or small ensemble if feasible.

#### Evaluation

Main metrics:

- correlation between predicted and true descriptors;
- noisy-label AUROC and AUPRC;
- OOD AUROC on SVHN;
- robustness/selective prediction on CIFAR-10-C;
- risk-coverage AURC;
- calibration metrics.

Required plots:

```text
predicted_vs_true_AUM.png
predicted_data_map.png
risk_coverage_baselines.png
noise_detection_roc.png
descriptor_correlation_matrix.png
```

#### Key ablations

The project must include sanity checks:

1. Train the head with shuffled descriptor targets.
2. Train the head with random descriptor targets.
3. Compare final embeddings vs earlier checkpoint embeddings.
4. Compare trained classifier features vs frozen DINOv2 or ImageNet features.
5. Train on one noisy split and test on another.

These ablations prevent overclaiming. The head may learn class difficulty, image ambiguity, or visual atypicality rather than “pure uncertainty.” That is acceptable, but the report must state it clearly.

#### Expected result

The desired claim is:

> Training dynamics can be distilled into an inference-time uncertainty estimator.

This is stronger than plotting Dataset Cartography and more feasible than immediately building a flow-matching model.

#### Deliverable

```text
experiments/02_descriptor_distillation/
  config.yaml
  metrics.csv
  report.md
  figures/
    predicted_vs_true_AUM.png
    predicted_data_map.png
    risk_coverage.png
    roc_noise_detection.png
```

---

### Project 3. Expert: Conditional Flow Matching over Doubt Trajectories

**Level:** expert  
**Main question:** Can a generative model learn possible learning trajectories conditioned on an input and a candidate label?

This is the most novel project. It should not be the first implementation step. It becomes meaningful only after Projects 1 and 2 show that trajectory descriptors contain useful information.

#### Safer version: training-dynamics flow model

Train a conditional flow matching model over trajectory descriptors:

```math
p(s \mid z, y_c)
```

where:

```text
z   = image embedding from the trained encoder
y_c = candidate label
s   = trajectory descriptor vector or compressed trajectory
```

The model generates possible learning histories for an image-label pair.

#### Why candidate labels matter

The same image may have multiple plausible labels under noise or ambiguity. We can ask counterfactual questions:

- If this image were labeled as class `k`, would the trajectory look stable?
- Does the provided label induce high predicted doubt?
- Which candidate label minimizes predicted trajectory instability?
- Does the generated descriptor distribution become multimodal for ambiguous examples?

#### Descriptor representation

Minimum viable descriptor vector:

```text
s = [
  AUM,
  mean_confidence,
  confidence_variability,
  forgetting_count,
  final_loss
]
```

Stronger alternatives:

- PCA over full epoch-wise probability trajectories;
- autoencoder over logit trajectories;
- early/mid/late trajectory summary;
- class-wise margin trajectories.

#### Training data for the flow model

A single training run gives one descriptor vector per sample. That is not enough to justify a distributional generative model.

Minimum viable setup:

- train 3-5 classifiers with different random seeds;
- collect descriptor vectors for each sample and seed;
- optionally collect descriptors under strong augmentations;
- condition on embedding and candidate label;
- train the conditional flow to model descriptor variability.

#### Uncertainty scores from generated samples

Sample multiple descriptor vectors from the flow model and compute:

- variance of generated AUM;
- probability of low-AUM region;
- expected forgetting count;
- generated descriptor entropy;
- disagreement between candidate labels;
- entropy over the best counterfactual label.

#### Baselines

- deterministic descriptor MLP from Project 2;
- Gaussian heteroscedastic regressor;
- ensemble of descriptor MLPs;
- direct AUM oracle;
- Mahalanobis distance;
- softmax confidence;
- energy score.

#### Critical risk

Flow matching may not improve over a deterministic or heteroscedastic regressor. This is a valid result if shown carefully.

The main scientific weakness is artificial uncertainty: if the descriptor distribution is created from too few seeds or too little variability, the flow model may only learn noise. The report must compare against simpler probabilistic baselines.

#### Riskier extension: decision-making dynamics in LLMs

After the CIFAR-10N version works, a second expert branch can use RAUQ-style features from LLM generation trajectories.

Possible trajectory inputs:

```text
token_probability
token_entropy
selected_attention_head_scores
RAUQ_recurrent_state
hidden_state_norm
residual_stream_change
layerwise_uncertainty_score
```

Datasets:

- TruthfulQA;
- SciQ;
- MMLU subset;
- Natural Questions subset if evaluation is manageable.

Models:

- Qwen2.5-0.5B or Qwen2.5-1.5B;
- Llama-3.2-1B or 3B;
- avoid 7B unless the infrastructure is stable.

Baselines:

- mean token entropy;
- sequence log-probability;
- RAUQ;
- semantic entropy if sampling budget allows;
- logistic regression over hand-crafted trajectory features;
- small GRU or Transformer over token-level features.

The realistic 6-week deliverable is modest:

> A trainable RAUQ-style probe improves, matches, or fails to improve over hand-crafted RAUQ features on one QA dataset.

#### Deliverable

```text
experiments/03_flow_matching_doubt/
  config.yaml
  metrics.csv
  report.md
  figures/
    generated_descriptor_distribution.png
    flow_vs_mlp_auc.png
    candidate_label_doubt.png
    ablation_num_seeds.png
```

---

## 3. Recommended scope for 6 weeks

The repository should support all three levels, but the project should not depend on all of them succeeding.

Recommended priority:

1. **Week 1-2:** CIFAR-10N training and trajectory logging.
2. **Week 2-3:** Dataset Cartography, AUM, forgetting, and baseline uncertainty metrics.
3. **Week 3-4:** descriptor distillation from embeddings.
4. **Week 4-5:** OOD, corruption, and selective-prediction evaluation.
5. **Week 5-6:** flow matching only if descriptor distillation already works.
6. **Week 6:** reports, figures, ablations, and workshop-style paper draft.

The minimum successful project is Project 1 plus a clean report.

The strongest workshop submission is Project 2 with careful ablations.

The most novel but risky submission is Project 3.

---

## 4. Datasets

### CIFAR-10N and CIFAR-100N

Primary datasets for the project.

Use them because they contain real human annotation noise and preserve the clean CIFAR labels for evaluation. This makes it possible to measure whether an uncertainty signal detects actual human label errors.

Expected use:

```text
train: CIFAR-10 images with noisy labels
evaluation: clean labels, noisy-label indicators, risk-coverage, calibration
```

Recommended order:

1. CIFAR-10N `aggregate`
2. CIFAR-10N `worse`
3. CIFAR-10N random annotator labels
4. CIFAR-100N only after the pipeline works

### SVHN

Use as a simple OOD benchmark for CIFAR-trained models.

### CIFAR-10-C

Use for corruption robustness and selective prediction.

### TruthfulQA, SciQ, MMLU subset

Use only for the LLM decision-dynamics extension. Start with one dataset and one small model.

---

## 5. Metrics

### Noisy-label detection

```text
AUROC
AUPRC
FPR@95TPR
```

Positive class:

```text
noisy_label != clean_label
```

### Selective classification

```text
risk-coverage curve
AURC
accuracy at fixed coverage: 50%, 70%, 90%, 95%
```

### Calibration

```text
ECE
NLL
Brier score
```

### OOD detection

```text
AUROC
AUPRC
FPR@95TPR
```

### Descriptor prediction

```text
Pearson correlation
Spearman correlation
MSE / MAE
rank correlation for doubt score
```

---

## 6. Repository structure

```text
room-for-doubt/
  README.md
  AGENTS.md
  LICENSE
  pyproject.toml
  configs/
    train_cifar10n.yaml
    cartography.yaml
    descriptor_distillation.yaml
    flow_matching.yaml
  data/
    README.md
  src/
    room_for_doubt/
      data/
      models/
      training/
      dynamics/
      uncertainty/
      evaluation/
      visualization/
  experiments/
    01_cifar10n_cartography/
      config.yaml
      report.md
      figures/
    02_descriptor_distillation/
      config.yaml
      report.md
      figures/
    03_flow_matching_doubt/
      config.yaml
      report.md
      figures/
  reports/
    beginner_report.md
    intermediate_report.md
    expert_report.md
  paper/
    main.tex
    refs.bib
  scripts/
    train_classifier.py
    compute_dynamics.py
    train_descriptor_head.py
    train_flow_matching.py
    evaluate_uncertainty.py
```

Large files must not be committed:

```text
data/
checkpoints/
wandb/
outputs/
*.pt
*.pth
*.ckpt
```

---

## 7. Expected experiment reports

Each experiment must produce a short `report.md`.

Required sections:

```text
# Experiment title

## Goal

## Configuration

## Dataset

## Method

## Metrics

## Results

## Figures

## What worked

## What failed

## Next steps
```

A result without a report is incomplete. A report without a reproducible config is incomplete.

---

## 8. Minimal technical stack

Recommended:

```text
Python 3.10+
PyTorch
torchvision
numpy
pandas
scikit-learn
matplotlib
tqdm
PyYAML
```

Optional:

```text
timm
transformers
datasets
accelerate
wandb or clearml
umap-learn
```

Keep the first version simple. Logging and reliable evaluation are more important than a complex model.

---

## 9. References

### Training dynamics and noisy labels

1. Swayamdipta et al. **Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics.** EMNLP 2020.  
   https://arxiv.org/abs/2009.10795

2. Toneva et al. **An Empirical Study of Example Forgetting during Deep Neural Network Learning.** ICLR 2019.  
   https://arxiv.org/abs/1812.05159

3. Pleiss et al. **Identifying Mislabeled Data using the Area Under the Margin Ranking.** NeurIPS 2020.  
   https://arxiv.org/abs/2001.10528

4. Wei et al. **Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations.** ICLR 2022.  
   https://arxiv.org/abs/2110.12088

5. Hendrycks and Gimpel. **A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.** ICLR 2017.  
   https://arxiv.org/abs/1610.02136

6. Liu et al. **Energy-based Out-of-distribution Detection.** NeurIPS 2020.  
   https://arxiv.org/abs/2010.03759

### Flow matching and generative uncertainty

7. Lipman et al. **Flow Matching for Generative Modeling.** ICLR 2023.  
   https://arxiv.org/abs/2210.02747

8. Tong et al. **Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport.** ICML 2024.  
   https://arxiv.org/abs/2302.00482

### LLM decision dynamics and hallucination uncertainty

9. Kuhn et al. **Detecting hallucinations in large language models using semantic entropy.** Nature 2024.  
   https://www.nature.com/articles/s41586-024-07421-0

10. Vazhentsev et al. **Efficient Unsupervised Uncertainty Quantification for LLMs.** 2025.  
    https://arxiv.org/abs/2505.20045

11. Lin et al. **TruthfulQA: Measuring How Models Mimic Human Falsehoods.** ACL 2022.  
    https://arxiv.org/abs/2109.07958

---

## 10. One-sentence summary

**Room for Doubt** studies whether uncertainty can be recovered from the dynamics of learning and deciding, then distilled into practical uncertainty estimators for noisy-label detection, OOD detection, calibration, and selective prediction.
