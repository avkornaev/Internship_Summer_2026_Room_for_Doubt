Create a new Git project named `room-for-doubt` for a 6-week student research internship on uncertainty estimation from training dynamics.

Use the README.md and AGENTS.md provided by the supervisor as the project specification.

Initial implementation goal:

1. Create a clean Python research repository for CIFAR-10N experiments.
2. Add a dataset-loading module that can load CIFAR-10 images together with CIFAR-10N noisy labels and original clean labels.
3. Add a small ResNet-18 training script that trains on a selected noisy-label split.
4. Log per-sample training dynamics across epochs: noisy-label probability, clean-label probability for evaluation only, max confidence, margin, prediction, loss, noisy-label correctness, clean-label correctness.
5. Add metric code for Dataset Cartography descriptors, forgetting count, and AUM-like score.
6. Add a first experiment script that compares final confidence, final loss, margin, variability, forgetting count, and AUM-like score for noisy-label detection.
7. Save metrics, figures, and a short experiment report.
8. Add smoke tests that run on a tiny subset without requiring a GPU.

Constraints:

- Do not implement the flow-matching expert project yet.
- Do not add unnecessary frameworks.
- Do not commit datasets, checkpoints, or large logs.
- Keep code simple and readable for students.
- Every generated result must be reproducible from a command in the README.
