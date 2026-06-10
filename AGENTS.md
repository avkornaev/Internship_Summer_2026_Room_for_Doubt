# AGENTS.md

This file defines how AI coding agents should work in this repository. It is written for students who may be using an agent-driven workflow for the first time.

## Project identity

Repository name: `room-for-doubt`

Scientific theme: uncertainty estimation from training dynamics and decision-making dynamics.

Main goal: build reproducible experiments showing whether trajectories of learning or internal computation provide better uncertainty estimates than static confidence scores.

## General rules for agents

1. Do not invent results.
2. Do not silently change the scientific question.
3. Do not add large frameworks unless they are necessary.
4. Prefer small, readable, testable code over clever abstractions.
5. Every experiment must produce a short `report.md` with results, figures, and limitations.
6. Every metric must be computed from saved predictions or saved trajectories, not copied manually.
7. Every plot must be reproducible from a script.
8. Every non-trivial change must update the README or the relevant experiment report.
9. If something fails, document the failure instead of hiding it.
10. Never commit datasets, large checkpoints, logs, or generated cache files.

## Expected agent behavior

Before editing code, the agent should state:

- what file it will change;
- why the change is needed;
- how the change will be tested;
- what output should be produced.

After editing code, the agent should state:

- what changed;
- how to run it;
- what files are expected as outputs;
- what still needs human review.

## Scientific constraints

The repository is not about maximizing CIFAR accuracy. Accuracy is useful, but the real target is uncertainty.

The main comparisons should include:

- final softmax confidence;
- entropy;
- energy score;
- margin;
- Dataset Cartography descriptors;
- forgetting count;
- AUM-like score;
- descriptor-prediction head;
- flow-matching model only after simpler baselines work.

The agent must not implement the expert model before the beginner and intermediate baselines are functional.

## Data rules

Datasets should be downloaded by scripts or documented instructions.

Do not commit:

- raw datasets;
- CIFAR binary files;
- checkpoints;
- tensor caches;
- wandb runs;
- large CSV logs;
- generated images larger than needed for reports.

Use `.gitignore` for these files.

If a dataset requires manual download, the agent should create a clear instruction in the README or in the relevant experiment report.

## Experiment rules

Each experiment should have one clear question.

Good experiment questions:

- Does AUM detect CIFAR-10N label noise better than final confidence?
- Can a small MLP predict AUM from a final embedding?
- Does predicted trajectory uncertainty improve selective classification?
- Does conditional flow matching improve over deterministic descriptor regression?

Bad experiment questions:

- Can we improve everything?
- Can we train a big model?
- Can we get the best CIFAR score?
- Can we add a transformer because it is modern?

## Reporting rules

Every experiment report should contain:

1. Research question.
2. Dataset and label split.
3. Model and training setup.
4. Metrics.
5. Main table.
6. Main figures.
7. What worked.
8. What failed.
9. Next steps.

Reports should be honest and compact. A negative result with a clean ablation is valuable.

## Coding style

Use clear names. Avoid hidden global state.

Preferred conventions:

- one script should do one thing;
- configuration should be saved with the experiment outputs;
- random seeds should be explicit;
- metrics should be saved as `.csv` or `.json`;
- figures should be saved as `.png` or `.pdf`;
- checkpoints should be ignored by Git.

## Testing expectations

For every new script, include at least one quick test mode.

Examples:

- train for 1 epoch on a tiny subset;
- compute metrics on 100 samples;
- generate one small diagnostic plot;
- run a smoke test without GPU.

The agent should not claim the code works unless it has run a smoke test or clearly says that it has not been run.

## Working with papers

When adding a paper reference, include:

- full paper title;
- author list or first author plus “et al.”;
- venue or year;
- one sentence explaining why it matters for this project.

Do not add irrelevant references to make the project look larger.

## Working with Codex or other coding agents

Use small prompts. Ask for one unit of work at a time.

Good prompt:

> Implement CIFAR-10N loading and verify that clean and noisy labels are both available. Add a tiny smoke test that prints label disagreement rate.

Bad prompt:

> Implement the whole project.

After every agent-generated change, the student must inspect the diff before committing.

## Commit rules

Use small commits with descriptive names.

Examples:

- `add cifar10n dataset loader`
- `log per-sample confidence trajectories`
- `compute aum and forgetting metrics`
- `add descriptor prediction baseline`
- `write first experiment report`

Avoid vague commit messages such as:

- `update`
- `fix`
- `changes`
- `final`

## Human review checklist

Before accepting an agent change, check:

- Does the code answer the intended research question?
- Are the metrics correct?
- Are clean labels used only for evaluation?
- Are noisy labels used for training when intended?
- Are random seeds fixed?
- Are outputs saved in a reproducible way?
- Is anything large accidentally added to Git?
- Is the report updated?

## Final reminder

The goal is not to make the repository look impressive. The goal is to produce a small number of reliable experiments that can support a workshop-style research story.
