# Sentient Things Recommendator


Persuasive Recommender System for Sentient Things project.

## Folder description

- `data`: Contains the data used in the experiments.
- `reports`: Contains the reports generated from the experiments and results from the models.
- `second_iteration`: Contains the source code for the second iteration of the experiments.
- `src`: Contains the source code for the experiments.
- `nudging_regression`: Contains the source code and README for the experiment regarding "Persuasion principle score prediction and fusion with optimal ranking"

## How to run the experiments

There are two different experiments in this repository. The first one is the baseline experiment created for the first iteration of the experiments and in the second iteration, we have merged several models to create a hybrid model and enhace the baseline. 

To run the experiments, follow the steps below:

1. Install the requirements:

```bash
pip install -r requirements.txt
```

2. Run the experiments:

```bash
# Baseline
cd src/models
python pre_prolific_exp.py

# Second iteration
cd second_iteration
python process_regression_3.py
```
