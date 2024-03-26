# Persuasion principle score prediction and fusion with optimal ranking

The data and code in this directory correspond to the persuasion principle score prediction using Random Forest regression and the fusion of the score prediction method with the optimal ranking method.

## Running the code

The code is written in R. The following libraries need to be installed:

* `tidyverse`
* `jsonlite`
* `glue`
* `caret`
* `randomForest`
* `rrr` (optional - for multi-variate regression)

You also need to create an `output` directory under this root directory, to store some output images.

To produce the results, you need to run the `src/main.R` file. It sources `src/load_data_cv.R`, where you can see how the data are loaded. The results are produced in the command-line window and as charts.

The following naming conventions may help the reader in reading the code and results. The different methods being compared are:

* Most frequent top-1: `*_dummy`
* Optimal ranking: `*_deusto`
* Score prediction: `*_certh`
* Fusion: `*_filt`

The above suffixes are added to the names of the different metrics computed. For example, the metric `macro_fscore_filt` is the Macro F-score computed for the Fusion (RF) method, while the `weighted_macro_fscore_certh` metric is the Weighted macro F-score computed for the score prediction (RF) method.


# Directory structure

The directory contains two sub-directories:

* `data/`, which contains the input data for the code
* `src/`, which contains the source code

The `data/` directory contains the following sub-directories (please also check `src/load_data_cv.R` for more details):

* `dataset`: The input data used. Contains the following files:
	- `users_all.json`: The demographic features for each user.
	- `gt_rankings_all.csv`: The ground truth principle rankings for each user.
	- `predictions_ranks_pre_prolific.csv`: The predictions made by the optimal ranking method for each user.
* `cross_validation_new/`: Contains the cross-validation splits used. These files are used to replicate the same cross-validation splits used in for the optimal ranking method, for comparison.

The `src/` directory contains the following files:

* `load_data.R`: Data loading and cross-validation split handling.
* `utils.R`: Utility functions and performance metrics.
* `main.R`: The main code. Loads input data, fits the prediction models and reports results and performance metrics.


