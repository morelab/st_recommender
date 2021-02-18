This folder contains all the data to conduct experiments with the new data acquired from prolific and previous data. This folder contains two different dataset with which we conducted three different experiments. Both datasets contain the same files:

- User_*.json --> Data from the users with their characteristics.
- Rankings_*.csv --> Ordered rankings for all the users.
- Ratings_*.json --> Ratings given by each user to each strategy.

### *_ALL FILE DESCRIPTION

[1:295] --> Data from PRE 

[296:360] --> Data from POST

[361:743] --> Data from Prolific

### First experiments

We used the *_prolific_post files for this experiment.

Splits:

- Train: Data from prolific. (PROLIFIC) IDX: :383
- Test: Data from the questionnaire done after Green Soul.(POST) IDX: 384:

### Second experiment

We used the *_all files for this experiment.

Splits:

- Train: Data from the first questionnaire and the questionnaire after Green Soul.(PRE+POST) IDX: :360
- Test: Data from prolific. (PROLIFIC) IDX: 361:

### Third experiment

We used the *_all files for this experiment. In this experiment we run 10 experiments and the splits are randomized at the start. We used 80% of the dataset to train the model and 20% for testing.

### Fourth experiment

We used the *_all files for this experiment.

Splits:

- Train: Data from the first questionnaire (PRE) IDX: :295
- Test: Data from prolific. (PROLIFIC) IDX: 361: