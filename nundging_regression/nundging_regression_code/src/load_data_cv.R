library(tidyverse)
library(jsonlite)
library(glue)
library(caret)

source("src/utils.R")

path = "data"

users <- fromJSON(glue("{path}/dataset/users_all.json"), flatten = TRUE) %>%
	as_tibble()
features = list(
	"Age"			= "features.0",
	"Gender"		= "features.1",
	"Education"		= "features.2",
	"Country"		= "features.3",
	"Work_culture"	= "features.4",
	"PST"			= "features.5",
	"Barriers"		= "features.6",
	"Intentions"	= "features.7",
	"Confidence"	= "features.8"
)
users <- rename(users, !!!features)

ordvars <- c("Age", "Education")
catvars <- c("Gender", "Country", "Work_culture", "PST", "Barriers", "Intentions")
numvars <- c("Confidence")

users <- mutate(users, across(all_of(catvars), factor))

onehot_mdl <- dummyVars(
	as.formula(glue("~ {paste(catvars, collapse = ' + ')}")),
	data = users,
	fullRank = TRUE
)
users_onehot <- as_tibble(predict(onehot_mdl, newdata = users))
catvars_onehot <- colnames(users_onehot)
users <- as_tibble(cbind(users, users_onehot))

idx_pre <- 1:295
idx_post <- 296:360
idx_prol <- 361:743
users <- users %>%
	arrange(userId) %>%
	mutate(pilot = "") %>%
	relocate(userId, pilot)
users$pilot[idx_pre] <- "pre"
users$pilot[idx_post] <- "post"
users$pilot[idx_prol] <- "prol"

gt_rankings <- read_csv(
	glue("{path}/dataset/gt_rankings_all.csv"),
	col_names = glue("top_{1:10}")
)

top_gt <- gt_rankings %>%
	rowwise() %>%
	mutate(top_gt = list(c_across(everything()))) %>%
	pull(top_gt)

principles <- as.character(gt_rankings[1, ])
principles <- principles[substring(principles, 2) %>% as.numeric() %>% order()]
n_princ <- length(principles)

princ_names <- list(
	"v2" = "Social recognition",
	"v5" = "Physical attractiveness",
	"v6" = "Conditioning",
	"v7" = "Reciprocity",
	"v10" = "Authority",
	"v11" = "Self-monitoring",
	"v15" = "Cause & effect",
	"v17" = "Social proof",
	"v19" = "Suggestion",
	"v20" = "Similarity"
)

princ_groups <- list(
	"v2" = "people",
	"v5" = "system",
	"v6" = "system",
	"v7" = "people",
	"v10" = "people",
	"v11" = "system",
	"v15" = "system",
	"v17" = "people",
	"v19" = "system",
	"v20" = "people"
)

order_gt <- map(top_gt, rankToOrder, principles)


d <- users
d$top_gt <- top_gt
d$order_gt <- order_gt


# Train-test splits and results
ids_cross <- read_lines(glue("{path}/cross_validation_new/ids_cross.txt")) %>%
	as.numeric()

split_idx <- list()
split_idx[[1]] <- 1:135
split_idx[[2]] <- 136:270
split_idx[[3]] <- 271:405
split_idx[[4]] <- 406:540
split_idx[[5]] <- 541:678

nsplits <- length(split_idx)
splits <- map(1:nsplits, function(i) {
	split <- list()
	split$test_idx <- ids_cross[split_idx[[i]]]
	split$train_idx <- ids_cross[setdiff(1:678, split_idx[[i]])]
	filename <- glue("{path}/cross_validation_new/ranks_split{i}.csv")
	split$order_deusto <- read_csv(filename) %>%
		select(all_of(principles)) %>%
		rowwise() %>%
		mutate(order_deusto = list(c_across(everything()))) %>%
		pull(order_deusto)
	split
})

splits[["pre_prol"]] <- list(
	test_idx = idx_prol,
	train_idx = idx_pre,
	order_deusto = read_csv(glue("{path}/dataset/predictions_ranks_pre_prolific.csv")) %>%
		select(all_of(principles)) %>%
		rowwise() %>%
		mutate(order_deusto = list(c_across(everything()))) %>%
		pull(order_deusto)
)

for (i in 1:length(splits)) {
	cname <- glue("order_deusto_{i}")
	d[[cname]] <- rep(NA, nrow(d))
	d[[cname]][1:length(splits[[i]]$order_deusto)] <- splits[[i]]$order_deusto
}


# d <- drop_na(d)
# # d <- filter(d, rowAny(across(starts_with("order_"), ~ map_lgl(.x, function(y) !any(is.na(y))))))
# d <- filter(d, map_lgl(order_deusto_1, ~ !any(is.na(.x))))
# d <- filter(d, map_lgl(order_deusto_2, ~ !any(is.na(.x))))
# d <- filter(d, map_lgl(order_deusto_3, ~ !any(is.na(.x))))
# d <- filter(d, map_lgl(order_deusto_4, ~ !any(is.na(.x))))
# d <- filter(d, map_lgl(order_deusto_5, ~ !any(is.na(.x))))





