library(tidyverse)
library(randomForest)
# library(rrr) 			# Uncomment for multi-variate regression

source("src/utils.R")

set.seed(100)

source("src/load_data_cv.R")
# Makes the following variables available:
# d
# ordvars
# catvars
# catvars_onehot
# principles
# n_princ
# splits

resAll <- list()
importancesAll <- list()
for (s in 1:5) {
	dtrain <- d[splits[[s]]$train_idx, ]
	dtest <- d[splits[[s]]$test_idx, ]

	dtrain <- removeNA(dtrain)	
	dtest <- removeNA(dtest)	

	# Regress on the order of each principle (random forest)
	rmses <- rep(0, length(principles))
	rmses_dummy <- rep(0, length(principles))
	names(rmses) <- principles
	names(rmses_dummy) <- principles
	preds_certh <- matrix(0, nrow = nrow(dtest), ncol = length(principles))
	importances <- list()
	for (i in 1:length(principles)) {
		print(glue("principle {i}"))
		p <- principles[i]
		dat <- dtrain %>%
			mutate(target = map_int(order_gt, `[`(p))) %>%
			select(all_of(c(numvars, ordvars, catvars_onehot)), target)
		dat_mat <- as.matrix(select(dat, -target))
		dat_targ <- dat$target
		dat_test <- dtest %>%
			mutate(target = map_int(order_gt, `[`(p))) %>%
			select(all_of(c(numvars, ordvars, catvars_onehot)), target)
		dat_test_mat <- as.matrix(select(dat_test, -target))

		dat_2 <- dtrain %>%
			mutate(target = map_int(order_gt, `[`(p))) %>%
			select(all_of(c(numvars, ordvars, catvars)), target)
		dat_test_2 <- dtest %>%
			mutate(target = map_int(order_gt, `[`(p))) %>%
			select(all_of(c(numvars, ordvars, catvars)), target)


		mdl <- randomForest(target ~ .,
							data = dat_2,
							ntree = 1000,
							mtry = 2
		)
		# print(mdl)
		pred <- predict(mdl, dat_test_2)
		rmses[i] <- rmse((pred - 1) / 2.25 + 1, (dat_test_2$target - 1) / 2.25 + 1)
		preds_certh[, i] <- pred

		# rmses_dummy[i] <- rmse(rep(mean(dat$target), nrow(dtest)), dat_test$target)

		importances[[principles[i]]] <- mdl$importance
	}
	importancesAll[[s]] <- importances



	# Multivariate regression code
	# Uncomment to use
	# --------------------------------------------------------------------

	# X_train <- dtrain[, c(numvars, ordvars, catvars_onehot)]
	# X_test <- dtest[, c(numvars, ordvars, catvars_onehot)]
	# Y_train <- as.matrix(do.call(rbind, dtrain$order_gt))
	# Y_test <- as.matrix(do.call(rbind, dtest$order_gt))

	# # # Classical (essentially not multivariate)
	# # mdl_mv_cl <- lm(Y_train ~ ., data = X_train)
	# # preds_certh <- predict(mdl_mv_cl, X_test)

	# # Reduced-rank regression
	# mdl_mv_rr <- rrr(X_train, Y_train, rank = 5, k = 1e-10)
	# mu <- mdl_mv_rr$mean
	# A <- mdl_mv_rr$A
	# B <- mdl_mv_rr$B
	# C <- mdl_mv_rr$C
	# preds_certh <- t(mu[, rep(1, nrow(X_test))] + C %*% t(X_test))

	# colnames(preds_certh) <- principles
	# rmses <- sqrt(colMeans(((preds_certh / 2.25 + 1) - ((Y_test - 1) / 2.25 + 1))^2))

	# importances <- mdl_mv_rr$C %>%
	# 	abs() %>%
	# 	colMeans()
	# 	# as_tibble(rownames = "predictor") %>%
	# 	# rename(importance = value)
	# importancesAll[[s]] <- importances

	# --------------------------------------------------------------------


	colnames(preds_certh) <- principles
	ratings_certh <- as_tibble(preds_certh) %>%
		rowwise() %>%
		mutate(ratings_certh = list(c_across(all_of(principles)))) %>%
		pull(ratings_certh)
	
	order_certh <- ratings_certh %>%
		map(~ 11 - rank(.x))

	order_deusto <- dtest[[glue("order_deusto_{s}")]]

	trials <- c("certh", "deusto", "filt", "parallel")

	# Predicted ranking, only with CERTH ratings
	top_certh <- map(order_certh, ~ orderToRank(.x, principles))
	# Predicted ranking, only with DEUSTO ranking
	top_deusto <- map(order_deusto, ~ orderToRank(.x, principles))
	# Predicted ranking, with CERTH ratings, after filtering through DEUSTO (sequential fusion)
	top_filt <- map2(ratings_certh, order_deusto, function(r_certh, o_deusto) {
		idx_deusto <- which(o_deusto < 5)
		principles[idx_deusto][order(r_certh[idx_deusto])]
	})
	# Predicted ranking, using parallel fusion
	top_parallel <- map2(order_certh, order_deusto, ~ orderToRank(combineOrders(.x, .y), principles))

	order_5_gt <- map(dtest$top_gt, ~ rankToOrder(.x[1:5], principles))
	order_5_certh <- map(top_certh, ~ rankToOrder(.x[1:5], principles))
	order_5_deusto <- map(top_deusto, ~ rankToOrder(.x[1:5], principles))
	order_filt <- map(top_filt, ~ rankToOrder(.x, principles))
	order_parallel <- map(top_parallel, ~ rankToOrder(.x, principles))

	tau_5_certh <- map2_dbl(order_5_certh, order_5_gt, cor, method = "kendall")
	tau_5_deusto <- map2_dbl(order_5_deusto, order_5_gt, cor, method = "kendall")
	tau_filt <- map2_dbl(order_filt, order_5_gt, cor, method = "kendall")
	
	mean_tau_5_certh <- mean(tau_5_certh)
	mean_tau_5_deusto <- mean(tau_5_deusto)
	mean_tau_filt <- mean(tau_filt)

	tau_certh <- map2_dbl(order_certh, dtest$order_gt, cor, method = "kendall")
	tau_deusto <- map2_dbl(order_deusto, dtest$order_gt, cor, method = "kendall")
	tau_parallel <- map2_dbl(order_parallel, dtest$order_gt, cor, method = "kendall")

	mean_tau_certh <- mean(tau_certh)
	mean_tau_deusto <- mean(tau_deusto)
	mean_tau_parallel <- mean(tau_parallel)

	ndpm_certh <- map2_dbl(order_certh, dtest$order_gt, ndpm)
	ndpm_deusto <- map2_dbl(order_deusto, dtest$order_gt, ndpm)
	ndpm_parallel <- map2_dbl(order_parallel, dtest$order_gt, ndpm)
	mean_ndpm_certh <- mean(ndpm_certh)
	mean_ndpm_deusto <- mean(ndpm_deusto)
	mean_ndpm_parallel <- mean(ndpm_parallel)

	ndpm_5_certh <- map2_dbl(order_5_certh, order_5_gt, ndpm)
	ndpm_5_deusto <- map2_dbl(order_5_deusto, order_5_gt, ndpm)
	ndpm_filt <- map2_dbl(order_filt, order_5_gt, ndpm)
	mean_ndpm_5_certh <- mean(ndpm_certh)
	mean_ndpm_5_deusto <- mean(ndpm_deusto)
	mean_ndpm_filt <- mean(ndpm_filt)

	acc_certh <- mean(map2_dbl(dtest$top_gt, top_certh, ~ ncommon(.x, .y, 1)))
	print(acc_certh)
	acc_deusto <- mean(map2_dbl(dtest$top_gt, top_deusto, ~ ncommon(.x, .y, 1)))
	print(acc_deusto)
	acc_filt <- mean(map2_dbl(dtest$top_gt, top_filt, ~ ncommon(.x, .y, 1)))
	print(acc_filt)
	most_freq_dtrain <- dtrain$top_gt %>% map_chr(~ .x[1]) %>% table() %>% which.max() %>% names()
	acc_parallel <- mean(map2_dbl(dtest$top_gt, top_parallel, ~ ncommon(.x, .y, 1)))
	print(acc_parallel)
	acc_dummy <- mean(map2_dbl(dtest$top_gt, rep(most_freq_dtrain, nrow(dtest)), ~ ncommon(.x, .y, 1)))
	print(acc_dummy)

	top_1_gt <- map_chr(dtest$top_gt, ~ .x[1])
	top_1_certh <- map_chr(top_certh, ~ .x[1])
	top_1_deusto <- map_chr(top_deusto, ~ .x[1])
	top_1_filt <- map_chr(top_filt, ~ .x[1])
	top_1_parallel <- map_chr(top_parallel, ~ .x[1])
	top_1_dummy <- rep(most_freq_dtrain, nrow(dtest))
	top_1_random <- sample(principles, nrow(dtest), replace = TRUE)

	acc_random <- mean(top_1_gt == top_1_random)
	
	fscore_certh <- fscore(top_1_gt, top_1_certh, principles)
	fscore_deusto <- fscore(top_1_gt, top_1_deusto , principles)
	fscore_filt <- fscore(top_1_gt, top_1_filt, principles)
	fscore_parallel <- fscore(top_1_gt, top_1_parallel, principles)
	fscore_dummy <- fscore(top_1_gt, top_1_dummy, principles)
	fscore_random <- fscore(top_1_gt, top_1_random, principles)


	# Grouped principles
	top_1_gt_grouped <- map_chr(top_1_gt, ~ princ_groups[[.x]])
	top_1_certh_grouped <- map_chr(top_1_certh, ~ princ_groups[[.x]])
	top_1_deusto_grouped <- map_chr(top_1_deusto, ~ princ_groups[[.x]])
	top_1_filt_grouped <- map_chr(top_1_filt, ~ princ_groups[[.x]])
	top_1_parallel_grouped <- map_chr(top_1_parallel, ~ princ_groups[[.x]])

	most_freq_group <- table(map_chr(dtrain$top_gt, ~ princ_groups[[.x[1]]])) %>% which.max() %>% names()
	top_1_dummy_grouped <- rep(most_freq_group, nrow(dtest))

	top_1_dummy_grouped <- map_chr(top_1_dummy, ~ princ_groups[[.x]])
	top_1_random_grouped <- map_chr(top_1_random, ~ princ_groups[[.x]])

	acc_certh_grouped <- mean(map2_dbl(top_1_gt_grouped, top_1_certh_grouped, ~ ncommon(.x, .y, 1)))
	acc_deusto_grouped <- mean(map2_dbl(top_1_gt_grouped, top_1_deusto_grouped, ~ ncommon(.x, .y, 1)))
	acc_filt_grouped <- mean(map2_dbl(top_1_gt_grouped, top_1_filt_grouped, ~ ncommon(.x, .y, 1)))
	acc_parallel_grouped <- mean(map2_dbl(top_1_gt_grouped, top_1_parallel_grouped, ~ ncommon(.x, .y, 1)))
	acc_dummy_grouped <- mean(map2_dbl(top_1_gt_grouped, top_1_dummy_grouped, ~ ncommon(.x, .y, 1)))

	bin_levels <- c("people", "system")
	fscore_certh_grouped <- fscore(top_1_gt_grouped, top_1_certh_grouped, bin_levels)
	fscore_deusto_grouped <- fscore(top_1_gt_grouped, top_1_deusto_grouped, bin_levels)
	fscore_filt_grouped <- fscore(top_1_gt_grouped, top_1_filt_grouped, bin_levels)
	fscore_parallel_grouped <- fscore(top_1_gt_grouped, top_1_parallel_grouped, bin_levels)
	fscore_dummy_grouped <- fscore(top_1_gt_grouped, top_1_dummy_grouped, bin_levels)
	fscore_random_grouped <- fscore(top_1_gt_grouped, top_1_random_grouped, bin_levels)

	rmses_order_certh <- rep(0, length(principles))
	rmses_order_deusto <- rep(0, length(principles))
	rmses_order_parallel <- rep(0, length(principles))
	for (i in 1:length(principles)) {
		o_certh <- map_dbl(order_certh, i)
		o_deusto <- map_dbl(order_deusto, i)
		o_parallel <- map_dbl(order_parallel, i)
		o_gt <- map_dbl(dtest$order_gt, i)
		rmses_order_certh[i] <- rmse(o_certh, o_gt)
		rmses_order_deusto[i] <- rmse(o_deusto, o_gt)
		rmses_order_parallel[i] <- rmse(o_parallel, o_gt)
	}


	res <- list(
		preds_cert = preds_certh,
		rmses = rmses,
		rmses_dummy = rmses_dummy,

		acc_certh = acc_certh,
		acc_deusto = acc_deusto,
		acc_filt = acc_filt,
		acc_parallel = acc_parallel,
		acc_dummy = acc_dummy,
		acc_random = acc_random,

		mean_tau_5_certh = mean_tau_5_certh,
		mean_tau_5_deusto = mean_tau_5_deusto,
		mean_tau_filt = mean_tau_filt,

		mean_tau_certh = mean_tau_certh,
		mean_tau_deusto = mean_tau_deusto,
		mean_tau_parallel = mean_tau_parallel,

		mean_ndpm_certh = mean_ndpm_certh,
		mean_ndpm_deusto = mean_ndpm_deusto,
		mean_ndpm_parallel = mean_ndpm_parallel,

		mean_ndpm_5_certh = mean_ndpm_5_certh,
		mean_ndpm_5_deusto = mean_ndpm_5_deusto,
		mean_ndpm_filt = mean_ndpm_filt,

		fscore_certh = fscore_certh,
		fscore_deusto = fscore_deusto,
		fscore_filt = fscore_filt,
		fscore_parallel = fscore_parallel,
		fscore_dummy = fscore_dummy,
		fscore_random = fscore_random,

		acc_certh_grouped = acc_certh_grouped,
		acc_deusto_grouped = acc_deusto_grouped,
		acc_filt_grouped = acc_filt_grouped,
		acc_parallel_grouped = acc_parallel_grouped,
		acc_dummy_grouped = acc_dummy_grouped,

		fscore_certh_grouped = fscore_certh_grouped,
		fscore_deusto_grouped = fscore_deusto_grouped,
		fscore_filt_grouped = fscore_filt_grouped,
		fscore_parallel_grouped = fscore_parallel_grouped,
		fscore_dummy_grouped = fscore_dummy_grouped,
		fscore_random_grouped = fscore_random_grouped,

		rmses_order_certh = rmses_order_certh,
		rmses_order_deusto = rmses_order_deusto,
		rmses_order_parallel = rmses_order_parallel
	)
	resAll[[s]] <- res
	# resAll[[1]] <- res
}

acc_certh <- mean(map_dbl(resAll, "acc_certh"))
acc_deusto <- mean(map_dbl(resAll, "acc_deusto"))
acc_filt <- mean(map_dbl(resAll, "acc_filt"))
acc_parallel <- mean(map_dbl(resAll, "acc_parallel"))
acc_dummy <- mean(map_dbl(resAll, "acc_dummy"))

mean_tau_5_certh <- mean(map_dbl(resAll, "mean_tau_5_certh"))
mean_tau_5_deusto <- mean(map_dbl(resAll, "mean_tau_5_deusto"))
mean_tau_filt <- mean(map_dbl(resAll, "mean_tau_filt"))

mean_tau_certh <- mean(map_dbl(resAll, "mean_tau_certh"))
mean_tau_deusto <- mean(map_dbl(resAll, "mean_tau_deusto"))
mean_tau_filt <- mean(map_dbl(resAll, "mean_tau_filt"))

print("accuracy")
print(acc_certh)
print(acc_deusto)
print(acc_filt)
print(acc_parallel)
print(acc_dummy)
print(acc_random)

print("tau 5")
print(mean_tau_5_certh)
print(mean_tau_5_deusto)
print(mean_tau_filt)

print("tau")
print(mean_tau_certh)
print(mean_tau_deusto)
print(mean_tau_parallel)

print("ndpm")
mean_ndpm_certh <- mean(map_dbl(resAll, "mean_ndpm_certh"))
mean_ndpm_deusto <- mean(map_dbl(resAll, "mean_ndpm_deusto"))
mean_ndpm_parallel <- mean(map_dbl(resAll, "mean_ndpm_parallel"))
print(mean_ndpm_certh)
print(mean_ndpm_deusto)
print(mean_ndpm_parallel)

print("ndpm 5")
mean_ndpm_5_certh <- mean(map_dbl(resAll, "mean_ndpm_5_certh"))
mean_ndpm_5_deusto <- mean(map_dbl(resAll, "mean_ndpm_5_deusto"))
mean_ndpm_filt <- mean(map_dbl(resAll, "mean_ndpm_filt"))
print(mean_ndpm_5_certh)
print(mean_ndpm_5_deusto)
print(mean_ndpm_filt)


print("macro fscore")
macro_fscore_certh <- mean(map_dbl(resAll, c("fscore_certh", "macro_fscore")))
macro_fscore_deusto <- mean(map_dbl(resAll, c("fscore_deusto", "macro_fscore")))
macro_fscore_filt <- mean(map_dbl(resAll, c("fscore_filt", "macro_fscore")))
macro_fscore_parallel <- mean(map_dbl(resAll, c("fscore_parallel", "macro_fscore")))
macro_fscore_dummy <- mean(map_dbl(resAll, c("fscore_dummy", "macro_fscore")))
macro_fscore_random <- mean(map_dbl(resAll, c("fscore_random", "macro_fscore")))
print(macro_fscore_certh)
print(macro_fscore_deusto)
print(macro_fscore_filt)
print(macro_fscore_parallel)
print(macro_fscore_dummy)
print(macro_fscore_random)

print("weighted macro fscore")
weighted_macro_fscore_certh <- mean(map_dbl(resAll, c("fscore_certh", "weighted_macro_fscore")))
weighted_macro_fscore_deusto <- mean(map_dbl(resAll, c("fscore_deusto", "weighted_macro_fscore")))
weighted_macro_fscore_filt <- mean(map_dbl(resAll, c("fscore_filt", "weighted_macro_fscore")))
weighted_macro_fscore_parallel <- mean(map_dbl(resAll, c("fscore_parallel", "weighted_macro_fscore")))
weighted_macro_fscore_dummy <- mean(map_dbl(resAll, c("fscore_dummy", "weighted_macro_fscore")))
weighted_macro_fscore_random <- mean(map_dbl(resAll, c("fscore_dummy", "weighted_macro_fscore")))
print(weighted_macro_fscore_certh)
print(weighted_macro_fscore_deusto)
print(weighted_macro_fscore_filt)
print(weighted_macro_fscore_parallel)
print(weighted_macro_fscore_dummy)
print(weighted_macro_fscore_random)

print("micro fscore")
micro_fscore_certh <- mean(map_dbl(resAll, c("fscore_certh", "micro_fscore")))
micro_fscore_deusto <- mean(map_dbl(resAll, c("fscore_deusto", "micro_fscore")))
micro_fscore_filt <- mean(map_dbl(resAll, c("fscore_filt", "micro_fscore")))
micro_fscore_parallel <- mean(map_dbl(resAll, c("fscore_parallel", "micro_fscore")))
micro_fscore_dummy <- mean(map_dbl(resAll, c("fscore_dummy", "micro_fscore")))
micro_fscore_random <- mean(map_dbl(resAll, c("fscore_random", "micro_fscore")))
print(micro_fscore_certh)
print(micro_fscore_deusto)
print(micro_fscore_filt)
print(micro_fscore_parallel)
print(micro_fscore_dummy)
print(micro_fscore_random)

print("accuracy grouped")
acc_certh_grouped <- mean(map_dbl(resAll, "acc_certh_grouped"))
acc_deusto_grouped <- mean(map_dbl(resAll, "acc_deusto_grouped"))
acc_filt_grouped <- mean(map_dbl(resAll, "acc_filt_grouped"))
acc_parallel_grouped <- mean(map_dbl(resAll, "acc_parallel_grouped"))
acc_dummy_grouped <- mean(map_dbl(resAll, "acc_dummy_grouped"))
print(acc_certh_grouped)
print(acc_deusto_grouped)
print(acc_filt_grouped)
print(acc_parallel_grouped)
print(acc_dummy_grouped)

print("macro fscore grouped")
macro_fscore_certh_grouped <- mean(map_dbl(resAll, c("fscore_certh_grouped", "macro_fscore")))
macro_fscore_deusto_grouped <- mean(map_dbl(resAll, c("fscore_deusto_grouped", "macro_fscore")))
macro_fscore_filt_grouped <- mean(map_dbl(resAll, c("fscore_filt_grouped", "macro_fscore")))
macro_fscore_parallel_grouped <- mean(map_dbl(resAll, c("fscore_parallel_grouped", "macro_fscore")))
macro_fscore_dummy_grouped <- mean(map_dbl(resAll, c("fscore_dummy_grouped", "macro_fscore")))
macro_fscore_random_grouped <- mean(map_dbl(resAll, c("fscore_random_grouped", "macro_fscore")))
print(macro_fscore_certh_grouped)
print(macro_fscore_deusto_grouped)
print(macro_fscore_filt_grouped)
print(macro_fscore_parallel_grouped)
print(macro_fscore_dummy_grouped)
print(macro_fscore_random_grouped)

print("weighted_macro fscore grouped")
weighted_macro_fscore_certh_grouped <- mean(map_dbl(resAll, c("fscore_certh_grouped", "weighted_macro_fscore")))
weighted_macro_fscore_deusto_grouped <- mean(map_dbl(resAll, c("fscore_deusto_grouped", "weighted_macro_fscore")))
weighted_macro_fscore_filt_grouped <- mean(map_dbl(resAll, c("fscore_filt_grouped", "weighted_macro_fscore")))
weighted_macro_fscore_parallel_grouped <- mean(map_dbl(resAll, c("fscore_parallel_grouped", "weighted_macro_fscore")))
weighted_macro_fscore_dummy_grouped <- mean(map_dbl(resAll, c("fscore_dummy_grouped", "weighted_macro_fscore")))
weighted_macro_fscore_random_grouped <- mean(map_dbl(resAll, c("fscore_random_grouped", "weighted_macro_fscore")))
print(weighted_macro_fscore_certh_grouped)
print(weighted_macro_fscore_deusto_grouped)
print(weighted_macro_fscore_filt_grouped)
print(weighted_macro_fscore_parallel_grouped)
print(weighted_macro_fscore_dummy_grouped)
print(weighted_macro_fscore_random_grouped)

print("micro fscore grouped")
micro_fscore_certh_grouped <- mean(map_dbl(resAll, c("fscore_certh_grouped", "micro_fscore")))
micro_fscore_deusto_grouped <- mean(map_dbl(resAll, c("fscore_deusto_grouped", "micro_fscore")))
micro_fscore_filt_grouped <- mean(map_dbl(resAll, c("fscore_filt_grouped", "micro_fscore")))
micro_fscore_parallel_grouped <- mean(map_dbl(resAll, c("fscore_parallel_grouped", "micro_fscore")))
micro_fscore_dummy_grouped <- mean(map_dbl(resAll, c("fscore_dummy_grouped", "micro_fscore")))
micro_fscore_random_grouped <- mean(map_dbl(resAll, c("fscore_random_grouped", "micro_fscore")))
print(micro_fscore_certh_grouped)
print(micro_fscore_deusto_grouped)
print(micro_fscore_filt_grouped)
print(micro_fscore_parallel_grouped)
print(micro_fscore_dummy_grouped)
print(micro_fscore_random_grouped)



mean_rmses <- map(resAll, "rmses") %>%
	do.call(rbind, .) %>%
	colMeans()

lb_rmses <- map(resAll, "rmses") %>%
	do.call(rbind, .) %>%
	apply(2, quantile, 0.025)

ub_rmses <- map(resAll, "rmses") %>%
	do.call(rbind, .) %>%
	apply(2, quantile, 0.975)

mean_rmses_dummy <- map(resAll, "rmses_dummy") %>%
	do.call(rbind, .) %>%
	colMeans()

rmses_df <- tibble(
	principles = map_chr(principles, ~ princ_names[[.x]]),
	rmses = mean_rmses,
	lb_rmses = lb_rmses,
	ub_rmses = ub_rmses,
	rmses_dummy = mean_rmses_dummy
)

# g <- ggplot(rmses_df, aes(x = reorder(principles, mean_rmses), y = mean_rmses)) +
	# geom_col() +
    # geom_text(aes(label = round(rmses, 2)), nudge_y = 0.15, size = 3) + 
	# ylim(0, 2) +
	# xlab("Persuasion principle") +
	# ylab("RMSE (cross-validation)")
# print(g)
# ggsave("output/rmse_per_principle.png", width = 5, height = 4)

g <- ggplot(rmses_df, aes(x = reorder(principles, mean_rmses), y = mean_rmses)) +
	geom_col(fill = "#a0a0a0") +
	geom_point() +
	geom_errorbar(aes(ymin = lb_rmses, ymax = ub_rmses)) +
	geom_text(aes(label = round(rmses, 2)), nudge_y = 0.17, size = 3) + 
	ylim(0, 2) +
	theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
	xlab("Persuasion principle") +
	ylab("RMSE (cross-validation)")
print(g)
# ggsave("output/rmse_per_principle_errorbars.png", width = 5, height = 4)

# g <- ggplot(rmses_df_all, aes(x = reorder(principles, rmses), y = rmses)) +
# 	# geom_col(fill = "#a0a0a0") +
# 	geom_errorbar(
# 		aes(color = method, ymin = lb_rmses, ymax = ub_rmses, width = 0.4),
# 		alpha = 0.7
# 	) +
# 	geom_line(aes(color = method, group = method)) +
# 	geom_point(aes(color = method)) +
# 	# geom_text(aes(label = round(rmses, 2)), nudge_y = 0.17, size = 3) + 
# 	ylim(0, 1.8) +
# 	theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
# 	xlab("Persuasion principle") +
# 	ylab("RMSE (cross-validation)")
# print(g)
# ggsave("output/rmse_per_principle_errorbars_comp.png", width = 5, height = 4)

# g <- ggplot(rmses_df, aes(x = principles)) +
	# geom_point(aes(y = mean_rmses)) +
	# geom_point(aes(y = mean_rmses_dummy), color = "red")
# print(g)

print("rmses order")
mean_rmses_order_certh <- map(resAll, "rmses_order_certh") %>%
	do.call(rbind, .) %>%
	colMeans()
mean_rmses_order_deusto <- map(resAll, "rmses_order_deusto") %>%
	do.call(rbind, .) %>%
	colMeans()
mean_rmses_order_parallel <- map(resAll, "rmses_order_parallel") %>%
	do.call(rbind, .) %>%
	colMeans()
print(mean_rmses_order_certh)
print(mean_rmses_order_deusto)
print(mean_rmses_order_parallel)



print("Summary results")
print("-----------------------------------")

print("macro")
print(macro_fscore_certh)
print(macro_fscore_filt)

print("weighted macro")
print(weighted_macro_fscore_certh)
print(weighted_macro_fscore_filt)

print("micro")
print(micro_fscore_certh)
print(micro_fscore_filt)


mean_importances <- reduce(map(importancesAll, ~ reduce(.x, `+`) / length(principles)), `+`) / length(importancesAll)
importances_df <- as_tibble(mean_importances, rownames = "predictor")

# mean_importances <- reduce(importancesAll, `+`) / length(importancesAll)
# importances_df <- as_tibble(mean_importances, rownames = "predictor")
# importances_df <- importances_df %>%
	# mutate(group = map_chr(str_split(predictor, "\\."), 1)) %>%
	# group_by(group) %>%
	# summarise(IncNodePurity = mean(value)) %>%
	# rename(predictor = group)

g <- ggplot(importances_df, aes(x = reorder(predictor, -IncNodePurity), y = IncNodePurity)) +
	geom_col() +
	theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
	xlab("Predictor") +
	ylab("Importance (residual sum of squares)")
print(g)
ggsave("output/importances_rf.png", width = 5, height = 4)


