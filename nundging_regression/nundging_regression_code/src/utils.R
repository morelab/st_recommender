library(tidyverse)

# Transforms a ranking to an order.
# E.g.
# rankToOrder(c("v6", "v2", "v7", "v5"), c("v2", "v5", "v6", "v7"))
# #> v2 v5 v6 v7
# #>  2  4  1  3
rankToOrder <- function(r, rnames, maxval = 20) {
	o <- map_int(rnames, function(x) {
		idx <- which(r == x)
		ifelse(length(idx) > 0, idx, as.integer(maxval))
	})
	names(o) <- rnames
	o
}
# rankToOrder <- function(r, rnames) {
	# o <- map_int(rnames, ~ which(r == .x))
	# names(o) <- rnames
	# o
# }

orderToRank <- function(o, onames) {
	onames[order(o)]
}

# Combines two orders (each starting at 1).
combineOrders <- function(o1, o2) {
	s <- o1 + o2
	# rank(s, ties.method = "min")		# can return ties
	rank(s, ties.method = "first")
}

# Root Mean Squared Error
rmse <- function(x, y) {
	sqrt(mean((x - y)^2))
}

# Accuracy
accuracy <- function(x, y) {
	mean(x == y)
}

# Counts the common elements in x and y, up to index n
ncommon <- function(x, y, n) {
	length(intersect(x[1:n], y[1:n])) / n
}

rowAny <- function(x) rowSums(x) > 0

removeNA <- function(d) {
	res <- d %>%
		drop_na() %>%
		filter(map_lgl(order_deusto_1, ~ !any(is.na(.x)))) %>%
		filter(map_lgl(order_deusto_2, ~ !any(is.na(.x)))) %>% 
		filter(map_lgl(order_deusto_3, ~ !any(is.na(.x)))) %>%
		filter(map_lgl(order_deusto_4, ~ !any(is.na(.x)))) %>%
		filter(map_lgl(order_deusto_5, ~ !any(is.na(.x))))
	res
}

fscore <- function(gt, pred, levels) {
	# see https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1

	conf_mat <- table(factor(gt, levels = levels), factor(pred, levels = levels))
	precision <- diag(conf_mat) / rowSums(conf_mat)
	recall <- diag(conf_mat) / colSums(conf_mat)
	fscores <- ifelse(is.na(recall), 2 * precision, 2 / (1 / precision + 1 / recall))

	macro_fscore <- mean(fscores)
	weights <- table(gt)[levels]
	weighted_macro_fscore <- sum(weights * fscores) / sum(weights)

	# micro_fscore <- sum(diag(conf_mat)) / (sum(conf_mat) - sum(diag(conf_mat)))
	micro_fscore <- sum(diag(conf_mat)) / sum(conf_mat)

	list(
		macro_fscore = macro_fscore,
		weighted_macro_fscore = weighted_macro_fscore,
		micro_fscore = micro_fscore
	)
}

sgn <- function(x) {
	# if (x > 0) return(1)
	# if (x < 0) return(-1)
	# if (x == 0) return(0)
	ifelse(x > 0, 1, 0)
}

# See https://www.bgu.ac.il/~shanigu/Publications/EvaluationMetrics.17.pdf
ndpm <- function(r, ref) {
	n <- length(r)
	C_plus <- 0
	C_minus <- 0
	C_u <- 0
	C_s <- 0
	for (i in 1:n) {
		for (j in 1:n) {
			if (j <= i)
				next
			C_plus <- C_plus + sgn(ref[i] - ref[j]) * sgn(r[i] - r[j])
			C_minus <- C_minus + sgn(ref[i] - ref[j]) * sgn(r[j] - r[i])
			C_u <- C_u + (sgn(ref[i] - ref[j]))^2
			C_s <- C_s + (sgn(r[i] - r[j]))^2
		}
	}
	C_u0 <- C_u - (C_plus + C_minus)
	res <- (C_minus + 0.5 * C_u0) / C_u
	res
}


