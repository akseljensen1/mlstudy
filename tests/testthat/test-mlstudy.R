# Relatively simple tests for each module

library(modeldata)

make_tiny_data <- function(n = 60, seed = 1) {
  set.seed(seed)
  data.frame(
    y  = factor(rep(c("bad", "good"), each = n / 2)),
    x1 = rnorm(n),
    x2 = rnorm(n),
    x3 = rnorm(n)
  )
}

tiny <- make_tiny_data()

# ---- ml_split ---------------------------------------------------------------

test_that("ml_split returns an rsplit object", {
  sp <- ml_split(tiny, target = "y")
  expect_s3_class(sp, "rsplit")
})

test_that("ml_split partitions all rows without overlap", {
  sp    <- ml_split(tiny, target = "y", prop_train = 0.75, seed = 7)
  tr_n  <- nrow(rsample::training(sp))
  te_n  <- nrow(rsample::testing(sp))
  expect_equal(tr_n + te_n, nrow(tiny))
  expect_true(tr_n > te_n)
})

test_that("ml_split validates inputs", {
  expect_error(ml_split("not_a_df", target = "y"))
  expect_error(ml_split(tiny, target = "no_such_column"))
  expect_error(ml_split(tiny, target = "y", prop_train = 1.5))
})

# ---- new_mlstudy_result -----------------------------------------------------

test_that("new_mlstudy_result constructs the S3 object", {
  obj <- new_mlstudy_result(
    split        = NULL,
    target       = "Status",
    cv_compare   = data.frame(model = "RF", mean = 0.84, std_err = 0.01),
    best         = "RF",
    tune_results = list(),
    best_params  = list(),
    final_wf     = NULL,
    final_res    = NULL,
    test_preds   = data.frame(),
    test_metrics = data.frame()
  )
  expect_s3_class(obj, "mlstudy_result")
  expect_equal(obj$target, "Status")
  expect_equal(obj$best, "RF")
})

test_that("new_mlstudy_result validates target and best", {
  expect_error(
    new_mlstudy_result(
      split = NULL, target = 123, cv_compare = data.frame(),
      best = "RF", tune_results = list(), best_params = list(),
      final_wf = NULL, final_res = NULL,
      test_preds = data.frame(), test_metrics = data.frame()
    )
  )
  expect_error(
    new_mlstudy_result(
      split = NULL, target = "y", cv_compare = data.frame(),
      best = 99L, tune_results = list(), best_params = list(),
      final_wf = NULL, final_res = NULL,
      test_preds = data.frame(), test_metrics = data.frame()
    )
  )
})

# ---- print / summary --------------------------------------------------------

test_that("print.mlstudy_result outputs without error", {
  obj <- new_mlstudy_result(
    split        = ml_split(tiny, "y"),
    target       = "y",
    cv_compare   = data.frame(model   = c("Random Forest", "Elastic Net", "XGBoost"),
                               mean    = c(0.84, 0.80, 0.83),
                               std_err = c(0.01, 0.02, 0.01),
                               n       = c(5L, 5L, 5L)),
    best         = "Random Forest",
    tune_results = list(),
    best_params  = list(),
    final_wf     = NULL,
    final_res    = NULL,
    test_preds   = data.frame(),
    test_metrics = data.frame(.metric = c("roc_auc", "accuracy"),
                               .estimate = c(0.85, 0.79))
  )
  expect_output(print(obj),   "mlstudy_result")
  expect_output(print(obj),   "Random Forest")
  expect_output(summary(obj), "mlstudy_result")
})

# ---- ml_compare (integration, skipped on CRAN) ------------------------------

test_that("ml_compare returns mlstudy_result with correct structure", {
  skip_on_cran()
  skip_if_not_installed("ranger")
  skip_if_not_installed("glmnet")
  skip_if_not_installed("xgboost")

  result <- ml_compare(
    tiny,
    target      = "y",
    cv_folds    = 2,
    grid_levels = 2,
    rf_trees    = 50,
    xgb_trees   = 50
  )

  expect_s3_class(result, "mlstudy_result")
  expect_true(result$best %in% c("Random Forest", "Elastic Net", "XGBoost"))
  expect_true(nrow(result$cv_compare) == 3)
  expect_true(nrow(result$test_metrics) >= 2)
  expect_true(all(c("roc_auc", "accuracy") %in% result$test_metrics$.metric))
})
