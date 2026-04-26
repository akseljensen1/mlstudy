#' Compare three model families on a binary classification task
#'
#' Runs the full tidymodels pipeline mirroring the self-study 1 analysis:
#'
#' 1. **Split** - stratified 80/20 train/test partition via [ml_split()].
#' 2. **Recipe** - dummy-encodes nominal predictors, removes zero-variance
#'    columns, and normalises numeric predictors.
#' 3. **Models** - specifies three families with tunable hyper-parameters:
#'    Random Forest ([ranger][ranger::ranger]), Elastic Net logistic regression
#'    ([glmnet][glmnet::glmnet]), and XGBoost ([xgboost][xgboost::xgboost]).
#' 4. **Grid search** - runs [tune::tune_grid()] on `cv_folds`-fold
#'    cross-validation for each model family using [dials::grid_regular()].
#' 5. **Selection** - picks the model with the highest mean CV ROC AUC.
#' 6. **Final fit** - finalises the winning workflow with its best
#'    hyper-parameters and evaluates it on the held-out test set via
#'    [tune::last_fit()].
#'
#' @param data        A data frame containing all predictors and the response.
#' @param target      Character string: name of the binary factor response column.
#' @param prop_train  Fraction of rows for training (passed to [ml_split()]).
#'   Default `0.8`.
#' @param seed_split  Random seed for the train/test split. Default `3`.
#' @param seed_tune   Random seed for cross-validation and tuning. Default `42`.
#' @param cv_folds    Number of cross-validation folds. Default `5`.
#' @param rf_trees    Number of trees for the Random Forest. Default `300`.
#' @param xgb_trees   Number of trees for XGBoost. Default `300`.
#' @param grid_levels Number of values per hyper-parameter in the tuning grid
#'   (passed to [dials::grid_regular()]). Default `4`.
#'
#' @return An object of class `mlstudy_result` (see [new_mlstudy_result()]).
#' @export
#'
#' @examples
#' \donttest{
#' library(modeldata)
#' data(credit_data)
#' credit_data <- tidyr::drop_na(credit_data)
#' result <- ml_compare(credit_data, target = "Status")
#' print(result)
#' plot(result)
#' }
ml_compare <- function(data, target,
                       prop_train  = 0.8,
                       seed_split  = 3,
                       seed_tune   = 42,
                       cv_folds    = 5,
                       rf_trees    = 300,
                       xgb_trees   = 300,
                       grid_levels = 4) {

  stopifnot(is.data.frame(data))
  stopifnot(is.character(target), length(target) == 1, target %in% names(data))

  split      <- ml_split(data, target, prop_train = prop_train, seed = seed_split)
  train_data <- rsample::training(split)

  formula_obj <- stats::as.formula(paste(target, "~ ."))

  rec <- recipes::recipe(formula_obj, data = train_data) |>
    recipes::step_dummy(recipes::all_nominal_predictors()) |>
    recipes::step_zv(recipes::all_predictors()) |>
    recipes::step_normalize(recipes::all_numeric_predictors())

  rf_spec <- parsnip::rand_forest(
    mode  = "classification",
    trees = rf_trees,
    mtry  = tune::tune(),
    min_n = tune::tune()
  ) |> parsnip::set_engine("ranger")

  elastic_spec <- parsnip::logistic_reg(
    mode    = "classification",
    penalty = tune::tune(),
    mixture = tune::tune()
  ) |> parsnip::set_engine("glmnet")

  xgb_spec <- parsnip::boost_tree(
    mode       = "classification",
    trees      = xgb_trees,
    tree_depth = tune::tune(),
    learn_rate = tune::tune(),
    min_n      = tune::tune()
  ) |> parsnip::set_engine("xgboost")

  rf_wf <- workflows::workflow() |>
    workflows::add_model(rf_spec) |>
    workflows::add_recipe(rec)

  elastic_wf <- workflows::workflow() |>
    workflows::add_model(elastic_spec) |>
    workflows::add_recipe(rec)

  xgb_wf <- workflows::workflow() |>
    workflows::add_model(xgb_spec) |>
    workflows::add_recipe(rec)

  set.seed(seed_tune)
  cv_folds_obj <- rsample::vfold_cv(train_data, v = cv_folds, strata = target)

  rf_grid <- dials::grid_regular(
    dials::mtry(range = c(2L, 10L)),
    dials::min_n(range = c(2L, 20L)),
    levels = grid_levels
  )

  elastic_grid <- dials::grid_regular(
    dials::penalty(range = c(-4, 0)),
    dials::mixture(range = c(0, 1)),
    levels = grid_levels
  )

  xgb_grid <- dials::grid_regular(
    dials::tree_depth(range = c(2L, 8L)),
    dials::learn_rate(range = c(-3, -1)),
    dials::min_n(range = c(2L, 20L)),
    levels = grid_levels
  )

  metric_set_obj <- yardstick::metric_set(yardstick::roc_auc, yardstick::accuracy)

  rf_tune_res <- tune::tune_grid(
    rf_wf,
    resamples = cv_folds_obj,
    grid      = rf_grid,
    metrics   = metric_set_obj
  )

  elastic_tune_res <- tune::tune_grid(
    elastic_wf,
    resamples = cv_folds_obj,
    grid      = elastic_grid,
    metrics   = metric_set_obj
  )

  xgb_tune_res <- tune::tune_grid(
    xgb_wf,
    resamples = cv_folds_obj,
    grid      = xgb_grid,
    metrics   = metric_set_obj
  )

  cv_compare <- dplyr::bind_rows(
    tune::show_best(rf_tune_res,      metric = "roc_auc", n = 1) |>
      dplyr::mutate(model = "Random Forest"),
    tune::show_best(elastic_tune_res, metric = "roc_auc", n = 1) |>
      dplyr::mutate(model = "Elastic Net"),
    tune::show_best(xgb_tune_res,     metric = "roc_auc", n = 1) |>
      dplyr::mutate(model = "XGBoost")
  ) |> dplyr::arrange(dplyr::desc(.data[["mean"]]))

  best_model_name <- cv_compare$model[1]

  best_rf      <- tune::select_best(rf_tune_res,      metric = "roc_auc")
  best_elastic <- tune::select_best(elastic_tune_res, metric = "roc_auc")
  best_xgb     <- tune::select_best(xgb_tune_res,     metric = "roc_auc")

  best_params <- list(
    `Random Forest` = best_rf,
    `Elastic Net`   = best_elastic,
    `XGBoost`       = best_xgb
  )

  final_wf <- switch(
    best_model_name,
    "Random Forest" = tune::finalize_workflow(rf_wf,      best_rf),
    "Elastic Net"   = tune::finalize_workflow(elastic_wf, best_elastic),
    "XGBoost"       = tune::finalize_workflow(xgb_wf,     best_xgb)
  )

  final_res <- tune::last_fit(
    final_wf,
    split   = split,
    metrics = metric_set_obj
  )

  test_preds   <- tune::collect_predictions(final_res)
  test_metrics <- tune::collect_metrics(final_res)

  tune_results <- list(
    `Random Forest` = rf_tune_res,
    `Elastic Net`   = elastic_tune_res,
    `XGBoost`       = xgb_tune_res
  )

  new_mlstudy_result(
    split        = split,
    target       = target,
    cv_compare   = cv_compare,
    best         = best_model_name,
    tune_results = tune_results,
    best_params  = best_params,
    final_wf     = final_wf,
    final_res    = final_res,
    test_preds   = test_preds,
    test_metrics = test_metrics
  )
}
