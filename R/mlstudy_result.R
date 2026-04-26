#' Create an mlstudy_result object
#'
#' Constructor for the S3 class `mlstudy_result`. This object stores the
#' results of a full tidymodels classification pipeline: the train/test split,
#' cross-validated tuning results for three model families, the selected best
#' model, the finalised workflow, and test-set predictions and metrics.
#'
#' @param split        An `rsplit` object from [rsample::initial_split()].
#' @param target       Character string: name of the binary factor response column.
#' @param cv_compare   A data frame (one row per model) with at least columns
#'   `model`, `mean` (CV ROC AUC), and `std_err`.
#' @param best         Character string: name of the winning model.
#' @param tune_results Named list of `tune_results` objects - one per model family.
#' @param best_params  Named list of data frames - best hyper-parameters per model.
#' @param final_wf     The finalised `workflow` object for the best model.
#' @param final_res    The `last_fit` result (from [tune::last_fit()]).
#' @param test_preds   Data frame of test-set predictions from
#'   [tune::collect_predictions()].
#' @param test_metrics Data frame of test-set metrics from
#'   [tune::collect_metrics()].
#'
#' @return An object of class `mlstudy_result`.
#' @export
#'
#' @examples
#' # Normally produced by ml_compare(); direct construction is for testing only.
#' obj <- new_mlstudy_result(
#'   split        = NULL,
#'   target       = "Status",
#'   cv_compare   = data.frame(model = "RF", mean = 0.84, std_err = 0.01),
#'   best         = "RF",
#'   tune_results = list(),
#'   best_params  = list(),
#'   final_wf     = NULL,
#'   final_res    = NULL,
#'   test_preds   = data.frame(),
#'   test_metrics = data.frame()
#' )
#' class(obj)
new_mlstudy_result <- function(split, target, cv_compare, best,
                                tune_results, best_params,
                                final_wf, final_res,
                                test_preds, test_metrics) {
  stopifnot(is.character(target), length(target) == 1)
  stopifnot(is.character(best),   length(best)   == 1)
  stopifnot(is.data.frame(cv_compare))
  stopifnot(is.list(tune_results))
  stopifnot(is.list(best_params))

  structure(
    list(
      split        = split,
      target       = target,
      cv_compare   = cv_compare,
      best         = best,
      tune_results = tune_results,
      best_params  = best_params,
      final_wf     = final_wf,
      final_res    = final_res,
      test_preds   = test_preds,
      test_metrics = test_metrics
    ),
    class = "mlstudy_result"
  )
}

#' Print an mlstudy_result
#'
#' Prints a compact overview of the experiment: data dimensions, cross-validated
#' model comparison, the selected best model, and test-set performance metrics.
#'
#' @param x   An object of class `mlstudy_result`.
#' @param ... Further arguments (currently unused).
#'
#' @return Invisibly returns `x`.
#' @export
#'
#' @examples
#' \donttest{
#' library(modeldata)
#' data(credit_data)
#' credit_data <- tidyr::drop_na(credit_data)
#' result <- ml_compare(credit_data, target = "Status")
#' print(result)
#' }
print.mlstudy_result <- function(x, ...) {
  if (!is.null(x$split)) {
    train_n <- nrow(rsample::training(x$split))
    test_n  <- nrow(rsample::testing(x$split))
  } else {
    train_n <- NA_integer_
    test_n  <- NA_integer_
  }

  cat("=== mlstudy_result ===\n")
  cat(sprintf("Target variable : %s\n", x$target))
  cat(sprintf("Training rows   : %s\n", train_n))
  cat(sprintf("Test rows       : %s\n", test_n))
  cat(sprintf("Best model      : %s\n\n", x$best))

  cmp <- x$cv_compare[, intersect(c("model", "mean", "std_err", "n"),
                                   names(x$cv_compare)), drop = FALSE]
  names(cmp)[names(cmp) == "mean"]    <- "cv_roc_auc"
  names(cmp)[names(cmp) == "std_err"] <- "std_err"
  cat("CV comparison (sorted by ROC AUC):\n")
  print(cmp, row.names = FALSE, digits = 4)

  if (nrow(x$test_metrics) > 0) {
    cat("\nTest-set performance:\n")
    m <- x$test_metrics[, c(".metric", ".estimate"), drop = FALSE]
    print(m, row.names = FALSE, digits = 4)
  }

  invisible(x)
}

#' Summarise an mlstudy_result
#'
#' Extends [print.mlstudy_result()] with the best hyper-parameters for each
#' model family and the top-5 cross-validated configurations for the winning
#' model.
#'
#' @param object An object of class `mlstudy_result`.
#' @param ...    Further arguments (currently unused).
#'
#' @return Invisibly returns `object`.
#' @export
#'
#' @examples
#' \donttest{
#' library(modeldata)
#' data(credit_data)
#' credit_data <- tidyr::drop_na(credit_data)
#' result <- ml_compare(credit_data, target = "Status")
#' summary(result)
#' }
summary.mlstudy_result <- function(object, ...) {
  print(object)

  cat("\n--- Best hyper-parameters per model ---\n")
  for (nm in names(object$best_params)) {
    cat(sprintf("%s:\n", nm))
    print(object$best_params[[nm]], row.names = FALSE)
  }

  if (length(object$tune_results) > 0 &&
      object$best %in% names(object$tune_results)) {
    cat(sprintf("\n--- Top 5 tuning configurations for %s (by ROC AUC) ---\n",
                object$best))
    top5 <- tune::show_best(object$tune_results[[object$best]],
                             metric = "roc_auc", n = 5)
    print(top5, row.names = FALSE, digits = 4)
  }

  invisible(object)
}

#' Plot an mlstudy_result
#'
#' Produces two ggplot2 panels: (1) a bar chart of cross-validated ROC AUC
#' for each model family (best model highlighted in green), and (2) the ROC
#' curve for the best model evaluated on the held-out test set.
#'
#' @param x   An object of class `mlstudy_result`.
#' @param ... Further arguments (currently unused).
#'
#' @return Invisibly returns `x`.
#' @export
#'
#' @examples
#' \donttest{
#' library(modeldata)
#' data(credit_data)
#' credit_data <- tidyr::drop_na(credit_data)
#' result <- ml_compare(credit_data, target = "Status")
#' plot(result)
#' }
plot.mlstudy_result <- function(x, ...) {

  cmp <- x$cv_compare[, intersect(c("model", "mean", "std_err"),
                                   names(x$cv_compare)), drop = FALSE]

  p1 <- ggplot2::ggplot(
    cmp,
    ggplot2::aes(
      x    = stats::reorder(.data[["model"]], .data[["mean"]]),
      y    = .data[["mean"]],
      fill = .data[["model"]] == x$best
    )
  ) +
    ggplot2::geom_col(show.legend = FALSE) +
    ggplot2::geom_errorbar(
      ggplot2::aes(
        ymin = .data[["mean"]] - .data[["std_err"]],
        ymax = .data[["mean"]] + .data[["std_err"]]
      ),
      width = 0.25
    ) +
    ggplot2::scale_fill_manual(
      values = c("FALSE" = "#95a5a6", "TRUE" = "#2ecc71")
    ) +
    ggplot2::coord_flip() +
    ggplot2::labs(
      title = "Cross-validated ROC AUC by model (best in green)",
      x     = NULL,
      y     = "Mean ROC AUC"
    ) +
    ggplot2::theme_minimal()

  target_vals <- x$test_preds[[x$target]]
  event_level <- if (is.factor(target_vals)) {
    levels(target_vals)[1]
  } else {
    pred_cols <- grep("^\\.pred_(?!class$)", names(x$test_preds),
                      perl = TRUE, value = TRUE)
    sub("^\\.pred_", "", pred_cols[1])
  }
  pred_col <- paste0(".pred_", event_level)

  roc_df <- rlang::inject(
    yardstick::roc_curve(
      x$test_preds,
      truth = !!rlang::sym(x$target),
      !!rlang::sym(pred_col)
    )
  )

  test_auc <- x$test_metrics$.estimate[x$test_metrics$.metric == "roc_auc"]

  p2 <- ggplot2::autoplot(roc_df) +
    ggplot2::labs(
      title    = sprintf("ROC curve - %s (test set)", x$best),
      subtitle = sprintf("AUC = %.4f", test_auc[1])
    ) +
    ggplot2::theme_minimal()

  print(p1)
  print(p2)

  invisible(x)
}
