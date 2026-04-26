#' Split a data frame into stratified training and test sets
#'
#' A thin wrapper around [rsample::initial_split()] that enforces stratified
#' sampling so that both the training and test sets preserve the class
#' distribution of the response variable.
#'
#' @param data        A data frame containing all predictors and the response.
#' @param target      Character string: name of the binary factor response column.
#'   Used as the stratification variable.
#' @param prop_train  Numeric in (0, 1): fraction of rows assigned to training.
#'   Default is `0.8` (80 / 20 split).
#' @param seed        Integer random seed for reproducibility. Default `3`.
#'
#' @return An `rsplit` object. Use [rsample::training()] and
#'   [rsample::testing()] to extract the two sets, or pass the object
#'   directly to [ml_compare()].
#' @export
#'
#' @examples
#' library(modeldata)
#' data(credit_data)
#' credit_data <- tidyr::drop_na(credit_data)
#' split <- ml_split(credit_data, target = "Status")
#' nrow(rsample::training(split))
#' nrow(rsample::testing(split))
ml_split <- function(data, target, prop_train = 0.8, seed = 3) {
  stopifnot(is.data.frame(data))
  stopifnot(is.character(target), length(target) == 1, target %in% names(data))
  stopifnot(is.numeric(prop_train), prop_train > 0, prop_train < 1)

  set.seed(seed)
  rsample::initial_split(data, prop = prop_train, strata = target)
}
