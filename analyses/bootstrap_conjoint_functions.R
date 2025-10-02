library(cregg)
library(parallel)

# This code contains helper functions to apply bootstrap estimation for
# estimators in cregg. All functions include parallelization for speed.

# Bootstrap function for calculating AMCEs
cj_bsp <- function(data, formula, n_bootstrap = 1000, ci_level = 0.95, n_cores = 12) {
    # Use mclapply for parallel bootstrapping
    bootstrap_estimates <- mclapply(1:n_bootstrap, function(i) {
        # Resample the data with replacement
        resample_data <- data[sample(1:nrow(data), replace = TRUE), ]

        # Recalculate AMCEs on the resampled data
        amce_resample <- cj(resample_data, formula)

        # Return the estimates from the resampled data
        return(amce_resample$estimate)
    }, mc.cores = n_cores)

    # Convert the list of bootstrap estimates to a matrix
    bootstrap_estimates <- do.call(cbind, bootstrap_estimates)

    # Calculate the original AMCE
    original_amce <- cj(data, formula)

    # Get the desired confidence intervals (percentile-based)
    lower_bound <- (1 - ci_level) / 2
    upper_bound <- 1 - lower_bound

    ci_lower <- apply(bootstrap_estimates, 1, quantile, probs = lower_bound)
    ci_upper <- apply(bootstrap_estimates, 1, quantile, probs = upper_bound)

    # Combine the original estimates with the CIs
    result <- data.frame(
        outcome = original_amce$outcome,
        statistic = original_amce$statistic,
        feature = original_amce$feature,
        level = original_amce$level,
        estimate = original_amce$estimate,
        std.error = original_amce$std.error,
        z = original_amce$z,
        p = original_amce$p,
        lower = ci_lower,
        upper = ci_upper,
        out = original_amce$out
    )

    return(result)
}

# Define the bootstrap function with parallelization for AMCE differences
amce_diffs_bsp <- function(data, formula, by_formula, n_bootstrap = 1000, ci_level = 0.95, n_cores = 12) {
    # Use mclapply for parallel bootstrapping
    bootstrap_estimates <- mclapply(1:n_bootstrap, function(i) {
        # Resample the data with replacement
        resample_data <- data[sample(1:nrow(data), replace = TRUE), ]

        # Recalculate AMCE differences on the resampled data
        amce_resample <- amce_diffs(resample_data, formula, by = by_formula)

        # Return the estimates from the resampled data
        return(amce_resample$estimate)
    }, mc.cores = n_cores)

    # Convert the list of bootstrap estimates to a matrix
    bootstrap_estimates <- do.call(cbind, bootstrap_estimates)

    # Calculate the original AMCE differences
    original_amce <- amce_diffs(data, formula, by = by_formula)

    # Get the desired confidence intervals (percentile-based)
    lower_bound <- (1 - ci_level) / 2
    upper_bound <- 1 - lower_bound

    ci_lower <- apply(bootstrap_estimates, 1, quantile, probs = lower_bound)
    ci_upper <- apply(bootstrap_estimates, 1, quantile, probs = upper_bound)

    # Combine the original estimates with the CIs
    result <- data.frame(
        BY = original_amce$BY,
        outcome = original_amce$outcome,
        statistic = original_amce$statistic,
        feature = original_amce$feature,
        level = original_amce$level,
        identity = original_amce$identity,
        estimate = original_amce$estimate,
        std.error = original_amce$std.error,
        z = original_amce$z,
        p = original_amce$p,
        lower = ci_lower,
        upper = ci_upper,
        out = original_amce$out
    )

    return(result)
}

# Define the bootstrap function with parallelization for MM differences
mm_diffs_bsp <- function(data, formula, by_formula, n_bootstrap = 1000, ci_level = 0.95, n_cores = 12) {
    # Use mclapply for parallel bootstrapping
    bootstrap_estimates <- mclapply(1:n_bootstrap, function(i) {
        # Resample the data with replacement
        resample_data <- data[sample(1:nrow(data), replace = TRUE), ]

        # Recalculate AMCE differences on the resampled data
        mm_resample <- mm_diffs(resample_data, formula, by = by_formula)

        # Return the estimates from the resampled data
        return(mm_resample$estimate)
    }, mc.cores = n_cores)

    # Convert the list of bootstrap estimates to a matrix
    bootstrap_estimates <- do.call(cbind, bootstrap_estimates)

    # Calculate the original AMCE differences
    original_mm <- mm_diffs(data, formula, by = by_formula)

    # Get the desired confidence intervals (percentile-based)
    lower_bound <- (1 - ci_level) / 2
    upper_bound <- 1 - lower_bound

    ci_lower <- apply(bootstrap_estimates, 1, quantile, probs = lower_bound)
    ci_upper <- apply(bootstrap_estimates, 1, quantile, probs = upper_bound)

    # Combine the original estimates with the CIs
    result <- data.frame(
        BY = original_mm$BY,
        outcome = original_mm$outcome,
        statistic = original_mm$statistic,
        feature = original_mm$feature,
        level = original_mm$level,
        identity = original_mm$identity,
        estimate = original_mm$estimate,
        std.error = original_mm$std.error,
        z = original_mm$z,
        p = original_mm$p,
        lower = ci_lower,
        upper = ci_upper,
        out = original_mm$out
    )

    return(result)
}
