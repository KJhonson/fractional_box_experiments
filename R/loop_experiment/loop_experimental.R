library(Matrix)
library(fmesher)
library(rSPDE)
library(future)
library(future.apply)

source("Basic_functions.R")
source("Dualmesh(final+parallelize).R")
source("sinc_solver_opt.R")

kappa <- 1


## Creating overkill base env

overkill_env <- base_env(8, "LM", 2, kappa)

levels <- c(2,3,4,5,6,7)

approx_env <- base_env(levels, "LM", 2, kappa)

## Auxiliary matrices

print("Start building B")

B <- list()
for (i in seq_along(approx_env)) {
B[[i]] <- dm_evaluator(approx_env[[i]]$mesh, overkill_env$mesh, boundary=TRUE)
}

print("B builted")

A <- list()
for (i in seq_along(approx_env)) {
A[[i]] <- fm_evaluator(approx_env[[i]]$mesh, overkill_env$mesh$loc[, 1:2])$proj$A
}

# Save and reload a runtime-generated checkpoint. This keeps the experiment
# self-contained while allowing the expensive mesh/matrix setup to be inspected
# or reused after a run; checkpoint.RData is generated output and is ignored by Git.
save.image(file = "checkpoint.RData")

# Remove all objects from the environment
# rm(list = ls())

# Load the saved workspace

load("checkpoint.RData")

kappa <- 1


# Experiment configuration
betas <- c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
beta_for_k <- min(betas)

# Function f
x <- overkill_env$mesh$loc[, 1]
y <- overkill_env$mesh$loc[, 2]

func <- function(x, y) {
  inside <- (x >= 0 & x <= 1 & y >= 0 & y <= 1)
  val <- ifelse((x - 0.5) * (y - 0.5) < 0, 1, -1)
  val[!inside] <- NA
  return(val)
}

f <- func(x, y)

# RHS for overkill mesh
RHS_over <- f * diag(overkill_env$fem_matrices$c0)

# Mesh sizes
h <- as.numeric(sapply(approx_env, function(env) env$h))

# h <- numeric(length(levels))
# for (i in seq_along(levels)) {
#   h[i] <- 1 / sqrt(approx_env[[i]]$mesh$n)
# }

h_overkill <- overkill_env$h

# Pre-compute shifts (cache) per mesh with dynamic k
# Dynamic k for overkill
k_overkill <- sinc_dynamic_k(h_overkill, beta_for_k)
rng_over <- sinc_range_for_betas(betas, k_overkill)

# Overkill cache
cache_over <- sinc_precompute_shifts(
  A = overkill_env$operator_matrix,
  C = overkill_env$fem_matrices$m2,
  RHS = RHS_over,
  k = k_overkill,
  lmin = rng_over$lmin, lmax = rng_over$lmax,
  parallel = TRUE, workers = 4
)

# Caches for approximate meshes: each with its dynamic k
caches_approx <- vector("list", length(approx_env))
k_approx <- numeric(length(approx_env))

for (i in seq_along(approx_env)) {
  g_approx <- t(B[[i]]) %*% RHS_over
  k_approx[i] <- sinc_dynamic_k(h[i], beta_for_k)
  rng_approx <- sinc_range_for_betas(betas, k_approx[i])
  
  caches_approx[[i]] <- sinc_precompute_shifts(
    A = approx_env[[i]]$operator_matrix,
    C = approx_env[[i]]$fem_matrices$m2,
    RHS = g_approx,
    k = k_approx[i],
    lmin = rng_approx$lmin, lmax = rng_approx$lmax,
    parallel = TRUE, workers = 4
  )
}

# Error computation loop
errs <- list()
for (beta in betas) {
  # Build u_overkill for current beta
  u_over <- sinc_assemble_from_cache(cache_over, beta)
  # For each approx mesh, build u_approx and compute error
  err_vec <- numeric(length(approx_env))
  for (i in seq_along(approx_env)) {
    u_approx <- sinc_assemble_from_cache(caches_approx[[i]], beta)
    r_i <- u_over - A[[i]] %*% u_approx
    err_vec[i] <- as.numeric(t(r_i) %*% overkill_env$fem_matrices$c1 %*% r_i)
  }
  errs[[as.character(beta)]] <- sqrt(err_vec)
}

# Output tables
error_matrix2 <- do.call(cbind, errs)
error_matrix <- error_matrix2[2:6,]
beta_names <- names(errs)
colnames(error_matrix) <- beta_names

p<-1/h 
p<- p[2:6]

# Slopes per column (log-log)
col_slopes <- numeric(ncol(error_matrix))
for (i in seq_len(ncol(error_matrix))) {
  fit <- lm(log(error_matrix[, i]) ~ log(p))
  col_slopes[i] <- -coef(fit)[2]
}

# Write output files
df <- data.frame(h = p, error_matrix, check.names = FALSE)
write.table(df, "errors2d.dat",
            sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)

df_2 <- data.frame(x = as.numeric(beta_names), y = col_slopes, check.names = FALSE)
write.table(df_2, "betaxslope.dat",
            sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)

# Debug: show dynamic k values used
cat("Dynamic k values used:\n")
cat("Overkill (h =", h_overkill, "):", k_overkill, "\n")
for (i in seq_along(approx_env)) {
  cat("Mesh", i, "(h =", p[i], "):", k_approx[i], "\n")
}
cat("\n")

print(col_slopes)
