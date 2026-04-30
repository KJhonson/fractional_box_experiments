library(Matrix)
library(fmesher)
library(rSPDE)




source("Basic_functions.R")
source("Dualmesh(final+parallelize).R")


kappa <- 1


# ## Creating overkill base

# overkill_env <- base_env(8, "LM", 2, kappa)

# ## Creating approximate base

# # aux.fun <- function(x,step){
# #   return(log2(sqrt((x*step))-1))
# # }

# # step <- 1000
# # quant <- 1:20
# # levels <- aux.fun(quant,step)


# levels <- c(2,3,4,5,6,7)

# approx_env <- base_env(levels, "LM", 2, kappa)


# ## L^{2}-error

# print("Start building B")

# B <- list()

# for (i in seq_along(approx_env)) {
#   B[[i]] <- dm_evaluator(approx_env[[i]]$mesh, overkill_env$mesh, boundary=TRUE)
# }

# print("B builted")

# A <- list()

# for (i in seq_along(approx_env)) {
# A[[i]] <- fm_evaluator(approx_env[[i]]$mesh, overkill_env$mesh$loc[, 1:2])$proj$A
# }



# # Save the whole workspace up to this point
# save.image(file = "checkpoint.RData")

# # Remove all objects from the environment
# rm(list = ls())

# Load the saved workspace
load("checkpoint.RData")


# set.seed(13)
# b <- rnorm(overkill_env$mesh$n, 0, 1)
# W <- b * sqrt(diag(overkill_env$fem_matrices$c0))
# f <- rSPDE_solver(overkill_env$mesh, overkill_env$operator_matrix, overkill_env$fem_matrices$c0, W, beta = 4, scale.factor = 1, m = 1, tau = 1)

# x <- overkill_env$mesh$loc[, 1]
# y <- overkill_env$mesh$loc[, 2]
# f <- cos(20 * pi * x) * cos(20 * pi * y)


beta <- 0.5
m <- 1


n <- 1
p <- 1

x <- overkill_env$mesh$loc[, 1]
y <- overkill_env$mesh$loc[, 2]
u_exact <- cos(p*pi*x)*cos(n*pi*y)
f <- ((kappa + pi^2 * (p^2 + n^2))^beta) * u_exact

# func <- function(x, y, r = 0.397) {
#   ifelse((x - 0.5)^2 + (y - 0.5)^2 <= r^2, 1, 0)
# }


# func <- function(x, y) {
#   r <- sqrt((x-0.5)^2 + (y-0.5)^2)

#   # para evitar divisao por zero, substituimos r=0 por um numero muito pequeno
#   r[r == 0] <- .Machine$double.eps

#   # calcular a funcao
#   val <- 1 / ( r * log(2 / r) )

#   # tambem podemos remover valores NaN ou Inf (por exemplo, no caso de r=0)
#   val[!is.finite(val)] <- NA
#   return(val)
# }




# f<- func(x, y)

print("Overkill load term builted")

g <- f * diag(overkill_env$fem_matrices$c0) # approximate version of a vector containing int_{b_{ok}} f dx

u <- rSPDE_solver(overkill_env$mesh, overkill_env$operator_matrix, overkill_env$fem_matrices$c0, g, beta = beta, scale.factor = 1, m = m, tau = 1)


print("Overkill solution builted")



error <- numeric(length(levels))

for (i in seq_along(approx_env)) {
    g_approx <- t(B[[i]]) %*% g
    u_approx <- rSPDE_solver(approx_env[[i]]$mesh, approx_env[[i]]$operator_matrix, approx_env[[i]]$fem_matrices$c0, g_approx, beta = beta, scale.factor = 1, m = m, tau = 1)
    error[i] <- t(u - A[[i]] %*% u_approx) %*% overkill_env$fem_matrices$c1 %*% (u - A[[i]] %*% u_approx)
}

final_error <- sqrt(error)

cat("Error computed:", final_error, "\n")


# h <- sqrt(sapply(approx_env, function(env) max(diag(env$fem_matrices$c0)))) #same as the obe below!!!!

h <- as.numeric(sapply(approx_env, function(env) env$h))
E <- final_error

# Plot E(h) in log-log scale
plot(h, E, log = "xy", type = "b", pch = 19, col = "blue",
     xlab = "h (mesh size)", ylab = "Error (E)",
     main = "Log-Log Plot of Error vs Mesh Size")

# Add points for clarity (optional, since type="b" already does this)
points(h, E, pch = 19, col = "blue")

# Fit a linear model in log-log scale
log_h <- log(h)
log_E <- log(E)
fit <- lm(log_E ~ log_h)

# Add the fitted line in red
lines(h, exp(fit$coefficients[1] + fit$coefficients[2] * log_h), col = "red", lwd = 2)

# Extract and print the slope
slope <- coef(fit)[2]
cat("Slope log-log:", slope, "\n")
cat("mesh sizes:", h, "\n")
cat("beta:", beta, "\n")
