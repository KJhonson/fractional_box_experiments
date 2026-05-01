# ===============================
# Sinc quadrature with shift cache
# ===============================
# Este arquivo expõe três funções:
# 1) sinc_range_for_betas(betas, k)
# 2) sinc_precompute_shifts(A, C, RHS, k, lmin, lmax, parallel, workers)
# 3) sinc_assemble_from_cache(cache, beta)
#
# Fluxo típico:
#   rng   <- sinc_range_for_betas(betas, k)
#   cache <- sinc_precompute_shifts(A, C, RHS, k, rng$lmin, rng$lmax, TRUE, 4)
#   u     <- sinc_assemble_from_cache(cache, beta)

suppressPackageStartupMessages({
  library(Matrix)
  library(future)
  library(future.apply)
})

# --- 0) Calcula k dinâmico baseado no mesh size ---
sinc_dynamic_k <- function(h, beta) {
  stopifnot(h > 0 & h < 1)
  pi^2 / (-6 * log(h))
}

# --- 1) Descobrir intervalo [lmin, lmax] que cobre TODAS as betas ---
sinc_range_for_betas <- function(betas, k) {
  stopifnot(all(betas > 0 & betas < 1), k > 0)
  N_of <- function(b) as.integer(ceiling(pi^2 / (2 * b       * k^2)))
  M_of <- function(b) as.integer(ceiling(pi^2 / (2 * (1 - b) * k^2)))
  lmax <- max(vapply(betas, N_of, integer(1)))
  lmin <- -max(vapply(betas, M_of, integer(1)))
  list(lmin = lmin, lmax = lmax)
}

# --- 2) Pré-cálculo (cache) de (A + t_l C)^{-1} * RHS em l ∈ [lmin, lmax] ---
sinc_precompute_shifts <- function(A, C, RHS, k, lmin, lmax,
                                   parallel = TRUE,
                                   workers  = max(1L, parallel::detectCores() - 2L)) {
  stopifnot(inherits(A, "sparseMatrix"), inherits(C, "sparseMatrix"), k > 0)
  if (!inherits(A, "dgCMatrix")) A <- as(A, "CsparseMatrix")
  if (!inherits(C, "dgCMatrix")) C <- as(C, "CsparseMatrix")
  RHS <- as.matrix(RHS)

  ls  <- seq.int(lmin, lmax)
  tks <- exp(ls * k)

  # Fatoração simbólica de referência (mesmo padrão esparso e ordenação)
  tk_ref <- 1.0
  Af_ref  <- forceSymmetric(A + tk_ref * C, uplo = "L")
  Af_refS <- as(Af_ref, "dsCMatrix")
  Fsym <- Cholesky(Af_refS, LDL = FALSE, perm = TRUE, super = TRUE)
  cat("✅ Cache: symbolic pattern pronto.\n")

  solve_one <- function(tk) {
    Af  <- forceSymmetric(A + tk * C, uplo = "L")
    AfS <- as(Af, "dsCMatrix")
    Fnum <- try(update(Fsym, AfS), silent = TRUE)
    if (inherits(Fnum, "try-error")) {
      Fnum <- Cholesky(AfS, LDL = FALSE, perm = TRUE, super = TRUE)
    }
    solve(Fnum, RHS, system = "A")
  }

  if (!parallel) {
    sols <- lapply(tks, solve_one)
  } else {
    if (workers < 1L) workers <- 1L
    old_plan <- future::plan(); on.exit(future::plan(old_plan), add = TRUE)
    future::plan(future::multisession, workers = workers)

    old_omp <- Sys.getenv("OMP_NUM_THREADS", unset = NA)
    on.exit({ if (!is.na(old_omp)) Sys.setenv(OMP_NUM_THREADS = old_omp) }, add = TRUE)
    Sys.setenv(OMP_NUM_THREADS = "1")

    sols <- future.apply::future_lapply(tks, solve_one, future.seed = TRUE)
  }

  list(
    lmin = lmin,
    lmax = lmax,
    k    = k,
    ls   = ls,
    tks  = tks,
    sols = sols,        # lista; cada elemento é n x r (mesma shape de RHS)
    ncol_rhs = ncol(RHS)
  )
}

# --- 3) Montagem para um beta específico a partir do cache ---
sinc_assemble_from_cache <- function(cache, beta) {
  stopifnot(beta > 0 && beta < 1)
  k  <- cache$k
  ls <- cache$ls

  # Calcula M e N específicos para este beta exato
  N  <- as.integer(ceiling(pi^2 / (2 * beta       * k^2)))
  M  <- as.integer(ceiling(pi^2 / (2 * (1 - beta) * k^2)))
  need_ls <- (-M):N

  # Verifica se o cache cobre todos os l necessários
  if (min(need_ls) < min(ls) || max(need_ls) > max(ls)) {
    stop("Cache insuficiente para este beta; aumente lmin/lmax no precompute.")
  }
  idx <- match(need_ls, ls)

  # Calcula os pesos corretos usando M e N específicos
  c0  <- (k * sin(pi * beta)) / pi
  wks <- c0 * exp((1 - beta) * need_ls * k)

  U <- 0
  for (j in seq_along(idx)) {
    U <- U + wks[j] * cache$sols[[ idx[j] ]]
  }
  if (cache$ncol_rhs == 1) drop(U) else U
}