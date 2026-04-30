# ==============================================================================
# DUAL MESH CONSTRUCTION AND EVALUATION FOR FINITE ELEMENT MESHES
# ==============================================================================
# This script implements functions to construct dual meshes from triangular 
# finite element meshes and evaluate functions between different mesh resolutions.
# 
# Author: [Your Name]
# Date: [Date]
# ==============================================================================

# Required Libraries
# ------------------
library(fmesher)  # For finite element mesh generation and manipulation
library(Matrix)   # For sparse matrix operations and linear algebra
library(splancs)  # For spatial point pattern analysis and polygon operations
library(sf)       # For spatial data handling and geometric operations
library(doSNOW)   # For parallel computing using SNOW clusters
library(parallel) # For detectCores() and parallel processing functions


# ==============================================================================
# HELPER FUNCTIONS FOR DUAL MESH CONSTRUCTION
# ==============================================================================

# Function: swap_fun
# ------------------
# Purpose: Swaps rows in a matrix I according to the swap instructions in matrix S
# 
# Parameters:
#   I - Matrix where rows need to be swapped
#   S - Matrix containing pairs of row indices to swap (each row contains two indices)
# 
# Returns:
#   Modified matrix I with swapped rows
# 
# Description:
#   This function is used to reorder the sparse matrix indices to ensure proper
#   correspondence between row and column indices in the finite element connectivity matrix.
swap_fun <- function(I, S) {
  # Iterate through each swap instruction
  for (i in 1:nrow(S)) {
    rows <- S[i, ]  # Get the pair of row indices to swap
    # Swap the specified rows in matrix I
    I[c(rows[1], rows[2]), ] <- I[c(rows[2], rows[1]), ]
  }
  return(I)
}

# Function: node_nbr
# ------------------
# Purpose: Creates a neighborhood list showing which triangles contain each vertex
# 
# Parameters:
#   C - Triangle connectivity matrix (each row contains 3 vertex indices forming a triangle)
# 
# Returns:
#   List where each element contains the triangle indices that include that vertex
# 
# Description:
#   For each vertex in the mesh, this function identifies all triangles that contain
#   that vertex. This is essential for constructing dual mesh cells, as each dual
#   cell corresponds to a vertex and is formed by connecting the centroids of all
#   triangles that share that vertex.
node_nbr <- function(C) {
  # Initialize a list with one element per vertex (max vertex index determines size)
  Nbrd <- vector("list", max(C))
  
  # Iterate through each triangle
  for (i in seq_len(nrow(C))) {
    # For each vertex in the current triangle
    for (j in 1:3) {
      vertex <- C[i, j]  # Get the vertex index
      
      # Add this triangle to the vertex's neighborhood list
      if (is.null(Nbrd[[vertex]])) {
        Nbrd[[vertex]] <- i  # First triangle for this vertex
      } else {
        Nbrd[[vertex]] <- c(Nbrd[[vertex]], i)  # Add to existing list
      }
    }
  }
  return(Nbrd)
}

# Function: ord_poly
# ------------------
# Purpose: Orders polygon vertices in counterclockwise direction around a center point
# 
# Parameters:
#   coords - Matrix of coordinate points (x, y) forming a polygon
#   boundary - Logical flag indicating if this is a boundary polygon
# 
# Returns:
#   Matrix of reordered coordinates forming a closed polygon (first point repeated at end)
# 
# Description:
#   This function ensures that polygon vertices are ordered consistently in a 
#   counterclockwise direction. For boundary polygons, it uses the centroid as 
#   the center point. For interior polygons, it uses the first point as center.
#   This consistent ordering is crucial for proper polygon operations and area calculations.
ord_poly <- function(coords, boundary=FALSE) {
  if(boundary){
    # For boundary polygons: use the geometric center (centroid)
    center <- colMeans(coords)
    # Calculate angles from center to each point
    angles <- atan2(coords[, 2] - center[2], coords[, 1] - center[1])
    # Order points by angle (counterclockwise)
    ord_points <- coords[order(angles), ]
    # Close the polygon by repeating the first point
    ord_coords <- rbind(ord_points, ord_points[1, ]) 
  } else {
    # For interior polygons: use the first point as center
    center <- coords[1, ]
    points <- coords[-1, ]  # All points except the first
    # Calculate angles from center to surrounding points
    angles <- atan2(points[, 2] - center[2], points[, 1] - center[1])
    # Order surrounding points by angle
    ord_points <- points[order(angles), ]
    # Close the polygon by repeating the first point
    ord_coords <- rbind(ord_points, ord_points[1, ]) 
  }
  # Remove row names for clean output
  rownames(ord_coords) <- NULL
  return(ord_coords)
}

# ==============================================================================
# MAIN FUNCTION: dual_mesh
# ==============================================================================
# 
# Purpose: Constructs the dual mesh associated with a triangular finite element mesh
# 
# Parameters:
#   mesh - A fmesher object containing triangulation information
#   parallelize - Logical flag to enable parallel processing (default: FALSE)
#   n.cores - Number of cores to use for parallel processing (default: detectCores() - 2)
# 
# Returns:
#   List containing:
#     $loc - Coordinates of dual mesh points (medium points and barycenters)
#     $idx - List with indices for medium points ($med) and barycenters ($bary)
#     $dual - List of ordered dual cell coordinates for each vertex
#     $neibr - List of neighboring vertex coordinates for each vertex
# 
# Description:
#   The dual mesh is constructed by creating polygonal cells around each vertex
#   of the original triangular mesh. Each dual cell is formed by connecting:
#   1. Midpoints of edges incident to the vertex
#   2. Barycenters (centroids) of triangles incident to the vertex
#   This creates a tessellation that is "dual" to the original triangulation.
dual_mesh <- function(mesh, parallelize = FALSE, n.cores = detectCores() - 2) {

  # Extract finite element matrices and mesh connectivity
  c1 <- fm_fem(mesh)$c1          # Sparse matrix for finite element assembly
  TV <- mesh$graph$tv            # Triangle-vertex connectivity matrix
  V_loc <- mesh$loc[,1:2]        # Vertex coordinates (x, y only)
  logical_bnd <- (1: mesh$n) %in% mesh$segm$bnd$idx[,1]  # Boundary vertex flags

  # Auxiliary Indexing for Edge Processing
  # --------------------------------------
  # Create edge connectivity from sparse matrix structure
  I <- cbind(c1@i+1, c1@j+1)     # Convert 0-based to 1-based indexing
  J <- which(I[,1] == I[,2])     # Find diagonal entries (self-connections)
  K <- !duplicated(I[,1])        # Find unique row indices
  Swap <- cbind(J, which(K == TRUE))  # Create swap instructions
  colnames(Swap) <- NULL
  I <- swap_fun(I, Swap)         # Reorder edges for consistent processing

  # Compute Key Geometric Points
  # ----------------------------
  # Medium Points: Midpoints of edges connecting vertices
  M <- (V_loc[I[, 1], ] + V_loc[I[, 2], ]) / 2
  
  # Barycentric Points: Centroids of triangles
  B <- (V_loc[TV[,1], ] + V_loc[TV[,2], ] + V_loc[TV[,3], ]) / 3

  # Add unique indices to distinguish point types
  M <- cbind(M, 1:nrow(M))                                    # Medium point indices
  B <- cbind(B, (nrow(M)+2):(nrow(M)+nrow(B)+1))            # Barycenter indices

  # Build Vertex Neighborhood Information
  # ------------------------------------
  V_tri <- node_nbr(TV)          # For each vertex, list triangles containing it

  # Prepare Edge Index Variables for Dual Cell Construction
  # -------------------------------------------------------
  rowidx_c1 <- I[,1]  # Row indices (first vertex of each edge)
  colidx_c1 <- I[,2]  # Column indices (second vertex of each edge)

  # PARALLEL PROCESSING BRANCH
  # ==========================
  if (parallelize) {
    # Set up SOCK cluster for parallel processing across multiple cores
    cl <- makeCluster(n.cores, type = "SOCK")
    registerDoSNOW(cl)

    # Parallel computation of dual cells for each vertex
    results <- foreach(k = 1:nrow(c1), .packages = c("base")) %dopar% {
      # For vertex k, get barycentric coordinates of incident triangles
      b_k <- B[V_tri[[k]], , drop= FALSE] 
      
      # Get midpoint coordinates of edges incident to vertex k
      m_k <- M[which(rowidx_c1 == k), , drop= FALSE] 
      
      # Combine midpoints and barycenters to form dual cell vertices
      mesh_dual_k <- rbind(m_k[,1:2], b_k[,1:2])
      
      # Get coordinates of neighboring vertices
      mesh_Vneib_k <- V_loc[colidx_c1[which(rowidx_c1 == k)], ]
      
      # Store location data with indices (excluding first midpoint for indexing)
      loc_k <- rbind(m_k[-1,], b_k)
      
      # Return structured results for this vertex
      list(mesh_dual = mesh_dual_k, mesh_Vneib = mesh_Vneib_k, loc = loc_k)
    }

    # Clean up parallel cluster
    stopCluster(cl)

    # Extract results from parallel computation
    mesh.dual <- lapply(results, `[[`, "mesh_dual")    # Dual cell coordinates
    mesh.Vneib <- lapply(results, `[[`, "mesh_Vneib")  # Neighbor coordinates
    loc <- do.call(rbind, lapply(results, `[[`, "loc")) # Combined location matrix

  # SEQUENTIAL PROCESSING BRANCH
  # ============================
  } else {
    # Initialize storage lists
    mesh.dual <- list()
    mesh.Vneib <- list()
    loc <- list()

    # Sequential computation for each vertex
    for(k in 1:nrow(c1)){
      # For vertex k, get barycentric coordinates of incident triangles
      b_k <- B[V_tri[[k]], , drop= FALSE] 
      
      # Get midpoint coordinates of edges incident to vertex k
      m_k <- M[which(rowidx_c1==k), , drop= FALSE] 
      
      # Store dual cell coordinates (midpoints + barycenters)
      mesh.dual[[k]] <- rbind(m_k[,1:2], b_k[,1:2])
      
      # Store neighboring vertex coordinates
      mesh.Vneib[[k]] <- V_loc[colidx_c1[which(rowidx_c1 == k)], ]
      
      # Store location data with indices
      loc[[k]] <- rbind(m_k[-1,], b_k)
    }

    # Combine all location data into single matrix
    loc <- do.call(rbind, loc)
  }

  # POST-PROCESSING AND FINALIZATION
  # ================================
  
  # Adjust indices to be 0-based for consistency
  loc[,3] <- loc[,3]-1
  
  # Separate indices for medium points and barycenters
  M_idx <- which(loc[,3]<=nrow(M))    # Medium point indices
  B_idx <- which(loc[,3]>nrow(M))     # Barycenter indices

  # Order polygon vertices consistently (counterclockwise)
  dual.ordered <- mapply(ord_poly, mesh.dual, logical_bnd)
  neibr.ordered <- mapply(ord_poly, mesh.Vneib, logical_bnd)

  # Return comprehensive dual mesh structure
  return(list(
    loc = loc[,-3],                                    # Point coordinates (without indices)
    idx = list(med = M_idx, bary = B_idx),            # Point type indices
    dual = dual.ordered,                               # Ordered dual cell polygons
    neibr = mesh.Vneib                                # Neighboring vertex coordinates
  ))
}


# ==============================================================================
# EXAMPLE USAGE AND TEST SETUP
# ==============================================================================

# Create a fine reference mesh (for demonstration)
N.ok <- 2
lattice.ok <- fm_lattice_2d(
      seq(0, 1, length.out = 2^{N.ok}+1),
      seq(0, 1, length.out = 2^{N.ok}+1)
    )
mesh2d.ok <- fm_rcdt_2d_inla(lattice = lattice.ok, boundary = lattice.ok$segm, extend = FALSE)

# Create a coarse mesh (for demonstration)
N <- 1
lattice <- fm_lattice_2d(
      seq(0, 1, length.out = 2^{N}+1),
      seq(0, 1, length.out = 2^{N}+1)
    )
mesh2d <- fm_rcdt_2d_inla(lattice = lattice, boundary = lattice$segm, extend = FALSE)

# Construct dual mesh from the coarse mesh
dmesh <- dual_mesh(mesh2d)





# ==============================================================================
# MESH EVALUATOR FUNCTIONS
# ==============================================================================





# Function: rdm_evaluator  
# -----------------------
# Purpose: Creates a projection matrix between two meshes using area-weighted dual mesh intersections
# 
# Parameters:
#   mesh - Source mesh (coarse mesh to project from)
#   mesh1 - Target mesh (fine mesh to project to)
#   parallelize - Enable parallel processing (default: FALSE)
#   n.cores - Number of cores for parallel processing
# 
# Returns:
#   Sparse matrix A where A[i,j] represents the contribution of source vertex j 
#   to target vertex i, weighted by the intersection area of their dual cells
# 
# Description:
#   This function implements a sophisticated mesh-to-mesh projection method that:
#   1. Constructs dual meshes for both source and target meshes
#   2. Computes geometric intersections between dual cells
#   3. Weights contributions by the ratio of intersection area to target cell area
#   This provides accurate projection while preserving mass/integral properties.
rdm_evaluator <- function(mesh, mesh1, parallelize = FALSE, n.cores = detectCores() - 2) {
  
  # Extract target mesh coordinates and create initial projection matrix
  loc <- mesh1$loc[, 1:2, drop = FALSE]
  A <- fm_evaluator(mesh, loc)$proj$A  # Initial sparse projection matrix

  # Extract sparse matrix structure for efficient processing
  point <- A@p        # Column pointers (compressed sparse column format)
  d <- diff(A@p)      # Number of non-zeros per column
  row_ind <- A@i + 1  # Row indices (convert to 1-based indexing)

  # Construct dual meshes for both source and target meshes
  dmesh <- dual_mesh(mesh, parallelize = TRUE, n.cores = n.cores)   # Source dual mesh
  dmesh1 <- dual_mesh(mesh1, parallelize = TRUE, n.cores = n.cores) # Target dual mesh

  # Convert dual cell coordinates to spatial geometry objects
  sfdual <- lapply(dmesh$dual, function(x) st_polygon(list(x)))    # Source dual polygons
  sfdual1 <- lapply(dmesh1$dual, function(x) st_polygon(list(x)))  # Target dual polygons
  
  # Calculate areas of target dual cells for normalization
  area_dual1 <- diag(fm_fem(mesh1)$c0)

  # PARALLEL PROCESSING BRANCH
  # ==========================
  if (parallelize) {
    # Set up parallel cluster
    cl <- makeCluster(n.cores, type = "SOCK")
    registerDoSNOW(cl)

    # Parallel computation of intersection weights
    A@x <- foreach(k = seq_along(d), .combine = 'c', 
                   .packages = c("sf", "splancs"), 
                   .noexport = c("dmesh1")) %dopar% {
      if (d[k] > 0) {
        # Get target vertices that interact with source vertex k
        j_k <- row_ind[point[k] + seq_len(d[k])]
        
        # Compute geometric intersections between source dual cell k and target dual cells
        intersec <- lapply(sfdual1[j_k], function(x) st_intersection(sfdual[[k]], x))
        
        # Calculate intersection areas
        area <- sapply(intersec, st_area) 
        
        # Normalize by target cell areas to get proportional weights
        prop_area <- area / area_dual1[j_k]
        
        # Alternative point-in-polygon method (commented out for performance)
        # indicator1 <- inout(loc[j_k, ], dmesh$dual[[k]], bound = TRUE) #v0.1
        # prop_area[!indicator1] <- 0  #v0.1
        
        return(prop_area)
      } else {
        return(rep(NA, d[k]))  # No interactions for this vertex
      }
    }

    stopCluster(cl)
    
  # SEQUENTIAL PROCESSING BRANCH
  # ============================
  } else {
    # Sequential computation for each source vertex
    for (k in seq_along(d)) {
      if (d[k] > 0) {
        # Get target vertices that interact with source vertex k
        j_k <- row_ind[point[k] + seq_len(d[k])]
        
        # Compute geometric intersections
        intersec <- lapply(sfdual1[j_k], function(x) st_intersection(sfdual[[k]], x))
        
        # Calculate and normalize intersection areas
        area <- sapply(intersec, st_area) 
        prop_area <- area / area_dual1[j_k]
        
        # Alternative method (commented out)
        # indicator1 <- inout(loc[j_k, ], dmesh$dual[[k]], bound = TRUE) #v0.1
        # prop_area[!indicator1] <- 0  #v0.1
        
        # Store computed weights in sparse matrix
        A@x[seq_len(d[k]) + point[k]] <- prop_area
      }
    }
  }

  # Remove explicitly zero entries and return optimized sparse matrix
  A <- drop0(A)
  return(A)
}



# Function: dm_evaluator
# ----------------------
# Purpose: Creates a binary projection matrix using point-in-polygon tests with dual mesh cells
# 
# Parameters:
#   mesh - Source mesh (coarse mesh to project from)
#   mesh1 - Target mesh (fine mesh to project to)
#   boundary - Handle boundary conditions specially (default: FALSE)
# 
# Returns:
#   Sparse binary matrix A where A[i,j] = 1 if target vertex i is inside source dual cell j,
#   with special handling for boundary vertices when boundary=TRUE
# 
# Description:
#   This function creates a simpler projection than rdm_evaluator by using binary
#   inclusion tests. When boundary=TRUE, it applies sophisticated logic for vertices
#   on mesh boundaries, including fractional weights (1/2) for boundary midpoints.
dm_evaluator <- function(mesh, mesh1, boundary = FALSE){

  # Extract target mesh information
  loc <- mesh1$loc[, 1:2, drop = FALSE]  # Target vertex coordinates
  bnd_idx <- mesh1$segm$bnd$idx[,1]      # Boundary vertex indices
  A <- fm_evaluator(mesh, loc)$proj$A    # Initial projection matrix

  # Extract sparse matrix structure
  d <- diff(A@p)      # Number of non-zeros per column
  row_ind <- A@i + 1  # Row indices (1-based)
  
  # Construct source dual mesh
  dmesh <- dual_mesh(mesh, parallelize = TRUE, n.cores = 5)

  if (boundary) {
  rows_dmesh <- apply(dmesh$loc[dmesh$idx$bary, ], 1, paste, collapse = ",")
  rows_dmed <- apply(dmesh$loc[dmesh$idx$med, ], 1 , paste, collapse = ",")
  rows_bnd <- apply(loc[bnd_idx, ], 1 , paste, collapse = ",")
    for (k in seq_along(d)) {
      if (d[k] > 0) {
        j_k <- row_ind[seq_len(d[k])]
        indicator1 <- inout(loc[j_k, , drop = FALSE], dmesh$dual[[k]], bound = TRUE)
        indicator2 <- inout(loc[j_k, , drop = FALSE], dmesh$dual[[k]], bound = FALSE)
        rows_loc <- apply(loc[j_k, , drop = FALSE], 1, paste, collapse = ",")
        which_bary <- rows_loc %in% rows_dmesh
        which_med <- rows_loc  %in%  rows_dmed
        bnd_medindicator <- indicator1 & !indicator2 & !which_bary & !which_med
        which_locbnd <- rows_loc  %in%  rows_bnd
        which_locbnd1 <- which_locbnd & bnd_medindicator
        which_locbndmed <- which_locbnd & which_med & indicator1 & !indicator2
        # bnd_nomedonlybnd <- which_locbnd1 & !which_med
        # indmesh_bndnobary <- bnd_medindicator & !bnd_nomedonlybnd

        A@x[which(bnd_medindicator) + A@p[k]] <- 1 / 2
        A@x[which(which_med) + A@p[k]] <- 1/2
        A@x[which(!indicator1) + A@p[k]] <- 0
        A@x[which(indicator2) + A@p[k]] <- 1
        A@x[which(which_locbnd1) + A@p[k]] <- 1
        A@x[which(which_locbndmed) + A@p[k]] <- 1 / 2


        row_ind <- row_ind[-seq_len(d[k])]
      }
    }
    return(A)

    
  # SIMPLE PROCESSING BRANCH (No boundary special treatment)
  # ========================================================
  } else {
    # Simple binary inclusion test for each source vertex
    for (k in seq_along(d)) {
      if (d[k] > 0) {
        # Get target vertices that potentially interact with source vertex k
        j_k <- row_ind[seq_len(d[k])]
        
        # Point-in-polygon test (strict interior only)
        indicator <- inout(loc[j_k, 1:2, drop = FALSE], dmesh$dual[[k]], bound = FALSE)
        
        # Set weights: 0 for outside, leave existing for inside
        A@x[which(!indicator) + A@p[k]] <- 0
        
        # Move to next set of target vertices
        row_ind <- row_ind[-seq_len(d[k])]
      }
    }
    
    # Set all remaining non-zero entries to 1 (binary projection)
    A@x[A@x != 0] <- 1
    return(A)
  }
}



# ==============================================================================
# PERFORMANCE TESTING AND COMPREHENSIVE EXAMPLES (COMMENTED OUT)
# ==============================================================================
# 
# This section contains extensive test code for performance evaluation and 
# visual verification of the dual mesh construction and evaluation functions.
# Uncomment sections as needed for testing and analysis.

# PERFORMANCE BENCHMARKING
# ========================
# Test with larger meshes to evaluate computational performance

# N.ok <- 5  # Creates a 2^5+1 x 2^5+1 = 33x33 grid (fine mesh)
# print(system.time({
#   lattice.ok <- fm_lattice_2d(
#         seq(0, 1, length.out = 2^{N.ok}+1),
#         seq(0, 1, length.out = 2^{N.ok}+1)
#       )
#   mesh2d.ok <- fm_rcdt_2d_inla(lattice = lattice.ok, boundary = lattice.ok$segm, extend = FALSE)
# }))

# N <- 3  # Creates a 2^3+1 x 2^3+1 = 9x9 grid (coarse mesh)
# print(system.time({
#   lattice <- fm_lattice_2d(
#         seq(0, 1, length.out = 2^{N}+1),
#         seq(0, 1, length.out = 2^{N}+1)
#       )
#   mesh2d <- fm_rcdt_2d_inla(lattice = lattice, boundary = lattice$segm, extend = FALSE
#   # , refine=list(max.edge = 1/2^{N})  # Optional mesh refinement
#   )
# }))

# DUAL MESH CONSTRUCTION TESTING
# ==============================
# Test parallel vs sequential performance for dual mesh construction

# print(system.time({
#   dmesh <- dual_mesh(mesh2d, parallelize=TRUE, n.cores=2)  # Test with 2 cores
# }))

# EVALUATOR FUNCTION TESTING
# ==========================
# Test the mesh-to-mesh projection functions

# D <- dm_evaluator(mesh2d, mesh2d.ok, boundary = TRUE)    # Binary projection with boundary handling
# A <- fm_evaluator(mesh2d.ok, mesh2d$loc[,1:2])$proj$A   # Standard fmesher projection for comparison



# VISUALIZATION AND ANALYSIS CODE
# ================================
# Multi-panel plotting for visual verification of dual mesh construction

# par(mfcol = c(1, 3))  # Create 3-panel figure
# par(mar = c(0.5, 1, 4, 1))  # Set plot margins

# PANEL 1: Original triangular mesh
# plot(mesh2d, lwd=2, main = "Figure 1")
# title("Triangulation", cex.main = 5)

# PANEL 2: Dual mesh overlay
# plot(mesh2d, lwd=2)
# title("Dual Mesh", cex.main = 5)

# # Draw dual mesh cells for each vertex
# for(k in 1:mesh2d$n){
#   lines(dmesh$dual[[k]], pch=1, col="blue")   # Draw dual cell boundaries
#   points(dmesh$dual[[k]])                     # Mark dual cell vertices
# }

# # Select specific vertex for detailed analysis
# vertex <- 2  # Choose vertex index for testing
# # vertex2 <- ncol(D)-16  # Alternative vertex selection

# # Visualize different point types in dual mesh
# # points(mesh2d.ok$loc[,1:2])  # All target mesh vertices
# points(dmesh$loc[dmesh$idx$med, ], pch=19, col="#24ad24")   # Medium points (green)
# points(dmesh$loc[dmesh$idx$bary, ], pch=19, col=2)         # Barycenters (red)

# # Visualize projection results (uncomment to see specific weights)
# # points(mesh2d.ok$loc[which(D[,vertex] == 1), 1:2], pch=19, col=5)         # Full weight
# # points(mesh2d.ok$loc[which(D[,vertex] == 1/2), 1:2], pch=19, col="orange") # Half weight
# # points(mesh2d.ok$loc[which(D[,vertex] == 0), 1:2], pch=19, col="#e1ff00")  # Zero weight
# # points(mesh2d.ok$loc[which(D[,vertex] == 1/5), 1:2], pch=19)              # Custom weight

# # Additional vertex analysis (alternative vertex)
# # points(mesh2d.ok$loc[which(D[,vertex2] == 1), 1:2], pch=19, col=5)
# # points(mesh2d.ok$loc[which(D[,vertex2] == 1/2), 1:2], pch=19, col="orange")
# # points(mesh2d.ok$loc[which(D[,vertex] == 0), 1:2], pch=19, col="#e1ff00")
# # points(mesh2d.ok$loc[which(D[,vertex] == 1/5), 1:2], pch=19)

# # Show original mesh vertices
# # points(mesh2d$loc, pch=19)





# DETAILED BOUNDARY ANALYSIS (DEBUGGING CODE)
# ===========================================
# Advanced testing for boundary vertex classification and point-in-polygon logic

# # Extract interior vertices (excluding boundary vertices)
# # loc.ok <- mesh2d.ok$loc[-mesh2d.ok$segm$bnd$idx[,1], 1:2, drop=FALSE]

# # Visualize specific dual cell for analysis
# # points(dmesh$dual[[length(dmesh$dual)]])    # Mark vertices of last dual cell
# # lines(dmesh$dual[[length(dmesh$dual)]])     # Draw boundary of last dual cell
# # points(loc.ok)                              # Show interior target vertices

# # Alternative point-in-polygon testing (legacy functions)
# # # indicator1 <- inpip(mesh2d.ok$loc[,1:2], dmesh$dual[[length(dmesh$dual)-1]], bound = TRUE)
# # # indicator2 <- inpip(mesh2d.ok$loc[,1:2], dmesh$dual[[length(dmesh$dual)-1]], bound = FALSE)

# # Current point-in-polygon testing with splancs::inout
# # indicator1 <- inout(loc.ok, dmesh$dual[[length(dmesh$dual)]], bound = TRUE)   # Include boundary
# # indicator2 <- inout(loc.ok, dmesh$dual[[length(dmesh$dual)]], bound = FALSE)  # Exclude boundary

# # Coordinate-based vertex classification for debugging
# # rows_A <- apply(loc.ok, 1, paste, collapse = ",")                    # Target vertex coordinates as strings
# # rows_B <- apply(dmesh$loc[dmesh$idx$bary, ], 1, paste, collapse = ",") # Barycenter coordinates as strings
# # which_bary <- rows_A %in% rows_B                                     # Check if target vertex is a barycenter
# # bnd_medindicator <- indicator1 & !indicator2 & !which_bary           # Boundary classification logic

# # Color-coded visualization of classification results
# # points(loc.ok[!indicator1,], pch=19, col=4)           # Blue: Outside dual cell
# # points(loc.ok[indicator2,], pch=19, col=5)            # Cyan: Strictly inside dual cell  
# # points(loc.ok[bnd_medindicator, ], pch=19, col=1)     # Black: On dual cell boundary

# PANEL 3: COMPLEX DOMAIN TESTING
# ===============================
# Test dual mesh construction on irregular/complex domains

# # Generate random points and create non-convex boundary
# inp <- matrix(rnorm(20), 10, 2)                           # 10 random 2D points
# out <- fm_nonconvex_hull(inp, convex = 1)                 # Create non-convex hull

# # Create coarse and fine meshes on complex domain
# mesh2dif <- fm_rcdt_2d(boundary = out, refine = list(max.edge = 2))      # Coarse mesh
# mesh2dif.ok <- fm_rcdt_2d(boundary = out, refine = list(max.edge = 0.08)) # Fine mesh

# # Visualize complex domain mesh
# plot(mesh2dif, lwd=2)
# title("Complex Domain", cex.main = 5)

# # Construct and visualize dual mesh on complex domain
# dmesh <- dual_mesh(mesh2dif)
# invisible(lapply(dmesh$dual, lines, col="blue"))          # Draw all dual cells in blue

# # Optional: visualize dual mesh points
# # points(dmesh$loc[ dmesh$idx$bary, ], pch=19, col=2)       # Barycenters in red
# # points(dmesh$loc[ dmesh$idx$med, ], pch=19, col="#4ea84e") # Midpoints in green

# # Test evaluator on complex domain
# B <- dm_evaluator(mesh2dif, mesh2dif.ok, boundary = TRUE)

# RANDOM SAMPLING FOR STATISTICAL ANALYSIS
# ========================================
# Select random vertices for detailed analysis and validation

# set.seed(NULL)  # Ensure truly random sampling

# # Sample random interior vertex (excluding boundary vertices)
# index <- sample(setdiff(1:nrow(mesh2dif$loc), mesh2dif$segm$bnd$idx[,1] ), 1, replace = TRUE)

# # Sample random boundary vertex
# index2 <- sample(mesh2dif$segm$bnd$idx[,1], 1, replace = TRUE)