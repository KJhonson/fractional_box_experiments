# plot_visualization.py
# Functions for plotting FEM solutions and domains
#
# STANDARD FEATURES (applied by default):
# Solid blue lines for numerical solutions (with node markers)
# Dashed red lines for exact solutions (smooth continuous)
# Smart y-axis scaling with breathing room (5% below, 20% above)
# Linear scale for zero errors, log scale for non-zero ranges
# Memory-based display for Jupyter (no unwanted files)
# High-resolution exact solutions (200 points) vs discrete numerical (at nodes)

import numpy as np
import matplotlib.pyplot as plt
from dolfinx import fem
from IPython.display import Image, display
import io

def visualize_domain(mesh, save_plot=False, filename="domain.png"):
    """
    Visualize the domain/mesh for your main.py.
    
    Parameters:
    -----------
    mesh : dolfinx.mesh.Mesh
        The mesh to visualize
    save_plot : bool
        Whether to save the plot to file
    filename : str
        Filename for saved plot
    """
    if mesh.comm.rank == 0:
        if mesh.topology.dim == 2:
            # 2D mesh visualization
            points = mesh.geometry.x[:, :2]
            cells = mesh.topology.connectivity(mesh.topology.dim, 0).array
            num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
            cells_reshaped = cells.reshape(num_cells, -1)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Draw mesh edges
            from matplotlib.collections import LineCollection
            lines = []
            for cell in cells_reshaped:
                cell_points = points[cell]
                closed_cell = np.vstack([cell_points, cell_points[0]])
                lines.append(closed_cell)
            
            line_collection = LineCollection(lines, colors='blue', linewidths=0.8)
            ax.add_collection(line_collection)
            
            # Set plot properties
            margin = 0.05
            x_range = points[:, 0].max() - points[:, 0].min()
            y_range = points[:, 1].max() - points[:, 1].min()
            
            ax.set_xlim(points[:, 0].min() - margin * x_range, 
                        points[:, 0].max() + margin * x_range)
            ax.set_ylim(points[:, 1].min() - margin * y_range, 
                        points[:, 1].max() + margin * y_range)
            
            ax.set_aspect('equal')
            ax.set_title('Domain Mesh', fontsize=14, fontweight='bold')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            plt.tight_layout()
            
            if save_plot:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Domain plot saved to: {filename}")
                plt.show()
                try:
                    display(Image(filename))
                except:
                    pass
            else:
                # Memory display (our special approach)
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                plt.close()
                
                img_buffer.seek(0)
                try:
                    display(Image(img_buffer.getvalue()))
                except:
                    plt.show()
            
        else:
            print(f"Domain visualization for {mesh.topology.dim}D meshes not implemented")

def plot_1d_dirichlet_solution(mesh, u_h, exact_solution_func, save_plot=False, filename="solution_1d_dirichlet.png", 
                              use_breathing_room=True, show_node_markers=True, use_scientific_notation=True,
                              l2_error=None, h1_error=None, linf_error=None):
    """
    Plot 1D Dirichlet solution comparison with exact solution.
    
    Parameters:
    -----------
    mesh : dolfinx.mesh.Mesh
        1D mesh
    u_h : dolfinx.fem.Function
        Numerical solution
    exact_solution_func : callable
        Function that takes x and returns exact solution
    save_plot : bool
        Whether to save the plot
    filename : str
        Filename for saved plot
    use_breathing_room : bool
        Whether to add breathing room to y-axis (default: True)
    show_node_markers : bool
        Whether to show markers at mesh nodes (default: True)
    use_scientific_notation : bool
        Whether to use scientific notation (1e-6) for small numbers (default: True)
    """
    if mesh.comm.rank == 0 and mesh.topology.dim == 1:
        # Get mesh coordinates (nodes where numerical solution is computed)
        x_coords = mesh.geometry.x[:, 0]
        x_sorted_idx = np.argsort(x_coords)
        x_nodes = x_coords[x_sorted_idx]  # Mesh nodes
        
        # Get numerical solution values (only at nodes)
        u_h_values = u_h.x.array[x_sorted_idx]
        
        # Create fine grid for exact solution (continuous everywhere)
        x_fine = np.linspace(x_nodes[0], x_nodes[-1], 200)  # Fine grid for smooth exact solution
        u_exact_fine = exact_solution_func(x_fine)  # Exact solution everywhere
        
        # Exact solution at mesh nodes (for error computation)
        u_exact_nodes = exact_solution_func(x_nodes)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot solutions with standard styling
        if show_node_markers:
            ax1.plot(x_nodes, u_h_values, 'b-o', label='Numerical Solution', linewidth=2, markersize=4)  # Solid blue with node markers
        else:
            ax1.plot(x_nodes, u_h_values, 'b-', label='Numerical Solution', linewidth=2)  # Solid blue without markers
        
        ax1.plot(x_fine, u_exact_fine, 'r--', label='Exact Solution', linewidth=2)  # Dashed red continuous (standard)
        ax1.set_xlabel('x')
        ax1.set_ylabel('u(x)')
        ax1.set_title('1D Dirichlet Problem: Solution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot error (computed at mesh nodes)
        error = np.abs(u_h_values - u_exact_nodes)
        ax2.plot(x_nodes, error, 'g-o', label='Absolute Error', linewidth=2, markersize=3)
        ax2.set_xlabel('x')
        ax2.set_ylabel('|u_h - u_exact|')
        ax2.set_title('Absolute Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Smart y-axis scaling for error plot
        max_error_val = np.max(error)
        min_error_val = np.min(error[error > 0]) if np.any(error > 0) else 1e-16
        
        if max_error_val < 1e-15:  # Essentially zero
            ax2.set_yscale('linear')
            ax2.set_ylim(-1e-15, 1e-15)
            ax2.text(0.5, 0.5, 'Error ≈ 0 (Machine Precision)', 
                    transform=ax2.transAxes, ha='center', va='center',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
        elif np.any(error == 0):  # Some zeros - handle carefully for log plot
            # Use linear scale to show the full error pattern including zeros (STANDARD)
            ax2.set_yscale('linear')
            if use_breathing_room:
                # Standard: Give more space at bottom and top for better visualization
                ax2.set_ylim(-max_error_val * 0.05, max_error_val * 1.2)
            else:
                ax2.set_ylim(0, max_error_val * 1.1)
        else:  # All errors non-zero - can use log scale
            ax2.set_yscale('log')
            ax2.set_ylim(min_error_val * 0.5, max_error_val * 2)
        
        # Control scientific notation on y-axis
        if use_scientific_notation and max_error_val < 1e-3:
            # Force scientific notation for small numbers
            ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        elif not use_scientific_notation:
            # Force regular notation
            ax2.ticklabel_format(style='plain', axis='y')
        
        # Use provided errors or compute them
        if l2_error is None or h1_error is None or linf_error is None:
            # Fallback to simple computation
            l2_error_computed = np.sqrt(np.trapz(error**2, x_nodes))
            max_error_computed = np.max(error)
            error_title = f'Dirichlet BC - L2 Error: {l2_error_computed:.2e}, Max Error: {max_error_computed:.2e}'
        else:
            # Use provided comprehensive errors
            error_title = f'Dirichlet BC - L² Error: {l2_error:.2e}, H¹ Error: {h1_error:.2e}, L∞ Error: {linf_error:.2e}'
        
        # Add error statistics
        fig.suptitle(error_title, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Dirichlet solution plot saved to: {filename}")
            plt.show()
            try:
                display(Image(filename))
            except:
                pass
        else:
            # Memory display
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            img_buffer.seek(0)
            try:
                display(Image(img_buffer.getvalue()))
            except:
                plt.show()
        
        # Return appropriate errors based on what was computed
        if l2_error is not None and h1_error is not None and linf_error is not None:
            return l2_error, h1_error, linf_error
        else:
            return l2_error_computed, max_error_computed
    else:
        print("Plot only available on rank 0 for 1D meshes")
        return None, None

def plot_1d_neumann_solution(mesh, u_h, exact_solution_func, save_plot=False, filename="solution_1d_neumann.png",
                             use_breathing_room=True, show_node_markers=True, use_scientific_notation=True,
                             l2_error=None, h1_error=None, linf_error=None):
    """
    Plot 1D Neumann solution comparison with exact solution.
    
    Parameters:
    -----------
    mesh : dolfinx.mesh.Mesh
        1D mesh
    u_h : dolfinx.fem.Function
        Numerical solution
    exact_solution_func : callable
        Function that takes x and returns exact solution
    save_plot : bool
        Whether to save the plot
    filename : str
        Filename for saved plot
    use_breathing_room : bool
        Whether to add breathing room to y-axis (default: True)
    show_node_markers : bool
        Whether to show markers at mesh nodes (default: True)
    use_scientific_notation : bool
        Whether to use scientific notation (1e-6) for small numbers (default: True)
    """
    if mesh.comm.rank == 0 and mesh.topology.dim == 1:
        # Get mesh coordinates (nodes where numerical solution is computed)
        x_coords = mesh.geometry.x[:, 0]
        x_sorted_idx = np.argsort(x_coords)
        x_nodes = x_coords[x_sorted_idx]  # Mesh nodes
        
        # Get numerical solution values (only at nodes)
        u_h_values = u_h.x.array[x_sorted_idx]
        
        # Create fine grid for exact solution (continuous everywhere)
        x_fine = np.linspace(x_nodes[0], x_nodes[-1], 200)  # Fine grid for smooth exact solution
        u_exact_fine = exact_solution_func(x_fine)  # Exact solution everywhere
        
        # Exact solution at mesh nodes (for error computation and adjustment)
        u_exact_nodes = exact_solution_func(x_nodes)
        
        # For Neumann problems, solutions are unique up to a constant
        # Adjust numerical solution to match exact solution at x=0
        offset = u_exact_nodes[0] - u_h_values[0]
        u_h_adjusted = u_h_values + offset
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot solutions with standard styling
        if show_node_markers:
            ax1.plot(x_nodes, u_h_adjusted, 'b-o', label='Numerical Solution (adjusted)', linewidth=2, markersize=4)  # Solid blue with node markers
        else:
            ax1.plot(x_nodes, u_h_adjusted, 'b-', label='Numerical Solution (adjusted)', linewidth=2)  # Solid blue without markers
        
        ax1.plot(x_fine, u_exact_fine, 'r--', label='Exact Solution', linewidth=2)  # Dashed red continuous (standard)
        ax1.set_xlabel('x')
        ax1.set_ylabel('u(x)')
        ax1.set_title('1D Neumann Problem: Solution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot error (computed at mesh nodes)
        error = np.abs(u_h_adjusted - u_exact_nodes)
        ax2.plot(x_nodes, error, 'g-o', label='Absolute Error', linewidth=2, markersize=3)
        ax2.set_xlabel('x')
        ax2.set_ylabel('|u_h - u_exact|')
        ax2.set_title('Absolute Error (after constant adjustment)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Smart y-axis scaling for error plot
        max_error_val = np.max(error)
        min_error_val = np.min(error[error > 0]) if np.any(error > 0) else 1e-16
        
        if max_error_val < 1e-15:  # Essentially zero error
            ax2.set_yscale('linear')
            ax2.set_ylim(-1e-15, 1e-15)
            ax2.text(0.5, 0.5, 'Error ≈ 0 (Machine Precision)', 
                    transform=ax2.transAxes, ha='center', va='center',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
        elif np.any(error == 0):  # Some errors are exactly zero
            ax2.set_yscale('linear')
            if use_breathing_room:
                # Standard: Give more space at bottom and top for better visualization
                ax2.set_ylim(-max_error_val * 0.05, max_error_val * 1.2)
            else:
                ax2.set_ylim(0, max_error_val * 1.1)
        else:  # All errors non-zero - can use log scale
            ax2.set_yscale('log')
            ax2.set_ylim(min_error_val * 0.5, max_error_val * 2)
        
        # Control scientific notation on y-axis
        if use_scientific_notation and max_error_val < 1e-3:
            # Force scientific notation for small numbers
            ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        elif not use_scientific_notation:
            # Force regular notation
            ax2.ticklabel_format(style='plain', axis='y')
        
        # Use provided errors or compute them
        if l2_error is None or h1_error is None or linf_error is None:
            # Fallback to simple computation
            l2_error_computed = np.sqrt(np.trapz(error**2, x_nodes))
            max_error_computed = np.max(error)
            if verbose:
                print(f"Warning: Using simple error computation (L2: {l2_error_computed:.2e}, Max: {max_error_computed:.2e})")
            error_title = f'Neumann BC - L2 Error: {l2_error_computed:.2e}, Max Error: {max_error_computed:.2e}'
        else:
            # Use provided comprehensive errors
            error_title = f'Neumann BC - L² Error: {l2_error:.2e}, H¹ Error: {h1_error:.2e}, L∞ Error: {linf_error:.2e}'
        
        # Add error statistics
        fig.suptitle(error_title, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Neumann solution plot saved to: {filename}")
            plt.show()
            try:
                display(Image(filename))
            except:
                pass
        else:
            # Memory display
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            img_buffer.seek(0)
            try:
                display(Image(img_buffer.getvalue()))
            except:
                plt.show()
        
        # Return appropriate errors based on what was computed
        if l2_error is not None and h1_error is not None and linf_error is not None:
            return l2_error, h1_error, linf_error
        else:
            return l2_error_computed, max_error_computed
    else:
        print("Plot only available on rank 0 for 1D meshes")
        return None, None

print("Plot visualization functions loaded and ready for main.py")
