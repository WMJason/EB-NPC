from typing import Dict, Any, List, Tuple
import numpy as np
import networkx as nx
import pandas as pd
import ast # For safely handling string-to-tuple node conversion
import matplotlib.pyplot as plt # ADDED for visualization
from scipy.linalg import cholesky, eigh
import os

# --- Utility Functions (load_graphml_with_tuple_nodes, get_shortest_path_distance_matrix) are unchanged ---

def load_graphml_with_tuple_nodes(filename: str) -> nx.Graph:
    """Helper to load GraphML where nodes were stringified tuples."""
    try:
        H = nx.read_graphml(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"GraphML file not found: {filename}")
        
    G = nx.Graph()
    node_map = {}
    
    for node_str in H.nodes:
        try:
            node_tuple = ast.literal_eval(node_str)
            node_map[node_str] = node_tuple
            G.add_node(node_tuple)
        except:
            node_map[node_str] = node_str
            G.add_node(node_str)

    for u_str, v_str, data in H.edges(data=True):
        u_tuple = node_map.get(u_str)
        v_tuple = node_map.get(v_str)
        
        if u_tuple and v_tuple:
            data['AADT'] = int(data.get('AADT', 0))
            data['length'] = float(data.get('length', 1.0))
            G.add_edge(u_tuple, v_tuple, **data)
            
    return G

def get_shortest_path_distance_matrix(G: nx.Graph, nodes: List[Any]) -> np.ndarray:
    """Computes the shortest path distance matrix (hop count) for the given nodes."""
    N = len(nodes)
    D = np.zeros((N, N))
    
    # Use unweighted shortest path length (since grid edges have length 1.0)
    #path_lengths = dict(nx.shortest_path_length(G))
    path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="length"))
    
    for i in range(N):
        for j in range(N):
            source = nodes[i]
            target = nodes[j]
            # Use .get(target, inf) for disconnected nodes
            distance = path_lengths.get(source, {}).get(target, np.inf) 
            D[i, j] = distance if distance != np.inf else 1e5 # Use large number instead of inf

    print("Num entries equal to 1e5:", np.sum(D == 1e5))
            
    return D

# --- Core GP Generation Function (squared_exponential_kernel and generate_gp_field are unchanged) ---

def squared_exponential_kernel(D: np.ndarray, lambda_range: float, sigma_spatial: float, sigma_noise: float) -> np.ndarray:
    """
    Computes the covariance matrix (Sigma) using the Squared Exponential Kernel.
    """
    D_finite = np.where(D == 1e5, 0.0, D)
    K_spatial = (sigma_spatial ** 2) * np.exp(-0.5 * (D_finite / lambda_range) ** 2)
    K_total = K_spatial + (sigma_noise ** 2) * np.eye(D.shape[0])
    
    return K_total


def generate_gp_field(
    G: nx.Graph, 
    lambda_range: float, 
    sigma_spatial: float, 
    sigma_noise: float, 
    mu_speed: float, 
    seed: int = None
) -> Dict[str, float]:
    """
    Generates the spatially correlated X_Speed covariate using a Gaussian Process.
    """
    if seed is not None:
        np.random.seed(seed)
        
    nodes = list(G.nodes())
    N = len(nodes)
    
    # 1. Compute Distance Matrix (D)
    D = get_shortest_path_distance_matrix(G, nodes)
    
    # 2. Compute Covariance Matrix (Sigma)
    Sigma = squared_exponential_kernel(D, lambda_range, sigma_spatial, sigma_noise)

    w, _ = np.linalg.eigh(Sigma)
    print("Min eigenvalue:", w.min())
    
    # --- CRITICAL FIX: Add Epsilon Nugget for Numerical Stability ---
    # Introduce a very small positive definite term (epsilon) to the diagonal 
    # to ensure the matrix is numerically stable for Cholesky decomposition.
    epsilon = 1e-6
    Sigma_stable = Sigma + epsilon * np.eye(N)
    # -----------------------------------------------------------------
    
    # ... (D and Sigma calculation remain the same) ...
    
    # Add small epsilon for stability if needed (keep 1e-4 from Solution 1)
    epsilon = 1e-4
    Sigma_stable = Sigma + epsilon * np.eye(N)
    
    # --- Robust Sampling via Spectral Decomposition (Recommended Alternative) ---
    
    try:
        # 3a. Compute Eigenvalues (Lambda) and Eigenvectors (U)
        # eigh is used for symmetric matrices (like Sigma) and is more stable than eig
        Lambda, U = eigh(Sigma_stable)
        
        # 3b. Filter out any tiny/negative eigenvalues
        # Set a tolerance (e.g., 1e-10) to treat small values as zero
        tol = 1e-10
        Lambda_sqrt = np.maximum(0, Lambda) ** 0.5 
        
        # Check if any crucial eigenvalues were filtered (optional check)
        if np.min(Lambda) < -tol:
            print("\n⚠️ WARNING: Matrix had small negative eigenvalues. Filtered them out.")

        # 3c. Construct the square root of Sigma: (U * sqrt(Lambda))
        Sigma_sqrt = U @ np.diag(Lambda_sqrt)
        
        # 3d. Sample the field: X_correlated = Sigma_sqrt @ Z, where Z ~ N(0, I)
        Z = np.random.normal(0, 1, N)
        X_correlated = Sigma_sqrt @ Z
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: Spectral decomposition also failed: {e}. Falling back to i.i.d. noise.")
        X_correlated = np.random.normal(0, sigma_spatial, N)

    # 4. Final Covariate (X_Speed)
    X_speed_values = {str(nodes[i]): float(mu_speed + X_correlated[i]) for i in range(N)}
    
    print(f"✅ Generated {N} spatially correlated X_Speed values using Spectral GP.")
    return X_speed_values

    # 4. Final Covariate (X_Speed)
    X_speed_values = {str(nodes[i]): float(mu_speed + X_correlated[i]) for i in range(N)}
    
    print(f"✅ Generated {N} spatially correlated X_Speed values using GP.")
    return X_speed_values


# --- NEW VISUALIZATION FUNCTION ---

def visualize_speed_field(G: nx.Graph, X_speed_values: Dict[str, float], filename: str, lambda_range: float, sigma_spatial: float):
    """
    Visualizes the generated X_Speed field, coloring nodes by speed level.
    """
    # 1. Map speed values to node tuples
    speed_map = {ast.literal_eval(k): v for k, v in X_speed_values.items()}
    
    # 2. Get node positions and speed colors
    nodes = list(G.nodes())
    node_colors = [speed_map[n] for n in nodes]
    pos = {node: (node[1], -node[0]) for node in nodes} # (col, -row) for standard plot layout

    plt.figure(figsize=(10, 10))
    
    # Draw edges first (gray background)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1.0)
    
    # Draw nodes, colored by speed
    nodes_scatter = nx.draw_networkx_nodes(
        G, 
        pos, 
        node_color=node_colors, 
        cmap=plt.cm.RdYlGn_r, # Reverse Red-Yellow-Green (Red=Slow, Green=Fast)
        node_size=200, 
        alpha=0.9,
        label='Intersection Speed'
    )
    
    # 3. Add Colorbar
    if nodes_scatter is not None:
        cbar = plt.colorbar(nodes_scatter, ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Intersection Driving Speed (X_speed)')

    # 4. Add Title and Metadata
    plt.title(f"Spatially Correlated Intersection Speed (GP Field)", fontsize=16)
    plt.suptitle(f"$\lambda={lambda_range}$, $\sigma_s={sigma_spatial}$", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    
    #plt.savefig(filename, dpi=300)
    print(f"✅ Exported speed field visualization to {filename}")

# --- I/O and Execution (update_and_export_node_data is unchanged) ---

def update_and_export_node_data(node_csv_filename: str, X_speed_values: Dict[str, float], output_csv_filename: str):
    """Reads node CSV, adds 'X_speed' column, and exports the updated file."""
    
    try:
        df_nodes = pd.read_csv(node_csv_filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Node CSV file not found: {node_csv_filename}")
    
    df_nodes['X_speed'] = df_nodes['node_id'].map(X_speed_values)
    
    if df_nodes['X_speed'].isnull().any():
        mean_speed = df_nodes['X_speed'].mean()
        df_nodes['X_speed'] = df_nodes['X_speed'].fillna(mean_speed)
    
    df_nodes.to_csv(output_csv_filename, index=False)
    print(f"✅ Exported updated node data (including 'X_speed') to {output_csv_filename}")


if __name__ == "__main__":
    
    # --- Configuration ---
    GRAPHML_FILE = "../1_network.graphml"
    NODES_CSV_FILE = "../1_nodes.csv"

    LAMBDA_RANGES = [1, 5, 12]
    SIGMA_SPATIALS = {1:1.0, 5:1.0, 12:1.0}
    RANDOM_SEEDS = range(300)

    for LAMBDA_RANGE in LAMBDA_RANGES:

        ##################
        output_folder = f'3_adding_speeding_info_forloop_R{LAMBDA_RANGE}'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        else:
            try:
                for ea in os.listdir(output_folder):
                    os.remove(output_folder + '/' + ea)
            except:
                for ea in os.listdir(output_folder):
                    shutil.rmtree(output_folder + '/' + ea)
                
        for RANDOM_SEED in RANDOM_SEEDS:
    
            # GP Parameters (Control Spatial Correlation and Magnitude)
            LAMBDA_RANGE = LAMBDA_RANGE      # Increased for strong, smooth correlation
            SIGMA_SPATIAL = SIGMA_SPATIALS[LAMBDA_RANGE]      # Increased magnitude of spatial variation, originally 5
            SIGMA_NOISE = 0.0        # Decreased noise for cleaner pattern, originally 0.2
            MU_SPEED = 20.0          # Baseline Mean Speed, originally 20
            
            RANDOM_SEED = RANDOM_SEED
            OUTPUT_CSV_FILE = os.path.join(output_folder, f"3_nodes_with_X_speed_R{LAMBDA_RANGE}_S{RANDOM_SEED}.csv")
            OUTPUT_VIZ_FILE = os.path.join(output_folder, f"3_X_speed_R{LAMBDA_RANGE}_S{RANDOM_SEED}.png") # NEW VIZ OUTPUT
            
            print(f"--- Generating Spatially Correlated Observed Covariate (X_Speed) via GP ---")
            print(f"Range ($\lambda$): {LAMBDA_RANGE}, Spatial Magnitude ($\sigma_s$): {SIGMA_SPATIAL}, Noise ($\sigma_n$): {SIGMA_NOISE}")
            
            try:
                # 1. Load the Graph
                G = load_graphml_with_tuple_nodes(GRAPHML_FILE)

                # Insert this check inside the __main__ block, after loading G
                if not nx.is_connected(G):
                    print("\n❌ CRITICAL ERROR: The network graph is DISCONNECTED. The GP cannot model correlation across disconnected components.")
                    print("Please check your graph generation (1_network.graphml).")
                
                # 2. Generate the GP Field
                X_speed_values = generate_gp_field(
                    G=G, 
                    lambda_range=LAMBDA_RANGE, 
                    sigma_spatial=SIGMA_SPATIAL, 
                    sigma_noise=SIGMA_NOISE, 
                    mu_speed=MU_SPEED, 
                    seed=RANDOM_SEED
                )
                
                # 3. Update and export
                update_and_export_node_data(NODES_CSV_FILE, X_speed_values, OUTPUT_CSV_FILE)
                
                # Quick check of the generated covariate
                all_speeds = np.array(list(X_speed_values.values()))
                print(f"X_Speed Summary: Mean={np.mean(all_speeds):.2f}, StdDev={np.std(all_speeds):.2f}")
                
                # 4. Visualize the Field (NEW STEP)
                visualize_speed_field(
                    G, 
                    X_speed_values, 
                    filename=OUTPUT_VIZ_FILE, 
                    lambda_range=LAMBDA_RANGE, 
                    sigma_spatial=SIGMA_SPATIAL
                )
                
            except FileNotFoundError as e:
                print(f"\n❌ ERROR: {e}")
                print("Ensure `network_builder.py` has been run successfully to create the required files.")
            except Exception as e:
                print(f"\n❌ An unexpected error occurred: {e}")
