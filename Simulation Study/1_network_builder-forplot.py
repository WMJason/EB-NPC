import random
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import ast 
from typing import Dict, Any, List, Tuple

# --- Existing Functions (build_grid_network, assign_aadt_grid, compute_node_aadts) ---
# (Functions remain the same as provided in the prompt)

def build_grid_network(rows=20, cols=20) -> nx.Graph:
    # ... (function body remains the same)
    G = nx.Graph()
    for i in range(rows):
        for j in range(cols):
            G.add_node((i, j))
    for i in range(rows):
        for j in range(cols):
            if i + 1 < rows:
                G.add_edge((i, j), (i + 1, j))
            if j + 1 < cols:
                G.add_edge((i, j), (i, j + 1))
    return G


def assign_aadt_grid(G, major_aadt_low=8000, major_aadt_high=30000, minor_aadt_low=1500, minor_aadt_high=9000, seed=42):
    """
    Assigns AADT to all links randomly. Half of the links are assigned a 'major'
    range AADT, and the other half a 'minor' range AADT, independent of grid location.
    
    NOTE: This is a suggested fix to remove the spatial structure while retaining
    the *range difference* between 'major' and 'minor' volumes. 
    """
    np.random.seed(seed)
    
    # Randomly categorize links as 'conceptual major' or 'conceptual minor'
    # based on a simple coin flip, ignoring grid coordinates.
    link_types = [random.choice([True, False]) for _ in G.edges()]
    
    for (u, v), is_major in zip(list(G.edges()), link_types):
        
        if is_major:
            # Randomly draw from the 'major' range
            aadt = int(np.random.randint(major_aadt_low, major_aadt_high + 1))
        else:
            # Randomly draw from the 'minor' range
            aadt = int(np.random.randint(minor_aadt_low, minor_aadt_high + 1))

        G[u][v]["AADT"] = aadt
        G[u][v]["length"] = 1.0
        G[u][v]["is_major"] = is_major # Still track the type for range assignment

def compute_node_aadts(G) -> Dict[Tuple[int, int], Dict[str, int]]:
    # ... (function body remains the same)
    node_aadts = {}
    for n in G.nodes():
        incident = list(G.edges(n, data=True))
        incident_aadts = [d.get("AADT", 0) for (a, b, d) in incident]
        
        if len(incident_aadts) == 0:
            major, minor = 0, 0
        elif len(incident_aadts) == 1:
            major, minor = incident_aadts[0], incident_aadts[0]
        else:
            sorted_a = sorted(incident_aadts, reverse=True)
            major, minor = sorted_a[0], sorted_a[1]
            
        node_aadts[n] = {"majorAADT": int(major), "minorAADT": int(minor)}
    return node_aadts

# --- NEW FUNCTION: Select Knots ---

def select_knots(G, fraction: float = 0.5) -> List[Tuple[int, int]]:
    """
    Selects approximately 'fraction' of the nodes as knots using a 
    deterministic, evenly distributed (checkerboard) pattern.
    """
    knots = []
    total_nodes = len(G.nodes())
    
    # Use a parity check for an even (approx. 50%) distribution
    for node in G.nodes():
        # Check if row + col is even
        if (node[0] + node[1]) % 2 == 0:
            knots.append(node)
            
    print(f"Selected {len(knots)} knots out of {total_nodes} nodes ({len(knots)/total_nodes:.1%}).")
    return knots

# --- Export and Visualization Functions ---

def export_knots_data(knots: List[Tuple[int, int]], filename: str) -> pd.DataFrame:
    """Exports knot identity information to a CSV file."""
    knot_data = []
    for node in knots:
        knot_data.append({
            "node_id": str(node),
            "is_knot": 1
        })
        
    knots_df = pd.DataFrame(knot_data)
    knots_df.to_csv(filename.replace('.csv','_plot.csv'), index=False)
    print(f"✅ Exported knot identity data to {filename}")
    return knots_df

def export_graphml(G, filename="simulated_network_plot.graphml"):
    """Export the networkx graph object to a GraphML file."""
    H = nx.Graph()
    for u, v, data in G.edges(data=True):
        H.add_edge(str(u), str(v), **data)
    
    nx.write_graphml(H, filename)
    print(f"✅ Exported network graph (GraphML) to {filename}")
    return filename

def export_network_data(G, node_aadts, filename_prefix="simulated_network"):
    """Export link and node data to CSV files."""
    
    # 1. Links Data (Edges)
    link_data = []
    for u, v, data in G.edges(data=True):
        link_data.append({
            "link_id": f"{u}-{v}",
            "node_u": str(u),
            "node_v": str(v),
            "AADT": data.get("AADT"),
            "is_major": data.get("is_major")
        })
    links_df = pd.DataFrame(link_data)
    links_df.to_csv(f"{filename_prefix}_links_plot.csv", index=False)
    print(f"✅ Exported link data to {filename_prefix}_links.csv")

    # 2. Nodes Data
    node_data = []
    for node, aadt_info in node_aadts.items():
        node_data.append({
            "node_id": str(node),
            "majorAADT": aadt_info["majorAADT"],
            "minorAADT": aadt_info["minorAADT"]
        })
    nodes_df = pd.DataFrame(node_data)
    nodes_df.to_csv(f"{filename_prefix}_nodes_plot.csv", index=False)
    print(f"✅ Exported node data to {filename_prefix}_nodes.csv")
    
    return links_df, nodes_df


def visualize_network(G, knots: List[Tuple[int, int]], filename="simulated_network_viz_plot.png"):
    """Visualize the network, highlighting links by AADT and nodes selected as knots."""
    
    # --- Sizing Parameters for Visibility ---
    NODE_SIZE_KNOTS = 300 # Increased size for highlighted knots
    NODE_SIZE_OTHERS = 100
    EDGE_WIDTH = 3       # Thicker edges
    TITLE_FONTSIZE = 30  # Larger title
    LEGEND_FONTSIZE = 20 # Larger legend text
    LABEL_FONTSIZE = 20  # Larger color bar label

    link_aadt = [d['AADT'] for u, v, d in G.edges(data=True)]
    pos = {node: (node[1], -node[0]) for node in G.nodes()}
    
    knot_nodes = [n for n in G.nodes() if n in knots]
    other_nodes = [n for n in G.nodes() if n not in knots]
    
    # Increase the figure size slightly for better presentation
    plt.figure(figsize=(14, 14)) 
    
    # Draw non-knot nodes (gray)
    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_size=NODE_SIZE_OTHERS, node_color='gray', alpha=0.8, label='Non-Knot Site')
    
    # Draw knot nodes (red)
    nx.draw_networkx_nodes(G, pos, nodelist=knot_nodes, node_size=NODE_SIZE_KNOTS, node_color='red', alpha=1.0, label='Knot Site')
    
    # Draw edges
    edges = nx.draw_networkx_edges(
        G, pos,
        edge_color=link_aadt,
        edge_cmap=plt.cm.coolwarm,
        width=EDGE_WIDTH,
        alpha=0.8
    )
    
    if edges is not None:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm)
        sm.set_array(link_aadt)
        
        # Draw the color bar
        cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)
        
        # Adjust Color Bar Label Size
        cbar.set_label('Link AADT', fontsize=LABEL_FONTSIZE) 
        # Adjust Color Bar Tick Label Size
        cbar.ax.tick_params(labelsize=LABEL_FONTSIZE - 2) 

    # Adjust Legend Size and Location
    plt.legend(scatterpoints=1, frameon=True, loc='lower left', fontsize=LEGEND_FONTSIZE)
    
    # Adjust Title Size
    plt.title("Artificial Grid Network\n(Links by AADT, Knots Highlighted)", fontsize=TITLE_FONTSIZE)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"✅ Exported network visualization to {filename}")


if __name__ == "__main__":
    # --- Configuration ---
    ROWS = 10
    COLS = 10
    SEED = 42
    OUTPUT_PREFIX = "1"
    
    # --- 1. Build Network ---
    print(f"Building {ROWS}x{COLS} grid network...")
    G = build_grid_network(rows=ROWS, cols=COLS)

    # --- 2. Assign AADT to Links ---
    print("Assigning AADT to links (edges)...")
    assign_aadt_grid(G, major_aadt_low=8000, major_aadt_high=30000, minor_aadt_low=1500, minor_aadt_high=9000, seed=SEED)

    # --- 3. Compute Node AADTs ---
    print("Computing major/minor AADT for nodes...")
    node_aadts = compute_node_aadts(G)
    
    # --- 4. Select Knots ---
    print("\n--- Selecting Knots ---")
    knots = select_knots(G, fraction=0.5)
    
    # --- 5. Export Data ---
    print("\n--- Exporting Data ---")
    export_network_data(G, node_aadts, filename_prefix=OUTPUT_PREFIX)
    export_knots_data(knots, filename=f"{OUTPUT_PREFIX}_knots_plot.csv")
    export_graphml(G, filename=f"{OUTPUT_PREFIX}_network_plot.graphml")

    # --- 6. Visualize Network ---
    print("\n--- Visualizing Network ---")
    visualize_network(G, knots, filename=f"{OUTPUT_PREFIX}_network_viz_plot.png")

    print("\nSetup complete.")
