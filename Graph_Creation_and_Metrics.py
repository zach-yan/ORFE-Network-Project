import networkx as nx
import pandas as pd

# 1. Load the finalized data directly into GPU memory using cuDF
nodes_df = pd.read_csv("orfe_nodes_final_gravity.csv")
edges_df = pd.read_csv("orfe_edges_fuzzy_cleaned.csv")

# 2. Initialize the Graph
G = nx.Graph()

# 3. Add Nodes with Attributes
print("Adding nodes to the graph...")
for _, row in nodes_df.iterrows():
    # We use .get() for Subjects/Applications in case any rows are missing them
    subjects = row.get('Subjects', '[]')
    applications = row.get('Applications', '[]')

    G.add_node(
        row['Id'],
        label=row['Label'],
        layer=row['Layer'],
        affiliation=row['Affiliation'],
        subjects=subjects,
        applications=applications
    )

# 4. Add Edges with Weights
print("Adding edges to the graph...")
for _, row in edges_df.iterrows():
    # A quick safety check to ensure we only add edges between nodes that actually exist in our nodes_df
    if G.has_node(row['Source']) and G.has_node(row['Target']):
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'])

# 5. Output Basic Graph Metrics
print("\n--- Network Summary ---")
print(f"Total Nodes: {G.number_of_nodes():,}")
print(f"Total Edges: {G.number_of_edges():,}")

# Density calculation
# Calculate network density
density = nx.density(G)
print(f"Network Density: {density:.6f}")


# 1. Number of cycles
# We convert the GPU edgelist to pandas to feed into networkx just for this step
num_cycles = nx.cycle_basis(G)
print(f"Number of cycles in the graph: {len(num_cycles)}")

# 2. Vectorized GPU calculation for nodes connected to high-degree nodes
nodes_connected_to_high_degree_nodes = set()

for node in G.nodes():
    # Get neighbors of the current node
    neighbors = list(G.neighbors(node))

    for neighbor in neighbors:
        # Check if the neighbor has more than one edge (degree > 1)
        if G.degree(neighbor) > 1:
            nodes_connected_to_high_degree_nodes.add(node)
            break # Once one such neighbor is found, move to the next node

print(f"Total nodes connected to a node with more than one edge: {len(nodes_connected_to_high_degree_nodes)}")

# Compute centrality metrics
print("Computing Degree Centrality...")
degree_centrality = nx.degree_centrality(G)

# Sort nodes by degree centrality in descending order
sorted_degree = sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)

# Convert the dictionary into a DataFrame
dc_df = pd.DataFrame(list(degree_centrality.items()), columns=['vertex', 'degree_centrality'])

print("Top 10 nodes by Degree Centrality:")
for node, dc in sorted_degree[:10]:
    label = G.nodes[node].get('label', 'N/A') # Get the label, default to 'N/A' if not found
    print(f"Node: {node} (Label: {label}), Degree Centrality: {dc:.4f}")

print("Computing Closeness Centrality...")
closeness_centrality = nx.closeness_centrality(G)

# Sort nodes by closeness centrality in descending order
sorted_closeness = sorted(closeness_centrality.items(), key=lambda item: item[1], reverse=True)

print("Top 10 nodes by Closeness Centrality:")
for node, cc in sorted_closeness[:10]:
    label = G.nodes[node].get('label', 'N/A')
    print(f"Node: {node} (Label: {label}), Closeness Centrality: {cc:.4f}")

# Convert the dictionary into a DataFrame
close_df = pd.DataFrame(list(closeness_centrality.items()), columns=['vertex', 'degree_centrality'])

print("Computing Betweenness Centrality...")
betweenness_centrality = nx.betweenness_centrality(G)

# Sort nodes by betweenness centrality in descending order
sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse=True)

print("Top 10 nodes by Betweenness Centrality:")
for node, bc in sorted_betweenness[:10]:
    label = G.nodes[node].get('label', 'N/A')
    print(f"Node: {node} (Label: {label}), Betweenness Centrality: {bc:.4f}")

print("Centrality calculations complete.")

# Convert the dictionary into a DataFrame
btw_df = pd.DataFrame(list(betweenness_centrality.items()), columns=['vertex', 'degree_centrality'])

# Ensure our node attributes dataframe is ready for merging
# We will use 'Id' as the merge key, matching it to the 'vertex' output from cuGraph
metrics_df = nodes_df[['Id', 'Label']].rename(columns={'Id': 'vertex'})

print("Merging metrics...")
# Merge all centrality dataframes based on the vertex ID
metrics_df = metrics_df.merge(dc_df, on='vertex', how='left')
metrics_df = metrics_df.merge(close_df, on='vertex', how='left')
metrics_df = metrics_df.merge(btw_df, on='vertex', how='left')

# Sort by Degree Centrality as an example output
metrics_df = metrics_df.sort_values(by='degree_centrality', ascending=False)

print("\n--- Top 5 Nodes by Degree Centrality ---")
print(metrics_df[['vertex', 'Label', 'degree_centrality']].head(5))

# Save the final consolidated metrics to a CSV file
output_filename = "orfe_graph_metrics.csv"
metrics_df.to_csv(output_filename, index=False)
print(f"\nAll metrics successfully saved to {output_filename}")
