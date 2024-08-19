import os
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import StandardScaler
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GraphConv
from digae_traffic import DirectedGNN, train_digae, test_digae, train_digae_quantile_regression, test_digae_quantile_regression
from autoencoder import GAE, DirectedEdgeDecoder
from scipy.stats import norm
import pickle

# Load Anaheim data
nodefile = os.path.join(os.path.dirname(__file__), 'traffic', 'anaheim_nodes.geojson')
nodes = gpd.read_file(nodefile)
nodes_df = pd.DataFrame(nodes)
nodes_df[['X', 'Y']] = nodes_df['geometry'].astype(str).str.split('(').str[-1].str.split(')').str[0].str.split(' ', expand=True).astype(np.float32)
node = nodes_df.rename(columns={'id': 'node'})

node_rename = {node: id for id, node in enumerate(range(1, 417))}
node['node'] = node['node'].map(node_rename)

# Load flow data
flowfile = os.path.join(os.path.dirname(__file__), 'traffic', 'Anaheim_flow.tntp')
colname = 'Volume '
flow = pd.read_csv(flowfile, sep='\t', usecols=['From ', 'To ', colname])

flow['From '] = flow['From '].map(node_rename)
flow['To '] = flow['To '].map(node_rename)
flow = flow[(flow['From '].notna()) & (flow['To '].notna())]
flow.drop(flow[flow[colname] <= 0].index, inplace=True)
flow[colname] = np.log(flow[colname])

# Preprocess node features
scaler = StandardScaler()
node[['X', 'Y']] = scaler.fit_transform(node[['X', 'Y']].values)

# Merge node and flow data
df = flow.rename(columns={'From ': 's', 'To ': 'r', colname: 'w'})
df1 = pd.merge(df, node, how='left', left_on='s', right_on='node')[['s', 'r', 'w', 'X', 'Y']].rename(columns={'X': 'X1', 'Y': 'Y1'})
df2 = pd.merge(df1, node, how='left', left_on='r', right_on='node')[['s', 'r', 'w', 'X1', 'Y1', 'X', 'Y']].rename(columns={'X': 'X2', 'Y': 'Y2'})
df2['feat'] = df2[['X1', 'Y1', 'X2', 'Y2']].values.tolist()

edge_name_to_y = {(s, r): w for s, r, w in df2[['s', 'r', 'w']].values}
edge_name_to_x = {(s, r): feat for s, r, feat in df2[['s', 'r', 'feat']].values}


'''
Uncomment for Chicago data
'''
# node_rename = {node: id for id, node in enumerate(range(388, 934))}
# nodefile = os.path.join(os.path.dirname(__file__), 'traffic', 'ChicagoSketch_node.tntp')
# node = pd.read_csv(nodefile, sep='\t', usecols=['node', 'X', 'Y'])
# flowfile = os.path.join(os.path.dirname(__file__), 'traffic', 'ChicagoSketch_flow.tntp')
# colname = 'Volume '
# flow = pd.read_csv(flowfile, sep='\t', usecols=['From ', 'To ', colname])
#
# node['node'] = node['node'].map(node_rename)
# node = node[node['node'].notna()]
#
# flow['From '] = flow['From '].map(node_rename)
# flow['To '] = flow['To '].map(node_rename)
# flow = flow[(flow['From '].notna()) & (flow['To '].notna())]
# flow.drop(flow[flow[colname] <= 0].index, inplace=True)
# flow[colname] = np.log(flow[colname])
#
# scaler = StandardScaler()
# node[['X', 'Y']] = scaler.fit_transform(node[['X', 'Y']].values)
# # minmax = MinMaxScaler()
# # flow[[colname]] = minmax.fit_transform(flow[[colname]].values)
#
# df = flow.rename(columns={'From ': 's', 'To ': 'r', colname: 'w'})
# df1 = pd.merge(df, node, how='left', left_on='s', right_on='node')[['s', 'r', 'w', 'X', 'Y']].rename(columns={'X': 'X1', 'Y': 'Y1'})
# df2 = pd.merge(df1, node, how='left', left_on='r', right_on='node')[['s', 'r', 'w', 'X1', 'Y1', 'X', 'Y']].rename(columns={'X': 'X2', 'Y': 'Y2'})
# df2['feat'] = df2[['X1', 'Y1', 'X2', 'Y2']].values.tolist()
#
# edge_name_to_y = {(s, r): w for s, r, w in df2[['s', 'r', 'w']].values}
# edge_name_to_x = {(s, r): feat for s, r, feat in df2[['s', 'r', 'feat']].values}


# Construct traffic network graph
G = nx.from_pandas_edgelist(df2, source='s', target='r', edge_attr='w', create_using=nx.DiGraph())
G = nx.convert_node_labels_to_integers(G)
traffic_network = from_networkx(G)
traffic_network.x = torch.from_numpy(node[['X', 'Y']].values).to(torch.float32)

# Split edges into training, validation, and cal_test sets
train_ratio = 0.5
val_ratio = 0.1
cal_test_ratio = 0.4

edge_index, edge_weight = traffic_network.edge_index, traffic_network.w

transform = RandomLinkSplit(is_undirected=False, num_val=val_ratio, num_test=cal_test_ratio, add_negative_train_samples=False, neg_sampling_ratio=0.0)
train_data, val_data, cal_test_data = transform(traffic_network)

# Extract edge indices for each set
edge_index_train = train_data.edge_label_index
edge_index_val = val_data.edge_label_index
edge_index_cal_test = cal_test_data.edge_label_index
edge_index_numpy = edge_index.numpy()

# Find indices of edges in each set
train_index = np.array(np.all((edge_index_numpy[:, None, :] == edge_index_train.numpy()[:, :, None]), axis=0).nonzero()).T[:, -1]
val_index = np.array(np.all((edge_index_numpy[:, None, :] == edge_index_val.numpy()[:, :, None]), axis=0).nonzero()).T[:, -1]
cal_test_index = np.array(np.all((edge_index_numpy[:, None, :] == edge_index_cal_test.numpy()[:, :, None]), axis=0).nonzero()).T[:, -1]

# Split edge weights for each set
edge_weight_train = edge_weight[train_index]
edge_weight_val = edge_weight[val_index]
edge_weight_cal_test = edge_weight[cal_test_index]

alpha_values = np.linspace(0.01, 0.1, 10)
quantile_levels = [0.25, 0.75]

# Create the encoder and decoder for edge weight prediction
out = len(quantile_levels) * 4
hidden = out * 8
encoder = DirectedGNN(in_channels=traffic_network.x.shape[-1], hidden_channels=hidden, out_channels=out, gconv=GraphConv)
decoder = DirectedEdgeDecoder(out, out_channels=1)
model = GAE(encoder, decoder)
print("Edge Weight Predictor:")
print(model)

# Train the edge weight predictor
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
best_val_loss = float('inf')
best_model = None
epochs = 2000
for epoch in range(epochs):
    model.train()
    model, loss = train_digae(model, traffic_network.x, edge_index_train, edge_weight_train, optimizer=optimizer)
    if epoch % 50 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    _, val_loss = train_digae(model, traffic_network.x, edge_index_train, edge_weight_train, val=True, edge_index_val=edge_index_val, edge_weight_val=edge_weight_val)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        print(f'Epoch: {epoch:03d}, Best validation loss: {val_loss:.4f}')

# Test the edge weight predictor
test_loss = test_digae(best_model, traffic_network.x, edge_index_train, edge_weight_train, edge_index_cal_test, edge_weight_cal_test)
print(f"Test Loss: {test_loss:.4f}")

# Create the encoder and decoder for edge weight quantile regression
decoder_quantile = DirectedEdgeDecoder(out//len(quantile_levels), out_channels=1)  # Output 3 quantiles (0.1, 0.5, 0.9)
encoder_quantile = DirectedGNN(in_channels=traffic_network.x.shape[-1], hidden_channels=hidden, out_channels=out, gconv=GraphConv)
model_quantile = GAE(encoder_quantile, decoder_quantile)
print("\nEdge Weight Quantile Regression Model:")
print(model_quantile)

# Train the edge weight quantile regression model
optimizer_quantile = torch.optim.Adam(model_quantile.parameters(), lr=0.01, weight_decay=1e-4)
best_val_loss_quantile = float('inf')
best_model_quantile = None
epochs_quantile = 100
for epoch in range(epochs_quantile):
    model_quantile.train()
    model_quantile, loss_quantile = train_digae_quantile_regression(model_quantile, traffic_network.x, edge_index_train, edge_weight_train, quantiles=quantile_levels, optimizer=optimizer_quantile)
    if epoch % 50 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss_quantile:.4f}')
    _, val_loss_quantile = train_digae_quantile_regression(model_quantile, traffic_network.x, edge_index_train, edge_weight_train, quantiles=quantile_levels, val=True, edge_index_val=edge_index_val, edge_weight_val=edge_weight_val)
    if val_loss_quantile < best_val_loss_quantile:
        best_val_loss_quantile = val_loss_quantile
        best_model_quantile = model_quantile
        print(f'Epoch: {epoch:03d}, Best validation loss: {val_loss_quantile:.4f}')

# Test the edge weight quantile regression model
test_loss_quantile = test_digae_quantile_regression(best_model_quantile, traffic_network.x, edge_index_train, edge_weight_train, edge_index_cal_test, quantiles=quantile_levels, edge_weight_test=edge_weight_cal_test)
print(f"Test Loss (Quantile Prediction): {test_loss_quantile[0]:.4f}")
# Conformal prediction

# Compute non-conformity scores for calibration set
predicted_weights_cal_test = test_digae(best_model, traffic_network.x, torch.hstack([edge_index_train, edge_index_val]), torch.hstack([edge_weight_train, edge_weight_val]), edge_index_cal_test).detach().numpy()
quantiles_calibration = test_digae_quantile_regression(best_model_quantile, traffic_network.x, torch.hstack([edge_index_train, edge_index_val]), torch.hstack([edge_weight_train, edge_weight_val]), edge_index_cal_test, quantiles=quantile_levels)
predicted_upper_cal_test = quantiles_calibration[1].squeeze().detach().numpy()
predicted_lower_cal_test = quantiles_calibration[0].squeeze().detach().numpy()

# Create a new graph with true edge weights for training and validation sets
G_pred = nx.DiGraph()
for u, v, w in zip(edge_index_train[0], edge_index_train[1], edge_weight_train):
    G_pred.add_edge(u.item(), v.item(), weight=w.item())
for u, v, w in zip(edge_index_val[0], edge_index_val[1], edge_weight_val):
    G_pred.add_edge(u.item(), v.item(), weight=w.item())
for u, v, w in zip(edge_index_cal_test[0], edge_index_cal_test[1], predicted_weights_cal_test):
    G_pred.add_edge(u.item(), v.item(), weight=w.item())

results_df = pd.DataFrame([])
# Sample pairs of source and target nodes randomly
num_paths = 2000
sampled_pairs = []
paths = []
for _ in range(num_paths):
    source = np.random.choice([x for x in G_pred.nodes() if G_pred.out_degree(x) >= 1])
    target = np.random.choice([x for x in G_pred.nodes() if G_pred.in_degree(x) >= 1])
    while source == target:
        target = np.random.choice([x for x in G_pred.nodes() if G.in_degree(x) >= 1])
    sampled_pairs.append((source, target))
    paths.append(nx.shortest_path(G_pred, source, target, weight='weight'))

# Load the NumPy arrays
y = edge_weight_cal_test.detach().numpy()
yhat = predicted_weights_cal_test
yhat_upper = predicted_upper_cal_test
yhat_lower = predicted_lower_cal_test

# Assume you have the following variables:
# - paths: a list of 2000 lists representing the paths
# - edge_index: a 2 by 450 array representing the edge indices

# Create an empty array of shape (450, 2000) initialized with zeros
output_array = np.zeros((len(y), num_paths))

# Iterate over each path in the paths list
for path_index, path in enumerate(paths):
    # Iterate over the consecutive pairs of nodes in the path
    for i in range(len(path) - 1):
        start_node = path[i]
        end_node = path[i + 1]

        # Find the corresponding edge index in the edge_index array
        edge_indices = np.where((edge_index_cal_test[0] == start_node) & (edge_index_cal_test[1] == end_node))[0]

        if len(edge_indices) > 0:
            edge_index_found = edge_indices[0]
            # Set the value in the output array to 1
            output_array[edge_index_found, path_index] = 1

result_array_abs = (yhat - y)[:, np.newaxis] * output_array
result_array_quantile = (y - yhat_upper)[:, np.newaxis] * output_array
true_array = y[:, np.newaxis] * output_array
predicted_array = yhat[:, np.newaxis] * output_array
upper_array = yhat_upper[:, np.newaxis] * output_array
num_iterations = 100

coverage_abs_nonstratified = np.zeros((num_iterations, len(alpha_values)))
coverage_quantile_nonstratified = np.zeros((num_iterations, len(alpha_values)))
efficiency_abs_nonstratified = np.zeros((num_iterations, len(alpha_values)))
efficiency_quantile_nonstratified = np.zeros((num_iterations, len(alpha_values)))

coverage_normal_conformal = np.zeros((num_iterations, len(alpha_values)))
coverage_normal_nonconformal = np.zeros((num_iterations, len(alpha_values)))
efficiency_normal_conformal = np.zeros((num_iterations, len(alpha_values)))
efficiency_normal_nonconformal = np.zeros((num_iterations, len(alpha_values)))

coverage_abs_stratified = np.zeros((num_iterations, len(alpha_values)))
coverage_quantile_stratified = np.zeros((num_iterations, len(alpha_values)))
efficiency_abs_stratified = np.zeros((num_iterations, len(alpha_values)))
efficiency_quantile_stratified = np.zeros((num_iterations, len(alpha_values)))
coverage_abs_naive = np.zeros((num_iterations, len(alpha_values)))
coverage_quantile_naive = np.zeros((num_iterations, len(alpha_values)))
efficiency_abs_naive = np.zeros((num_iterations, len(alpha_values)))
efficiency_quantile_naive = np.zeros((num_iterations, len(alpha_values)))
coverage_abs_bonferroni = np.zeros((num_iterations, len(alpha_values)))
coverage_quantile_bonferroni = np.zeros((num_iterations, len(alpha_values)))
efficiency_abs_bonferroni = np.zeros((num_iterations, len(alpha_values)))
efficiency_quantile_bonferroni = np.zeros((num_iterations, len(alpha_values)))

coverage_abs_stratified_null = np.zeros((num_iterations, len(alpha_values)))
coverage_quantile_stratified_null = np.zeros((num_iterations, len(alpha_values)))
efficiency_abs_stratified_null = np.zeros((num_iterations, len(alpha_values)))
efficiency_quantile_stratified_null = np.zeros((num_iterations, len(alpha_values)))

for i in range(num_iterations):
    # Randomly partition the rows into two parts
    row_indices = np.arange(len(y))
    np.random.shuffle(row_indices)
    partition_1_abs = result_array_abs[row_indices[:len(y) // 2], :]
    partition_2_abs = result_array_abs[row_indices[len(y) // 2:], :]
    partition_1_quantile = result_array_quantile[row_indices[:len(y) // 2], :]
    partition_2_quantile = result_array_quantile[row_indices[len(y) // 2:], :]

    # Compute the column sums for each partition
    column_sums_1_abs = np.abs(np.sum(partition_1_abs, axis=0))
    column_sums_2_abs = np.abs(np.sum(partition_2_abs, axis=0))
    column_sums_1_quantile = np.sum(partition_1_quantile, axis=0)
    column_sums_2_quantile = np.sum(partition_2_quantile, axis=0)

    # Disregard any paths that either do not contain test edges or calibration edges
    calib_null = np.where(output_array[row_indices[:len(y) // 2]].sum(axis=0) == 0)[0]
    calib_extant = np.where(output_array[row_indices[:len(y) // 2]].sum(axis=0) > 0)[0]
    test_null = np.where(output_array[row_indices[len(y) // 2:]].sum(axis=0) == 0)[0]
    test_extant = np.where(output_array[row_indices[len(y) // 2:]].sum(axis=0) > 0)[0]

    column_sums_1_abs = column_sums_1_abs[calib_extant]
    column_sums_1_quantile = column_sums_1_quantile[calib_extant]
    column_sums_2_abs = column_sums_2_abs[test_extant]
    column_sums_2_quantile = column_sums_2_quantile[test_extant]

    '''
    nonstratified
    '''
    for idx, alpha in enumerate(alpha_values):

        # Find the percentile values (nonstratified)
        percentile_threshold_nonstratified = min(100, (100 - 100 * alpha) * (len(calib_extant) + 1) / len(calib_extant))
        percentile_value_abs_nonstratified = np.percentile(column_sums_1_abs, percentile_threshold_nonstratified)
        percentile_value_quantile_nonstratified = np.percentile(column_sums_1_quantile, percentile_threshold_nonstratified)

        # Count the number of entries in the second vector below the percentile value (nonstratified)
        count_abs_nonstratified = np.sum(column_sums_2_abs <= percentile_value_abs_nonstratified)
        count_quantile_nonstratified = np.sum(column_sums_2_quantile <= percentile_value_quantile_nonstratified)

        # Record coverage and efficiency (nonstratified)
        coverage_abs_nonstratified[i, idx] = count_abs_nonstratified / len(column_sums_2_abs)
        coverage_quantile_nonstratified[i, idx] = count_quantile_nonstratified / len(column_sums_2_quantile)
        efficiency_abs_nonstratified[i, idx] = (yhat[:, np.newaxis] * output_array)[row_indices[len(y) // 2:]][:, test_extant].sum()/len(column_sums_2_abs) + percentile_value_abs_nonstratified
        efficiency_quantile_nonstratified[i, idx] = (yhat_upper[:, np.newaxis] * output_array)[row_indices[len(y) // 2:]][:, test_extant].sum()/len(column_sums_2_quantile) + percentile_value_quantile_nonstratified
    '''
    stratified
    '''
    # Stratified version
    column_sums_1 = np.sum(output_array[row_indices[:len(y) // 2], :], axis=0)[calib_extant]
    column_sums_2 = np.sum(output_array[row_indices[len(y) // 2:], :], axis=0)[test_extant]
    K = len(calib_extant) // 200  # Number of quantiles
    level_quantiles_1 = np.percentile(np.concatenate([column_sums_1, column_sums_2]), np.linspace(0, 100, K + 1))
    level_quantiles_1 = np.ceil(level_quantiles_1).astype(int)
    level_quantiles_1 = np.unique(level_quantiles_1)
    level_quantiles_1 = np.concatenate([np.array([0]), level_quantiles_1])

    for idx, alpha in enumerate(alpha_values):
        count_abs_stratified = 0
        count_quantile_stratified = 0
        total_columns_abs = 0
        total_columns_quantile = 0
        efficiency_sum_abs = 0
        efficiency_sum_quantile = 0

        for j in range(len(level_quantiles_1) - 1):
            lower_threshold = level_quantiles_1[j]
            upper_threshold = level_quantiles_1[j + 1]

            # Find the indices of elements within the current quantile range
            indices_1 = np.where((column_sums_1 > lower_threshold) & (column_sums_1 <= upper_threshold))[0]
            indices_2 = np.where((column_sums_2 > lower_threshold) & (column_sums_2 <= upper_threshold))[0]

            # Partition the data based on the indices
            partitioned_value_1_abs = column_sums_1_abs[indices_1]
            partitioned_value_2_abs = column_sums_2_abs[indices_2]
            percentile_value = np.percentile(partitioned_value_1_abs,
                                             min(100, (100 - 100 * alpha) * (len(indices_1)+1)/len(indices_1)))
            efficiency_sum_abs += (
                        (yhat[:, np.newaxis] * output_array)[row_indices[len(y) // 2:]][:, test_extant[indices_2]].sum()
                        + percentile_value * len(indices_2))
            count = np.sum(partitioned_value_2_abs <= percentile_value)
            count_abs_stratified += count
            total_columns_abs += len(indices_2)

            partitioned_value_1_quantile = column_sums_1_quantile[indices_1]
            partitioned_value_2_quantile = column_sums_2_quantile[indices_2]
            percentile_value = np.percentile(partitioned_value_1_quantile,
                                             min(100, (100 - 100 * alpha) * (len(indices_1)+1)/len(indices_1)))
            count = np.sum(partitioned_value_2_quantile <= percentile_value)
            count_quantile_stratified += count
            total_columns_quantile += len(indices_2)
            efficiency_sum_quantile += (
                        (yhat_upper[:, np.newaxis] * output_array)[row_indices[len(y) // 2:]][:, test_extant[indices_2]].sum()
                        + percentile_value * len(indices_2))

        coverage_abs_stratified[i, idx] = count_abs_stratified / total_columns_abs
        coverage_quantile_stratified[i, idx] = count_quantile_stratified / total_columns_quantile
        efficiency_abs_stratified[i, idx] = efficiency_sum_abs / total_columns_abs
        efficiency_quantile_stratified[i, idx] = efficiency_sum_quantile / total_columns_quantile

    '''
    stratified and bootstrapped
    '''
    # Stratified version
    column_sums_1 = np.sum(output_array[row_indices[:len(y) // 2], :], axis=0)[calib_extant]
    column_sums_2 = np.sum(output_array[row_indices[len(y) // 2:], :], axis=0)[test_extant]

    num_samples = 100  # User-defined number of samples
    for idx, alpha in enumerate(alpha_values):
        count_abs_stratified = 0
        count_quantile_stratified = 0
        total_columns_abs = 0
        total_columns_quantile = 0
        efficiency_sum_abs = 0
        efficiency_sum_quantile = 0

        for level in np.unique(column_sums_2):
            percentile_threshold_stratified = min(100, (100 - 100 * alpha) * (1+num_samples)/num_samples)

            # Sampling without replacement k edges in the calibration set
            k = int(level)
            abs_errors = [np.abs((yhat - y)[row_indices[:len(y) // 2]][np.random.choice(np.arange(len(y) // 2), size=(k,), replace=False)].sum()) for _ in range(num_samples)]
            quantile_errors = [(y - yhat_upper)[row_indices[:len(y) // 2]][np.random.choice(np.arange(len(y) // 2), size=(k,), replace=False)].sum() for _ in range(num_samples)]
            percentile_value_abs_stratified = np.percentile(abs_errors, percentile_threshold_stratified)
            percentile_value_quantile_stratified = np.percentile(quantile_errors, percentile_threshold_stratified)

            column_indices_2 = np.where(column_sums_2 == level)[0]
            count_abs_stratified += np.sum(column_sums_2_abs[column_indices_2] <= percentile_value_abs_stratified)
            count_quantile_stratified += np.sum(column_sums_2_quantile[column_indices_2] <= percentile_value_quantile_stratified)

            total_columns_abs += len(column_indices_2)
            total_columns_quantile += len(column_indices_2)
            efficiency_sum_abs += (
                    (yhat[:, np.newaxis] * output_array)[row_indices[len(y) // 2:]][:, test_extant[column_indices_2]].sum()
                    + percentile_value_abs_stratified * len(column_indices_2))
            efficiency_sum_quantile += (
                    (yhat_upper[:, np.newaxis] * output_array)[row_indices[len(y) // 2:]][:, test_extant[column_indices_2]].sum()
                    + percentile_value_quantile_stratified * len(column_indices_2))
        coverage_abs_stratified_null[i, idx] = count_abs_stratified / total_columns_abs
        coverage_quantile_stratified_null[i, idx] = count_quantile_stratified / total_columns_quantile
        efficiency_abs_stratified_null[i, idx] = efficiency_sum_abs / total_columns_abs
        efficiency_quantile_stratified_null[i, idx] = efficiency_sum_quantile / total_columns_quantile


    score_abs = np.abs(yhat - y)[row_indices[:len(y) // 2]]
    score_quantile = (y - yhat_upper)[row_indices[:len(y) // 2]]
    for idx, alpha in enumerate(alpha_values):
        percentile_threshold_bonferroni = np.minimum(100, (100 - 100 * alpha / np.maximum(1, column_sums_2)))
        result_bonferroni_abs = predicted_array[row_indices[len(y) // 2:], :][:, test_extant].sum(axis=0) + column_sums_2 * np.percentile(score_abs,
                                                                                                                percentile_threshold_bonferroni)  # 128 * 5000
        result_bonferroni_quantile = upper_array[row_indices[len(y) // 2:], :][:, test_extant].sum(axis=0) + column_sums_2 * np.percentile(score_quantile,percentile_threshold_bonferroni)
        '''
        conformal normal approximation
        '''
        std = np.sqrt(1/(len(y)//2-1) * ((yhat - y)**2)[row_indices[:len(y) // 2]].sum())
        result_normal_conformal = predicted_array[row_indices[len(y) // 2:], :][:, test_extant].sum(axis=0) + np.sqrt(
            column_sums_2) * std * norm.ppf(1 - alpha)
        '''
        nonconformal normal approximation
        '''
        std = (yhat_upper - yhat_lower) / (norm.ppf(0.75) - norm.ppf(0.25))
        result_normal_nonconformal = predicted_array[row_indices[len(y) // 2:], :][:, test_extant].sum(axis=0) + np.sqrt(
            ((std[:, np.newaxis] * output_array)[row_indices[len(y) // 2:], :][:, test_extant] ** 2).sum(axis=0)) * norm.ppf(1 - alpha)

        # Record coverage and efficiency (stratified)
        coverage_normal_conformal[i, idx] = (result_normal_conformal >= true_array[row_indices[len(y) // 2:], :][:, test_extant].sum(axis=0)).mean()
        coverage_normal_nonconformal[i, idx] = (result_normal_nonconformal >= true_array[row_indices[len(y) // 2:], :][:, test_extant].sum(axis=0)).mean()
        efficiency_normal_conformal[i, idx] = result_normal_conformal.mean()
        efficiency_normal_nonconformal[i, idx] = result_normal_nonconformal.mean()
        '''
        bonferroni
        '''
        coverage_abs_bonferroni[i, idx] = (result_bonferroni_abs >= true_array[row_indices[len(y) // 2:], :][:, test_extant].sum(axis=0)).mean()
        coverage_quantile_bonferroni[i, idx] = (result_bonferroni_quantile >= true_array[row_indices[len(y) // 2:], :][:, test_extant].sum(axis=0)).mean()
        efficiency_abs_bonferroni[i, idx] = result_bonferroni_abs.mean()
        efficiency_quantile_bonferroni[i, idx] = result_bonferroni_quantile.mean()

results = [coverage_abs_nonstratified, efficiency_abs_nonstratified, coverage_quantile_nonstratified, efficiency_quantile_nonstratified, \
    coverage_normal_nonconformal, efficiency_normal_nonconformal, coverage_normal_conformal, efficiency_normal_conformal, \
    coverage_abs_stratified, efficiency_abs_stratified, coverage_quantile_stratified, efficiency_quantile_stratified, \
    coverage_abs_stratified_null, efficiency_abs_stratified_null, coverage_quantile_stratified_null, efficiency_quantile_stratified_null, \
    coverage_abs_bonferroni, efficiency_abs_bonferroni, coverage_quantile_bonferroni, efficiency_quantile_bonferroni]

with open('anaheim.pkl', 'wb') as f:
    pickle.dump(results, f)

methods = [
    "CIA (Split)",
    "CIA (CQR)",
    "CIA (Split) Stratified",
    "CIA (CQR) Stratified",
    "Group (Split)",
    "Group (CQR)",
    "Normal (Hetero)",
    "Normal (Homo)",
    "Bonferroni (Split)",
    "Bonferroni (CQR)"
]

coverage_data = [
    coverage_abs_nonstratified,
    coverage_quantile_nonstratified,
    coverage_abs_stratified,
    coverage_quantile_stratified,
    coverage_abs_stratified_null,
    coverage_quantile_stratified_null,
    coverage_normal_conformal,
    coverage_normal_nonconformal,
    coverage_abs_bonferroni,
    coverage_quantile_bonferroni
]

efficiency_data = [
    efficiency_abs_nonstratified,
    efficiency_quantile_nonstratified,
    efficiency_abs_stratified,
    efficiency_quantile_stratified,
    efficiency_abs_stratified_null,
    efficiency_quantile_stratified_null,
    efficiency_normal_conformal,
    efficiency_normal_nonconformal,
    efficiency_abs_bonferroni,
    efficiency_quantile_bonferroni
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
for i in range(len(methods)):
    ax1.errorbar(alpha_values, coverage_data[i].mean(axis=0), yerr=coverage_data[i].std(axis=0),
                 label=methods[i], linewidth=2,
                 markersize=8, marker='o')
    ax2.plot(coverage_data[i].mean(axis=0), efficiency_data[i].mean(axis=0), linewidth=2,
             markersize=8, marker='o')

ax1.plot(alpha_values, 1 - alpha_values, '--', color='red', linewidth=2)
ax1.set_xlabel(r'$ \alpha $', fontsize=22)
ax1.set_ylabel('Coverage', fontsize=22)
ax1.tick_params(axis='both', labelsize=14)
ax2.set_xlabel('Coverage', fontsize=22)
ax2.set_ylabel('Size', fontsize=22)
ax2.tick_params(axis='both', labelsize=14)

fig.suptitle(r'Anaheim ($|\mathcal{I}_{\text{train}}|:|\mathcal{I}_{\text{cal}}|:|\mathcal{I}_{\text{test}}|=5:2:2$)', fontsize=30)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), loc='lower center', ncol=3, fontsize=20)

plt.tight_layout(rect=[0, 0.22, 1, 0.99])  # Adjust the spacing to accommodate the legend
plt.show()

'''
main text plot
'''

methods = [
    "CIA",
    "Group",
    "Normal",
    "Bonferroni"
]

coverage_data = [
    coverage_quantile_stratified,
    coverage_quantile_stratified_null,
    coverage_normal_nonconformal,
    coverage_quantile_bonferroni
]

efficiency_data = [
    efficiency_quantile_stratified,
    efficiency_quantile_stratified_null,
    efficiency_normal_nonconformal,
    efficiency_quantile_bonferroni
]

colors = ['red', 'blue', 'green', 'orange']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
for i in range(len(methods)):
    ax1.errorbar(alpha_values, coverage_data[i].mean(axis=0), yerr=coverage_data[i].std(axis=0),
                 label=methods[i], linewidth=2, color=colors[i],
                 markersize=8, marker='o')
    ax2.plot(coverage_data[i].mean(axis=0),
             efficiency_data[i].mean(axis=0),
             linewidth=2, color=colors[i], markersize=8, marker='o')

ax1.plot(alpha_values, 1 - alpha_values, '--', color='black', linewidth=2)
ax1.set_xlabel(r'$ \alpha $', fontsize=24)
ax1.set_ylabel('Coverage', fontsize=24)
ax1.tick_params(axis='both', labelsize=18)
ax2.set_xlabel('Coverage', fontsize=24)
ax2.set_ylabel('Size', fontsize=24)
ax2.tick_params(axis='both', labelsize=18)

fig.suptitle("Anaheim", fontsize=30)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), loc='lower center', ncol=4, fontsize=20)

plt.tight_layout(rect=[0, 0.1, 1, 0.99])  # Adjust the spacing to accommodate the legend
plt.show()
