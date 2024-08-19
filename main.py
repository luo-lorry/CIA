import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from common.dataset import build_reg_data
from common.utils import build_regression_model

from quantile_forest import RandomForestQuantileRegressor
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle

# Constants
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def group_by_dimensions(x, y, yhat, yhat_upper, yhat_lower, q1, q3, dimensions, num_iterations, alpha_values):
    """
    Perform grouping by dimensions and compute coverage and efficiency metrics.

    Parameters:
    x (numpy array): Input data
    y (numpy array): Target values
    yhat (numpy array): Predicted values
    yhat_upper (numpy array): Upper bounds of prediction intervals
    yhat_lower (numpy array): Lower bounds of prediction intervals
    q1 (numpy array): First quartiles of predicted values
    q3 (numpy array): Third quartiles of predicted values
    dimensions (tuple): Dimensions to group by
    num_iterations (int): Number of iterations for bootstrapping
    alpha_values (numpy array): Alpha values for confidence intervals

    Returns:
    tuple: Coverage and efficiency metrics for different methods
    """
    # Convert dimensions to tuple if it's not already
    if not isinstance(dimensions, tuple):
        dimensions = tuple(dimensions)

    # Check if dimensions are within the range of x's dimensions
    if any(dim < 0 or dim >= x.shape[1] for dim in dimensions):
        raise ValueError("Invalid dimensions specified.")

    # Get the unique value combinations along the specified dimensions
    unique_combinations = np.unique(x[:, dimensions], axis=0)

    # Create the binary matrix
    binary_matrix = np.zeros((unique_combinations.shape[0], x.shape[0]), dtype=int)

    # Fill the binary matrix based on the unique value combinations
    for i, combination in enumerate(unique_combinations):
        mask = np.all(x[:, dimensions] == combination, axis=1)
        binary_matrix[i, mask] = 1

    # Sanity check: column sum of binary matrix should be all ones
    column_sum = np.sum(binary_matrix, axis=0)
    if not np.all(column_sum == 1):
        raise ValueError("Binary matrix column sum is not all ones.")

    # Print the height of the binary matrix (number of unique combinations)
    print(f"Number of unique combinations: {binary_matrix.shape[0]}")

    output_array = binary_matrix.T

    result_array_abs = (yhat - y)[:, np.newaxis] * output_array
    result_array_quantile_1 = (y - yhat_upper)[:, np.newaxis] * output_array
    result_array_quantile_2 = (yhat_lower - y)[:, np.newaxis] * output_array
    result_array_quantile = (yhat_upper - yhat_lower)[:, np.newaxis] * output_array

    # Initialize arrays to store coverage and efficiency metrics
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

    coverage_abs_stratified_null = np.zeros((num_iterations, len(alpha_values)))
    coverage_quantile_stratified_null = np.zeros((num_iterations, len(alpha_values)))
    efficiency_abs_stratified_null = np.zeros((num_iterations, len(alpha_values)))
    efficiency_quantile_stratified_null = np.zeros((num_iterations, len(alpha_values)))

    coverage_abs_bonferroni = np.zeros((num_iterations, len(alpha_values)))
    coverage_quantile_bonferroni = np.zeros((num_iterations, len(alpha_values)))
    efficiency_abs_bonferroni = np.zeros((num_iterations, len(alpha_values)))
    efficiency_quantile_bonferroni = np.zeros((num_iterations, len(alpha_values)))

    for i in range(num_iterations):
        # Randomly partition the rows into two parts
        row_indices = np.arange(len(y))
        np.random.shuffle(row_indices)
        partition_1_abs = result_array_abs[row_indices[:len(y) // 2], :]
        partition_2_abs = result_array_abs[row_indices[len(y) // 2:], :]
        partition_1_quantile_1 = result_array_quantile_1[row_indices[:len(y) // 2], :]
        partition_2_quantile_1 = result_array_quantile_1[row_indices[len(y) // 2:], :]
        partition_1_quantile_2 = result_array_quantile_2[row_indices[:len(y) // 2], :]
        partition_2_quantile_2 = result_array_quantile_2[row_indices[len(y) // 2:], :]
        # Compute the column sums for each partition
        column_sums_1_abs = np.abs(np.sum(partition_1_abs, axis=0))
        column_sums_2_abs = np.abs(np.sum(partition_2_abs, axis=0))
        column_sums_1_quantile = np.maximum(np.sum(partition_1_quantile_1, axis=0),
                                            np.sum(partition_1_quantile_2, axis=0))
        column_sums_2_quantile = np.maximum(np.sum(partition_2_quantile_1, axis=0),
                                            np.sum(partition_2_quantile_2, axis=0))

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
        nonstratified CIA
        '''
        for alpha_i, alpha in enumerate(alpha_values):
            # Find the percentile values (nonstratified)
            percentile_threshold_nonstratified = min(100, (100 - 100 * alpha) * (len(column_sums_1_abs) + 1) / len(
                column_sums_1_abs))
            percentile_value_abs_nonstratified = np.percentile(column_sums_1_abs, percentile_threshold_nonstratified)
            percentile_value_quantile_nonstratified = np.percentile(column_sums_1_quantile,
                                                                    percentile_threshold_nonstratified)

            # Count the number of entries in the second vector below the percentile value (nonstratified)
            count_abs_nonstratified = np.sum(column_sums_2_abs <= percentile_value_abs_nonstratified)
            count_quantile_nonstratified = np.sum(column_sums_2_quantile <= percentile_value_quantile_nonstratified)

            # Record coverage and efficiency (nonstratified)
            coverage_abs_nonstratified[i, alpha_i] = count_abs_nonstratified / len(column_sums_2_abs)
            coverage_quantile_nonstratified[i, alpha_i] = count_quantile_nonstratified / len(column_sums_2_quantile)
            efficiency_abs_nonstratified[i, alpha_i] = 2 * percentile_value_abs_nonstratified
            efficiency_quantile_nonstratified[i, alpha_i] = result_array_quantile[row_indices[len(y) // 2:]][:,
                                                            test_extant].sum(
                axis=0).mean() + 2 * percentile_value_quantile_nonstratified

        '''
        stratified CIA
        '''
        # Stratified version
        column_sums_1 = np.sum(output_array[row_indices[:len(y) // 2], :], axis=0)[calib_extant]
        column_sums_2 = np.sum(output_array[row_indices[len(y) // 2:], :], axis=0)[test_extant]
        K = len(calib_extant) // 50  # Number of quantiles
        level_quantiles_1 = np.percentile(np.concatenate([column_sums_1, column_sums_2]), np.linspace(0, 100, K + 1))
        level_quantiles_1 = np.ceil(level_quantiles_1).astype(int)
        level_quantiles_1 = np.unique(level_quantiles_1)
        level_quantiles_1 = np.concatenate([np.array([0]), level_quantiles_1])

        for alpha_i, alpha in enumerate(alpha_values):
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
                                                 min(100, (100 - 100 * alpha) * (len(indices_1) + 1) / len(indices_1)))
                count = np.sum(partitioned_value_2_abs <= percentile_value)
                count_abs_stratified += count
                total_columns_abs += len(indices_2)
                efficiency_sum_abs += 2 * percentile_value * len(indices_2)

                partitioned_value_1_quantile = column_sums_1_quantile[indices_1]
                partitioned_value_2_quantile = column_sums_2_quantile[indices_2]
                percentile_value = np.percentile(partitioned_value_1_quantile,
                                                 min(100, (100 - 100 * alpha) * (len(indices_1) + 1) / len(indices_1)))
                count = np.sum(partitioned_value_2_quantile <= percentile_value)
                count_quantile_stratified += count
                total_columns_quantile += len(indices_2)
                efficiency_sum_quantile += result_array_quantile[row_indices[len(y) // 2:]][:,
                                           test_extant[indices_2]].sum(axis=0).mean() + 2 * percentile_value * len(
                    indices_2)

            coverage_abs_stratified[i, alpha_i] = count_abs_stratified / total_columns_abs
            coverage_quantile_stratified[i, alpha_i] = count_quantile_stratified / total_columns_abs
            efficiency_abs_stratified[i, alpha_i] = efficiency_sum_abs / total_columns_quantile
            efficiency_quantile_stratified[i, alpha_i] = efficiency_sum_quantile / total_columns_quantile

        '''
        Group sampling
        '''
        column_sums_1 = np.sum(output_array[row_indices[:len(y) // 2], :], axis=0)[calib_extant]
        column_sums_2 = np.sum(output_array[row_indices[len(y) // 2:], :], axis=0)[test_extant]

        for alpha_i, alpha in enumerate(alpha_values):

            num_samples = 100  # User-defined number of samples

            count_abs_stratified = 0
            count_quantile_stratified = 0
            total_columns_abs = 0
            total_columns_quantile = 0
            efficiency_sum_abs = 0
            efficiency_sum_quantile = 0

            for level in np.unique(column_sums_2):
                percentile_threshold_stratified = min(100, (100 - 100 * alpha) * (1 + num_samples) / num_samples)

                # Sampling without replacement k edges in the calibration set
                k = int(level)
                abs_errors = [np.abs((yhat - y)[row_indices[:len(y) // 2]][
                                         np.random.choice(np.arange(len(y) // 2), size=(k,), replace=False)].sum()) for
                              _ in range(num_samples)]
                samples = [np.random.choice(np.arange(len(y) // 2), size=(k,), replace=False) for _ in
                           range(num_samples)]
                quantile_errors = [np.maximum((yhat_lower - y)[row_indices[:len(y) // 2]][sample].sum(),
                                              (y - yhat_upper)[row_indices[:len(y) // 2]][sample].sum()) for sample in
                                   samples]
                percentile_value_abs_stratified = np.percentile(abs_errors, percentile_threshold_stratified)
                percentile_value_quantile_stratified = np.percentile(quantile_errors, percentile_threshold_stratified)

                column_indices_2 = np.where(column_sums_2 == level)[0]
                count_abs_stratified += np.sum(column_sums_2_abs[column_indices_2] <= percentile_value_abs_stratified)
                count_quantile_stratified += np.sum(
                    column_sums_2_quantile[column_indices_2] <= percentile_value_quantile_stratified)

                total_columns_abs += len(column_indices_2)
                total_columns_quantile += len(column_indices_2)
                efficiency_sum_abs += (
                        2 * percentile_value_abs_stratified * len(column_indices_2))
                efficiency_sum_quantile += (
                        ((yhat_upper - yhat_lower)[row_indices[len(y) // 2:]] @ output_array[
                            row_indices[len(y) // 2:]])[test_extant[column_indices_2]].sum()
                        + 2 * percentile_value_quantile_stratified * len(column_indices_2))

            coverage_abs_stratified_null[i, alpha_i] = count_abs_stratified / total_columns_abs
            coverage_quantile_stratified_null[i, alpha_i] = count_quantile_stratified / total_columns_quantile
            efficiency_abs_stratified_null[i, alpha_i] = efficiency_sum_abs / total_columns_abs
            efficiency_quantile_stratified_null[i, alpha_i] = efficiency_sum_quantile / total_columns_quantile

        '''
        Bonferroni
        '''
        for alpha_i, alpha in enumerate(alpha_values):
            score_abs = np.abs(yhat - y)[row_indices[:len(y) // 2]]
            score_quantile = (y - yhat_upper)[row_indices[:len(y) // 2]]
            percentile_threshold_bonferroni = np.minimum(100, (100 - 100 * alpha / np.maximum(1, column_sums_2)))
            upper_bonferroni_abs = (yhat[:, np.newaxis] * output_array)[row_indices[len(y) // 2:], :][:,
                                   test_extant].sum(axis=0) + column_sums_2 * np.percentile(score_abs,
                                                                                            percentile_threshold_bonferroni)  # 128 * 5000
            lower_bonferroni_abs = (yhat[:, np.newaxis] * output_array)[row_indices[len(y) // 2:], :][:,
                                   test_extant].sum(
                axis=0) - column_sums_2 * np.percentile(score_abs,
                                                        percentile_threshold_bonferroni)  # 128 * 5000

            upper_bonferroni_quantile = (yhat_upper[:, np.newaxis] * output_array)[row_indices[len(y) // 2:], :][:,
                                        test_extant].sum(axis=0) + column_sums_2 * np.percentile(score_quantile,
                                                                                                 percentile_threshold_bonferroni)
            lower_bonferroni_quantile = (yhat_lower[:, np.newaxis] * output_array)[row_indices[len(y) // 2:], :][:,
                                        test_extant].sum(axis=0) - column_sums_2 * np.percentile(score_quantile,
                                                                                                 percentile_threshold_bonferroni)
            coverage_abs_bonferroni[i, alpha_i] = np.mean(((y[row_indices[len(y) // 2:]] @ output_array[
                row_indices[len(y) // 2:]])[test_extant] >= lower_bonferroni_abs) & ((y[row_indices[len(y) // 2:]] @
                                                                                      output_array[
                                                                                          row_indices[len(y) // 2:]])[
                                                                                         test_extant] <= upper_bonferroni_abs))
            efficiency_abs_bonferroni[i, alpha_i] = np.mean(upper_bonferroni_abs - lower_bonferroni_abs)
            coverage_quantile_bonferroni[i, alpha_i] = np.mean(((y[row_indices[len(y) // 2:]] @ output_array[
                row_indices[len(y) // 2:]])[test_extant] >= lower_bonferroni_quantile) & (
                                                                           (y[row_indices[len(y) // 2:]] @
                                                                            output_array[row_indices[len(y) // 2:]])[
                                                                               test_extant] <= upper_bonferroni_quantile))
            efficiency_quantile_bonferroni[i, alpha_i] = np.mean(upper_bonferroni_quantile - lower_bonferroni_quantile)

        '''
        normal homogeneous approximation
        '''
        std = np.sqrt(1 / (len(y) // 2 - 1) * ((yhat - y) ** 2)[row_indices[:len(y) // 2]].sum())
        for alpha_i, alpha in enumerate(alpha_values):
            upper_bound_conformal, lower_bound_conformal = compute_normal_bounds(
                (yhat[:, np.newaxis] * output_array)[row_indices[len(y) // 2:]][:, test_extant].sum(axis=0),
                np.tile(std, (len(y) - len(y) // 2,)), binary_matrix[test_extant][:, row_indices[len(y) // 2:]], alpha)
            coverage_normal_conformal[i, alpha_i] = np.mean(((y[:, np.newaxis] * output_array)[
                                                                 row_indices[len(y) // 2:]][:, test_extant].sum(
                axis=0) >= lower_bound_conformal) & ((y[:, np.newaxis] * output_array)[row_indices[len(y) // 2:]][:,
                                                     test_extant].sum(axis=0) <= upper_bound_conformal))
            efficiency_normal_conformal[i, alpha_i] = np.mean(upper_bound_conformal - lower_bound_conformal)

        '''
        normal IQR approximation
        '''
        std = (q3 - q1) / (norm.ppf(0.75) - norm.ppf(0.25))
        for alpha_i, alpha in enumerate(alpha_values):
            upper_bound_iqr, lower_bound_iqr = compute_normal_bounds(
                (yhat[:, np.newaxis] * output_array)[row_indices[len(y) // 2:]][:, test_extant].sum(axis=0),
                std[len(y) // 2:], binary_matrix[test_extant][:, row_indices[len(y) // 2:]], alpha)
            coverage_normal_nonconformal[i, alpha_i] = np.mean(((y[row_indices[len(y) // 2:]] @ output_array[
                row_indices[len(y) // 2:]])[test_extant] >= lower_bound_iqr) & ((y[row_indices[len(y) // 2:]] @
                                                                                 output_array[
                                                                                     row_indices[len(y) // 2:]])[
                                                                                    test_extant] <= upper_bound_iqr))
            efficiency_normal_nonconformal[i, alpha_i] = np.mean(upper_bound_iqr - lower_bound_iqr)

    return (coverage_abs_nonstratified, efficiency_abs_nonstratified, coverage_quantile_nonstratified,
            efficiency_quantile_nonstratified,
            coverage_normal_nonconformal, efficiency_normal_nonconformal, coverage_normal_conformal,
            efficiency_normal_conformal,
            coverage_abs_stratified, efficiency_abs_stratified, coverage_quantile_stratified,
            efficiency_quantile_stratified,
            coverage_abs_stratified_null, efficiency_abs_stratified_null, coverage_quantile_stratified_null,
            efficiency_quantile_stratified_null,
            coverage_abs_bonferroni, efficiency_abs_bonferroni, coverage_quantile_bonferroni,
            efficiency_quantile_bonferroni)


def compute_normal_bounds(yhat_sums, std, binary_matrix, alpha):
    """
    Compute normal bounds for conformal prediction.

    Parameters:
    yhat_sums (numpy array): Sums of predicted values
    std (numpy array): Standard deviations
    binary_matrix (numpy array): Binary matrix
    alpha (float): Alpha value for confidence interval

    Returns:
    tuple: Upper and lower bounds
    """
    std = np.sqrt(binary_matrix @ (std ** 2))
    z_upper = norm.ppf(1 - alpha / 2)
    z_lower = norm.ppf(alpha / 2)
    upper_bound = yhat_sums + z_upper * std
    lower_bound = yhat_sums + z_lower * std
    return upper_bound, lower_bound


def train(model, device, train_data_loader, criterion, optimizer):
    """
    Train a model for one epoch.

    Parameters:
    model (nn.Module): Model to train
    device (torch.device): Device to use
    train_data_loader (DataLoader): Data loader for training data
    criterion (nn.Module): Loss function
    optimizer (nn.Module): Optimizer
    """
    for index, (tmp_x, tmp_y) in enumerate(train_data_loader):
        outputs = model(tmp_x.to(device))
        loss = criterion(outputs, tmp_y.unsqueeze(dim=1).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def run_split_predictor(X_train, train_data_loader, cal_test_data_loader, epochs=2000):
    """
    Run a split predictor model.

    Parameters:
    X_train (numpy array): Training data
    train_data_loader (DataLoader): Data loader for training data
    cal_test_data_loader (DataLoader): Data loader for calibration and test data
    epochs (int): Number of epochs to train

    Returns:
    tuple: Predicted values, actual values, and input data
    """
    model = build_regression_model("NonLinearNet")(X_train.shape[1], 1, 64, 0.5).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        train(model, DEVICE, train_data_loader, criterion, optimizer)

    model.eval()
    predicts_list = []
    y_truth_list = []
    x_list = []
    with torch.no_grad():
        for examples in cal_test_data_loader:
            tmp_x, tmp_labels = examples[0], examples[1]
            tmp_predicts = model(tmp_x).detach()
            predicts_list.append(tmp_predicts)
            y_truth_list.append(tmp_labels)
            x_list.append(tmp_x)
        predicts = torch.cat(predicts_list).float()
        y_truth = torch.cat(y_truth_list)
        xs = torch.cat(x_list)

    return xs.detach().numpy(), predicts.detach().numpy().flatten(), y_truth.detach().numpy().flatten()


def run_cqr(X_train, y_train, cal_test_data_loader, alpha):
    """
    Run a CQR model.

    Parameters:
    X_train (numpy array): Training data
    y_train (numpy array): Training target values
    cal_test_data_loader (DataLoader): Data loader for calibration and test data
    alpha (float): Alpha value for confidence interval

    Returns:
    tuple: Predicted upper and lower bounds, first and third quartiles, and actual values
    """
    quantiles = [alpha / 2, 1 - alpha / 2, 0.25, 0.75]
    model = RandomForestQuantileRegressor(n_estimators=500, max_depth=5, max_samples_leaf=None)
    model.fit(X_train, y_train)
    predicts_list = []
    y_truth_list = []
    x_list = []
    for examples in cal_test_data_loader:
        tmp_x, tmp_labels = examples[0].numpy(), examples[1].numpy()
        tmp_predicts = model.predict(tmp_x, quantiles=quantiles)
        predicts_list.append(tmp_predicts)
        y_truth_list.append(tmp_labels)
        x_list.append(tmp_x)
    predicts = np.concatenate(predicts_list)
    xs = np.concatenate(x_list)
    upper = predicts[..., 1]
    lower = predicts[..., 0]
    q3 = predicts[..., 3]
    q1 = predicts[..., 2]
    y_truth = np.concatenate(y_truth_list)
    return xs, upper, lower, q3, q1, y_truth


def run_experiments(data_names, num_runs, ratio_train_values, test_ratio, alpha_values, dimension_dict):
    """
    Run experiments for different datasets and methods.

    Parameters:
    data_names (list): List of dataset names
    num_runs (int): Number of runs for each experiment
    ratio_train_values (list): List of ratio train values
    test_ratio (float): Test ratio
    alpha_values (numpy array): Alpha values for confidence intervals
    dimension_dict (dict): Dictionary of dimensions for each dataset
    """
    for data_name in data_names:
        dimension = dimension_dict[data_name]
        for ratio_train in ratio_train_values:
            for run in range(num_runs):
                print(f"Running dataset: {data_name}, Ratio_Train: {ratio_train}, Run: {run + 1}/{num_runs}")
                seed = run + 1
                X_train, X_calib, X_test, y_train, y_calib, y_test = build_reg_data(data_name, ratio_train, test_ratio,
                                                                                    seed, normalize=True)
                X_cal_test = np.vstack([X_calib, X_test])
                y_cal_test = np.concatenate([y_calib, y_test])
                # Prepare datasets and data loaders
                train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
                cal_test_dataset = TensorDataset(torch.from_numpy(X_cal_test), torch.from_numpy(y_cal_test))
                train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=True,
                                                                pin_memory=True)
                cal_test_data_loader = torch.utils.data.DataLoader(cal_test_dataset, batch_size=1000, shuffle=False,
                                                                   pin_memory=True)

                # SplitPredictor
                xs, yhat, ytrue = run_split_predictor(X_train, train_data_loader, cal_test_data_loader)

                # CQR
                _, yhat_upper, yhat_lower, q3, q1, _ = run_cqr(X_train, y_train, cal_test_data_loader, 0.5)

                results = group_by_dimensions(xs, ytrue, yhat, yhat_upper, yhat_lower, q1, q3, dimension, 100,
                                              alpha_values)

                with open(f'CIA_results/pkl/{data_name}.pkl', 'wb') as f:
                    pickle.dump(results, f)


if __name__ == '__main__':
    # Usage
    data_names = ['meps_19', 'meps_20', 'meps_21', 'community', 'bike']
    dimension_dict = {'bike': (0, 2, 3), 'meps_19': (0, 9, 11), 'meps_20': (0, 9, 11), 'meps_21': (0, 9, 11),
                      'community': (0, 1)}
    num_runs = 1
    ratio_train_values = [0.8]
    test_ratio = 1 / 8
    alpha_values = np.linspace(0.01, 0.1,10)
    run_experiments(data_names, num_runs, ratio_train_values, test_ratio, alpha_values, dimension_dict)

    for data_name in data_names:
        with open(f'{data_name}.pkl', 'rb') as f:
            results = pickle.load(f)

        coverage_abs, efficiency_abs, coverage_quantile, efficiency_quantile, \
            coverage_normal_iqr, efficiency_normal_iqr, coverage_normal_conformal, efficiency_normal_conformal, \
            coverage_abs_stratified, efficiency_abs_stratified, coverage_quantile_stratified, efficiency_quantile_stratified, \
            coverage_abs_stratified_null, efficiency_abs_stratified_null, coverage_quantile_stratified_null, efficiency_quantile_stratified_null, \
            coverage_abs_bonferroni, efficiency_abs_bonferroni, coverage_quantile_bonferroni, efficiency_quantile_bonferroni = results

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
            coverage_abs,
            coverage_quantile,
            coverage_abs_stratified,
            coverage_quantile_stratified,
            coverage_abs_stratified_null,
            coverage_quantile_stratified_null,
            coverage_normal_iqr,
            coverage_normal_conformal,
            coverage_abs_bonferroni,
            coverage_quantile_bonferroni
        ]

        efficiency_data = [
            efficiency_abs,
            efficiency_quantile,
            efficiency_abs_stratified,
            efficiency_quantile_stratified,
            efficiency_abs_stratified_null,
            efficiency_quantile_stratified_null,
            efficiency_normal_iqr,
            efficiency_normal_conformal,
            efficiency_abs_bonferroni,
            efficiency_quantile_bonferroni
        ]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        for i in range(len(methods)):
            ax1.errorbar(alpha_values[:10], coverage_data[i].mean(axis=0)[:10], yerr=coverage_data[i].std(axis=0)[:10],
                         label=methods[i], linewidth=2,
                         markersize=8, marker='o')
            ax2.plot(coverage_data[i].mean(axis=0)[:10], efficiency_data[i].mean(axis=0)[:10], linewidth=2,
                     markersize=8, marker='o')

        ax1.plot(alpha_values[:10], 1 - alpha_values[:10], '--', color='red', linewidth=2)
        ax1.set_xlabel(r'$ \alpha $', fontsize=22)
        ax1.set_ylabel('Coverage', fontsize=22)
        ax1.tick_params(axis='both', labelsize=14)
        ax2.set_xlabel('Coverage', fontsize=22)
        ax2.set_ylabel('Size', fontsize=22)
        ax2.tick_params(axis='both', labelsize=14)

        fig.suptitle(data_name.replace('_', ''), fontsize=30)

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), loc='lower center', ncol=3, fontsize=20)

        plt.tight_layout(rect=[0, 0.22, 1, 0.99])  # Adjust the spacing to accommodate the legend
        plt.show()
        fig.savefig(f"CIA_results/appendix/{data_name}.pdf", dpi=300)

        methods = [
            "CIA",
            "Group",
            "Normal",
            "Bonferroni"
        ]

        coverage_data = [
            coverage_quantile_stratified,
            coverage_quantile_stratified_null,
            coverage_normal_iqr,
            coverage_quantile_bonferroni
        ]

        efficiency_data = [
            efficiency_quantile_stratified,
            efficiency_quantile_stratified_null,
            efficiency_normal_iqr,
            efficiency_quantile_bonferroni
        ]

        colors = ['red', 'blue', 'green', 'orange']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
        for i in range(len(methods)):
            ax1.errorbar(alpha_values[:10], coverage_data[i].mean(axis=0)[:10], yerr=coverage_data[i].std(axis=0)[:10],
                         label=methods[i], linewidth=2, color=colors[i],
                         markersize=8, marker='o')
            ax2.plot(coverage_data[i].mean(axis=0)[:10],
                     efficiency_data[i].mean(axis=0)[:10],
                     linewidth=2, color=colors[i], markersize=8, marker='o')

        ax1.plot(alpha_values[:10], 1 - alpha_values[:10], '--', color='black', linewidth=2)
        ax1.set_xlabel(r'$ \alpha $', fontsize=24)
        ax1.set_ylabel('Coverage', fontsize=24)
        ax1.tick_params(axis='both', labelsize=18)
        ax2.set_xlabel('Coverage', fontsize=24)
        ax2.set_ylabel('Size', fontsize=24)
        ax2.tick_params(axis='both', labelsize=18)

        fig.suptitle(data_name.replace('_', ''), fontsize=30)

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), loc='lower center', ncol=4, fontsize=20)

        plt.tight_layout(rect=[0, 0.1, 1, 0.99])  # Adjust the spacing to accommodate the legend
        plt.show()
        fig.savefig(f"CIA_results/main/{data_name}.pdf", dpi=300)
