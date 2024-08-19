import torch
from torch_geometric.nn import GraphConv, SAGEConv, GCNConv, GATConv
import torch.nn.functional as F

class DirectedGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gconv=SAGEConv):
        super().__init__()
        self.layers = [in_channels, hidden_channels, out_channels]
        self.num_layers = len(self.layers) - 1
        self.source = torch.nn.ModuleList()
        self.target = torch.nn.ModuleList()
        for n_in, n_out in zip(self.layers[:-1], self.layers[1:]):
            self.source.append(gconv(n_in, n_out))
            self.target.append(gconv(n_in, n_out))

    def forward(self, s, t, edge_index, edge_weight):
        edge_weight = (edge_weight).sigmoid()
        for layer_id, (layer_s, layer_t) in enumerate(zip(self.source, self.target)):
            s_new = layer_s(t, edge_index, edge_weight)
            t_new = layer_t(s, torch.flip(edge_index, [0]), edge_weight)
            if layer_id < self.num_layers - 1:
                s_new = s_new.relu()
                t_new = t_new.relu()
                s_new = F.dropout(s_new, p=0.5, training=self.training)
                t_new = F.dropout(t_new, p=0.5, training=self.training)
            s = s_new
            t = t_new

        return s, t

def train_digae(model, x, edge_index, edge_weight, optimizer=None, val=False, edge_index_val=None, edge_weight_val=None):
    if val:
        model.eval()
    else:
        model.train()
    Z_source, Z_target = model(x, x, edge_index, edge_weight)
    if val:
        predicted_weights = model.decoder(Z_source, Z_target, edge_index_val)
    else:
        predicted_weights = model.decoder(Z_source, Z_target, edge_index)

    label = edge_weight_val if val else edge_weight

    criterion = torch.nn.MSELoss()
    loss = criterion(predicted_weights, label)

    if not val:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, float(loss)

def test_digae(best_model, x, edge_index_train, edge_weight_train, edge_index_test, edge_weight_test=None):
    best_model.eval()
    Z_source, Z_target = best_model(x, x, edge_index_train, edge_weight_train)
    predicted_weights = best_model.decoder(Z_source, Z_target, edge_index_test)
    if edge_weight_test == None:
        return predicted_weights.squeeze()

    true_weights = edge_weight_test

    # Calculate mean squared error (MSE)
    mse = torch.nn.functional.mse_loss(predicted_weights, true_weights)

    # Calculate mean absolute error (MAE)
    # mae = torch.nn.functional.l1_loss(predicted_weights, true_weights)
    return mse


# def train_digae_quantile_regression(model, x, edge_index, edge_weight, alpha=0.1, optimizer=None, val=False, edge_index_val=None, edge_weight_val=None, sigmoid=False):
#     if val:
#         model.eval()
#     else:
#         model.train()
#     Z_source, Z_target = model(x, x, edge_index, edge_weight)
#     out_dim = Z_source.shape[-1] // 2
#     z_lower_source = Z_source[:, :out_dim]
#     z_upper_source = Z_source[:, out_dim:]
#     z_lower_target = Z_target[:, :out_dim]
#     z_upper_target = Z_target[:, out_dim:]
#     if val:
#         lower = model.decoder(z_lower_source, z_lower_target, edge_index_val)
#         upper = model.decoder(z_upper_source, z_upper_target, edge_index_val)
#     else:
#         lower = model.decoder(z_lower_source, z_lower_target, edge_index)
#         upper = model.decoder(z_upper_source, z_upper_target, edge_index)
#
#     label = edge_weight_val if val else edge_weight
#     low_bound = alpha / 2; upp_bound = 1 - alpha / 2
#     low_loss = torch.mean(torch.max((low_bound - 1) * (label - lower), low_bound * (label - lower)))
#     upp_loss = torch.mean(torch.max((upp_bound - 1) * (label - upper), upp_bound * (label - upper)))
#     loss = low_loss + upp_loss
#
#     if not val:
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     return model, float(loss)
#
#
# def test_digae_quantile_regression(best_model, x, edge_index_train, edge_weight_train, edge_index_test, edge_weight_test=None, alpha=0.1, sigmoid=False):
#     best_model.eval()
#     Z_source, Z_target = best_model(x, x, edge_index_train, edge_weight_train)
#     out_dim = Z_source.shape[-1] // 2
#     z_lower_source = Z_source[:, :out_dim]
#     z_upper_source = Z_source[:, out_dim:]
#     z_lower_target = Z_target[:, :out_dim]
#     z_upper_target = Z_target[:, out_dim:]
#     lower = best_model.decoder(z_lower_source, z_lower_target, edge_index_test)
#     upper = best_model.decoder(z_upper_source, z_upper_target, edge_index_test)
#     if edge_weight_test == None:
#         return lower.squeeze(), upper.squeeze()
#
#     true_weights = edge_weight_test
#
#     # Calculate quantile losses
#     low_bound = alpha / 2; upp_bound = 1 - alpha / 2
#     low_loss = torch.mean(torch.max((low_bound - 1) * (true_weights - lower), low_bound * (true_weights - lower)))
#     upp_loss = torch.mean(torch.max((upp_bound - 1) * (true_weights - upper), upp_bound * (true_weights - upper)))
#     quantile_loss = low_loss + upp_loss
#
#     return float(quantile_loss)

def train_digae_quantile_regression(model, x, edge_index, edge_weight, quantiles, optimizer=None, val=False,
                                    edge_index_val=None, edge_weight_val=None):
    if val:
        model.eval()
    else:
        model.train()

    Z_source, Z_target = model(x, x, edge_index, edge_weight)
    out_dim = Z_source.shape[-1] // len(quantiles)

    if val:
        edge_index_used = edge_index_val
        edge_weight_used = edge_weight_val
    else:
        edge_index_used = edge_index
        edge_weight_used = edge_weight

    quantile_losses = []
    for i, q in enumerate(quantiles):
        z_source_q = Z_source[:, i * out_dim:(i + 1) * out_dim]
        z_target_q = Z_target[:, i * out_dim:(i + 1) * out_dim]

        pred_q = model.decoder(z_source_q, z_target_q, edge_index_used)

        q_loss = torch.mean(torch.max((q - 1) * (edge_weight_used - pred_q), q * (edge_weight_used - pred_q)))
        quantile_losses.append(q_loss)

    loss = sum(quantile_losses)

    if not val:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, float(loss)


def test_digae_quantile_regression(best_model, x, edge_index_train, edge_weight_train, edge_index_test, quantiles,
                                   edge_weight_test=None):
    best_model.eval()
    Z_source, Z_target = best_model(x, x, edge_index_train, edge_weight_train)
    out_dim = Z_source.shape[-1] // len(quantiles)

    quantile_preds = []
    for i, q in enumerate(quantiles):
        z_source_q = Z_source[:, i * out_dim:(i + 1) * out_dim]
        z_target_q = Z_target[:, i * out_dim:(i + 1) * out_dim]

        pred_q = best_model.decoder(z_source_q, z_target_q, edge_index_test)
        quantile_preds.append(pred_q)

    if edge_weight_test is None:
        return quantile_preds

    quantile_losses = []
    for i, q in enumerate(quantiles):
        pred_q = quantile_preds[i]
        q_loss = torch.mean(torch.max((q - 1) * (edge_weight_test - pred_q), q * (edge_weight_test - pred_q)))
        quantile_losses.append(float(q_loss))

    return quantile_losses