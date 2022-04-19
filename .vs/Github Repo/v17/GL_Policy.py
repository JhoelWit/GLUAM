from torch import nn,Tensor
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv, global_mean_pool,BatchNorm



class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.Leaky_ReLU = nn.LeakyReLU(negative_slope=0.1)
        self.BatchNorm = BatchNorm(in_channels,track_running_stats=True)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edge]
        x = self.BatchNorm(x)
        x = self.conv1(x, edge_index)
        x = self.Leaky_ReLU(x)
        x = self.conv2(x, edge_index)
        # print('x without pooling',x)

        #Pooling for the graph embedding

        x = global_mean_pool(x,batch=None)

        # print('\n x with pooling',x)
        return x

class MLP(nn.Module):
    def __init__(self,input_dim, output_dim) -> None:
        super().__init__()

        hidden1 = 128
        hidden2 = 64
        self.input = nn.Linear(input_dim,hidden1)
        self.hidden = nn.Linear(hidden1,hidden2)
        self.output = nn.Linear(hidden2,output_dim)
        self.Leaky_ReLU = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):

        #x should be (input_dim,-1)
        # batch_size = x.shape[0]
        # x = x.view(batch_size,-1)
 
        #Forward propogation
        h1 = self.input(x)
        l1 = self.Leaky_ReLU(h1)

        h2 = self.hidden(l1)
        l2 = self.Leaky_ReLU(h2)

        ypred = self.output(l2).tanh()

        return ypred




#Below is Steve's code, using it as a reference to create the policy. 

# from torch import nn
# import torch
# class GCAPCNFeatureExtractorNTDA(nn.Module):

#     def __init__(self,
#                  n_layers=2,
#                  n_dim=128,
#                  n_p=1,
#                  node_dim=2,
#                  n_K=2
#                  ):
#         super(GCAPCNFeatureExtractorNTDA, self).__init__()
#         self.n_layers = n_layers
#         self.n_dim = n_dim
#         self.n_p = n_p
#         self.n_K = n_K
#         self.node_dim = node_dim
#         self.init_embed = nn.Linear(node_dim, n_dim * n_p)
#         self.init_embed_depot = nn.Linear(2, n_dim)

#         self.W_L_1_G1 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)

#         self.normalization_1 = nn.BatchNorm1d(n_dim * n_p)

#         self.W_F = nn.Linear(n_dim * n_p, n_dim)

#         self.activ = nn.Tanh()

#     def forward(self, data, mask=None):

#         X = data['location'][:,1:,:]
#         # X = torch.cat((data['loc'], data['deadline']), -1)
#         X_loc = data['location'][:,1:,:]
#         distance_matrix = ((((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5)
#         num_samples, num_locations, _ = X.size()
#         A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
#             (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
#         A[A != A] = 0
#         D = torch.mul(torch.eye(num_locations, device=distance_matrix.device).expand((num_samples, num_locations, num_locations)),
#                       (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))

#         # Layer 1

#         # p = 3
#         F0 = self.init_embed(X)

#         # K = 3
#         L = D - A
#         L_squared = torch.matmul(L, L)
#         # L_cube = torch.matmul(L, L_squared)

#         g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
#                                           torch.matmul(L, F0)[:, :, :],
#                                           torch.matmul(L_squared, F0)[:, :, :]
#                                           ),
#                                          -1))


#         F1 = g_L1_1#torch.cat((g_L1_1), -1)
#         F1 = self.activ(F1) #+ F0
#         # F1 = self.normalization_1(F1)

#         F_final = self.activ(self.W_F(F1))

#         # init_depot_embed = self.init_embed_depot(data['depot'])[:]
#         h = F_final#torch.cat((init_depot_embed, F_final), 1)
#         return (
#             h,  # (batch_size, graph_size, embed_dim)
#             h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
#         )

# data = {
#     'location': self.locations,
#     'depot': self.depot.reshape(1, 2),
#     'mask': self.nodes_visited,
#     'agents_destination_coordinates': self.agents_destination_coordinates,
#     'agent_taking_decision_coordinates': self.agents_destination_coordinates[
#                                          self.agent_taking_decision, :].reshape(1,
#                                                                                 2),
#     'topo_laplacian':self.topo_laplacian,
#     'first_dec': self.first_dec
# }