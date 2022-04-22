import string
import numpy as np
from torch import long, nn,Tensor,tensor
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv, global_mean_pool,BatchNorm



class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.Leaky_ReLU = nn.LeakyReLU(negative_slope=0.1)
        self.BatchNorm = BatchNorm(in_channels,track_running_stats=True)

    def create_edge_connect(self,num_nodes=5,adj_mat=None,direct_connect=0) -> Tensor:
        if direct_connect == 0: #undirected graph, every node is connected and shares info with each other
            k = num_nodes-1
            num_edges = k*num_nodes
            blank_edge = np.ones( (2 , num_edges) )
            top_index = 0
            bot_index = np.arange(num_nodes)
            index = 1
            for i in range(num_nodes):
                blank_edge[0][k*i:k*index] = top_index
                blank_edge[1][k*i:k*index] = np.delete(bot_index,top_index)
                index+=1
                top_index+=1
        elif direct_connect == 1: #directed graph, in which case we need the adjacency matrix to create the edge list tensor
            blank_edge = np.array([]).reshape(2,0)
            for i in range(adj_mat.shape[0]):
                for j in range(adj_mat.shape[1]):
                    print(adj_mat[i,j])
                    if adj_mat[i,j] == 1:
                        blank_edge = np.concatenate((blank_edge,np.array([i,j]).reshape(2,1)),axis=1)
        return tensor(blank_edge,dtype=long)

    def create_feature_mat(self,data:dict,name:string) -> Tensor:
        pass

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



