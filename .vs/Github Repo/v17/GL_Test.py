import numpy as np
import torch
from GL_Policy import GCN,MLP

#State Variable Initialization
vtol_edge = np.matrix([
    [0,1,1,1,1],
    [1,0,1,1,1],
    [1,1,0,1,1],
    [1,1,1,0,1],
    [1,1,1,1,0]])

pad_edge = np.matrix([
    [0,1,1],
    [1,0,1],
    [1,1,0]])


vtol_features = np.matrix([
    [0.3,0,3,4,5], #[Battery capacity, Current mode, X,Y,Z]
    [1,2,0,0,0],
    [0.9,0,7,9,5],
    [0.5,2,1,1,0],
    [0.7,3,3,3,5]])

pad_features = np.matrix([
    [35,4,1,0], #[Time interval, eVTOL number, availability, type of node]
    [40,1,1,1],
    [20,2,0,2]])



#GCN Initialization
pads_in_channels = pad_features.shape[1]
vtols_in_channels = vtol_features.shape[1]
print('pads have',pads_in_channels,'inputs')
print('vtols have',vtols_in_channels,'inputs')
hidden_channels = 150
out_channels = 64
pads = GCN(pads_in_channels,hidden_channels,out_channels)
vtols = GCN(vtols_in_channels,hidden_channels,out_channels)


test_pad_edge = pads.create_edge_connect(num_nodes=3)
adj_test_pad_edge = pads.create_edge_connect(adj_mat=pad_edge,direct_connect=1)
test_vtol_edge = vtols.create_edge_connect(num_nodes=5)
adj_test_vtol_edge = vtols.create_edge_connect(adj_mat=vtol_edge,direct_connect=1)
print('Test_pad_edge\n',test_pad_edge,'\nadj mat Test_pad_edge\n',adj_test_pad_edge,'\nTest_vtol_edge\n',test_vtol_edge,'\nadj_Test_vtol_edge\n',adj_test_vtol_edge)


#Trying out forward propogation and basic encoding

vtol_edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
                                [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]],
                                dtype=torch.long)
pad_edge_index = torch.tensor([[0, 0, 1, 1, 2, 2],
                               [1, 2, 0, 2, 0, 1]],
                               dtype=torch.long)


# tvtol_edge = torch.tensor(vtol_edge,dtype=torch.long)
# tpad_edge = torch.tensor(pad_edge,dtype=torch.long)

tvtol_features = torch.tensor(vtol_features,dtype=torch.float)
tpad_features = torch.tensor(pad_features,dtype=torch.float)

pad_x = pads.forward(tpad_features,pad_edge_index)
vtol_x = vtols.forward(tvtol_features,vtol_edge_index)

print('\npad feature vector: \n',pad_x)
print('\nvtol feature vector: \n',vtol_x)

#MLP test
ypred_vtol = [[] for vtol in range(vtols_in_channels)]
num_actions = 7
final_features = torch.cat((pad_x.flatten(),vtol_x.flatten()))
print('\n final feature vector:\n',final_features,'\n length is',len(final_features))
input_dim = len(final_features)
output_dim = num_actions*vtols_in_channels
with torch.no_grad():
    action_space = MLP(input_dim,output_dim)
    ypred = action_space.forward(final_features).numpy().reshape((vtols_in_channels,num_actions))
    print('\nypred is:\n',ypred)
    # ypred = torch.split(ypred,num_actions)

# for i in range(vtols_in_channels):
#     ypred_vtol[i].append(ypred[i])

actions_to_take = np.argmax(ypred,axis=1)
for i in range(vtols_in_channels):
    print('\nVtol',i,'will take action:',actions_to_take[i])

# With Learnable Parameters
m = torch.nn.BatchNorm1d(100)
# Without Learnable Parameters
m = torch.nn.BatchNorm1d(100, affine=False)
input = torch.randn(20, 100)
output = m(input)