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
hidden_channels = 15
out_channels = 64
pads = GCN(pads_in_channels,hidden_channels,out_channels)
vtols = GCN(vtols_in_channels,hidden_channels,out_channels)

#Trying out forward propogation and basic encoding

vtol_basic_connect = np.matrix([
    [0,1,1,1,1],
    [1,0,1,1,1]])
pads_basic_connect = np.matrix([
    [0,1,1],
    [1,0,1]])

tvtol_edge = torch.tensor(vtol_edge,dtype=torch.long)
tpad_edge = torch.tensor(pad_edge,dtype=torch.long)

tvtol_features = torch.tensor(vtol_features,dtype=torch.float)
tpad_features = torch.tensor(pad_features,dtype=torch.float)

tvtol_basic_connect = torch.tensor(vtol_basic_connect,dtype=torch.long)
tpads_basic_connect = torch.tensor(pads_basic_connect,dtype=torch.long)

pad_x = pads.forward(tpad_features,tpads_basic_connect)
vtol_x = vtols.forward(tvtol_features,tvtol_basic_connect)

print('\npad feature vector: \n',pad_x)
print('\nvtol feature vector: \n',vtol_x)

#MLP test
final_features = torch.cat((pad_x.flatten(),vtol_x.flatten()))
print('\n final feature vector:\n',final_features)
input_dim = len(final_features)
output_dim = 7
action_space = MLP(input_dim,output_dim)
ypred = action_space.forward(final_features)

print('\nypred is:\n',ypred)