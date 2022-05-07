import numpy as np
import torch
import gym
from gym import spaces
from GL_Policy import CustomGLPolicy
from stable_baselines3.common.type_aliases import Schedule





def create_edge_connect(num_nodes=5,adj_mat=None,direct_connect=0):
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
                    if adj_mat[i,j] == 1:
                        blank_edge = np.concatenate((blank_edge,np.array([i,j]).reshape(2,1)),axis=1)
        # return torch.tensor(blank_edge,dtype=torch.long)
        return blank_edge

action_space = spaces.Discrete(4) 

observation_space = spaces.Dict(
        dict(
            vertiport_features = spaces.Box(low=0.0,high=2.0,shape=(10,2), dtype=np.float32),
            vertiport_edge = spaces.Box(low=0.0,high=9.0,shape=(2,90), dtype=np.float32),
            evtol_features = spaces.Box(low=0.0,high=3.0,shape=(5,3), dtype=np.float32),
            evtol_edge = spaces.Box(low=0.0,high=4.0,shape=(2,20), dtype=np.float32),
            next_drone_embedding = spaces.Box(low=0,high=2,shape=(6,),dtype=np.float32)
        ))

vtol_features = np.matrix([
    [1,0,0], #[Battery capacity, Current mode, Collision]
    [1,0,0],
    [1,2,0],
    [1,0,0],
    [1,0,0]])

pad_features = np.matrix([
    [1,0], #[Availability, Port type]
    [0,1],
    [0,2],
    [0,2],
    [0,2],
    [0,2],
    [0,2],
    [0,2],
    [1,2],
    [1,2]])

next_drone = np.array([1,0,1,0,1,0])
tnext_drone = torch.tensor([1,0,1,0,1,0],dtype=torch.float)
tvtol_features = torch.tensor(vtol_features,dtype=torch.float)
tpad_features = torch.tensor(pad_features,dtype=torch.float)

test_policy = CustomGLPolicy(observation_space=observation_space,action_space=action_space)
test_obs = {'vertiport_features':pad_features,'vertiport_edge':create_edge_connect(num_nodes=10),
            'evtol_features':vtol_features,'evtol_edge':create_edge_connect(num_nodes=5),
            'next_drone_embedding':next_drone}
output = test_policy(test_obs)
print(output)














#State Variable Initialization
# vtol_edge = np.matrix([
#     [0,1,1,1,1],
#     [1,0,1,1,1],
#     [1,1,0,1,1],
#     [1,1,1,0,1],
#     [1,1,1,1,0]])

# pad_edge = np.matrix([
#     [0,1,1],
#     [1,0,1],
#     [1,1,0]])






# #GCN Initialization
# pads_in_channels = pad_features.shape[1]
# vtols_in_channels = vtol_features.shape[1]
# print('pads have',pads_in_channels,'inputs')
# print('vtols have',vtols_in_channels,'inputs')
# hidden_channels = 150
# out_channels = 64
# pads = GCN(pads_in_channels,hidden_channels,out_channels)
# vtols = GCN(vtols_in_channels,hidden_channels,out_channels)


# test_pad_edge = pads.create_edge_connect(num_nodes=3)
# adj_test_pad_edge = pads.create_edge_connect(adj_mat=pad_edge,direct_connect=1)
# test_vtol_edge = vtols.create_edge_connect(num_nodes=5)
# adj_test_vtol_edge = vtols.create_edge_connect(adj_mat=vtol_edge,direct_connect=1)
# print('Test_pad_edge\n',test_pad_edge,'\nadj mat Test_pad_edge\n',adj_test_pad_edge,'\nTest_vtol_edge\n',test_vtol_edge,'\nadj_Test_vtol_edge\n',adj_test_vtol_edge)


#Trying out forward propogation and basic encoding

# vtol_edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
#                                 [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]],
#                                 dtype=torch.long)
# pad_edge_index = torch.tensor([[0, 0, 1, 1, 2, 2],
#                                [1, 2, 0, 2, 0, 1]],
#                                dtype=torch.long)


# tvtol_edge = torch.tensor(vtol_edge,dtype=torch.long)
# tpad_edge = torch.tensor(pad_edge,dtype=torch.long)



# pad_x = pads(tpad_features,pad_edge_index)
# vtol_x = vtols(tvtol_features,vtol_edge_index)

# print('\npad feature vector: \n',pad_x)
# print('\nvtol feature vector: \n',vtol_x)

# #MLP test
# ypred_vtol = [[] for vtol in range(vtols_in_channels)]
# num_actions = 8
# mask = np.array([1, 1, 1, 0, 0, 0, 0, 0,1, 1, 1, 0, 0, 0, 0, 0,1, 1, 1, 0, 0, 0, 0, 0,1, 1, 1, 0, 0, 0, 0, 0,1, 1, 1, 0, 0, 0, 0, 0,]) 
# final_features = torch.cat((pad_x.flatten(),vtol_x.flatten()))
# print('\n final feature vector:\n',final_features,'\n length is',len(final_features))
# input_dim = len(final_features)
# output_dim = num_actions*vtols_in_channels
# with torch.no_grad():
#     action_space = MLP(input_dim,output_dim)
#     ypred = action_space.forward(final_features,mask=mask).numpy().reshape((vtols_in_channels,num_actions))
#     print('\nypred is:\n',ypred)
#     # ypred = torch.split(ypred,num_actions)

# # for i in range(vtols_in_channels):
# #     ypred_vtol[i].append(ypred[i])

# actions_to_take = np.argmax(ypred,axis=1)
# for i in range(vtols_in_channels):
#     print('\nVtol',i,'will take action:',actions_to_take[i])

# # With Learnable Parameters
# m = torch.nn.BatchNorm1d(100)
# # Without Learnable Parameters
# m = torch.nn.BatchNorm1d(100, affine=False)
# input = torch.randn(20, 100)
# output = m(input)