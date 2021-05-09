import torch
import  torch.nn  as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, NNConv, FeaStConv
from cgat import CGAT
import numpy as np



class GeometricEncoder(torch.nn.Module):
    def __init__(self, hidden_size,nheads=8):
        super(GeometricEncoder, self).__init__()
        num_edge_feature = 4
        num_atom_feature = 33
        nf = hidden_size
 
        GAT = GATConv
        self.conv1 = CGAT(num_atom_feature, int(nf/nheads), heads=nheads)
        self.lin1 = torch.nn.Linear(num_atom_feature, nf)
        self.conv2 = GAT(nf,   int(nf/nheads), heads=nheads)
        self.lin2 = torch.nn.Linear(nf, nf)
        self.conv3 = GAT(nf,   int(nf/nheads),  heads=nheads)
        self.lin3 = torch.nn.Linear(nf, nf)
        n_out = hidden_size
        self.conv4 = GAT(nf,  int(n_out/nheads),  heads=nheads)
        self.lin4 = torch.nn.Linear(nf, n_out)

        self.regressor =torch.nn.Linear(2*n_out, 3)
        self.predictor =torch.nn.Linear(4*n_out, 1)

    def step(self, X, E, Ea):
        #x = X[:,:-1]
        x_init = X[:,:-1]
        resid_max = torch.max(x_init[:,27]).item()/6.283
        sinx = torch.sin(x_init[:,27:28]/resid_max)
        cosx = torch.cos(x_init[:,27:28]/resid_max)
        x = torch.cat([x_init[:,:27],sinx, cosx,x_init[:,28:]],1)
        edge_index,edge_attr = E.t(), Ea.view(-1)

        x = F.elu(self.conv1(x,   edge_index,edge_attr) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x3 = F.elu(self.conv3(x, edge_index) + self.lin3(x))
        x4 = F.elu(self.conv4(x3, edge_index) + self.lin4(x3)) # K, d

        return x3,x4

    def gen_features(self, X, E, Ea, X_m, E_m, Ea_m):

        x3,x4 = self.step(X,E,Ea)
        x3_m,x4_m = self.step(X_m,E_m,Ea_m)

        idx = torch.nonzero((X[:,-1]==1).float()).view(-1)
        idxm = torch.nonzero((X_m[:,-1]==1).float()).view(-1)
        maxx4 = torch.max(x4[idx,:],0)[0]
        maxx4m = torch.max(x4_m[idxm,:],0)[0]
        meanx4 = torch.sum(x4[idx,:],0)
        meanx4m = torch.sum(x4_m[idxm,:],0)

        maxx3 = torch.max(x3[idx,:],0)[0]
        maxx3m = torch.max(x3_m[idxm,:],0)[0]
        meanx3 = torch.mean(x3[idx,:],0)
        meanx3m = torch.mean(x3_m[idxm,:],0)

        id_contact = torch.nonzero((X[:,24]==1).float()).view(-1)
        idm_contact = torch.nonzero((X_m[:,24]==1).float()).view(-1)

        maxx4_contact = torch.max(x4[id_contact,:],0)[0]
        maxx4m_contact = torch.max(x4_m[idm_contact,:],0)[0]
        meanx4_contact = torch.sum(x4[id_contact,:],0)
        meanx4m_contact = torch.sum(x4_m[idm_contact,:],0)


        maxx3_contact = torch.max(x3[id_contact,:],0)[0]
        maxx3m_contact = torch.max(x3_m[idm_contact,:],0)[0]
        meanx3_contact = torch.mean(x3[id_contact,:],0)
        meanx3m_contact = torch.mean(x3_m[idm_contact,:],0)


        rep1 = [maxx4, maxx4m, meanx4, meanx4m, maxx4_contact, maxx4m_contact, meanx4_contact,meanx4m_contact,\
                maxx3, maxx3m, meanx3, meanx3m, maxx3_contact, maxx3m_contact, meanx3_contact,meanx3m_contact]
        
        rep = torch.cat(rep1+[maxx4-maxx4m, meanx4-meanx4m],0) # (2d
        return rep


def GeoPPIpredict(A, E, A_m, E_m, model, forest, sorted_idx,flag):

    with torch.no_grad():
        fea = model.gen_features(A, E, E, A_m, E_m, E_m)

    features = np.round(fea.cpu().view(1,-1).numpy(),3)
    ddg = forest.predict(features[:,sorted_idx[:240]])
    ddg = np.round(ddg[0],2)
    if ddg>8.0:
        ddg =8.0
    elif ddg<-8.0:
        ddg = -8.0
    # Note that our model is able to predict a small value to the case of "no mutaiton" (e.g., TI17T). To further calibrate the prediction, we set the output of this case to zero.
    if flag: ddg=0.0
 
    return ddg


