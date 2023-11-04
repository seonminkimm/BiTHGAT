import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

from .spatial_conv import SpatialConvOrderK
from .gcrnn import GCGRUCell
from .spatial_attention import SpatialAttention
from ..utils.ops import reverse_tensor

from sklearn.impute import SimpleImputer
import pandas as pd
from fancyimpute import IterativeImputer

import torch.nn.functional as F
import math

class THGAT(nn.Module):
    def __init__(self, d_in, d_model, agg_type, init_type, alpha=0.2, dropout=0.):
        super(THGAT, self).__init__()

        self.alpha = alpha
        self.agg_type = agg_type
        self.init_type = init_type

        # self.activation = nn.PReLU()
        self.incidence = None

        # self.conv = nn.Conv1d(d_in, d_model, kernel_size=1)
        self.bias = nn.Parameter(torch.rand(d_model))

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None


    def forward(self, x, m, pri_e, pri_n, incidence):
        
        x_position = x.shape[1]
        
        # [batch, channels, nodes]
        # TODO x_in (x+masking) 대신 x 사용해보기 
        x_in = [x, m, pri_n]
        # x_in = [x, m]
        
        x_in = torch.cat(x_in, 1)
        # x_in = self.conv(x_in)
        
        weight = nn.Parameter(torch.rand(x_in.shape[1], 1)).to('cuda')

        x_in = x_in.permute(0,2,1)
        x_in = x_in.matmul(weight)
        x_in = x_in.permute(0,2,1)
        
        if self.bias is not None:
            x_in = x_in + self.bias

        incidence = incidence.expand(x_in.shape[0], incidence.shape[0], incidence.shape[1])

        # x : (batch, edge, node)
        node_num = incidence.shape[1] 
        edge_num = incidence.shape[2] 
        
        pair = incidence.nonzero().t()    # hyper-edge 좌표의 transpose
        
        # print(x_in.shape)               # [32, 1, 36]
        # print(incidence.shape)          # torch.Size([32, 36, 7])
        # print(pair.shape)               # torch.Size([3, 5664])
        
        edge = torch.matmul(x_in, incidence) / torch.unsqueeze(torch.sum(incidence, dim=1), 1)
        
        weight2 = nn.Parameter(torch.rand(incidence.shape[2], incidence.shape[2])).to('cuda')
        edge = edge.matmul(weight2)
        
        edge = edge.permute(0,2,1)        # torch.Size([32, 7, 1])
        x_in = x_in.permute(0,2,1)        # torch.Size([32, 36, 1])
        
        get = lambda i: x_in[i][incidence[i].nonzero().t()[0]]
        q1 = torch.cat([get(i) for i in torch.arange(x_in.shape[0]).long()])

        get = lambda i: edge[i][incidence[i].nonzero().t()[1]]
        y1 = torch.cat([get(i) for i in torch.arange(x_in.shape[0]).long()])
        
        pair_h = torch.cat((q1, y1), dim=-1)
   
        a = nn.Parameter(torch.rand(size=(x_in.shape[2]+edge.shape[2], 1))).to('cuda')
        pair_e = self.leakyrelu(torch.matmul(pair_h, a).squeeze()).t()
        assert not torch.isnan(pair_e).any()
        # pair_e = F.dropout(pair_e, self.dropout, training=self.training)
        
        # print(incidence.shape)    # torch.Size([32, 36, 7])
        # print(x_in.shape)         # torch.Size([32, 1, 36])
        # print(pair.shape)         # torch.Size([3, 5664])
        # print(pair_h.shape)       # torch.Size([5664, 2])
        # print(edge.shape)         # torch.Size([32, 7, 1])
        # print(pri_e.shape)        # torch.Size([32, 7, 1])
        # print(pair_e.shape)       # torch.Size([5664])
        
        if pri_e.shape[2] != 1: pri_e = pri_e[:, :, 1:]
        
        
        # time_e : 이전 시점과 현재 시점의 hyper-edge attention
        if self.agg_type == 'concat':
            time_edge = torch.cat((edge, pri_e), dim=-1)                        # torch.Size([32, 7, N]) --> # torch.Size([32, 7, 1])
            
            a2 = nn.Parameter(torch.rand(size=(time_edge.shape[2], 1))).to('cuda')   
            time_e = self.leakyrelu(torch.matmul(time_edge, a2))
            assert not torch.isnan(time_e).any()            
            
        elif self.agg_type == 'sum':
            time_e = torch.add(pri_e, edge)
            
        elif self.agg_type == 'mul':
            time_e = torch.mul(pri_e, edge) 
      
        
        pair_e = pair_e.unsqueeze(0).t()                                        # torch.Size([5664, 1])
        
        get = lambda i: time_e[i][incidence[i].nonzero().t()[1]]
        y2 = torch.cat([get(i) for i in torch.arange(x_in.shape[0]).long()])    # torch.Size([5664, 1])
        
        if self.agg_type == 'concat':
            pair_e = torch.cat((y2, pair_e), dim=-1)
            
            a3 = nn.Parameter(torch.rand(size=(pair_e.shape[1], 1))).to('cuda')
            pair_e = self.leakyrelu(torch.matmul(pair_e, a3).squeeze()).t()
            assert not torch.isnan(pair_e).any()
            
        elif self.agg_type == 'sum':
            pair_e = torch.add(pair_e, y2)
            pair_e = pair_e.squeeze()
            
        elif self.agg_type == 'mul':
            time_e = torch.mul(pair_e, y2)
            pair_e = pair_e.squeeze() 
        
        
        # [3, 5664], [5664] => 32, 36, 7
        e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([x_in.shape[0], node_num, edge_num])).to_dense()      
        
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(incidence > 0, e, zero_vec)

        attention_node = F.softmax(attention.transpose(1,2), dim=2) # torch.Size([32, 36, 7])
        
        # edge to node
        node = torch.matmul(attention_node.permute(0,2,1), edge)    # torch.Size([32, 36, 1])

        if torch.isnan(node).sum().item() > 0:    
            raise Exception("ERROR: There are some NaN values")

        return node.permute(0,2,1), edge


class THGATImputer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 agg_type,
                 init_type,
                 n_layers=1,
                 dropout=0.,
                 kernel_size=2,
                 n_nodes=None,
                 n_edges=None):
        super(THGATImputer, self).__init__()

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.n_layers = int(n_layers)

        self.init_type = init_type

        input_size = 3 * self.input_size  # input + mask 
        
        # message-passing
        self.thgat = THGAT(d_in=input_size,
                            d_model=self.hidden_size,
                            agg_type=agg_type,
                            init_type=init_type)


        # Hidden state initialization embedding
        if n_nodes is not None:
            self.h0 = self.init_hidden_states(n_nodes)
        else:
            self.register_parameter('h0', None)

        if n_edges is not None:
            self.h1 = self.init_hidden_states(n_edges)
        else:
            self.register_parameter('h1', None)


    def init_hidden_states(self, n_nodes):
        h0 = []
        for l in range(self.n_layers):
            std = 1. / torch.sqrt(torch.rand(self.input_size, dtype=torch.float))
            vals = torch.distributions.Normal(0, std).sample((self.input_size, n_nodes))
            h0.append(nn.Parameter(vals))
        return nn.ParameterList(h0)


    def get_h0(self, x, incidence):
        if self.h0 is not None:
            init_h = [h0.expand(x.shape[0], -1, -1).permute(0,2,1) for h0 in self.h0]
            init_h_e = [h1.expand(x.shape[0], -1, -1) for h1 in self.h1]
            
            return (init_h, init_h_e)
        
        init_h = [torch.zeros(size=(x.shape[0], self.input_size, x.shape[2])).to(x.device)] * self.n_layers
        init_h_e = [torch.zeros(size=(x.shape[0], incidence.shape[1], self.input_size)).to(x.device) * self.n_layers]
        
        return (init_h, init_h_e)


    def forward(self, x, incidence, mask=None, h=None, h_e0=None):
        *_, steps = x.size()

        # infer all valid if mask is None
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.uint8)

        # init hyper-edge state
        if h is None:
            (h_0, h_e0) = self.get_h0(x, incidence)
        elif not isinstance(h, list):
            h_0 = [*h_0]
            h_e0 = [*h_e0]
       
       
        imputations = []
        imputations2 = torch.tensor([]).to('cuda')
        
        h = torch.tensor([]).to('cuda')
        h = torch.cat((h, h_0[-1]))
        
        h_e = torch.tensor([]).to('cuda')
        h_e = torch.cat((h_e, h_e0[-1]))

        i = 1
        for step in range(steps):
            # x:[batch, features, nodes, steps(window)]
            m_s = mask[..., step]
            x_s = x[..., step]      # torch.Size([32, 1, 36])             
            e_s = h_e               # 이전 시점의 edge
            n_s = h[:, -1:, :]      # 현재 시점의 초기화된 node
            
            n_shape = x_s.shape[2]
            x_s_ = x_s.reshape(-1,n_shape)
            n2_shape = x_s_.shape[0]
            x_s_ = x_s_.tolist()

            for depth in range(n2_shape):
                for row in range(n_shape):
                    if x_s_[depth][row] == 0.0:
                        x_s_[depth][row]=np.nan

            # 현재 시점의 전체 node(h) 초기화
            # 1) mean : 평균값으로 대치
            # 2) median : 중앙값으로 대치
            # 3) next observation carrid backward (NOCB) : 이전값으로 대치
            # 4) deviation 
            # 5) distribution
            # 6) fc
            if len(imputations)==0:
                n_s = h[:, -1:, :] 
              
            else:
                # win = self.step_win + 1 #윈도우 크기만큼만 초기화에 필요한 데이터로 사용
                #print("win",win)
                b_shape = x_s.shape[0]
                m_shape = x_s.shape[1]
                n_shape = x_s.shape[2]
                _h_node_new = []
                _h_node = imputations2 #imputations2
                _h_node_fc = imputations2
                
                imputations2_s_shape = imputations2.shape[0]
                imputations2_b_shape = imputations2.shape[1]
                imputations2_m_shape = imputations2.shape[2]
                imputations2_n_shape = imputations2.shape[3]
            
                _h_node = _h_node.reshape(-1,n_shape).tolist() 

                if self.init_type == 'mean': #한개 이전 시점의 평균
                    # imp_mean = SimpleImputer(strategy='mean') 
                    # imp_mean.fit(_h_node)
                    # df_imputed = pd.DataFrame(imp_mean.transform(_h_node))
                    # #print("df_imputed",df_imputed)
                    # df_imputed = df_imputed.values
                    # #print("df_imputed",df_imputed)
                    
                    _h_node_mean = _h_node[-b_shape:]
                    _h_node_mean = torch.tensor(_h_node_mean)
                    _h_node_mean = _h_node_mean.mean()
                    _h_node_mean= _h_node_mean.tolist()
                    df_imputed = x_s_
                        
                    for depth in range(b_shape):
                        for row in range(n_shape):
                            if math.isnan(df_imputed[depth][row]): 
                                df_imputed[depth][row] =_h_node_mean
                                
                    df_imputed = pd.DataFrame(df_imputed)
                    #print("df_imputed",df_imputed)
                    df_imputed = df_imputed.values

                elif self.init_type == 'median':
                    imp_mean = SimpleImputer(strategy='median') 
                    imp_mean.fit(_h_node)
                    df_imputed = pd.DataFrame(imp_mean.transform(_h_node))
                    df_imputed = df_imputed.values
                    #print("df_imputed",df_imputed)
                    

                elif self.init_type == 'NOCB':
                    _h_node_test = _h_node[-b_shape:]
                    df_imputed = x_s_
                        
                    for depth in range(b_shape):
                        for row in range(n_shape):
                            if math.isnan(df_imputed[depth][row]): 
                                df_imputed[depth][row] =_h_node_test[depth][row]
                                
                    df_imputed = pd.DataFrame(df_imputed)
                    #print("df_imputed",df_imputed)
                    df_imputed = df_imputed.values
                    
                elif self.init_type == 'deviation':
                    std_list = []
                    df_imputed = x_s_
                    
                    for i in range(imputations2_s_shape): #2 imputations2_s_shape:전 시점 길이 정할 수 있음
                        i2 = i+1
                        if i ==0:
                            std_list.append(np.std(_h_node[-b_shape:]))
                        else:
                            std_list.append(np.std(_h_node[-b_shape*i2:-b_shape*i]))
              
                    _h_node_std_avg = sum(std_list)/len(std_list)
                    
                    for depth in range(b_shape):
                            for row in range(n_shape):
                                if math.isnan(df_imputed[depth][row]): 
                                    df_imputed[depth][row] = _h_node_std_avg
                                    
                    df_imputed = pd.DataFrame(df_imputed)
                    df_imputed = df_imputed.values
                    
                
                elif self.init_type == 'distribution':
                    std_list = []
                    mean_list = []
                    df_imputed = x_s_
                    
                    for i in range(imputations2_s_shape): #imputations2_s_shape:전 시점 길이 정할 수 있음
                        i2 = i+1
                        if i ==0:
                            std_list.append(np.std(_h_node[-b_shape:]))
                            mean_list.append(_h_node[-b_shape:])
                        else:
                            std_list.append(np.std(_h_node[-b_shape*i2:-b_shape*i]))
                            mean_list.append(_h_node[-b_shape*i2:-b_shape*i])
              
                    _h_node_std_avg = sum(std_list)/len(std_list)
                    
                    mean_list = torch.tensor(mean_list)
                    _h_node_mean = mean_list.mean()
                    _h_node_mean=_h_node_mean.tolist()
                    #print("_h_node_mean",_h_node_mean)
                    
                    dis = np.random.normal(loc=_h_node_mean,scale=_h_node_std_avg,size=b_shape*n_shape)
                    dis = dis.reshape(b_shape,n_shape)
                   
                    for depth in range(b_shape):
                            for row in range(n_shape):
                                if math.isnan(df_imputed[depth][row]): 
                                    df_imputed[depth][row] = dis[depth][row]
                                    
                    df_imputed = pd.DataFrame(df_imputed)
                    #print("df_imputed",df_imputed)
                    df_imputed = df_imputed.values
                    
                elif self.init_type == 'fc':
                    df_imputed = x_s_
                    _h_node_fc = _h_node_fc.permute(1,2,3,0)
                    
                    fc1 = nn.Linear(imputations2_s_shape,1).to('cuda')
                    
                    _h_node_fc = fc1(_h_node_fc)
                    
                    _h_node_fc = _h_node_fc.reshape(b_shape,n_shape)
                    _h_node_fc = _h_node_fc.tolist()
                    
                    for depth in range(b_shape):
                            for row in range(n_shape):
                                if math.isnan(df_imputed[depth][row]): 
                                    #print("imputaed value",_h_node_fc[depth][row])
                                    df_imputed[depth][row] = _h_node_fc[depth][row]
                                
                    df_imputed = pd.DataFrame(df_imputed)
                    df_imputed = df_imputed.values
                
                n_s = torch.tensor(df_imputed[-b_shape:], dtype = torch.float32).to('cuda')
                n_s = n_s.reshape(b_shape,m_shape,n_shape)

            
            if h.shape[1] != 1: x_s = torch.cat((x_s, h[:, 1:, :]), dim=1)
                
            # prepare inputs
            # retrieve maximum information from neighbors
            xs_hat, edge_ = self.thgat(x=x_s, m=m_s, pri_e=e_s, pri_n=n_s, incidence=incidence)  # receive messages from neighbors (no self-loop!)
            
            # grin에서 m_s는 0과 1이 바뀌어져 있음
            #print(torch.where(m_s>0, 0, 1))
            #print(m_s)
            #print(x_s)
            x_s = x[..., step]
            
            # x_s = torch.mul(x_s, m_s) + torch.mul(xs_hat, torch.where(m_s>0, 0, 1))
            x_s = torch.mul(x_s+(step+1), m_s) + torch.mul(xs_hat+(step+1), torch.where(m_s>0, 0, 1))
            
            bn = nn.BatchNorm1d(x_s.shape[-1]).to("cuda")
            x_s = x_s.permute(0,2,1)
            x_s = bn(x_s)
            x_s = x_s.permute(0,2,1)
            
            # print(xs_hat)
            # print(x_s)
            # print("*"*30)
            #exit()
            
            h = torch.cat([h, x_s], dim=1)
            h_e = torch.cat((h_e, edge_), dim=-1)

            # store imputations and states
            imputations.append(xs_hat)
            imputations2=torch.cat((imputations2,torch.unsqueeze(xs_hat, 0)),dim=0)

        # Aggregate outputs -> [batch, features, nodes, steps]
        imputations = torch.stack(imputations, dim=-1)      # torch.Size([32, 1, 36, 40])

        return imputations


class BiTHGATImputer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size,
                 ff_dropout,
                 agg_type,
                 init_type,
                 n_layers=1,
                 dropout=0.,
                 n_nodes=None,
                 n_edges=None,
                 kernel_size=2,
                #  u_size=0,
                 embedding_size=0,
                 merge='mlp'):
        super(BiTHGATImputer, self).__init__()

        self.fwd_thgat = THGATImputer(input_size=input_size,
                            hidden_size=hidden_size,
                            agg_type=agg_type,
                            init_type=init_type,
                            n_layers=n_layers,
                            dropout=dropout,
                            kernel_size=kernel_size,
                            n_nodes=n_nodes,
                            n_edges=n_edges)
        self.bwd_thgat = THGATImputer(input_size=input_size,
                            hidden_size=hidden_size,
                            agg_type=agg_type,
                            init_type=init_type,
                            n_layers=n_layers,
                            dropout=dropout,
                            kernel_size=kernel_size,
                            n_nodes=n_nodes,    
                            n_edges=n_edges)

        if n_nodes is None:
            embedding_size = 0
        if embedding_size > 0:
            self.emb = nn.Parameter(torch.empty(embedding_size, n_nodes))
            nn.init.kaiming_normal_(self.emb, nonlinearity='relu')
        else:
            self.register_parameter('emb', None)

        # self._impute_from_states == True : representation을 사용한 imputation
        if merge == 'mlp':
            self._impute_from_states = True     # masking
            self.out = nn.Sequential(
                nn.Conv2d(in_channels=2,
                          out_channels=1, kernel_size=1),
                # nn.ReLU(),
                # nn.Dropout(ff_dropout),
                # nn.Conv2d(in_channels=ff_size, out_channels=input_size, kernel_size=1)
            )
        elif merge in ['mean', 'sum', 'min', 'max']:
            self._impute_from_states = False # non-masking
            self.out = getattr(torch, merge)
        else:
            raise ValueError("Merge option %s not allowed." % merge)
        self.supp = None

    def forward(self, x, incidence, mask=None):

        # Forward
        fwd_out = self.fwd_thgat(x, incidence, mask=mask)
        # Backward
        rev_x, rev_mask = [reverse_tensor(tens) for tens in (x, mask)]
        bwd_res = self.bwd_thgat(rev_x, incidence, mask=rev_mask)
        bwd_out = reverse_tensor(bwd_res)

        # _impute_from_states = true
        #if self._impute_from_states:
        if False:
            inputs = [fwd_repr, bwd_repr, mask]

            if self.emb is not None:
                b, *_, s = fwd_repr.shape  # fwd_h: [batches, channels, nodes, steps]
                inputs += [self.emb.view(1, *self.emb.shape, 1).expand(b, -1, -1, s)]  # stack emb for batches and steps
            imputation = torch.cat(inputs, dim=1) 
            imputation = self.out(imputation)

        else:
            # torch.Size([32, 2, 36, 40])
            imputation = torch.cat([fwd_out, bwd_out], dim=1)
            # torch.Size([32, 1, 36, 40])
            imputation = self.out(imputation)

        return imputation