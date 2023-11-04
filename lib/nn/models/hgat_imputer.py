import torch
from einops import rearrange
from torch import nn

from ..layers import BiHGATImputer
from ...utils.parser_utils import str_to_bool
 

class HGATNet(nn.Module):
    def __init__(self,
                 adj,
                 d_in,
                 d_hidden,
                 d_ff,
                 ff_dropout,
                 agg_type,
                 init_type,
                 adj_dist=[],
                 n_layers=1,
                 kernel_size=2,
                 decoder_order=1,
                 global_att=False,
                 d_u=0,
                 d_emb=0,
                 layer_norm=False,
                 merge='mlp',
                 impute_only_holes=True):
        super(HGATNet, self).__init__()

        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_u = int(d_u) if d_u is not None else 0
        self.d_emb = int(d_emb) if d_emb is not None else 0
        self.register_buffer('adj', torch.tensor(adj).float())
        self.register_buffer('adj_dist', torch.tensor(adj_dist).float())
        self.impute_only_holes = impute_only_holes 

        self.bihgat_imputer = BiHGATImputer(input_size=self.d_in,
                              hidden_size=self.d_hidden,
                              ff_size=d_ff,
                              ff_dropout=ff_dropout,
                              agg_type=agg_type,
                              init_type=init_type,
                              n_layers=n_layers,
                              n_nodes=self.adj.shape[0],
                              n_edges=self.adj.shape[1],
                              n_dedges=self.adj_dist.shape[1],
                              kernel_size=kernel_size,
                            #   u_size=self.d_u,
                              embedding_size=self.d_emb,
                              merge=merge)

    def forward(self, x, mask=None, u=None, **kwargs):
        # x: [batches, steps, nodes, channels] -> [batches, channels, nodes, steps]
        x = rearrange(x, 'b s n c -> b c n s')
        
        if mask is not None:
            mask = rearrange(mask, 'b s n c -> b c n s')
        
        if u is not None:
            u = rearrange(u, 'b s n c -> b c n s')
         
        # imputation, prediction --> imputation
        # imputation = self.bihgat_imputer(x, self.adj, mask=mask)                # torch.Size([32, 1, 36, 40])
        imputation, prediction = self.bihgat_imputer(x, self.adj, self.adj_dist, mask=mask)
        
        # In evaluation stage impute only missing values
        if self.impute_only_holes and not self.training:
            imputation = torch.where(mask, x, imputation)

        # out: [batches, channels, nodes, steps] -> [batches, steps, nodes, channels]
        imputation = torch.transpose(imputation, -3, -1)
        prediction = torch.transpose(prediction, -3, -1)
        
        
        # NOTE loss 여러 개 사용 시
        if self.training:
            # return imputation
            return imputation, prediction
        
        return imputation

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--d-hidden', type=int, default=64)
        parser.add_argument('--d-ff', type=int, default=64)
        parser.add_argument('--ff-dropout', type=int, default=0.)
        parser.add_argument('--n-layers', type=int, default=1)
        parser.add_argument('--kernel-size', type=int, default=2)
        parser.add_argument('--decoder-order', type=int, default=1)
        parser.add_argument('--d-u', type=int, default=0)
        parser.add_argument('--d-emb', type=int, default=8)
        parser.add_argument('--layer-norm', type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument('--global-att', type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument('--merge', type=str, default='mlp')
        parser.add_argument('--impute-only-holes', type=str_to_bool, nargs='?', const=True, default=True)
        return parser
