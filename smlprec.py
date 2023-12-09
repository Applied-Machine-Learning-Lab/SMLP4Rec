r"""
MLP4Rec v1.7

无线性output层 一维卷积加速

"""

import torch
from torch import nn
from functools import partial
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer, VanillaAttention
from recbole.model.loss import BPRLoss
# 引入smlp
from recbole.ext_lib.model.mlp.sMLP_block import sMLPBlock


# 根据dwconv和smlp定义tokenmixing
class TokenMixing(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels=in_shape[0],
                                    out_channels=in_shape[0],
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_shape[0])
        self.smlp = sMLPBlock(h=in_shape[1], w=in_shape[2], c=in_shape[0])
        self.bn0 = nn.BatchNorm2d(in_shape[0])
        self.bn1 = nn.BatchNorm2d(in_shape[0])
        # self.bn2 = nn.BatchNorm2d(in_shape[0])
    def forward(self, x):
        x = x+self.dwconv(self.bn0(x))
        x = x+self.smlp(self.bn1(x))
        # x = self.bn2(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, in_shape_0):
        super().__init__()
        self.PM = nn.Conv2d(in_shape_0, 2 * in_shape_0, 2, stride=2, padding=0, groups=in_shape_0)
    def forward(self, x):
        return self.PM(x)

    
def FeedForward(dim, expansion_factor = 2, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

class SMLP(nn.Module):
    def __init__(self, in_shape, expansion_factor = 2, dropout=0.):
        super().__init__()
        self.c = nn.Sequential(nn.Linear(in_shape[0], expansion_factor * in_shape[0], bias=False),
                               nn.GELU(),
                               nn.Dropout(dropout),
                               nn.Linear(expansion_factor * in_shape[0], in_shape[0], bias=False),
                               nn.Dropout(dropout),)
        self.h = nn.Sequential(nn.Linear(in_shape[1], expansion_factor * in_shape[1], bias=False),
                               nn.GELU(),
                               nn.Dropout(dropout),
                               nn.Linear(expansion_factor * in_shape[1], in_shape[1], bias=False),
                               nn.Dropout(dropout),)
        self.w = nn.Sequential(nn.Linear(in_shape[2], expansion_factor * in_shape[2], bias=False),
                               nn.GELU(),
                               nn.Dropout(dropout),
                               nn.Linear(expansion_factor * in_shape[2], in_shape[2], bias=False),
                               nn.Dropout(dropout),)
        self.norm2 = nn.LayerNorm(in_shape[2])
    def forward(self, x):
        xn = self.norm2(x)
        x0 = (self.c(xn.transpose(1,3).contiguous())).transpose(3,1).contiguous()
        x1 = (self.h(xn.transpose(2,3).contiguous())).transpose(3,2).contiguous()
        x2 = self.w(xn)
        y = x0 + x1 + x2
        return y

class SMLPREC(SequentialRecommender):
    r"""
    FDSA is similar with the GRU4RecF implemented in RecBole, which uses two different Transformer encoders to
    encode items and features respectively and concatenates the two subparts' outputs as the final output.

    """

    def __init__(self, config, dataset, seq_len = 50):
        super(SMLPREC, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        expansion_factor = 3
        chan_first = partial(nn.Conv1d, kernel_size = 1)
        chan_last = nn.Linear
        self.num_feature_field = len(config['selected_features'])
        self.layerSize = self.num_feature_field + 1

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)

        self.feature_embed_layer = FeatureSeqEmbLayer(
            dataset, self.hidden_size, self.selected_features, self.pooling_mode, self.device
        )

        self.layers = nn.ModuleList([])
        dim_new = self.num_feature_field+1
        self.layers.append(SMLP([dim_new, seq_len, self.hidden_size], expansion_factor, self.hidden_dropout_prob))
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        # item_seq shape: torch.Size([256, 50]), item_seq_len shape: torch.Size([256])
        item_emb = self.item_embedding(item_seq)
        # item_emb shape:  torch.Size([256, 50, 128])
        sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
        sparse_embedding = sparse_embedding['item']
        # sparse_embedding shape:  torch.Size([256, 50, 2, 128])
        dense_embedding = dense_embedding['item']
        # dense_embedding shape:  torch.Size([256, 50, 2, 128])

        # 将两个embedding合成一个feature embedding
        if sparse_embedding is not None:
            feature_embeddings = sparse_embedding
        if dense_embedding is not None:
            if sparse_embedding is not None:
                feature_embeddings = torch.cat((sparse_embedding,dense_embedding),2)
            else:
                feature_embeddings = dense_embedding
        # feature_embeddings shape:  torch.Size([256, 50, 4, 128])

        # 加入物体的embedding
        item_emb = torch.unsqueeze(item_emb,2)
        # item_emb1 shape:  torch.Size([256, 50, 1, 128])
        item_emb = torch.cat((item_emb,feature_embeddings),2)
        # mixer_outputs = item_emb
        # item_emb2 shape:  torch.Size([256, 50, 5, 128])

        # 将feature个数所在的维度从2变为1：拆分重组
        # self.num_feature_field为selected_features数量，此处为4
        # mixer_outputs = torch.split(item_emb,[1]*(self.num_feature_field+1),2)
        # # mixer_outputs len:  5
        # mixer_outputs = torch.stack(mixer_outputs,1)
        # mixer_outputs = torch.squeeze(mixer_outputs)
        # print(mixer_outputs.shape)
        mixer_outputs = item_emb.transpose(1,2).contiguous()
        # mixer_outputs shape: torch.Size([256, 5, 50, 128]) (batch * item_feature * item_seq * channel)

        for x in range(self.n_layers):
            mixer_outputs = self.layers[0](mixer_outputs)
        # mixer_outputs = mixer_outputs.transpose(3, 1).contiguous()
        # output = self.pool(mixer_outputs).squeeze()

        mixer_outputs = mixer_outputs.transpose(1, 0).contiguous()
        output = self.gather_indexes(mixer_outputs[0], item_seq_len - 1)
        output = self.LayerNorm(output)
        # print(output.shape)
        return output



    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
