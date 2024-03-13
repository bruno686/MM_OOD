import numpy as np
import os
import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_normal_initialization
import torch.nn.functional as F
import wandb

class ProjectHead(nn.Module):
    def __init__(self, input_dim=2816, hidden_dim=2048, out_dim=128):
        super(ProjectHead, self).__init__()
        self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim)
            )
        
    def forward(self, feat):
        feat = F.normalize(self.head(feat), dim=1)
        return feat

class XVBPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """
    def __init__(self, config, dataloader):
        super(XVBPR, self).__init__(config, dataloader)

        # load parameters info
        self.u_embedding_size = self.i_embedding_size = config['embedding_size']
        self.explore_loss_coeff = config['explore_loss_coeff']
        self.alpha_contrast = config['alpha_contrast']
        self.temp = config['temp']
        self.hidden_dim = config['hidden_dim']
        self.out_dim = config['out_dim']
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton

        self.t_proj = ProjectHead(input_dim=32, hidden_dim=self.hidden_dim, out_dim=self.out_dim)
        self.v_proj = ProjectHead(input_dim=32, hidden_dim=self.hidden_dim, out_dim=self.out_dim)

        # define layers and loss
        self.u_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.u_embedding_size * 2)))
        self.i_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.i_embedding_size)))
        if self.v_feat is not None and self.t_feat is not None:
            self.item_raw_features = torch.cat((self.t_feat, self.v_feat), -1)
        elif self.v_feat is not None:
            self.item_raw_features = self.v_feat
        else:
            self.item_raw_features = self.t_feat

        self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.i_embedding_size)
        self.t_item_linear = nn.Linear(self.t_feat.shape[1], self.i_embedding_size)
        self.v_item_linear = nn.Linear(self.v_feat.shape[1], self.i_embedding_size)

        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        return self.u_embedding[user, :]

    def get_item_embedding(self, item):
        return self.item_embedding[item, :]

    def forward(self, dropout=0.0):
        t_item_embeddings = self.t_item_linear(self.t_feat)
        v_item_embeddings = self.v_item_linear(self.v_feat)

        # Prepare for Contrastive learning
        t_dim = int(t_item_embeddings.shape[1] / 2)
        v_dim = int(v_item_embeddings.shape[1] / 2)
        t_emd_proj = self.t_proj(t_item_embeddings[:, :t_dim])
        v_emd_proj = self.v_proj(v_item_embeddings[:, :v_dim])

        # concat multi-modal with free embedding
        item_embeddings = torch.cat([t_emd_proj, v_emd_proj], dim=1)
        item_embeddings = torch.cat((self.i_embedding, item_embeddings), -1)

        user_embeddings = F.dropout(self.u_embedding, 0.01)
        item_embeddings = F.dropout(item_embeddings, 0.01)
        return user_embeddings, item_embeddings, t_emd_proj, v_emd_proj, t_item_embeddings, v_item_embeddings, t_dim, v_dim


    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_embeddings, item_embeddings, t_emd_proj, v_emd_proj, t_item_embeddings, v_item_embeddings, t_dim, v_dim = self.forward()

        user_e = user_embeddings[user, :]
        pos_e = item_embeddings[pos_item, :]
        #neg_e = self.get_item_embedding(neg_item)
        neg_e = item_embeddings[neg_item, :]
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        mf_loss = self.loss(pos_item_score, neg_item_score)
        reg_loss = self.reg_loss(user_e, pos_e, neg_e)

        # INfoNCE
        loss_InfoNCE = self.InfoNCE(t_emd_proj, v_emd_proj, temperature=self.temp)

        # Feature Splitting with Distance
        loss_e1 = -F.mse_loss(t_item_embeddings[:, :t_dim], t_item_embeddings[:, t_dim:])
        loss_e2 = -F.mse_loss(v_item_embeddings[:, :v_dim], v_item_embeddings[:, v_dim:])

        loss = mf_loss + self.reg_weight * reg_loss + self.explore_loss_coeff * (loss_e1 + loss_e2)/2 + self.alpha_contrast * loss_InfoNCE
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings, _, _, _, _, _, _ = self.forward()
        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

    def InfoNCE(self, view1, view2, temperature: float, b_cos: bool = True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

        pos_score = (view1 @ view2.T) / temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()
