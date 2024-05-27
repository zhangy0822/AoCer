import torch
import torch.nn as nn

from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from utils.utils import *

class TextImageMatchingModel(nn.Module):
    def __init__(self, config):
        super(TextImageMatchingModel, self).__init__()
        self.cfg = config
        self.img_enc = ImageEncoder(config)
        self.txt_enc = TextEncoder(config)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif param.dim() < 2:
                nn.init.uniform_(param)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, sent_inds, sent_msks, region_feats):
        # encode image feature
        img_feats = self.img_enc(region_feats)
        if self.cfg.l2_norm:
            img_feats = l2norm(img_feats)

        # encode text feature
        bsize, nturns, nwords = sent_inds.size()
        reg_cap_feas, lang_feats, _ = self.txt_enc(sent_inds.view(-1, nwords), sent_msks.view(-1, nwords))
        reg_cap_feas = reg_cap_feas.view(bsize, nturns, -1, self.cfg.n_feature_dim)
        lang_feats = lang_feats.view(bsize, nturns, self.cfg.n_feature_dim)
        # lang_masks =lang_feats.new_ones(bsize, nturns)
        if self.cfg.l2_norm:
            lang_feats = l2norm(lang_feats)
            # reg_cap_feas = l2norm(reg_cap_feas)

        return img_feats, lang_feats
        # return img_feats, reg_cap_feas

    def triplet_loss(self, img_feats, lang_feats):
        sim = self.compute_sim(img_feats, lang_feats)
        diagonal = sim.diag().view(img_feats.size(0), 1)
        d1 = diagonal.expand_as(sim)
        d2 = diagonal.t().expand_as(sim)

        # compare every diagonal score to scores in its column
        # image retrieval
        cost_i = (self.cfg.margin + sim - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # caption retrieval
        cost_l = (self.cfg.margin + sim - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(sim.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_i = cost_i.masked_fill_(I, 0)
        cost_l = cost_l.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.cfg.max_violation:
            cost_i = cost_i.max(1)[0]
            cost_l = cost_l.max(0)[0]
        return cost_i.sum() + cost_l.sum()
        # return cost_l.sum()


    def compute_sim(self, img_feats, lang_feats):
        """

        :param img_feats: tensor, [batchsize, num_region, feature_dim]
        :param lang_feats: tensor, [batchsize, num_turn, feature_dim]
        :return: similarity: tensor, [batchsize, batchsize]
        """
        similarity = []
        for i in range(lang_feats.shape[0]):
            # -> (1, num_turn, feature_dim)
            lang_feat = lang_feats[i]
            # -> (batchsize, num_turn, feature_dim)
            lang_feat_expand = lang_feat.repeat(lang_feats.shape[0], 1, 1)
            # ->(batchsize, num_turn, num_region)
            raw_sim = torch.bmm(lang_feat_expand, img_feats.transpose(1, 2))
            img_feats_norm = torch.norm(img_feats, 2, dim=-1, keepdim=True)
            lang_feats_norm = torch.norm(lang_feats, 2,dim=-1, keepdim=True)
            raw_sim_norm = lang_feats_norm * img_feats_norm.transpose(1, 2)
            sim = raw_sim / raw_sim_norm
            # compute attention
            attn = torch.softmax(sim * self.cfg.lse_lambda, dim=-1)
            # -> (batchsize)
            sim = torch.mean(torch.mean(attn * sim, dim=-1, keepdim=True), dim=-2, keepdim=True)
            similarity.append(sim.squeeze(-1))
        # -> (batchsize, batchsize)
        similarity = torch.cat(similarity, dim=1).t()
        return similarity



    def compute_triplet_loss(self, sent_inds, sent_msks, region_feats):
        img_feats, lang_feats = self.forward(sent_inds, sent_msks, region_feats)
        loss = self.triplet_loss(img_feats, lang_feats)
        return loss


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    # return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
    return (w12 / (w1 * w2).clamp(min=eps))


def func_attention(query, context, smooth):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)


    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax(dim=-1)(attn*smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext