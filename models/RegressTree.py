# import torch.nn as nn
# import torch
# import torch.nn.functional as F
        
# class RegressTree(nn.Module):
#     def __init__(self, in_channel, hidden_channel, depth):
#         super(RegressTree, self).__init__()
#         self.depth = depth
#         self.num_leaf = 2**(depth-1)

#         self.first_layer = nn.Sequential(
#             nn.Linear(in_channel, hidden_channel),
#             nn.ReLU(inplace=True)
#         )

#         self.feature_layers = nn.ModuleList([self.get_tree_layer(2**d, hidden_channel) for d in range(self.depth - 1)])
#         self.clf_layers = nn.ModuleList([self.get_clf_layer(2**d, hidden_channel) for d in range(self.depth - 1)])
#         self.reg_layer = nn.Conv1d(self.num_leaf * hidden_channel, self.num_leaf, 1, groups=self.num_leaf)

#     @staticmethod
#     def get_tree_layer(num_node_in, hidden_channel=256):
#         return nn.Sequential(
#             nn.Conv1d(num_node_in * hidden_channel, num_node_in * 2 * hidden_channel, 1, groups=num_node_in),
#             nn.ReLU(inplace=True)
#         )

#     @staticmethod
#     def get_clf_layer(num_node_in, hidden_channel=256):
#         return nn.Conv1d(num_node_in * hidden_channel, num_node_in * 2, 1, groups=num_node_in)

#     def forward(self, input_feature):
#         out_prob = []
#         x = self.first_layer(input_feature)
#         bs = x.size(0)
#         x = x.unsqueeze(-1)
#         for i in range(self.depth - 1):
#             prob = self.clf_layers[i](x).squeeze(-1)
#             x = self.feature_layers[i](x)
#             if len(out_prob) > 0:
#                 prob = F.log_softmax(prob.view(bs, -1, 2), dim=-1)
#                 pre_prob = out_prob[-1].view(bs, -1, 1).expand(bs, -1, 2).contiguous()
#                 prob = pre_prob + prob
#                 out_prob.append(prob)
#             else:
#                 out_prob.append(F.log_softmax(prob.view(bs, -1, 2), dim=-1))  # 2 branch only
#         delta = self.reg_layer(x).squeeze(-1)
#         return out_prob, delta

import torch.nn as nn
import torch
import torch.nn.functional as F

class RegressTree(nn.Module):
    def __init__(self, in_channel, hidden_channel, depth):
        super(RegressTree, self).__init__()
        self.depth = depth
        self.num_leaf = 2**(depth-1)

        self.first_layer = nn.Sequential(
            nn.Linear(in_channel, hidden_channel),
            nn.ReLU(inplace=True)
        )

        self.feature_layers = nn.ModuleList([self.get_tree_layer(2**d, hidden_channel) for d in range(self.depth - 1)])
        self.clf_layers = nn.ModuleList([self.get_clf_layer(2**d, hidden_channel) for d in range(self.depth - 1)])

        # 修改：输出num_leaf * 2（μ和σ²），groups=num_leaf保持独立
        self.reg_layer = nn.Conv1d(self.num_leaf * hidden_channel, self.num_leaf * 2, 1, groups=self.num_leaf)
        # self.dropout = nn.Dropout(p=0.1)

    @staticmethod
    def get_tree_layer(num_node_in, hidden_channel=256):
        return nn.Sequential(
            nn.Conv1d(num_node_in * hidden_channel, num_node_in * 2 * hidden_channel, 1, groups=num_node_in),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def get_clf_layer(num_node_in, hidden_channel=256):
        return nn.Conv1d(num_node_in * hidden_channel, num_node_in * 2, 1, groups=num_node_in)

    def forward(self, input_feature, is_train=True):
        out_prob = []
        x = self.first_layer(input_feature)
        bs = x.size(0)
        x = x.unsqueeze(-1)
        # depth=1: single leaf, no internal nodes - add dummy log prob for compatibility
        if self.depth == 1:
            out_prob.append(torch.zeros(bs, self.num_leaf, device=x.device, dtype=x.dtype))
        for i in range(self.depth - 1):
            prob = self.clf_layers[i](x).squeeze(-1)
            x = self.feature_layers[i](x)
            # # 这里加了dropout，改了这里
            # x = self.dropout(x)
            if len(out_prob) > 0:
                prob = F.log_softmax(prob.view(bs, -1, 2), dim=-1)
                pre_prob = out_prob[-1].view(bs, -1, 1).expand(bs, -1, 2).contiguous()
                prob = pre_prob + prob
                out_prob.append(prob)
            else:
                out_prob.append(F.log_softmax(prob.view(bs, -1, 2), dim=-1))  # 2 branch only

        # 修改：输出[bs, num_leaf * 2, 1]，分离μ和σ²
        reg_out = self.reg_layer(x)  # [bs, num_leaf * 2, 1]
        reg_out = reg_out.squeeze(-1)  # [bs, num_leaf * 2]
        mu = reg_out[:, :self.num_leaf]  # [bs, num_leaf]
        log_var = reg_out[:, self.num_leaf:]  # [bs, num_leaf] (log(σ²) for stability)
        sigma = torch.exp(0.5 * log_var)  # σ = sqrt(exp(log_var))

        if is_train:
            # 重参数化采样：delta = μ + σ * ε
            epsilon = torch.randn_like(sigma)  # ε ~ N(0,1)
            delta = mu + sigma * epsilon
        else:
            # 推理时用均值（无随机性）
            delta = mu

        return out_prob, delta, mu, log_var  # 新增mu和log_var用于损失计算