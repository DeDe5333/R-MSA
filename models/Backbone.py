import torch
import torch.nn as nn
import torch.nn.functional as F
from .i3d import I3D
import logging
from .RegressTree import RegressTree

class CoAttentionModule(nn.Module):
    def __init__(self, feature_dim, r_max):
        super(CoAttentionModule, self).__init__()
        self.feature_dim = feature_dim
        self.r_max = r_max
        self.gamma = 0.2

    def generate_gaussian_matrix(self, r_max):
        G = torch.zeros(r_max, r_max, device='cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(r_max):
            for j in range(r_max):
                G[i, j] = torch.exp(torch.tensor(-((j - i) ** 2) / (2 * (self.gamma * r_max) ** 2)))
        return G

    def forward(self, F_k, F_l):
        batch_size = F_k.size(0)
        num_clips = F_k.size(1)  # Dynamically get num_clips
        S_kl = F.softmax(torch.bmm(F_l, F_k.transpose(1, 2)) / (self.feature_dim ** 0.5), dim=-1)
        S_lk = F.softmax(torch.bmm(F_k, F_l.transpose(1, 2)) / (self.feature_dim ** 0.5), dim=-1)
        # Adjust G_kl to match num_clips
        G_kl = self.generate_gaussian_matrix(num_clips)  # Use num_clips instead of r_max
        G_lk = G_kl.transpose(0, 1)
        S_kl_masked = S_kl * G_kl.unsqueeze(0)
        S_lk_masked = S_lk * G_lk.unsqueeze(0)
        F_k_co = torch.bmm(S_lk_masked, F_k)
        F_l_co = torch.bmm(S_kl_masked, F_l)
        F_k_enh = F_k_co  # [N, num_clips, feature_dim]
        F_l_enh = F_l_co  # [N, num_clips, feature_dim]
        return F_k_enh, F_l_enh

class RhythmEncodingModule(nn.Module):
    def __init__(self, encode_dim=128, num_groups=6):
        super(RhythmEncodingModule, self).__init__()
        self.encode_dim = encode_dim
        self.num_groups = num_groups

    def forward(self, num_clips, duration_group_idx):
        t = torch.arange(num_clips).float().unsqueeze(-1).to(duration_group_idx.device)  # [num_clips, 1]
        n = torch.arange(self.encode_dim).float().to(duration_group_idx.device)  # [encode_dim]
        omega = (duration_group_idx * self.num_groups) / (10000 ** (2 * n / self.encode_dim))
        omega = omega.unsqueeze(0).expand(num_clips, -1)  # [num_clips, encode_dim]
        
        rhythm = torch.where(
            n % 2 == 0,
            torch.sin(omega * t),
            torch.cos(omega * t)
        )  # [num_clips, encode_dim]
        
        return rhythm

class I3D_backbone(nn.Module):
    def __init__(self, args, I3D_class, r_max=50, encode_dim=128, num_groups=6, hidden_channel=256, tree_depth=3):
        super(I3D_backbone, self).__init__()

        self.backbone = I3D(num_classes=I3D_class, modality='rgb', dropout_prob=0.5)
        self.r_max = r_max
        self.encode_dim = encode_dim
        self.num_groups = num_groups
        self.benchmark = args.benchmark

        self._feature_dim = 1024 + encode_dim

        self.co_attention = CoAttentionModule(feature_dim=1024, r_max=r_max)
        self.rhythm_encoder = RhythmEncodingModule(encode_dim=encode_dim, num_groups=num_groups)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self._feature_dim, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=5)
        self.queries = nn.Parameter(torch.randn(num_groups, self._feature_dim))
        self.regress_tree = RegressTree(in_channel=self._feature_dim, hidden_channel=hidden_channel, depth=tree_depth)

    def load_pretrain(self, I3D_ckpt_path):
        self.backbone.load_state_dict(torch.load(I3D_ckpt_path))
        print('loading ckpt done')

    def get_feature_dim(self):
        return self._feature_dim

    def forward(self, target, exemplar, is_train, label, theta, duration_1, duration_2):
        # spatiotemporal feature
        # total_video = torch.cat((target, exemplar), 0)  # 2N, C, H, W
        # start_idx = [0, 10, 20, 30, 40, 50, 60, 70, 80, 86]
        # video_pack = torch.cat([total_video[:, :, i: i + 16] for i in start_idx])  # 10*2N, C, 16, H, W
        # total_feature = self.backbone(video_pack).reshape(10, len(total_video), -1).transpose(0, 1)  # 2N, 10, 1024

        # T = total_video.size(2)  # Get the time dimension T

        total_video = torch.cat((target, exemplar), 0)  # [2N, C, T, H, W]
        T = total_video.size(2)  # 当前采样的总帧数（如 JIG: ~120-200, Hei: 400+）

        # === 根据 benchmark 选择不同的 clip 提取策略 ===
        # if hasattr(self, 'benchmark'):  # 如果你在 __init__ 中传了 benchmark
        #     benchmark = self.benchmark
        # else:
        #     # 从外部传入（推荐方式：builder 时传 args.benchmark）
        #     # 临时方案：这里假设你能在 forward 里访问 args（如果不行，见下方备选）
        #     import sys
        #     if 'args' in sys.modules['__main__'].__dict__:
        #         args = sys.modules['__main__'].__dict__['args']
        #         benchmark = args.benchmark
        #     else:
        #         benchmark = 'JIG'  # 默认

        if self.benchmark in ['JIG', 'Seven', 'MTL']:  # 短视频数据集：用原来固定10个clips的方式
            # 固定10个clips，但需要根据实际视频长度调整
            clip_length = 16
            num_clips_target = 10
            # 如果视频长度足够，使用原来的固定索引
            if T >= 86 + clip_length:
                start_indices = [0, 10, 20, 30, 40, 50, 60, 70, 80, 86]
            else:
                # 如果视频长度不足，动态调整索引，确保每个clip都有16帧
                max_start = max(0, T - clip_length)
                if max_start == 0:
                    # 如果视频长度不足16帧，只提取一个clip（从0开始）
                    start_indices = [0]
                else:
                    # 均匀采样，确保最后一个clip的起始位置不超过 max_start
                    step = max(1, max_start // (num_clips_target - 1)) if num_clips_target > 1 else max_start
                    start_indices = list(range(0, max_start + 1, step))[:num_clips_target]
                    # 确保最后一个clip的起始位置能提取到16帧
                    if start_indices[-1] > max_start:
                        start_indices[-1] = max_start
            # print(f"[{self.benchmark}] 使用固定10个clips提取策略 (T={T}, clips={len(start_indices)})")
        
        else:  # HeiChole 等长视频：动态多clips均匀采样
            clip_length = 16
            max_clips = 40  # 最大提取40个clips（可根据显存调小）
            clip_step = max(8, (T - clip_length) // max_clips)  # 步长至少8，保证有重叠
            start_indices = list(range(0, T - clip_length + 1, clip_step))[:max_clips]
            print(f"[{self.benchmark}] 动态提取 {len(start_indices)} 个clips (T={T}, step={clip_step})")

        # 统一提取 clips，确保每个clip都有16帧（如果不够则padding）
        clip_list = []
        for i in start_indices:
            end_idx = min(i + clip_length, T)
            if end_idx - i >= clip_length:
                # 正常的clip提取
                clip_list.append(total_video[:, :, i:end_idx])
            else:
                # 如果帧数不足，使用最后一个clip并重复最后一帧进行padding
                clip = total_video[:, :, i:end_idx]
                padding_frames = clip_length - (end_idx - i)
                # 重复最后一帧进行padding
                if end_idx > i:
                    last_frame = total_video[:, :, end_idx-1:end_idx].expand(-1, -1, padding_frames, -1, -1)
                    padded_clip = torch.cat([clip, last_frame], dim=2)
                else:
                    # 如果完全没有帧，用第一帧填充
                    padded_clip = total_video[:, :, 0:1].expand(-1, -1, clip_length, -1, -1)
                clip_list.append(padded_clip)
        video_pack = torch.cat(clip_list, dim=0)
        num_clips = len(start_indices)

        # 提取 I3D 特征
        total_feature = self.backbone(video_pack).reshape(num_clips, len(total_video), -1).transpose(0, 1)  # [2N, num_clips, 1024]

        # 限制 num_clips 不超过 r_max（CoAttention 需要）
        # num_clips = min(self.r_max, num_clips)
        num_clips = min(self.r_max, max(1, (T - 16) // 10 + 1))  # Limit num_clips to r_max
        total_feature = total_feature[:, :num_clips, :]  # [2N, num_clips, 1024]




        # num_clips = min(self.r_max, max(1, (T - 16) // 10 + 1))  # Limit num_clips to r_max

        feature_1 = total_feature[:total_feature.shape[0] // 2]  # [N, num_clips, 1024]
        feature_2 = total_feature[total_feature.shape[0] // 2:]  # [N, num_clips, 1024]

        # SSAM (CoAttentionModule)
        feature_1_enh, feature_2_enh = self.co_attention(feature_1, feature_2)  # [N, num_clips, 1024]

        # TRME (RhythmEncodingModule)
        batch_size = feature_1_enh.size(0)
        Rhy_1 = torch.stack([self.rhythm_encoder(num_clips, idx) for idx in duration_1])  # [N, num_clips, encode_dim]
        Rhy_2 = torch.stack([self.rhythm_encoder(num_clips, idx) for idx in duration_2])  # [N, num_clips, encode_dim]
        feature_1_enh = torch.cat([feature_1_enh, Rhy_1], dim=-1)  # [N, num_clips, 1024+encode_dim]
        feature_2_enh = torch.cat([feature_2_enh, Rhy_2], dim=-1)

        # MSD (Transformer Decoder)
        queries = self.queries.to(feature_1_enh.device).unsqueeze(0).repeat(batch_size, 1, 1)
        memory = feature_1_enh.transpose(0, 1)
        dec_out_1 = self.decoder(queries.transpose(0, 1), memory).transpose(0, 1).mean(1)  # [N, feature_dim]
        memory = feature_2_enh.transpose(0, 1)
        dec_out_2 = self.decoder(queries.transpose(0, 1), memory).transpose(0, 1).mean(1)

        # UCIT (RegressTree in base_model)
        out_prob_1, delta_1, mu_1, log_var_1 = self.regress_tree(dec_out_1, is_train)
        out_prob_2, delta_2, mu_2, log_var_2 = self.regress_tree(dec_out_2, is_train)

        if is_train:
            combined_feature_1 = torch.cat((dec_out_1, dec_out_2, label[0] / theta), 1)
            combined_feature_2 = torch.cat((dec_out_2, dec_out_1, label[1] / theta), 1)
            return combined_feature_1, combined_feature_2, Rhy_1, Rhy_2, mu_1, log_var_1, mu_2, log_var_2
        else:
            combined_feature = torch.cat((dec_out_2, dec_out_1, label[0] / theta), 1)
            return combined_feature, Rhy_1, Rhy_2