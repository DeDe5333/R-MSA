import torch
import numpy as np
import os
import csv
import random
import glob
from PIL import Image
import math
import random

def long_range_rand(seq_len, num_seg):
        if seq_len <= num_seg:
            # 如果视频太短，直接均匀采样或补帧
            return list(range(0, seq_len, max(1, seq_len // num_seg)))[:num_seg]
        r = int(seq_len / num_seg)
        real_num_seg = int(math.ceil(seq_len / r))

        frame_ind = []
        for i in range(real_num_seg - 1):
            frame_ind.append(random.randint(i * r, (i + 1) * r - 1))
        frame_ind.append(random.randint((real_num_seg - 1) * r, seq_len - 1))

        # 取最后 num_seg 个（保证覆盖尾部）
        frame_ind = frame_ind[-num_seg:]
        return frame_ind

def long_range_first(seq_len, num_seg):
    if seq_len <= num_seg:
        return list(range(0, seq_len, max(1, seq_len // num_seg)))[:num_seg]
    r = int(seq_len / num_seg)
    frame_ind = [i * r for i in range(num_seg)]
    return frame_ind

def long_range_last(seq_len, num_seg):
    if seq_len <= num_seg:
        indices = list(range(0, seq_len, max(1, seq_len // num_seg)))
        return indices[-num_seg:] if len(indices) >= num_seg else indices
    r = int(seq_len / num_seg)
    frame_ind = [(seq_len - 1) - i * r for i in range(num_seg)]
    frame_ind.reverse()
    return frame_ind

def long_range_sample(seq_len, num_seg, mode='random'):
    if mode == "random":
        return long_range_rand(seq_len, num_seg)
    elif mode == "first":
        return long_range_first(seq_len, num_seg)
    elif mode == "last":
        return long_range_last(seq_len, num_seg)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

class HeiPair_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        random.seed(args.seed)
        self.transforms = transform
        self.cls = args.cls
        self.mode = subset
        self.split_index = args.split_index  # 替代之前的 fold

        try:
            self.frames_dir = args.frames_dir
            self.info_dir = args.info_dir
        except AttributeError as e:
            raise AttributeError(f"缺少必要的 args 属性: {e}。请确保 args 中包含 'frames_dir' 和 'info_dir'。")

        self.video_info_dict = self.read_video_info()
        if not self.video_info_dict:
            raise ValueError(f"{self.cls} 任务的 video_info_dict 为空，没有找到任何有效的视频数据！")

        self.temporal_shift = [args.temporal_shift_min, args.temporal_shift_max]
        self.voter_number = args.voter_number
        self.length = args.num_frames
        self.num_duration_groups = args.num_duration_groups
        self.sample_mode = 'random' if self.mode == 'train' else 'last'

        

        # 加载视频列表
        self.name_list = [f'hei-chole{idx}_{self.cls}' for idx in self.indices if f'hei-chole{idx}_{self.cls}' in self.video_info_dict]
        self.train_name = [f'hei-chole{idx}_{self.cls}' for idx in train_indices if f'hei-chole{idx}_{self.cls}' in self.video_info_dict]
        if not self.name_list:
            raise ValueError(f"{self.mode} 模式下的 name_list 为空，请检查数据路径或 {self.cls} 任务数据！")
        
        self.durations, self.groups = self.compute_durations_and_groups()
        print(f"数据集初始化完成，任务: {self.cls}, 模式: {self.mode}, 帧数长度: {self.length}, 样本数: {len(self.name_list)}, 视频列表: {self.name_list}, 分组: {self.groups}")

    def compute_durations_and_groups(self, name_list=None):
        if name_list is None:
            name_list = self.name_list
        durations = []
        for sample in name_list:
            image_list = sorted(glob.glob(os.path.join(self.frames_dir, '5fps_160w', sample, '*.jpg')))
            num_frames_total = len(image_list)
            durations.append(num_frames_total)

        durations = np.array(durations)
        sorted_indices = np.argsort(durations)
        group_size = len(durations) // self.num_duration_groups if self.num_duration_groups > 0 else len(durations)
        groups = np.zeros_like(durations, dtype=int)
        for i in range(self.num_duration_groups):
            start = i * group_size
            end = (i + 1) * group_size if i < self.num_duration_groups - 1 else len(durations)
            groups[sorted_indices[start:end]] = i

        return durations, groups

    

    def read_video_info(self):
        return self.__read_taskwise_video_info(self.cls)

    def __read_taskwise_video_info(self, task_name):
        frame_dir = os.path.join(self.frames_dir, '5fps_160w')
        annot_dir = os.path.join(self.info_dir, 'Skill')
        video_info_dict = {}
        for vidx in range(1, 25):
            vname = f'hei-chole{vidx}_{task_name}'
            video_frame_dir = os.path.join(frame_dir, vname)
            video_annot_dir = os.path.join(annot_dir, f'Hei-Chole{vidx}_{task_name}_Skill.csv')
            if not os.path.exists(video_frame_dir) or not os.path.exists(video_annot_dir):
                print(f"警告: {vname} 的数据缺失 (帧路径: {video_frame_dir}, 标注路径: {video_annot_dir})")
                continue
            num_frame = len([f for f in os.listdir(video_frame_dir) if '.jpg' in f])
            if num_frame == 0:
                print(f"警告: {vname} 的帧目录为空，跳过")
                continue
            with open(video_annot_dir, 'r') as annot_csv:
                csv_reader = csv.reader(annot_csv, delimiter=',')
                annot_scores = [int(s) for s in list(csv_reader)[0]]
            sum_score = sum(annot_scores)
            video_info_dict[vname] = {'num_frame': num_frame, 'sum_score': sum_score, 'sub_scores': annot_scores}
        return video_info_dict

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        data = {}
        sample_1 = self.name_list[item]
        data['video'] = self.load_video(sample_1)
        data['final_score'] = torch.tensor(self.video_info_dict[sample_1]['sum_score'])
        data['duration'] = torch.tensor(self.durations[item], dtype=torch.float32)
        data['group'] = torch.tensor(self.groups[item], dtype=torch.float32) if self.mode == 'train' else torch.tensor(0)

        if self.mode == 'train':
            file_list = [x for x in self.name_list if x != sample_1]
            if not file_list:
                raise ValueError("没有可用的配对样本")
            sample_2 = random.choice(file_list)
            target = {}
            target['video'] = self.load_video(sample_2)
            target['final_score'] = torch.tensor(self.video_info_dict[sample_2]['sum_score'])
            idx = self.name_list.index(sample_2)
            target['duration'] = torch.tensor(self.durations[idx], dtype=torch.float32)
            target['group'] = torch.tensor(self.groups[idx], dtype=torch.float32)
            # print(f"样本对: {sample_1} (group: {data['group']}), {sample_2} (group: {target['group']})")  # 调试
            return data, target
        else:
            train_file_list = self.train_name
            random.shuffle(train_file_list)
            durations, groups = self.compute_durations_and_groups(train_file_list)
            choosen_sample_list = train_file_list[:self.voter_number]
            
            target_list = []
            for item_name in choosen_sample_list:
                tmp = {}
                tmp['video'] = self.load_video(item_name)
                tmp['final_score'] = torch.tensor(self.video_info_dict[item_name]['sum_score'])
                idx = choosen_sample_list.index(item_name)
                tmp['duration'] = torch.tensor(durations[idx], dtype=torch.float32)
                tmp['group'] = torch.tensor(groups[idx], dtype=torch.float32)
                target_list.append(tmp)
            
            return data, target_list

    def delta(self):
        delta = []
        dataset = self.name_list.copy()
        for i in range(len(dataset)):
            for j in range(i + 1, len(dataset)):
                score_i = self.video_info_dict[dataset[i]]['sum_score']
                score_j = self.video_info_dict[dataset[j]]['sum_score']
                delta.append(abs(score_i - score_j))
        return delta