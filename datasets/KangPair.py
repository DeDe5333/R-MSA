import torch
import numpy as np
import os
import pickle
import random
import glob
import csv
import io
import torch.nn.functional as F

class KangPair_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform=None):
        random.seed(args.seed)
        self.cls = args.cls
        self.mode = subset
        self.info_dir = args.info_dir
        self.features_dir = args.features_dir
        self.label_file = os.path.join(self.info_dir, 'scores3.csv')  # CSV 文件名
        self.label_dict = self.read_csv_labels()  # 加载 CSV 标签
        self.split_index = args.fold  # 假设 args.fold 对应 split_index (0 到 3)
        self.load_fold(self.split_index)
        self.num_duration_groups = args.num_duration_groups
        self.voter_number = args.voter_number
        self.durations, self.groups = self.compute_durations_and_groups()
        # 动态计算 max_num_clips
        npy_files = sorted(glob.glob(os.path.join(self.features_dir, '*.npy')))
        max_num_clips = 0
        for f in npy_files:
            feature = np.load(f)
            max_num_clips = max(max_num_clips, feature.shape[0])
        self.num_frames = max_num_clips * 16  # 假设每剪辑 16 帧
        print(f"Dataset initialized for Kangduo with num_samples = {len(self.name_list)}")

    def read_csv_labels(self):
        if not os.path.exists(self.label_file):
            raise FileNotFoundError(f"Label file not found: {self.label_file}")
        label_dict = {}
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
        for encoding in encodings:
            try:
                with open(self.label_file, 'r', encoding=encoding) as f:
                    content = f.read()  # 读取整个文件内容
                    # 替换 NUL 字符
                    content = content.replace('\0', '')
                    reader = csv.reader(io.StringIO(content))
                    header = next(reader)  # 跳过标题行
                    print(f"Header: {header}")  # 调试
                    for i, row in enumerate(reader):
                        try:
                            if len(row) < 5:  # 确保有足够列
                                print(f"Skipped row {i} due to insufficient columns: {row}")
                                continue
                            seq_num = int(row[0])  # 第一列序号
                            avg_score = float(row[9]) if row[9].replace('.', '').replace('-', '').isdigit() else 1.0  # 第五列平均值
                            if not row[9] or isinstance(avg_score, str):  # 处理无效值
                                avg_score = 1.0  # 默认值
                            label_dict[seq_num] = avg_score
                        except (ValueError, IndexError) as e:
                            print(f"Skipped row {i} due to error: {e}, row: {row}")
                            continue
                    return label_dict  # 成功读取后返回
            except UnicodeDecodeError:
                continue  # 尝试下一个编码
        raise UnicodeDecodeError(f"Unable to decode {self.label_file} with tried encodings: {encodings}")

    def compute_durations_and_groups(self, name_list=None):
        if name_list is None:
            name_list = self.name_list
        durations = []
        for sample in name_list:
            feature_path = os.path.join(self.features_dir, sample + '.npy')
            if not os.path.exists(feature_path):
                raise FileNotFoundError(f"Feature file not found: {feature_path}")
            feature = np.load(feature_path)
            num_clips = feature.shape[0]  # [num_clips, 1024]
            num_frames_total = num_clips * 16  # 近似 duration
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

    def load_feature(self, video_file_name):
        feature_path = os.path.join(self.features_dir, video_file_name + '.npy')
        feature = np.load(feature_path)
        return torch.from_numpy(feature).float()  # [num_clips, 1024]

    def read_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def __len__(self):
        return len(self.name_list)

    # def load_fold(self, split_index):
    #     # 获取所有 .npy 文件名
    #     npy_files = sorted(glob.glob(os.path.join(self.features_dir, '*.npy')))
    #     self.name_list = []
    #     self.train_name = []

    #     # 提取序号和完整文件名映射
    #     seq_to_name = {}
    #     for f in npy_files:
    #         base_name = os.path.basename(f)[:-4]  # 去掉 .npy
    #         seq_num = int(base_name.split('.')[0])
    #         seq_to_name[seq_num] = base_name

    #     all_seq = sorted(seq_to_name.keys())
    #     if not all_seq:
    #         raise ValueError("No samples found in features_dir")

    #     # 均匀分割为 4 折
    #     num_samples = len(all_seq)
    #     fold_size = num_samples // 4  # 基础大小 27
    #     remainder = num_samples % 4  # 余数 2

    #     # 定义每个折的范围
    #     folds = []
    #     start = 0
    #     for i in range(4):
    #         end = start + fold_size + (1 if i < remainder else 0)  # 前两个折多分配 1 个
    #         folds.append(all_seq[start:end])
    #         start = end

    #     # 根据 split_index 分配验证集和训练集
    #     val_seq = folds[split_index]
    #     train_seq = [s for sublist in folds[:split_index] + folds[split_index + 1:] for s in sublist]

    #     # 构造 name_list 和 train_name
    #     if self.mode == 'train':
    #         self.name_list = [seq_to_name[s] for s in train_seq]
    #     else:
    #         self.name_list = [seq_to_name[s] for s in val_seq]

    #     self.train_name = [seq_to_name[s] for s in train_seq]

    def load_fold(self, split_index):
        # 获取所有 .npy 文件名
        npy_files = sorted(glob.glob(os.path.join(self.features_dir, '*.npy')))
        self.name_list = []
        self.train_name = []

        # 提取序号和完整文件名映射
        seq_to_name = {}
        for f in npy_files:
            base_name = os.path.basename(f)[:-4]  # 去掉 .npy
            seq_num = int(base_name.split('.')[0])
            seq_to_name[seq_num] = base_name

        all_seq = sorted(seq_to_name.keys())
        if not all_seq:
            raise ValueError("No samples found in features_dir")

        # 均匀分割为 4 折
        num_samples = len(all_seq)
        fold_size = num_samples // 4  # 基础大小 27
        remainder = num_samples % 4  # 余数 2

        # 定义每个折的范围
        folds = []
        start = 0
        for i in range(4):
            end = start + fold_size + (1 if i < remainder else 0)  # 前两个折多分配 1 个
            folds.append(all_seq[start:end])
            start = end

        # 根据 split_index 分配验证集和训练集
        val_seq = folds[split_index]
        train_seq = [s for sublist in folds[:split_index] + folds[split_index + 1:] for s in sublist]

        # 构造 name_list 和 train_name
        if self.mode == 'train':
            self.name_list = [seq_to_name[s] for s in train_seq]
        else:
            self.name_list = [seq_to_name[s] for s in val_seq]

        self.train_name = [seq_to_name[s] for s in train_seq]


    # def __getitem__(self, item):
    #     data = {}
    #     sample_1 = self.name_list[item]
    #     # 从文件名提取序号 (e.g., '1.13-04-28_video' -> 1)
    #     seq_num = int(sample_1.split('.')[0])
    #     data['feature'] = self.load_feature(sample_1)
    #     data['final_score'] = torch.tensor(self.label_dict.get(seq_num, 1.0))  # 默认 1.0 如果缺失
    #     data['duration'] = torch.tensor(self.durations[item], dtype=torch.float32)
    #     data['group'] = torch.tensor(self.groups[item], dtype=torch.float32)

    #     if self.mode == 'train':
    #         file_list = [x for x in self.name_list if x != sample_1]
    #         if not file_list:
    #             raise ValueError("No other samples available for pairing")
    #         sample_2 = random.choice(file_list)
    #         seq_num_2 = int(sample_2.split('.')[0])
    #         target = {}
    #         target['feature'] = self.load_feature(sample_2)
    #         target['final_score'] = torch.tensor(self.label_dict.get(seq_num_2, 1.0))
    #         idx = self.name_list.index(sample_2)
    #         target['duration'] = torch.tensor(self.durations[idx], dtype=torch.float32)
    #         target['group'] = torch.tensor(self.groups[idx], dtype=torch.float32)
    #         return data, target
    #     else:
    #         train_file_list = self.train_name[:]
    #         random.shuffle(train_file_list)
    #         choosen_sample_list = train_file_list[:self.voter_number]
    #         durations, groups = self.compute_durations_and_groups(choosen_sample_list)
    #         target_list = []
    #         for idx, item_name in enumerate(choosen_sample_list):
    #             tmp = {}
    #             seq_num_tmp = int(item_name.split('.')[0])
    #             tmp['feature'] = self.load_feature(item_name)
    #             tmp['final_score'] = torch.tensor(self.label_dict.get(seq_num_tmp, 1.0))
    #             tmp['duration'] = torch.tensor(durations[idx], dtype=torch.float32)
    #             tmp['group'] = torch.tensor(groups[idx], dtype=torch.float32)
    #             target_list.append(tmp)
    #         return data, target_list

    

    def __getitem__(self, item):
        data = {}
        sample_1 = self.name_list[item]
        seq_num = int(sample_1.split('.')[0])
        feature = self.load_feature(sample_1)  # [num_clips, 1024]
        # 标准化到 num_frames // 16
        target_clips = self.num_frames // 16  # 从 args.num_frames=160 推导
        if feature.size(0) > target_clips:
            feature = feature[:target_clips, :]
        else:
            feature = F.pad(feature, (0, 0, 0, target_clips - feature.size(0)))
        data['feature'] = feature
        data['final_score'] = torch.tensor(self.label_dict.get(seq_num, 1.0))
        data['duration'] = torch.tensor(self.durations[item], dtype=torch.float32)
        data['group'] = torch.tensor(self.groups[item], dtype=torch.float32)

        if self.mode == 'train':
            file_list = [x for x in self.name_list if x != sample_1]
            if not file_list:
                raise ValueError("No other samples available for pairing")
            sample_2 = random.choice(file_list)
            seq_num_2 = int(sample_2.split('.')[0])
            feature_target = self.load_feature(sample_2)
            if feature_target.size(0) > target_clips:
                feature_target = feature_target[:target_clips, :]
            else:
                feature_target = F.pad(feature_target, (0, 0, 0, target_clips - feature_target.size(0)))
            target = {
                'feature': feature_target,
                'final_score': torch.tensor(self.label_dict.get(seq_num_2, 1.0)),
                'duration': torch.tensor(self.durations[self.name_list.index(sample_2)], dtype=torch.float32),
                'group': torch.tensor(self.groups[self.name_list.index(sample_2)], dtype=torch.float32)
            }
            return data, target
        else:
            train_file_list = self.train_name[:]
            random.shuffle(train_file_list)
            choosen_sample_list = train_file_list[:self.voter_number]
            durations, groups = self.compute_durations_and_groups(choosen_sample_list)
            target_list = []
            for idx, item_name in enumerate(choosen_sample_list):
                tmp = {}
                seq_num_tmp = int(item_name.split('.')[0])
                feature_tmp = self.load_feature(item_name)
                if feature_tmp.size(0) > target_clips:
                    feature_tmp = feature_tmp[:target_clips, :]
                else:
                    feature_tmp = F.pad(feature_tmp, (0, 0, 0, target_clips - feature_tmp.size(0)))
                tmp['feature'] = feature_tmp
                tmp['final_score'] = torch.tensor(self.label_dict.get(seq_num_tmp, 1.0))
                tmp['duration'] = torch.tensor(durations[idx], dtype=torch.float32)
                tmp['group'] = torch.tensor(groups[idx], dtype=torch.float32)
                target_list.append(tmp)
            return data, target_list

    def delta(self):
        delta = []
        dataset = self.name_list.copy()
        for i in range(len(dataset)):
            for j in range(i + 1, len(dataset)):
                name_i = dataset[i].split('.')[0]
                name_j = dataset[j].split('.')[0]
                score_i = self.label_dict.get(int(name_i), 1.0)
                score_j = self.label_dict.get(int(name_j), 1.0)
                delta.append(abs(score_i - score_j))
        return delta