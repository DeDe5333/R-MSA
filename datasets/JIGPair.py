# import torch
# import numpy as np
# import os
# import pickle
# import random
# import glob
# # from os.path import join
# from PIL import Image


# class JIGPair_Dataset(torch.utils.data.Dataset):
#     def __init__(self, args, subset, transform):
#         random.seed(args.seed)
#         self.transforms = transform
#         self.cls = args.cls
#         self.mode = subset
#         # file path
#         self.info_dir = args.info_dir
#         self.label_dict = self.read_pickle(os.path.join(self.info_dir, 'label.pkl'))
#         self.frames_dir = args.frames_dir
#         # setting
#         self.temporal_shift = [args.temporal_shift_min, args.temporal_shift_max]
#         self.voter_number = args.voter_number
#         self.length = args.num_frames
#         self.load_fold(0)
#         self.num_duration_groups = args.num_duration_groups
#         # build difficulty dict ( difficulty of each action, the cue to choose exemplar)
#         # 计算每个样本的原始时长和分组
#         self.durations, self.groups = self.compute_durations_and_groups()
#         print(f"Dataset initialized with self.length = {self.length}")  # Debug log
        


#     def compute_durations_and_groups(self):
#         durations = []
#         for sample in self.name_list:
#             image_list = sorted(glob.glob(os.path.join(self.frames_dir, sample, '*.jpg')))
#             num_frames_total = len(image_list)
#             durations.append(num_frames_total)

#         # 使用排序分组法
#         durations = np.array(durations)
#         sorted_indices = np.argsort(durations)
#         group_size = len(durations) // self.num_duration_groups
#         groups = np.zeros_like(durations, dtype=int)
#         for i in range(self.num_duration_groups):
#             start = i * group_size
#             end = (i + 1) * group_size if i < self.num_duration_groups - 1 else len(durations)
#             groups[sorted_indices[start:end]] = i

#         return durations, groups

#     def load_video(self, video_file_name):
#         # print(video_file_name)
#         # print(os.path.join(self.data_root, str('{:02d}'.format(video_file_name[0]))))
#         # image_list = sorted(os.listdir(os.path.join(self.frames_dir, video_file_name)))
#         image_list = sorted((glob.glob(os.path.join(self.frames_dir, video_file_name, '*.jpg'))))
#         frame_list = np.linspace(0, len(image_list) - 1, num=self.length, dtype=np.int32)
#         image_list = [image_list[frame_list[i]] for i in range(self.length)]
#         # if phase == 'train':
#         #     temporal_aug_shift = random.randint(self.temporal_shift[0], self.temporal_shift[1])
#         #     end_frame = end_frame + temporal_aug_shift
#         # start_frame = end_frame - self.length
#         video = [Image.open(image_list[i]) for i in range(self.length)]
#         return self.transforms(video)

#     # def load_video(self, video_file_name):
#     #     image_list = sorted(glob.glob(os.path.join(self.frames_dir, video_file_name, '*.jpg')))
#     #     frame_list = np.linspace(0, len(image_list) - 1, num=self.length, dtype=np.int32)
#     #     image_list = [image_list[frame_list[i]] for i in range(self.length)]
#     #     video = [Image.open(image_list[i]).convert('RGB') for i in range(self.length)]  # 确保 RGB 格式
#     #     video = [np.array(img) for img in video]  # 转换为 [H, W, 3]
#     #     video = np.stack(video, axis=0)  # [T, H, W, 3]
#     #     video = video.transpose(0, 3, 1, 2)  # [T, 3, H, W]
#     #     video = torch.from_numpy(video).float()  # [T, 3, H, W]
#     #     if self.transforms:
#     #         video = self.transforms(video)  # 应用 transform，确保输出 [3, T, H, W]
#     #     return video

#     def read_pickle(self, pickle_path):
#         with open(pickle_path, 'rb') as f:
#             pickle_data = pickle.load(f)
#         return pickle_data

#     def __len__(self):
#         return len(self.name_list)

#     def load_fold(self, fold):
#         with open(os.path.join(self.info_dir, 'splits.pkl'), 'rb') as f:
#             cv_file = pickle.load(f)  # info of cross validation

#         self.name_list = []
#         self.train_name = []
#         all_list = cv_file[self.cls]
#         folds = [0, 1, 2, 3]
#         train_folds = [0, 1, 2, 3]
#         train_folds.pop(fold)
#         if self.mode == 'train':
#             folds.pop(fold)
#         else:
#             folds = [fold]
#         for fold in train_folds:
#             for vid in all_list[fold]:
#                 self.train_name.append(vid + '_capture1')  # only loads left view

#         for fold in folds:
#             for vid in all_list[fold]:
#                 self.name_list.append(vid + '_capture1')  # only loads left view

#     # def __getitem__(self, item):
#     #     data = {}
#     #     sample_1 = self.name_list[item]
#     #     # anchor_idx = item
#     #     data['video'] = self.load_video(sample_1)
#     #     name = sample_1[:-9]  # *_capture1
#     #     data['final_score'] = torch.tensor(self.label_dict[name]).sum()
#     #     # data['group'] = torch.tensor(self.groups[anchor_idx])
#     #     # data['duration'] = torch.tensor(self.durations[anchor_idx])


#     #     cp = data['final_score']
#     #     if self.mode == 'train':
#     #         file_list = self.name_list.copy()
#     #         if len(file_list) > 1:
#     #             file_list.pop(file_list.index(sample_1))
#     #         idx = random.randint(0, len(file_list) - 1)
#     #         sample_2 = file_list[idx]
#     #         target = {}
#     #         target['video'] = self.load_video(sample_2)
#     #         name2 = sample_2[:-9]  # *_capture1
#     #         target['final_score'] = torch.tensor(self.label_dict[name2]).sum()

#     #         return data, target
#     #     else:
#     #         train_file_list = self.train_name
#     #         random.shuffle(train_file_list)
#     #         choosen_sample_list = train_file_list[:self.voter_number]
#     #         target_list = []
#     #         for item in choosen_sample_list:
#     #             tmp = {}
#     #             tmp['video'] = self.load_video(item)
#     #             name2 = item[:-9]
#     #             tmp['final_score'] = torch.tensor(self.label_dict[name2]).sum()
#     #             target_list.append(tmp)
#     #         return data, target_list
#     def __getitem__(self, item):
#         data = {}
#         sample_1 = self.name_list[item]
#         data['video'] = self.load_video(sample_1)
#         name = sample_1[:-9]  # 去掉 '_capture1'，例如 'Suturing_E01'
#         data['final_score'] = torch.tensor(self.label_dict[name]).sum()
#         # 添加 duration 属性
#         data['duration'] = torch.tensor(self.durations[item], dtype=torch.float32)
#         data['group'] = torch.tensor(self.groups[item], dtype=torch.float32)

#         cp = data['final_score']
#         if self.mode == 'train':
#             file_list = self.name_list.copy()
#             if len(file_list) > 1:
#                 file_list.pop(file_list.index(sample_1))
#             idx = random.randint(0, len(file_list) - 1)
#             sample_2 = file_list[idx]
#             target = {}
#             target['video'] = self.load_video(sample_2)
#             name2 = sample_2[:-9]
#             target['final_score'] = torch.tensor(self.label_dict[name2]).sum()
#             # 添加 duration 属性
#             target['duration'] = torch.tensor(self.durations[idx], dtype=torch.float32)
#             target['group'] = torch.tensor(self.groups[idx], dtype=torch.float32)

#             return data, target
#         else:
#             train_file_list = self.train_name
#             random.shuffle(train_file_list)
#             choosen_sample_list = train_file_list[:self.voter_number]
#             target_list = []
#             for item_name in choosen_sample_list:
#                 tmp = {}
#                 tmp['video'] = self.load_video(item_name)
#                 name2 = item_name[:-9]
#                 tmp['final_score'] = torch.tensor(self.label_dict[name2]).sum()
#                 # 添加 duration 属性
#                 # 查找 item_name 在 self.name_list 中的索引
#                 idx = self.name_list.index(item_name) if item_name in self.name_list else self.train_name.index(item_name)
#                 tmp['duration'] = torch.tensor(self.durations[idx], dtype=torch.float32)
#                 tmp['group'] = torch.tensor(self.groups[idx], dtype=torch.float32)
#                 target_list.append(tmp)
#             # for item_name in choosen_sample_list:
#             #     tmp = {}
#             #     tmp['video'] = self.load_video(item_name)
#             #     name2 = item_name[:-9]
#             #     tmp['final_score'] = torch.tensor(self.label_dict[name2]).sum()

#             #     # ✅ 重新计算 duration
#             #     image_list = sorted(glob.glob(os.path.join(self.frames_dir, item_name, '*.jpg')))
#             #     duration = len(image_list)
#             #     tmp['duration'] = torch.tensor(duration, dtype=torch.float32)
#             #     # ✅ 可选：重新计算分组（如果需要）
#             #     group = np.searchsorted(np.linspace(np.min(self.durations), np.max(self.durations), self.num_duration_groups + 1), duration, side='right') - 1
#             #     group = min(group, self.num_duration_groups - 1)
#             #     tmp['group'] = torch.tensor(group, dtype=torch.float32)

#             #     target_list.append(tmp)

#             return data, target_list
    
#     def delta(self):
#         delta = []
#         dataset = self.name_list.copy()  # 使用当前模式下的样本列表
#         for i in range(len(dataset)):
#             for j in range(i + 1, len(dataset)):
#                 # 获取样本 i 和 j 的名称（去掉 _capture1）
#                 name_i = dataset[i][:-9]
#                 name_j = dataset[j][:-9]
#                 # 计算 final_score
#                 score_i = torch.tensor(self.label_dict[name_i]).sum().item()
#                 score_j = torch.tensor(self.label_dict[name_j]).sum().item()
#                 # 计算原始差值
#                 delta.append(abs(score_i - score_j))
#         return delta


import torch
import numpy as np
import os
import pickle
import random
import glob
from PIL import Image

class JIGPair_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        random.seed(args.seed)
        self.transforms = transform
        self.cls = args.cls
        self.mode = subset
        # file path
        self.info_dir = args.info_dir
        self.label_dict = self.read_pickle(os.path.join(self.info_dir, 'label.pkl'))
        self.frames_dir = args.frames_dir
        self.fold = args.fold
        self.cv_method = getattr(args, 'cv_method', '4fold')  # 默认使用4折交叉验证
        # 支持指定Experimental_setup目录路径（用于LOSO从文件读取划分）
        self.experimental_setup_dir = args.experimental_setup_dir
        
        # setting
        self.temporal_shift = [args.temporal_shift_min, args.temporal_shift_max]
        self.voter_number = args.voter_number
        self.length = args.num_frames
        self.load_fold(self.fold)
        self.num_duration_groups = args.num_duration_groups
        # 计算每个样本的原始时长和分组（基于 self.name_list）
        self.durations, self.groups = self.compute_durations_and_groups()
        print(f"Dataset initialized with self.length = {self.length}, cv_method = {self.cv_method}")  # Debug log

    def compute_durations_and_groups(self, name_list=None):
        if name_list is None:
            name_list = self.name_list
        durations = []
        for sample in name_list:
            image_list = sorted(glob.glob(os.path.join(self.frames_dir, sample, '*.jpg')))
            num_frames_total = len(image_list)
            durations.append(num_frames_total)

        # 使用排序分组法
        durations = np.array(durations)
        sorted_indices = np.argsort(durations)
        group_size = len(durations) // self.num_duration_groups if self.num_duration_groups > 0 else len(durations)
        groups = np.zeros_like(durations, dtype=int)
        for i in range(self.num_duration_groups):
            start = i * group_size
            end = (i + 1) * group_size if i < self.num_duration_groups - 1 else len(durations)
            groups[sorted_indices[start:end]] = i

        return durations, groups

    def load_video(self, video_file_name):
        image_list = sorted(glob.glob(os.path.join(self.frames_dir, video_file_name, '*.jpg')))
        frame_list = np.linspace(0, len(image_list) - 1, num=self.length, dtype=np.int32)
        image_list = [image_list[frame_list[i]] for i in range(self.length)]
        video = [Image.open(image_list[i]) for i in range(self.length)]
        return self.transforms(video)

    def read_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def __len__(self):
        return len(self.name_list)

    def extract_user_id(self, video_name):
        """
        从视频名称中提取用户ID，例如 'Suturing_B001_T1' -> 'B'
        LOUO应该按用户字母（B-I）分组，而不是按完整的用户ID（B001, B002等）分组
        JIGSAWS格式通常是: Task_UserID_TrialID
        用户ID以 B, C, D, E, F, G, H, I 开头，后面跟数字（trial编号），例如 B001, B002, C001 等
        LOUO中，B001, B002, B003, B004, B005 都属于同一个用户B，应该只提取字母部分
        """
        parts = video_name.split('_')
        if len(parts) >= 2:
            # 第二个部分通常是用户ID（第一个是任务名，如Suturing, Needle_Passing, Knot_Tying）
            user_id_candidate = parts[1]
            # JIGSAWS用户ID格式：以B-I开头的字母，后跟数字（trial编号）
            if len(user_id_candidate) >= 2 and user_id_candidate[0].upper() in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'] and user_id_candidate[1:].isdigit():
                # 只返回用户字母部分（B, C, D, E, F, G, H, I），不包含trial编号
                return user_id_candidate[0].upper()
            # 如果第二个部分不是用户ID，尝试查找符合用户ID格式的部分
            for part in parts[1:]:  # 跳过第一个部分（任务名）
                if len(part) >= 2 and part[0].upper() in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'] and part[1:].isdigit():
                    return part[0].upper()
        return None
    
    def extract_task_name(self, video_name):
        """从视频名称中提取任务名称，例如 'Suturing_B001_T1' -> 'Suturing'"""
        # JIGSAWS格式通常是: Task_UserID_TrialID
        parts = video_name.split('_')
        if len(parts) >= 1:
            return parts[0]
        return None

    def load_fold(self, fold):
        if self.cv_method == 'LOUO':
            self.load_fold_louo(fold)
        elif self.cv_method == 'LOSO':
            self.load_fold_loso(fold)
        else:  # 默认4折交叉验证
            self.load_fold_4fold(fold)
    
    def load_fold_4fold(self, fold):
        """原有的4折交叉验证方法"""
        with open(os.path.join(self.info_dir, 'splits.pkl'), 'rb') as f:
            cv_file = pickle.load(f)  # info of cross validation

        self.name_list = []
        self.train_name = []
        all_list = cv_file[self.cls]
        folds = [0, 1, 2, 3]
        train_folds = [0, 1, 2, 3]
        train_folds.pop(fold)
        if self.mode == 'train':
            folds.pop(fold)
        else:
            folds = [fold]
        for fold in train_folds:
            for vid in all_list[fold]:
                self.train_name.append(vid + '_capture1')  # only loads left view

        for fold in folds:
            for vid in all_list[fold]:
                self.name_list.append(vid + '_capture1')  # only loads left view
    
    def load_fold_louo(self, fold):
        """
        Leave-One-User-Out交叉验证：每次留出一个用户的所有数据作为测试集
        参考JIGSAWS标准实现：fold范围应该是0-7（对应8个用户B001-I001）
        """
        with open(os.path.join(self.info_dir, 'splits.pkl'), 'rb') as f:
            cv_file = pickle.load(f)  # info of cross validation

        self.name_list = []
        self.train_name = []
        
        # 尝试从所有任务中收集视频（如果splits.pkl包含多个任务）
        all_videos = []
        if isinstance(cv_file, dict):
            # 如果splits.pkl包含多个任务，可以选择是否使用所有任务
            # 这里先使用指定的cls（任务），如果需要所有任务，可以遍历所有key
            task_keys = [self.cls] if self.cls in cv_file else list(cv_file.keys())
            for task_key in task_keys:
                if task_key in cv_file:
                    all_list = cv_file[task_key]
                    for fold_list in all_list:
                        for vid in fold_list:
                            all_videos.append(vid)
        else:
            # 如果splits.pkl是其他格式，使用原有逻辑
            all_list = cv_file[self.cls] if isinstance(cv_file, dict) and self.cls in cv_file else cv_file
            if isinstance(all_list, list):
                for fold_list in all_list:
                    if isinstance(fold_list, list):
                        for vid in fold_list:
                            all_videos.append(vid)
                    else:
                        all_videos.append(fold_list)
        
        # 按用户ID分组（JIGSAWS标准：B001, C001, D001, E001, F001, G001, H001, I001）
        user_groups = {}
        failed_videos = []
        for vid in all_videos:
            user_id = self.extract_user_id(vid)
            if user_id is None:
                failed_videos.append(vid)
                continue  # 跳过无法提取用户ID的视频，而不是使用整个名称
            if user_id not in user_groups:
                user_groups[user_id] = []
            user_groups[user_id].append(vid)
        
        # 获取所有用户ID并按标准顺序排序
        # JIGSAWS标准用户顺序：B, C, D, E, F, G, H, I (8个用户)
        standard_user_order = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        found_users = set(user_groups.keys())
        # 先按标准顺序排序，然后添加其他未在标准列表中的用户
        user_ids = [uid for uid in standard_user_order if uid in found_users]
        other_users = sorted([uid for uid in found_users if uid not in standard_user_order])
        user_ids.extend(other_users)
        
        if not user_ids:
            raise ValueError(f"无法从数据中找到任何有效的用户ID！检查数据格式是否正确。")
        
        if fold >= len(user_ids):
            raise ValueError(f"Fold {fold} 超出范围。找到的用户数: {len(user_ids)} ({', '.join(user_ids)})。期望8个用户(B-I)")
        
        # 选择测试用户（fold是0-based，对应8个用户）
        test_user = user_ids[fold]
        
        # 分配训练集和测试集
        for user_id, videos in user_groups.items():
            for vid in videos:
                vid_name = vid + '_capture1'
                if user_id == test_user:
                    # 测试集
                    if self.mode == 'test':
                        self.name_list.append(vid_name)
                    # 注意：测试集用户的数据不加入train_name，避免数据泄露
                else:
                    # 训练集
                    if self.mode == 'train':
                        self.name_list.append(vid_name)
                    self.train_name.append(vid_name)
        
        # 打印调试信息
        train_users = [uid for uid in user_ids if uid != test_user]
        print(f"LOUO Fold {fold} (0-based, 对应第{fold+1}个用户): Test user = {test_user}, Train users = {train_users}")
        print(f"  Total users found: {len(user_ids)}/{len(standard_user_order)} ({', '.join(user_ids)})")
        print(f"  Train samples: {len(self.train_name)}, Test samples: {len(self.name_list) if self.mode == 'test' else 0}")
        
        # 检查缺失的用户
        missing_users = [uid for uid in standard_user_order if uid not in found_users]
        if missing_users:
            print(f"  警告：以下标准用户未在数据中找到: {missing_users}")
            print(f"  这可能是因为当前任务({self.cls})只包含部分用户的数据")
        
        # 检查是否有未正确提取用户ID的视频
        if failed_videos:
            print(f"  警告：以下视频无法提取用户ID (共{len(failed_videos)}个):")
            for vid in failed_videos[:5]:  # 只显示前5个
                print(f"    - {vid}")
            if len(failed_videos) > 5:
                print(f"    ... 还有 {len(failed_videos) - 5} 个视频")
    
    def load_fold_loso(self, fold):
        """
        Leave-One-Subject-Out交叉验证：使用SuperTrialOut划分（5个fold）
        参考JIGSAWS标准实现：fold范围应该是0-4（对应5个SuperTrial）
        优先尝试从Experimental_setup目录读取划分文件，如果不存在则使用程序化划分
        """
        self.name_list = []
        self.train_name = []
        
        # 尝试从Experimental_setup目录读取划分文件（参考代码的方式）
        # split_index从1开始，fold从0开始，所以需要+1
        split_index = fold + 1
        experimental_setup_dir = self.experimental_setup_dir
        if experimental_setup_dir is None:
            # 尝试从info_dir的父目录查找Experimental_setup
            info_dir_parent = os.path.dirname(self.info_dir) if self.info_dir else None
            if info_dir_parent:
                experimental_setup_dir = os.path.join(info_dir_parent, 'Experimental_setup')
            else:
                experimental_setup_dir = None
        
        # 检查是否存在Experimental_setup目录结构
        split_file_path = None
        if experimental_setup_dir and os.path.exists(experimental_setup_dir):
            # 参考代码的路径结构：LOSO使用SuperTrialOut（5个fold），而不是UserOut（8个fold）
            # Experimental_setup/task_name/unBalanced/GestureRecognition/SuperTrialOut/{split_index}_Out/itr_1/Train.txt或Test.txt
            # 或者：Experimental_setup/task_name/FourFolds/{split_index}_Out/itr_1/Train.txt或Test.txt
            split_filename = 'Train.txt' if self.mode == 'train' else 'Test.txt'
            # 尝试多个可能的任务名称（因为self.cls可能是不同的格式）
            possible_task_names = [self.cls, self.cls.replace('_', ''), self.cls.lower(), self.cls.upper()]
            
            # 优先尝试SuperTrialOut（LOSO标准）
            for task_name in possible_task_names:
                potential_path = os.path.join(experimental_setup_dir, task_name, 'unBalanced', 
                                            'GestureRecognition', 'SuperTrialOut', 
                                            f'{split_index}_Out', 'itr_1', split_filename)
                if os.path.exists(potential_path):
                    split_file_path = potential_path
                    break
            
            # 如果SuperTrialOut不存在，尝试FourFolds路径
            if split_file_path is None:
                for task_name in possible_task_names:
                    potential_path = os.path.join(experimental_setup_dir, task_name, 'FourFolds', 
                                                f'{split_index}_Out', 'itr_1', split_filename)
                    if os.path.exists(potential_path):
                        split_file_path = potential_path
                        break
        
        # 如果找到了划分文件，使用文件中的划分
        if split_file_path and os.path.exists(split_file_path):
            self._load_fold_loso_from_file(split_file_path)
        else:
            # 否则使用程序化划分（类似LOUO的方式）
            self._load_fold_loso_programmatic(fold)
    
    def _load_fold_loso_from_file(self, split_file_path):
        """
        从Experimental_setup的划分文件中加载LOSO划分
        参考代码的格式：每行包含视频名称和帧范围信息
        """
        # 如果是测试模式，需要先读取Train.txt来填充train_name
        if self.mode == 'test':
            train_file_path = split_file_path.replace('Test.txt', 'Train.txt')
            if os.path.exists(train_file_path):
                with open(train_file_path, 'r') as train_file:
                    for line in train_file.readlines():
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(' ' * 11)
                        if len(parts) >= 2:
                            str1, str2 = parts[0], parts[1]
                            video_name = str2.replace('.txt', '')
                            vid_name = video_name + '_capture1'
                            self.train_name.append(vid_name)
        
        # 读取当前模式对应的文件（train模式读Train.txt，test模式读Test.txt）
        with open(split_file_path, 'r') as split_file:
            for line in split_file.readlines():
                line = line.strip()
                if not line:
                    continue
                # 参考代码格式：str1和str2用11个空格分隔
                # str1格式：video_name_startframe_endframe.txt
                # str2格式：video_name.txt
                parts = line.split(' ' * 11)
                if len(parts) >= 2:
                    str1, str2 = parts[0], parts[1]
                    video_name = str2.replace('.txt', '')
                    vid_name = video_name + '_capture1'
                    
                    if self.mode == 'train':
                        self.name_list.append(vid_name)
                        self.train_name.append(vid_name)
                    else:  # test mode
                        self.name_list.append(vid_name)
        
        print(f"LOSO: Loaded splits from file: {split_file_path}")
        print(f"  Train samples: {len(self.train_name)}, Test samples: {len(self.name_list) if self.mode == 'test' else 0}")
    
    def _load_fold_loso_programmatic(self, fold):
        """
        程序化生成LOSO划分（按用户分组，类似LOUO）
        这是当Experimental_setup目录不存在时的回退方案
        """
        with open(os.path.join(self.info_dir, 'splits.pkl'), 'rb') as f:
            cv_file = pickle.load(f)  # info of cross validation

        # 尝试从所有任务中收集视频（如果splits.pkl包含多个任务）
        all_videos = []
        if isinstance(cv_file, dict):
            # 如果splits.pkl包含多个任务，可以选择是否使用所有任务
            # 这里先使用指定的cls（任务），如果需要所有任务，可以遍历所有key
            task_keys = [self.cls] if self.cls in cv_file else list(cv_file.keys())
            for task_key in task_keys:
                if task_key in cv_file:
                    all_list = cv_file[task_key]
                    for fold_list in all_list:
                        for vid in fold_list:
                            all_videos.append(vid)
        else:
            # 如果splits.pkl是其他格式，使用原有逻辑
            all_list = cv_file[self.cls] if isinstance(cv_file, dict) and self.cls in cv_file else cv_file
            if isinstance(all_list, list):
                for fold_list in all_list:
                    if isinstance(fold_list, list):
                        for vid in fold_list:
                            all_videos.append(vid)
                    else:
                        all_videos.append(fold_list)
        
        # 按用户ID分组（JIGSAWS标准：B, C, D, E, F, G, H, I）
        user_groups = {}
        failed_videos = []
        for vid in all_videos:
            user_id = self.extract_user_id(vid)
            if user_id is None:
                failed_videos.append(vid)
                continue  # 跳过无法提取用户ID的视频
            if user_id not in user_groups:
                user_groups[user_id] = []
            user_groups[user_id].append(vid)
        
        # 获取所有用户ID并按标准顺序排序
        # JIGSAWS标准用户顺序：B, C, D, E, F, G, H, I (8个用户)
        standard_user_order = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        found_users = set(user_groups.keys())
        # 先按标准顺序排序，然后添加其他未在标准列表中的用户
        user_ids = [uid for uid in standard_user_order if uid in found_users]
        other_users = sorted([uid for uid in found_users if uid not in standard_user_order])
        user_ids.extend(other_users)
        
        if not user_ids:
            raise ValueError(f"无法从数据中找到任何有效的用户ID！检查数据格式是否正确。")
        
        if fold >= len(user_ids):
            raise ValueError(f"Fold {fold} 超出范围。找到的用户数: {len(user_ids)} ({', '.join(user_ids)})。期望8个用户(B-I)")
        
        # 选择测试用户（fold是0-based，对应8个用户）
        test_user = user_ids[fold]
        
        # 分配训练集和测试集
        for user_id, videos in user_groups.items():
            for vid in videos:
                vid_name = vid + '_capture1'
                if user_id == test_user:
                    # 测试集
                    if self.mode == 'test':
                        self.name_list.append(vid_name)
                    # 注意：测试集用户的数据不加入train_name，避免数据泄露
                else:
                    # 训练集
                    if self.mode == 'train':
                        self.name_list.append(vid_name)
                    self.train_name.append(vid_name)
        
        # 打印调试信息
        train_users = [uid for uid in user_ids if uid != test_user]
        print(f"LOSO Fold {fold} (0-based, 对应第{fold+1}个用户): Test user = {test_user}, Train users = {train_users}")
        print(f"  Total users found: {len(user_ids)}/{len(standard_user_order)} ({', '.join(user_ids)})")
        print(f"  Train samples: {len(self.train_name)}, Test samples: {len(self.name_list) if self.mode == 'test' else 0}")
        
        # 检查缺失的用户
        missing_users = [uid for uid in standard_user_order if uid not in found_users]
        if missing_users:
            print(f"  警告：以下标准用户未在数据中找到: {missing_users}")
            print(f"  这可能是因为当前任务({self.cls})只包含部分用户的数据")
        
        # 检查是否有未正确提取用户ID的视频
        if failed_videos:
            print(f"  警告：以下视频无法提取用户ID (共{len(failed_videos)}个):")
            for vid in failed_videos[:5]:  # 只显示前5个
                print(f"    - {vid}")
            if len(failed_videos) > 5:
                print(f"    ... 还有 {len(failed_videos) - 5} 个视频")

    # def __getitem__(self, item):
    #     data = {}
    #     sample_1 = self.name_list[item]
    #     data['video'] = self.load_video(sample_1)
    #     name = sample_1[:-9]  # 去掉 '_capture1'，例如 'Suturing_E01'
    #     data['final_score'] = torch.tensor(self.label_dict[name]).sum()
    #     # 添加 duration 属性
    #     data['duration'] = torch.tensor(self.durations[item], dtype=torch.float32)
    #     data['group'] = torch.tensor(self.groups[item], dtype=torch.float32)

    #     cp = data['final_score']
    #     if self.mode == 'train':
    #         file_list = self.name_list.copy()
    #         if len(file_list) > 1:
    #             file_list.pop(file_list.index(sample_1))
    #         idx = random.randint(0, len(file_list) - 1)
    #         sample_2 = file_list[idx]
    #         target = {}
    #         target['video'] = self.load_video(sample_2)
    #         name2 = sample_2[:-9]
    #         target['final_score'] = torch.tensor(self.label_dict[name2]).sum()
    #         # 添加 duration 属性
    #         target['duration'] = torch.tensor(self.durations[idx], dtype=torch.float32)
    #         target['group'] = torch.tensor(self.groups[idx], dtype=torch.float32)
    #         return data, target
    #     else:
    #         train_file_list = self.train_name
    #         random.shuffle(train_file_list)
    #         choosen_sample_list = train_file_list[:self.voter_number]
    #         # 为 choosen_sample_list 重新计算 durations 和 groups
    #         durations, groups = self.compute_durations_and_groups(choosen_sample_list)
    #         target_list = []
    #         for item_name in choosen_sample_list:
    #             tmp = {}
    #             tmp['video'] = self.load_video(item_name)
    #             name2 = item_name[:-9]
    #             tmp['final_score'] = torch.tensor(self.label_dict[name2]).sum()
    #             # 添加 duration 属性
    #             idx = choosen_sample_list.index(item_name)
    #             tmp['duration'] = torch.tensor(durations[idx], dtype=torch.float32)
    #             tmp['group'] = torch.tensor(groups[idx], dtype=torch.float32)
    #             target_list.append(tmp)
    #         return data, target_list

    def __getitem__(self, item):
        data = {}
        sample_1 = self.name_list[item]
        data['video'] = self.load_video(sample_1)
        name = sample_1[:-9]  # 去掉 '_capture1'，例如 'Suturing_E01'
        data['final_score'] = torch.tensor(self.label_dict[name]).sum()
        # 添加 duration 属性
        data['duration'] = torch.tensor(self.durations[item], dtype=torch.float32)
        data['group'] = torch.tensor(self.groups[item], dtype=torch.float32)

        cp = data['final_score']
        # if self.mode == 'train':
        #     file_list = self.name_list.copy()
        #     if len(file_list) > 1:
        #         file_list.pop(file_list.index(sample_1))
        #     idx = random.randint(0, len(file_list) - 1)
        #     sample_2 = file_list[idx]
        #     target = {}
        #     target['video'] = self.load_video(sample_2)
        #     name2 = sample_2[:-9]
        #     target['final_score'] = torch.tensor(self.label_dict[name2]).sum()
        #     # 添加 duration 属性
        #     target['duration'] = torch.tensor(self.durations[idx], dtype=torch.float32)
        #     target['group'] = torch.tensor(self.groups[idx], dtype=torch.float32)
        #     return data, target
        if self.mode == 'train':
            file_list = [x for x in self.name_list if x != sample_1]  # 更简洁的方式
            if not file_list:
                raise ValueError("No other samples available for pairing")
            sample_2 = random.choice(file_list)  # 使用 random.choice 更简洁
            target = {}
            target['video'] = self.load_video(sample_2)
            name2 = sample_2[:-9]
            target['final_score'] = torch.tensor(self.label_dict[name2]).sum()
            idx = self.name_list.index(sample_2)  # 获取 sample_2 在 name_list 中的索引
            target['duration'] = torch.tensor(self.durations[idx], dtype=torch.float32)
            target['group'] = torch.tensor(self.groups[idx], dtype=torch.float32)
            return data, target
        else:
            train_file_list = self.train_name
            random.shuffle(train_file_list)
            # 为 choosen_sample_list 重新计算 durations 和 groups
            durations, groups = self.compute_durations_and_groups(train_file_list)
            choosen_sample_list = train_file_list[:self.voter_number]
            
            target_list = []
            for item_name in choosen_sample_list:
                tmp = {}
                tmp['video'] = self.load_video(item_name)
                name2 = item_name[:-9]
                tmp['final_score'] = torch.tensor(self.label_dict[name2]).sum()
                # 添加 duration 属性
                idx = choosen_sample_list.index(item_name)
                tmp['duration'] = torch.tensor(durations[idx], dtype=torch.float32)
                tmp['group'] = torch.tensor(groups[idx], dtype=torch.float32)
                target_list.append(tmp)
            return data, target_list

    def delta(self):
        delta = []
        dataset = self.name_list.copy()  # 使用当前模式下的样本列表
        for i in range(len(dataset)):
            for j in range(i + 1, len(dataset)):
                # 获取样本 i 和 j 的名称（去掉 _capture1）
                name_i = dataset[i][:-9]
                name_j = dataset[j][:-9]
                # 计算 final_score
                score_i = torch.tensor(self.label_dict[name_i]).sum().item()
                score_j = torch.tensor(self.label_dict[name_j]).sum().item()
                # 计算原始差值
                delta.append(abs(score_i - score_j))
        return delta








    