#!/usr/bin/env python3
"""
运行LOUO (Leave-One-User-Out) 交叉验证
"""
import os
import sys
import subprocess
import argparse
from utils import parser

def extract_user_id(video_name):
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

def get_num_users(args):
    """获取用户数量"""
    import pickle
    with open(os.path.join(args.info_dir, 'splits.pkl'), 'rb') as f:
        cv_file = pickle.load(f)
    
    all_list = cv_file[args.cls]
    all_videos = []
    for fold_list in all_list:
        for vid in fold_list:
            all_videos.append(vid)
    
    # 提取所有用户ID（支持多种格式：B001, C001, D001等）
    user_ids = set()
    failed_extractions = []
    for vid in all_videos:
        user_id = extract_user_id(vid)
        if user_id is not None:
            user_ids.add(user_id)
        else:
            failed_extractions.append(vid)
    
    # 打印调试信息
    print(f"成功提取的用户ID: {sorted(user_ids)}")
    if failed_extractions:
        print(f"警告：以下视频无法提取用户ID（共{len(failed_extractions)}个）:")
        for vid in failed_extractions[:10]:  # 只显示前10个
            print(f"  - {vid}")
        if len(failed_extractions) > 10:
            print(f"  ... 还有 {len(failed_extractions) - 10} 个")
    
    return len(user_ids), sorted(user_ids)

def main():
    # 解析参数
    args = parser.get_args()
    
    # 设置交叉验证方法（在setup之前设置，以便保存到配置中）
    args.cv_method = 'LOUO'
    
    parser.setup(args)
    
    # 获取用户数量和用户列表
    num_users, user_ids_list = get_num_users(args)
    print(f"Found {num_users} users for LOUO cross-validation: {user_ids_list}")
    
    # 验证是否找到了所有8个用户（只按字母B-I分组）
    expected_users = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    missing_users = [uid for uid in expected_users if uid not in user_ids_list]
    if missing_users:
        print(f"警告：以下用户未找到: {missing_users}")
        print(f"这可能是数据集中确实没有这些用户的数据，或者用户ID格式不同")
    if len(user_ids_list) == 8:
        print(f"✓ 成功找到所有8个用户！")
    
    # 获取GPU设备
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    
    # 保存原始实验名称
    original_exp_name = args.exp_name
    
    # 为每个fold运行训练
    for fold in range(num_users):
        print(f"\n{'='*60}")
        print(f"Running LOUO fold {fold+1}/{num_users}")
        print(f"{'='*60}\n")
        
        # 设置fold和实验名称
        args.fold = fold
        args.exp_name = f"{original_exp_name}_louo_fold{fold}"
        
        # 重新设置实验路径（setup会创建目录）
        args.experiment_path = os.path.join('./experiments', 'CoRe_RT', args.benchmark, args.exp_name)
        parser.create_experiment_dir(args)
        
        # 确保cv_method被设置
        args.cv_method = 'LOUO'
        
        # 保存配置
        parser.save_experiment_config(args)
        
        # 运行训练
        from tools import run_net
        try:
            run_net(args)
            print(f"LOUO fold {fold+1}/{num_users} completed successfully")
        except Exception as e:
            print(f"Error in LOUO fold {fold+1}/{num_users}: {e}")
            import traceback
            traceback.print_exc()
    
    # 恢复原始实验名称
    args.exp_name = original_exp_name
    
    print(f"\n{'='*60}")
    print(f"All LOUO folds completed!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
