#!/usr/bin/env python3
"""
运行LOSO (Leave-One-Subject-Out) 交叉验证
注意：LOSO使用SuperTrialOut划分，固定为5个fold（split_index从1到5）
这与LOUO不同，LOUO使用UserOut，有8个fold（对应8个用户）
"""
import os
import sys
import subprocess
import argparse
from utils import parser

def get_num_folds(args):
    """
    获取LOSO的fold数量
    LOSO使用SuperTrialOut，固定为5个fold（split_index从1到5）
    """
    # LOSO使用SuperTrialOut，固定为5个fold
    num_folds = 5
    return num_folds

def main():
    # 解析参数
    args = parser.get_args()
    
    # 设置交叉验证方法（在setup之前设置，以便保存到配置中）
    args.cv_method = 'LOSO'
    
    parser.setup(args)
    
    # 获取LOSO的fold数量（固定为5，对应SuperTrialOut）
    num_folds = get_num_folds(args)
    print(f"LOSO cross-validation will run {num_folds} folds (SuperTrialOut)")
    
    # 获取GPU设备
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    
    # 保存原始实验名称
    original_exp_name = args.exp_name
    
    # 为每个fold运行训练
    for fold in range(num_folds):
        print(f"\n{'='*60}")
        print(f"Running LOSO fold {fold+1}/{num_folds}")
        print(f"{'='*60}\n")
        
        # 设置fold和实验名称
        args.fold = fold
        args.exp_name = f"{original_exp_name}_loso_fold{fold}"
        
        # 重新设置实验路径（setup会创建目录）
        args.experiment_path = os.path.join('./experiments', 'CoRe_RT', args.benchmark, args.exp_name)
        parser.create_experiment_dir(args)
        
        # 确保cv_method被设置
        args.cv_method = 'LOSO'
        
        # 保存配置
        parser.save_experiment_config(args)
        
        # 运行训练
        from tools import run_net
        try:
            run_net(args)
            print(f"LOSO fold {fold+1}/{num_folds} completed successfully")
        except Exception as e:
            print(f"Error in LOSO fold {fold+1}/{num_folds}: {e}")
            import traceback
            traceback.print_exc()
    
    # 恢复原始实验名称
    args.exp_name = original_exp_name
    
    print(f"\n{'='*60}")
    print(f"All LOSO folds completed!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
