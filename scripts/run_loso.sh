#!/usr/bin/env sh
# 运行LOSO (Leave-One-Subject-Out) 交叉验证
# 使用方法: bash scripts/run_loso.sh <GPU_ID> <BENCHMARK> <EXP_NAME> [其他参数]
# 例如: bash scripts/run_loso.sh 0 JIG loso_experiment

mkdir -p logs
now=$(date +"%m%d_%H%M")
log_name="LOG_LOSO_$2_$3_$now"
CUDA_VISIBLE_DEVICES=$1 python3 -u run_loso.py --benchmark $2 --exp_name $3 ${@:4} 2>&1 | tee logs/$log_name.log
