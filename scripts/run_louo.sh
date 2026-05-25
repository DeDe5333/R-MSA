#!/usr/bin/env sh
# 运行LOUO (Leave-One-User-Out) 交叉验证
# 使用方法: bash scripts/run_louo.sh <GPU_ID> <BENCHMARK> <EXP_NAME> [其他参数]
# 例如: bash scripts/run_louo.sh 0 JIG louo_experiment

mkdir -p logs
now=$(date +"%m%d_%H%M")
log_name="LOG_LOUO_$2_$3_$now"
CUDA_VISIBLE_DEVICES=$1 python3 -u run_louo.py --benchmark $2 --exp_name $3 ${@:4} 2>&1 | tee logs/$log_name.log
