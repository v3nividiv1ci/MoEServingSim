#!/bin/bash

# 创建日志文件名，使用时间戳使其唯一
LOG_FILE="output/simulation_$(date +%Y%m%d_%H%M%S).log"

# 使用tee命令同时输出到终端和日志文件
python main.py --model_name 'qwen2-moe' --hardware 'A800' --npu_num 1 --npu_group 1 --npu_mem 16 \
    --local_bw 1024 --remote_bw 512 --link_bw 256 --fp 16 --block_size 4 \
    --dataset 'dataset/share-gpt-req100-rate10.tsv' --output "output/example_run_$(date +%Y%m%d_%H%M%S).csv" \
    --verbose --req_num 10 2>&1 | tee "$LOG_FILE"

# 输出日志文件位置信息
echo "完整日志已保存至: $LOG_FILE"