#!/bin/bash

while true; do
    # 运行 generate_jacobian.py
    python generate_jacobian.py

    # 获取上一个命令的退出码
    exit_code=$?

    # 如果退出码为零，表示程序正常退出，退出循环
    if [ $exit_code -eq 0 ]; then
        echo "程序正常退出"
        break
    else
        echo "程序非正常退出，将重新运行"
        # 等待一段时间，可以根据需要调整
        sleep 5
    fi
done
