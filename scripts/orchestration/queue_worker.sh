#!/bin/bash
# queue_worker.sh - 从共享队列取任务，在指定 GPU 上串行训练
# 达到/略超论文目标 MAE 时自动停止当前任务，执行下一个
# 用法: bash queue_worker.sh <GPU_ID>

GPU_ID=$1
PYTHON=/mnt/nfs/yxwang/env/fract_env/bin/python
SCRIPT=/mnt/nfs/yxwang/code/multifract/train_metrla_optimized.py
WORKDIR=/mnt/nfs/yxwang/code/multifract
LOGDIR=$WORKDIR/logs
QUEUE=$WORKDIR/task_queue.txt
LOCKFILE=$WORKDIR/task_queue.lock

cd "$WORKDIR" || exit 1

# ── 论文 STMAN 目标 MAE（达到即可停止）──────────────────────────────
get_target() {
    local ds=$1 pred=$2
    case "${ds}__${pred}" in
        "METR-LA__12")   echo "3.28" ;;
        "METR-LA__48")   echo "5.19" ;;
        "METR-LA__96")   echo "7.80" ;;
        "METR-LA__288")  echo "9.44" ;;
        "METR-LA__864")  echo "10.50" ;;
        "METR-LA__2016") echo "11.13" ;;
        "PEMS-BAY__12")  echo "1.86" ;;
        "PEMS-BAY__48")  echo "2.59" ;;
        "PEMS-BAY__96")  echo "2.78" ;;
        "PEMS-BAY__288") echo "2.98" ;;
        "PEMS-BAY__864") echo "3.17" ;;
        "PEMS-BAY__2016") echo "3.39" ;;
        "PEMS03__12")    echo "14.42" ;;
        "PEMS03__48")    echo "15.36" ;;
        "PEMS03__96")    echo "17.04" ;;
        "PEMS03__288")   echo "18.36" ;;
        "PEMS03__864")   echo "20.09" ;;
        "PEMS03__2016")  echo "22.92" ;;
        "PEMS07__12")    echo "22.32" ;;
        "PEMS07__48")    echo "24.07" ;;
        "PEMS07__96")    echo "25.90" ;;
        "PEMS07__288")   echo "27.63" ;;
        "PEMS07__864")   echo "29.73" ;;
        "PEMS07__2016")  echo "31.99" ;;
        "PEMS08__12")    echo "13.75" ;;
        "PEMS08__48")    echo "17.31" ;;
        "PEMS08__96")    echo "21.76" ;;
        "PEMS08__288")   echo "22.37" ;;
        "PEMS08__864")   echo "24.46" ;;
        "PEMS08__2016")  echo "26.23" ;;
        *) echo "999" ;;
    esac
}

# ── 原子弹出队列第一行 ────────────────────────────────────────────────
pop_task() {
    (
        flock -x 9
        LINE=$(head -1 "$QUEUE" 2>/dev/null)
        if [ -n "$LINE" ]; then
            tail -n +2 "$QUEUE" > "${QUEUE}.tmp" && mv "${QUEUE}.tmp" "$QUEUE"
            echo "$LINE"
        fi
    ) 9>"$LOCKFILE"
}

log() { echo "[GPU$GPU_ID][$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Worker started"

while true; do
    TASK=$(pop_task)
    [ -z "$TASK" ] && { log "Queue empty, worker exiting."; break; }

    read -r DS PRED D_MODEL LR PATIENCE TAG SCALER BATCH <<< "$TASK"
    SCALER="${SCALER:-minmax}"   # 默认 minmax
    BATCH="${BATCH:-32}"          # 默认 batch_size=32
    TARGET=$(get_target "$DS" "$PRED")
    DS_LOWER=$(echo "$DS" | tr '[:upper:]' '[:lower:]' | tr -d '-')
    LOGFILE="$LOGDIR/${DS_LOWER}_${PRED}_${TAG}.log"

    PROGRESS_LOG="$LOGDIR/${DS_LOWER}_${PRED}_${TAG}_progress.log"

    log "START: $DS pred_len=$PRED d_model=$D_MODEL lr=$LR patience=$PATIENCE tag=$TAG scaler=$SCALER batch=$BATCH"
    log "Target MAE ≤ $TARGET | Log: $LOGFILE | Progress: $PROGRESS_LOG"

    # 启动训练（后台）
    CUDA_VISIBLE_DEVICES=$GPU_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PYTHON -u $SCRIPT \
        --dataset "$DS" --pred_len "$PRED" --d_model "$D_MODEL" \
        --lr "$LR" --scaler "$SCALER" --split_rate 0.6 --batch_size "$BATCH" \
        --patience "$PATIENCE" --epochs 200 --tag "$TAG" \
        --progress_log "$PROGRESS_LOG" --log_interval 5 \
        >> "$LOGFILE" 2>&1 &
    TRAIN_PID=$!
    log "Training PID=$TRAIN_PID"

    # 监控循环：每60秒检查一次最佳 MAE（从独立 progress 文件读取）
    BEST_MAE="999"
    STOP_REASON="natural"
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        sleep 60

        # 从 progress 文件末行读取 best_MAE（备选：扫描主 log）
        CUR_BEST=$(grep -o 'best_MAE=[0-9.]*' "$PROGRESS_LOG" 2>/dev/null \
                   | tail -1 | awk -F= '{print $2}')
        # 若 progress 文件尚未生成，回退到主 log
        if [ -z "$CUR_BEST" ]; then
            CUR_BEST=$(grep -o 'MAE=[0-9.]*' "$LOGFILE" 2>/dev/null \
                       | awk -F= '{print $2}' | sort -n | head -1)
        fi

        if [ -n "$CUR_BEST" ]; then
            BEST_MAE=$CUR_BEST
            # awk 比较浮点：CUR_BEST <= TARGET 时返回1
            REACHED=$(awk -v a="$CUR_BEST" -v b="$TARGET" 'BEGIN{print (a+0<=b+0)?1:0}')
            if [ "$REACHED" = "1" ]; then
                log "TARGET REACHED: best_MAE=$CUR_BEST <= target=$TARGET — stopping early"
                kill "$TRAIN_PID" 2>/dev/null
                # 等待进程退出（最多10s）
                for _ in $(seq 1 10); do
                    kill -0 "$TRAIN_PID" 2>/dev/null || break
                    sleep 1
                done
                kill -9 "$TRAIN_PID" 2>/dev/null
                STOP_REASON="target_reached"
                break
            else
                log "Monitor: $DS pred=$PRED best_MAE=$CUR_BEST (target=$TARGET) — continuing"
            fi
        fi
    done

    wait "$TRAIN_PID" 2>/dev/null
    log "DONE: $DS pred_len=$PRED tag=$TAG | best_MAE=$BEST_MAE | stop=$STOP_REASON"
done
