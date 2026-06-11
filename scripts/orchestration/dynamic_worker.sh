#!/bin/bash
# dynamic_worker.sh - 动态多任务并行 GPU worker
# 扫描队列找第一个能放下的任务，最大化 GPU 利用率
# 用法: bash dynamic_worker.sh <GPU_ID>

GPU_ID=$1
WORKER_INSTANCE_LOCK=${MULTIFRACT_WORKER_INSTANCE_LOCK:-/tmp/yxwang_multifract_dynamic_worker_v2_gpu${GPU_ID}.lock}
PYTHON=/mnt/nfs/yxwang/env/fract_env/bin/python
SCRIPT=/mnt/nfs/yxwang/code/multifract/train_metrla_optimized.py
WORKDIR=/mnt/nfs/yxwang/code/multifract
LOGDIR=$WORKDIR/logs
QUEUE=$WORKDIR/task_queue.txt
# NFS-hosted flock can leave workers stuck forever on stale/opaque locks.
# Use a node-local lock by default; override with MULTIFRACT_QUEUE_LOCK if needed.
LOCKFILE=${MULTIFRACT_QUEUE_LOCK:-/tmp/yxwang_multifract_task_queue.lock}
HOPEFUL_POOL=$WORKDIR/watchdog_hopeful_pool.txt
FALLBACK_POOL=$WORKDIR/watchdog_fallback_pool.txt
REQUEUE_STATE=$LOGDIR/auto_requeue_state.tsv
REQUEUE_STATE_LOCK=${MULTIFRACT_REQUEUE_STATE_LOCK:-/tmp/yxwang_multifract_auto_requeue_state.lock}
# Keep enough backlog to cover the full unmet-cell pool. A small hopeful target
# let lower-priority queued tasks block close-to-target tasks from being refilled.
QUEUE_FILL_HOPEFUL_TARGET=${WATCHDOG_QUEUE_FILL_HOPEFUL_TARGET:-10}
QUEUE_FILL_TARGET=${WATCHDOG_QUEUE_FILL_TARGET:-14}

cd "$WORKDIR" || exit 1

# Prevent watchdog and manual launches from creating duplicate workers for the
# same GPU. Duplicate workers can race the queue and start the same tag twice.
exec 8>"$WORKER_INSTANCE_LOCK"
if ! flock -n 8; then
    echo "[GPU${GPU_ID}][$(date '+%Y-%m-%d %H:%M:%S')] Another worker is already active for GPU${GPU_ID}; exiting." >&2
    exit 0
fi

. "$WORKDIR/worker_requeue_lib.sh"

task_signature_running() {
    # 全局查重：避免 queue 被重写后再次启动同一个 DS/pred/tag
    local ds=$1 pred=$2 tag=$3
    ps -eo args= 2>/dev/null | grep -F "train_metrla_optimized.py" | \
        grep -F -- "--dataset $ds" | \
        grep -F -- "--pred_len $pred" | \
        grep -F -- "--tag $tag" | \
        grep -v grep >/dev/null
}

# ── 各任务所需最低空闲显存（MiB），含安全缓冲 ────────────────────────
# 2026-04-16 重新校准：基于实测 nvidia-smi 数据
#   direct_pred 模式极其节省显存（无自回归解码器）：1-3GB
#   自回归模式：2-7GB（PEMS07/883节点最大约7GB）
min_vram_required() {
    # $1=d_model, $2=pred_len, $3=dataset, $4=direct_pred, $5=seq_len, $6=batch
    local dm=$1 pred=${2:-96} ds=${3:-} dp=${4:-0} sl=${5:-12} bs=${6:-8}
    local n_nodes=500
    case "$ds" in
        PEMS07)   n_nodes=883 ;;
        PEMS03)   n_nodes=358 ;;
        PEMS-BAY) n_nodes=325 ;;
        METR-LA)  n_nodes=207 ;;
        PEMS08)   n_nodes=170 ;;
    esac

    # seq_len 额外开销：编码器注意力 O(seq_len^2) 占主导
    # 实测 seq_len=12 基准，seq_len=96 额外 +1.5-2GB
    local seq_extra=0
    if [ "$sl" -ge 96 ]; then
        if   [ "$n_nodes" -ge 700 ]; then seq_extra=3072
        elif [ "$n_nodes" -ge 300 ]; then seq_extra=2048
        else seq_extra=1536; fi
    elif [ "$sl" -ge 48 ]; then
        seq_extra=768
    fi

    # Direct-prediction 模式
    if [ "$dp" = "1" ]; then
        # Empirical 2026-04-27: PEMS07 d32/seq96/batch2 direct jobs use
        # roughly 2.3-3.5GiB. The generic long-seq PEMS07 estimate was
        # >10GiB and prevented valid short-horizon direct repairs from
        # launching on otherwise usable cards.
        if [ "$n_nodes" -ge 700 ] && [ "$sl" -ge 96 ] && [ "$bs" -le 2 ]; then
            if [ "$dm" -le 32 ]; then
                echo 4096
                return
            fi
            if [ "$dm" -le 64 ]; then
                # Empirical 2026-04-28: PEMS07 d64/seq96/batch1 direct-attn
                # v34 uses ~2.7GiB for H48. Keep a larger but usable guard so
                # H96+ repairs are not blocked forever on 10-11GiB-free cards.
                echo 7168
                return
            fi
        fi

        local base
        if   [ "$dm" -le 32 ]; then base=2560
        elif [ "$dm" -le 64 ]; then base=3584
        else                        base=5632
        fi
        local extra=0
        [ "$n_nodes" -ge 700 ] && extra=1024
        local batch_extra=0
        if [ "$bs" -ge 16 ]; then
            batch_extra=1024
        elif [ "$bs" -ge 8 ]; then
            batch_extra=512
        fi
        local safety=512
        if [ "$n_nodes" -ge 700 ] && [ "$sl" -ge 96 ]; then
            safety=4096
        elif [ "$n_nodes" -ge 300 ] && [ "$sl" -ge 96 ] && [ "$bs" -ge 16 ]; then
            # PEMS-BAY seq96 batch16 direct probes measured ~7.3GiB and can
            # OOM on fragmented cards when launched with only the old 7.0GiB
            # threshold. Require more headroom so these land on freer GPUs.
            safety=2048
        elif [ "$n_nodes" -ge 700 ]; then
            safety=1024
        fi
        echo $((base + extra + seq_extra + batch_extra + safety))
        return
    fi

    # 自回归模式（2026-04-18：对 batch<=8 小模型重新校准）
    local base
    if   [ "$dm" -le 32 ]; then base=2560
    elif [ "$dm" -le 64 ]; then base=4096
    else                        base=8704
    fi

    local extra=0
    if [ "$pred" -ge 2016 ]; then
        if   [ "$n_nodes" -ge 700 ]; then extra=4096
        elif [ "$n_nodes" -ge 300 ]; then extra=2048
        else extra=1024; fi
    elif [ "$pred" -ge 864 ]; then
        if   [ "$n_nodes" -ge 700 ]; then extra=3072
        elif [ "$n_nodes" -ge 300 ]; then extra=1024
        else extra=512; fi
    elif [ "$pred" -ge 288 ]; then
        extra=512
    fi
    local batch_extra=0
    if [ "$bs" -ge 16 ]; then
        batch_extra=1536
    elif [ "$bs" -ge 8 ]; then
        batch_extra=768
    elif [ "$bs" -ge 4 ]; then
        batch_extra=256
    fi

    # Shared GPU safety margin to reduce optimistic launches on crowded cards.
    local safety=512
    if [ "$n_nodes" -ge 700 ] && [ "$sl" -ge 96 ]; then
        safety=4096
    elif [ "$n_nodes" -ge 700 ]; then
        safety=1024
    fi

    echo $((base + extra + seq_extra + batch_extra + safety))
}

# ── 查询 GPU 当前空闲显存（MiB）────────────────────────────────────────
free_vram() {
    local free_mib buffer
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        -i "$GPU_ID" 2>/dev/null | tr -d ' ')
    [ -n "$free_mib" ] || return 0
    if gpu_has_foreign_process; then
        buffer=${SHARED_GPU_BUFFER_MIB:-4096}
        free_mib=$((free_mib - buffer))
        [ "$free_mib" -lt 0 ] && free_mib=0
    fi
    echo "$free_mib"
}

gpu_uuid_for_index() {
    nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader 2>/dev/null | \
        awk -F', ' -v gpu_id="$GPU_ID" '$1 == gpu_id { print $2; exit }'
}

gpu_has_foreign_process() {
    local gpu_uuid current_user
    gpu_uuid=$(gpu_uuid_for_index)
    current_user=$(id -un)
    [ -n "$gpu_uuid" ] || return 1

    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | \
        awk -F', ' -v gpu_uuid="$gpu_uuid" '$1 == gpu_uuid { print $2 }' | \
        while IFS= read -r pid; do
            [ -n "$pid" ] || continue
            local owner
            owner=$(ps -p "$pid" -o user= 2>/dev/null | awk '{print $1}')
            if [ -n "$owner" ] && [ "$owner" != "$current_user" ]; then
                echo 1
                return 0
            fi
        done | grep -q 1
}

master_train_pids_on_gpu() {
    local gpu_uuid
    gpu_uuid=$(gpu_uuid_for_index)
    [ -n "$gpu_uuid" ] || return 0

    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | \
        awk -F', ' -v gpu_uuid="$gpu_uuid" '$1 == gpu_uuid { print $2 }' | \
        while IFS= read -r pid; do
            [ -n "$pid" ] || continue
            local args ppid parent_args parent_comm
            args=$(ps -p "$pid" -o args= 2>/dev/null)
            [ -n "$args" ] || continue
            [[ "$args" == *"train_metrla_optimized.py"* ]] || continue
            ppid=$(ps -p "$pid" -o ppid= 2>/dev/null | tr -d ' ')
            parent_args=$(ps -p "$ppid" -o args= 2>/dev/null)
            parent_comm=$(ps -p "$ppid" -o comm= 2>/dev/null | awk '{print $1}')
            if [[ "$parent_comm" == python* && "$parent_args" == *"train_metrla_optimized.py"* ]]; then
                continue
            fi
            echo "$pid"
        done | sort -u
}

# ── 论文目标 MAE ──────────────────────────────────────────────────────
get_target() {
    local ds=$1 pred=$2
    case "${ds}__${pred}" in
        "METR-LA__12")    echo "3.28"  ;;
        "METR-LA__48")    echo "5.19"  ;;
        "METR-LA__96")    echo "7.80"  ;;
        "METR-LA__288")   echo "9.44"  ;;
        "METR-LA__864")   echo "10.50" ;;
        "METR-LA__2016")  echo "11.13" ;;
        "PEMS-BAY__12")   echo "1.86"  ;;
        "PEMS-BAY__48")   echo "2.59"  ;;
        "PEMS-BAY__96")   echo "2.78"  ;;
        "PEMS-BAY__288")  echo "2.98"  ;;
        "PEMS-BAY__864")  echo "3.17"  ;;
        "PEMS-BAY__2016") echo "3.39"  ;;
        "PEMS03__12")     echo "14.42" ;;
        "PEMS03__48")     echo "15.36" ;;
        "PEMS03__96")     echo "17.04" ;;
        "PEMS03__288")    echo "18.36" ;;
        "PEMS03__864")    echo "20.09" ;;
        "PEMS03__2016")   echo "22.92" ;;
        "PEMS07__12")     echo "22.32" ;;
        "PEMS07__48")     echo "24.07" ;;
        "PEMS07__96")     echo "25.90" ;;
        "PEMS07__288")    echo "27.63" ;;
        "PEMS07__864")    echo "29.73" ;;
        "PEMS07__2016")   echo "31.99" ;;
        "PEMS08__12")     echo "13.75" ;;
        "PEMS08__48")     echo "17.31" ;;
        "PEMS08__96")     echo "21.76" ;;
        "PEMS08__288")    echo "22.37" ;;
        "PEMS08__864")    echo "24.46" ;;
        "PEMS08__2016")   echo "26.23" ;;
        *) echo "999" ;;
    esac
}

# ── 扫描队列，找第一个所需显存 <= FREE_MiB 的任务，原子取出 ──────────
# 返回: 任务行（找到）或空字符串（没有合适任务）
find_and_pop_fitting_task() {
    local free_mib=$1
    local result=""
    (
        flock -x 9
        local tmpq="${QUEUE}.tmp"
        : > "$tmpq"
        local idx=0
        while IFS= read -r LINE || [ -n "$LINE" ]; do
            idx=$((idx + 1))
            if [ -z "$LINE" ]; then
                continue
            fi
            if [[ "$LINE" =~ ^[[:space:]]*# ]]; then
                echo "$LINE" >> "$tmpq"
                continue
            fi
            if ! task_line_is_valid "$LINE"; then
                log "DROP malformed queue line: $LINE"
                continue
            fi
            local dm pred ds tag direct sl batch
            ds=$(awk '{print $1}' <<< "$LINE")
            pred=$(awk '{print $2}' <<< "$LINE")
            dm=$(awk '{print $3}' <<< "$LINE")
            tag=$(awk '{print $6}' <<< "$LINE")
            batch=$(awk '{print $8}' <<< "$LINE")
            sl=$(awk '{print $9}' <<< "$LINE")
            direct=$(awk '{print $10}' <<< "$LINE")
            batch=${batch:-8}
            direct=${direct:-0}
            sl=${sl:-12}

            if is_frozen_task_line "$LINE"; then
                log "DROP frozen cell from queue: $ds pred=$pred tag=$tag"
                continue
            fi
            if is_blocked_task_line "$LINE"; then
                log "DROP blocked task from queue: $ds pred=$pred tag=$tag"
                continue
            fi

            if task_signature_running "$ds" "$pred" "$tag"; then
                log "DROP queued duplicate: $ds pred=$pred tag=$tag already running"
                continue
            fi

            local need
            need=$(min_vram_required "${dm:-96}" "${pred:-96}" "$ds" "$direct" "$sl" "$batch")
            if [ "$free_mib" -ge "$need" ]; then
                if [ -z "$result" ]; then
                    result="$LINE"
                    continue
                fi
                echo "$LINE" >> "$tmpq"
                continue
            fi

            echo "$LINE" >> "$tmpq"
        done < "$QUEUE"
        mv "$tmpq" "$QUEUE"
        echo "$result"
    ) 9>"$LOCKFILE"
}

log() { echo "[GPU${GPU_ID}][$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }

log "Dynamic worker started (PID=$$)"
log "Auto requeue policy: close_ratio=${AUTO_REQUEUE_CLOSE_RATIO:-0.10} min_improve=${AUTO_REQUEUE_MIN_IMPROVE:-0.003} window=${AUTO_REQUEUE_WINDOW:-5} max_attempts=${AUTO_REQUEUE_MAX_ATTEMPTS:-2}"

# ── 运行中任务表：PID -> "DS|PRED|TAG|TARGET|PROGRESS_LOG" ────────────
declare -A RUNNING
declare -A RUNNING_TASK

attach_existing_tasks() {
    local attached=0
    local frozen_file blocked_file
    frozen_file=$(default_frozen_file)
    blocked_file=$(default_blocked_file)
    while IFS= read -r pid; do
        [ -n "$pid" ] || continue
        [ -n "${RUNNING[$pid]:-}" ] && continue

        local args ds pred tag plog target task_line
        args=$(ps -p "$pid" -o args= 2>/dev/null)
        [ -n "$args" ] || continue

        ds=$(extract_cli_arg "$args" "dataset")
        pred=$(extract_cli_arg "$args" "pred_len")
        tag=$(extract_cli_arg "$args" "tag")
        plog=$(extract_cli_arg "$args" "progress_log")
        [ -n "$ds" ] || continue
        [ -n "$pred" ] || continue
        [ -n "$tag" ] || continue
        [ -n "$plog" ] || continue

        target=$(get_target "$ds" "$pred")
        task_line=$(task_line_from_args "$args" 2>/dev/null || true)
        if [ -n "$task_line" ] && is_blocked_task_line "$task_line" "$blocked_file"; then
            log "ATTACH_BLOCKED: $ds pred=$pred tag=$tag — killing PID=$pid"
            kill "$pid" 2>/dev/null
            sleep 5
            kill -9 "$pid" 2>/dev/null
            continue
        fi
        if frozen_contains_cell "$frozen_file" "$ds" "$pred"; then
            log "ATTACH_FROZEN: $ds pred=$pred tag=$tag — killing PID=$pid"
            kill "$pid" 2>/dev/null
            sleep 5
            kill -9 "$pid" 2>/dev/null
            continue
        fi
        RUNNING[$pid]="${ds}|${pred}|${tag}|${target}|${plog}"
        RUNNING_TASK[$pid]="$task_line"
        attached=$((attached + 1))
        log "ATTACH PID=$pid: $ds pred=$pred tag=$tag progress=$(basename "$plog") | target=$target"
    done < <(master_train_pids_on_gpu)

    log "Attach scan complete: attached=${attached} tasks=${#RUNNING[@]}"
}

start_task() {
    local TASK="$1"
    local -a TASK_FIELDS EXTRA_TOKENS EXTRA_CLI CMD
    local DS PRED D_MODEL LR PATIENCE TAG SCALER BATCH SEQ_LEN DIRECT SPATIAL_MODE K_HOP ENC_LAYERS EPOCHS WEIGHT_DECAY T_MULT DIV_WEIGHT VMG_WEIGHT QML_WEIGHT
    local EXTRA_TOKEN KEY VALUE SCHEDULER RESUME_CKPT RESUME_MODE_CLI FINETUNE_LR SEED_BEST_MAE
    local SPLIT_RATE VAL_RATIO MONITOR_MODE_CLI
    local FRESH_START NO_AUTO_RESUME

    TASK=$(normalize_task_resume_line "$TASK" "$WORKDIR")

    read -r -a TASK_FIELDS <<< "$TASK"
    if [ "${#TASK_FIELDS[@]}" -lt 6 ]; then
        log "SKIP malformed task: $TASK"
        return 1
    fi

    DS="${TASK_FIELDS[0]}"
    PRED="${TASK_FIELDS[1]}"
    D_MODEL="${TASK_FIELDS[2]}"
    LR="${TASK_FIELDS[3]}"
    PATIENCE="${TASK_FIELDS[4]}"
    TAG="${TASK_FIELDS[5]}"
    SCALER="${TASK_FIELDS[6]:-minmax}"
    BATCH="${TASK_FIELDS[7]:-32}"
    SEQ_LEN="${TASK_FIELDS[8]:-12}"         # 第9字段：输入序列长度，缺省12（短任务兼容旧格式）
    DIRECT="${TASK_FIELDS[9]:-0}"           # 第10字段：1=使用直接预测头，0=自回归解码器
    SPATIAL_MODE="${TASK_FIELDS[10]:-attention}"  # 第11字段：空间混合模式 attention|gcn
    K_HOP="${TASK_FIELDS[11]:-8}"           # 第12字段：GCN多跳数
    ENC_LAYERS="${TASK_FIELDS[12]:-3}"      # 第13字段：编码器层数
    EPOCHS="${TASK_FIELDS[13]:-200}"        # 第14字段：最大epoch数
    WEIGHT_DECAY="${TASK_FIELDS[14]:-1e-5}" # 第15字段：weight decay
    T_MULT="${TASK_FIELDS[15]:-2}"          # 第16字段：cosine T_mult
    DIV_WEIGHT="${TASK_FIELDS[16]:-0}"      # 第17字段：diversity loss 权重 (防坍缩)
    VMG_WEIGHT="${TASK_FIELDS[17]:-0}"      # 第18字段：variance matching loss 权重 (推荐 0.3)
    QML_WEIGHT="${TASK_FIELDS[18]:-0}"      # 第19字段：quantile matching loss 权重
    EXTRA_TOKENS=("${TASK_FIELDS[@]:19}")
    FRESH_START=0
    NO_AUTO_RESUME=0
    SPLIT_RATE=0.6
    VAL_RATIO=""

    for EXTRA_TOKEN in "${EXTRA_TOKENS[@]}"; do
        case "$EXTRA_TOKEN" in
            *=*)
                KEY=${EXTRA_TOKEN%%=*}
                VALUE=${EXTRA_TOKEN#*=}
                case "$KEY" in
                    scheduler) SCHEDULER="$VALUE" ;;
                    split|split_rate) SPLIT_RATE="$VALUE" ;;
                    val_ratio) VAL_RATIO="$VALUE" ;;
                    monitor) MONITOR_MODE_CLI="$VALUE" ;;
                    monitor_mode) MONITOR_MODE_CLI="$VALUE" ;;
                    resume_ckpt) RESUME_CKPT="$VALUE" ;;
                    resume_mode) RESUME_MODE_CLI="$VALUE" ;;
                    finetune_lr) FINETUNE_LR="$VALUE" ;;
                    seed_best_mae) SEED_BEST_MAE="$VALUE" ;;
                    fresh)
                        case "${VALUE,,}" in
                            1|true|yes|on) FRESH_START=1 ;;
                        esac
                        ;;
                    no_auto_resume)
                        case "${VALUE,,}" in
                            1|true|yes|on) NO_AUTO_RESUME=1 ;;
                        esac
                        ;;
                    allow_resume_arch_mismatch|no_residual_delta|no_decoder_future_tod)
                        case "${VALUE,,}" in
                            1|true|yes|on) EXTRA_CLI+=("--$KEY") ;;
                        esac
                        ;;
                    *)
                        EXTRA_CLI+=("--$KEY" "$VALUE")
                        ;;
                esac
                ;;
            *)
                log "WARN malformed extra token ignored: $EXTRA_TOKEN"
                ;;
        esac
    done

    local TARGET
    TARGET=$(get_target "$DS" "$PRED")
    local DS_LOWER
    DS_LOWER=$(echo "$DS" | tr '[:upper:]' '[:lower:]' | tr -d '-')
    local LOGFILE="$LOGDIR/${DS_LOWER}_${PRED}_${TAG}.log"
    local PROGRESS_LOG="$LOGDIR/${DS_LOWER}_${PRED}_${TAG}_progress.log"
    local LAST_CKPT="$WORKDIR/checkpoints/${DS}_${PRED}_${TAG}_last.pt"
    local RESUME_ENABLED=0
    local RESUME_SRC=""
    if [ "$FRESH_START" = "1" ]; then
        RESUME_CKPT=""
        RESUME_MODE_CLI=""
        FINETUNE_LR=""
        SEED_BEST_MAE=""
    elif [ -n "$RESUME_CKPT" ]; then
        if [ ! -f "$RESUME_CKPT" ]; then
            log "SKIP task missing resume checkpoint: $TASK | resume_ckpt=$RESUME_CKPT"
            return 1
        fi
        RESUME_ENABLED=1
        RESUME_SRC=$(basename "$RESUME_CKPT")
    elif [ "$NO_AUTO_RESUME" != "1" ] && [ -f "$LAST_CKPT" ]; then
        RESUME_CKPT="$LAST_CKPT"
        RESUME_ENABLED=1
        RESUME_SRC=$(basename "$LAST_CKPT")
    fi

    # 强制丢弃旧 pyc，避免 NFS 缓存 / 旧代码混用。
    # 仅在从头训练时清空 progress log；resume 时必须保留已有历史和 best 轨迹。
    rm -f "$WORKDIR"/__pycache__/train*.pyc "$WORKDIR"/__pycache__/train_metrla_optimized*.pyc 2>/dev/null || true
    if [ "$RESUME_ENABLED" != "1" ]; then
        : > "$PROGRESS_LOG"
    fi

    CMD=(
        "$PYTHON" -u "$SCRIPT"
        --dataset "$DS" --pred_len "$PRED" --d_model "$D_MODEL"
        --lr "$LR" --scaler "$SCALER" --split_rate "$SPLIT_RATE" --batch_size "$BATCH"
        --patience "$PATIENCE" --epochs "$EPOCHS" --tag "$TAG"
        --seq_len "$SEQ_LEN" --enc_layers "$ENC_LAYERS"
        --weight_decay "$WEIGHT_DECAY" --cosine_T_mult "$T_MULT"
        --diversity_weight "$DIV_WEIGHT"
        --vmg_weight "$VMG_WEIGHT"
        --qml_weight "$QML_WEIGHT"
        --progress_log "$PROGRESS_LOG" --log_interval 5
    )
    [ "$DIRECT" = "1" ] && CMD+=(--use_direct_pred)
    [ "$SPATIAL_MODE" = "gcn" ] && CMD+=(--spatial_mode gcn --k_hop "$K_HOP")
    [ -n "$SCHEDULER" ] && CMD+=(--scheduler "$SCHEDULER")
    [ -n "$VAL_RATIO" ] && CMD+=(--val_ratio "$VAL_RATIO")
    [ -n "$MONITOR_MODE_CLI" ] && CMD+=(--monitor_mode "$MONITOR_MODE_CLI")
    [ -n "$RESUME_CKPT" ] && CMD+=(--resume_ckpt "$RESUME_CKPT")
    [ -n "$RESUME_MODE_CLI" ] && CMD+=(--resume_mode "$RESUME_MODE_CLI")
    [ -n "$FINETUNE_LR" ] && CMD+=(--finetune_lr "$FINETUNE_LR")
    [ -n "$SEED_BEST_MAE" ] && CMD+=(--seed_best_mae "$SEED_BEST_MAE")
    if [ "${#EXTRA_CLI[@]}" -gt 0 ]; then
        CMD+=("${EXTRA_CLI[@]}")
    fi

    (
        # The worker singleton lock must not be inherited by training jobs.
        # Otherwise a killed/restarted worker cannot reacquire the lock while
        # its detached training children are still alive.
        exec 8>&-
        exec env CUDA_VISIBLE_DEVICES=$GPU_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1 \
            setsid "${CMD[@]}"
    ) >> "$LOGFILE" 2>&1 &
    local PID=$!
    RUNNING[$PID]="${DS}|${PRED}|${TAG}|${TARGET}|${PROGRESS_LOG}"
    RUNNING_TASK[$PID]="$TASK"
    if [ "$RESUME_ENABLED" = "1" ]; then
        log "RESUME PID=$PID: $DS pred=$PRED d_model=$D_MODEL seq=$SEQ_LEN split=$SPLIT_RATE val_ratio=${VAL_RATIO:-0.2} enc=$ENC_LAYERS direct=$DIRECT spatial=$SPATIAL_MODE k_hop=$K_HOP tag=$TAG batch=$BATCH epochs=$EPOCHS wd=$WEIGHT_DECAY T_mult=$T_MULT div=$DIV_WEIGHT vmg=$VMG_WEIGHT qml=$QML_WEIGHT scheduler=${SCHEDULER:-cosine} monitor=${MONITOR_MODE_CLI:-mae} resume_mode=${RESUME_MODE_CLI:-states} finetune_lr=${FINETUNE_LR:-none} seed_best_mae=${SEED_BEST_MAE:-none} fresh=$FRESH_START no_auto_resume=$NO_AUTO_RESUME ckpt=$RESUME_SRC | target=$TARGET"
    else
        log "START PID=$PID: $DS pred=$PRED d_model=$D_MODEL seq=$SEQ_LEN split=$SPLIT_RATE val_ratio=${VAL_RATIO:-0.2} enc=$ENC_LAYERS direct=$DIRECT spatial=$SPATIAL_MODE k_hop=$K_HOP tag=$TAG batch=$BATCH epochs=$EPOCHS wd=$WEIGHT_DECAY T_mult=$T_MULT div=$DIV_WEIGHT vmg=$VMG_WEIGHT qml=$QML_WEIGHT scheduler=${SCHEDULER:-cosine} monitor=${MONITOR_MODE_CLI:-mae} fresh=$FRESH_START no_auto_resume=$NO_AUTO_RESUME | target=$TARGET"
    fi
    return 0
}

attach_existing_tasks

self_backfill_queue() {
    local added current
    added=$(backfill_queue_from_pools "$QUEUE" "$LOCKFILE" "$HOPEFUL_POOL" "$QUEUE_FILL_HOPEFUL_TARGET" "$FALLBACK_POOL" "$QUEUE_FILL_TARGET")
    current=$(count_queue_effective_tasks "$QUEUE")
    current=${current:-0}
    if [ "${added:-0}" -gt 0 ]; then
        log "SELF_REFILL added=$added queue=$current hopeful_target=$QUEUE_FILL_HOPEFUL_TARGET total_target=$QUEUE_FILL_TARGET"
    fi
}

# ── 主循环 ────────────────────────────────────────────────────────────
while true; do
    # 1. 清理已完成任务，记录结果
    for PID in "${!RUNNING[@]}"; do
        if ! kill -0 "$PID" 2>/dev/null; then
            IFS='|' read -r DS PRED TAG TARGET PLOG <<< "${RUNNING[$PID]}"
            TASK_LINE=${RUNNING_TASK[$PID]:-}
            LAST_LINE=$(tail -n 1 "$PLOG" 2>/dev/null)
            FINAL_EPOCH=$(extract_progress_metric "$LAST_LINE" "epoch")
            FINAL_MAE=$(extract_progress_metric "$LAST_LINE" "MAE")
            FINAL_BEST=$(extract_progress_metric "$LAST_LINE" "best_MAE")
            FINAL_DIV=$(extract_progress_metric "$LAST_LINE" "div")
            FINAL_VSP=$(extract_progress_metric "$LAST_LINE" "vsp")
            FINAL_ACC15=$(extract_progress_metric "$LAST_LINE" "acc15")
            KEY=$(task_requeue_key "$DS" "$PRED" "$TAG")
            ATTEMPTS=$(requeue_state_get_attempts "$REQUEUE_STATE" "$KEY")
            DECISION=$(auto_requeue_decision "$PLOG" "$TARGET" "$PRED" "$ATTEMPTS")
            IFS='|' read -r SHOULD_REQUEUE REQUEUE_REASON REQUEUE_DETAIL_1 REQUEUE_DETAIL_2 REQUEUE_DETAIL_3 REQUEUE_DETAIL_4 <<< "$DECISION"
            if frozen_contains_cell "$(default_frozen_file)" "$DS" "$PRED"; then
                log "DONE: $DS pred=$PRED tag=$TAG | epoch=${FINAL_EPOCH:-?} mae=${FINAL_MAE:-?} best_MAE=${FINAL_BEST:-?} div=${FINAL_DIV:-?} vsp=${FINAL_VSP:-?} acc15=${FINAL_ACC15:-?} | target=$TARGET | auto_requeue=no reason=frozen_cell"
            elif [ "$SHOULD_REQUEUE" = "1" ] && [ -n "$TASK_LINE" ]; then
                APPENDED=$(append_task_if_missing "$QUEUE" "$LOCKFILE" "$TASK_LINE")
                if [ "$APPENDED" = "1" ]; then
                    NEW_ATTEMPTS=$(requeue_state_increment "$REQUEUE_STATE" "$REQUEUE_STATE_LOCK" "$KEY" "$DS" "$PRED" "$TAG" "${FINAL_BEST:-?}" "$REQUEUE_REASON")
                    log "DONE: $DS pred=$PRED tag=$TAG | epoch=${FINAL_EPOCH:-?} mae=${FINAL_MAE:-?} best_MAE=${FINAL_BEST:-?} div=${FINAL_DIV:-?} vsp=${FINAL_VSP:-?} acc15=${FINAL_ACC15:-?} | target=$TARGET | auto_requeue=yes attempt=${NEW_ATTEMPTS}/${AUTO_REQUEUE_MAX_ATTEMPTS:-2} reason=$REQUEUE_REASON ${REQUEUE_DETAIL_1:-} ${REQUEUE_DETAIL_2:-} ${REQUEUE_DETAIL_3:-} ${REQUEUE_DETAIL_4:-}"
                else
                    log "DONE: $DS pred=$PRED tag=$TAG | epoch=${FINAL_EPOCH:-?} mae=${FINAL_MAE:-?} best_MAE=${FINAL_BEST:-?} div=${FINAL_DIV:-?} vsp=${FINAL_VSP:-?} acc15=${FINAL_ACC15:-?} | target=$TARGET | auto_requeue=skip_duplicate reason=already_queued"
                fi
            else
                log "DONE: $DS pred=$PRED tag=$TAG | epoch=${FINAL_EPOCH:-?} mae=${FINAL_MAE:-?} best_MAE=${FINAL_BEST:-?} div=${FINAL_DIV:-?} vsp=${FINAL_VSP:-?} acc15=${FINAL_ACC15:-?} | target=$TARGET | auto_requeue=no reason=$REQUEUE_REASON ${REQUEUE_DETAIL_1:-} ${REQUEUE_DETAIL_2:-} ${REQUEUE_DETAIL_3:-} ${REQUEUE_DETAIL_4:-}"
            fi
            unset RUNNING[$PID]
            unset RUNNING_TASK[$PID]
        fi
    done

    # 2. 检查运行中任务是否“真实达标”
    #    安全门槛：根据 pred_len 动态设置最低 epoch（长预测易坍缩假阳性）
    #    真实达标 = round2(MAE) / round2(best_MAE) <= round2(target)
    #             AND div > 0.5 AND vs.persistence < 0.95
    for PID in "${!RUNNING[@]}"; do
        IFS='|' read -r DS PRED TAG TARGET PLOG <<< "${RUNNING[$PID]}"
        if frozen_contains_cell "$(default_frozen_file)" "$DS" "$PRED"; then
            log "FROZEN_CELL: $DS pred=$PRED tag=$TAG — killing PID=$PID"
            kill "$PID" 2>/dev/null
            sleep 5; kill -9 "$PID" 2>/dev/null
            unset RUNNING[$PID]
            unset RUNNING_TASK[$PID]
            continue
        fi
        if blocked_contains_triplet "$(default_blocked_file)" "$DS" "$PRED" "$TAG"; then
            log "BLOCKED_TASK: $DS pred=$PRED tag=$TAG — killing PID=$PID"
            kill "$PID" 2>/dev/null
            sleep 5; kill -9 "$PID" 2>/dev/null
            unset RUNNING[$PID]
            unset RUNNING_TASK[$PID]
            continue
        fi
        LAST_LINE=$(tail -n 1 "$PLOG" 2>/dev/null)
        CUR_MAE=$(extract_progress_metric "$LAST_LINE" "MAE")
        CUR_BEST=$(extract_progress_metric "$LAST_LINE" "best_MAE")
        CUR_EPOCH=$(extract_progress_metric "$LAST_LINE" "epoch")
        CUR_DIV=$(extract_progress_metric "$LAST_LINE" "div")
        CUR_VSP=$(extract_progress_metric "$LAST_LINE" "vsp")
        CUR_ACC15=$(extract_progress_metric "$LAST_LINE" "acc15")
        CUR_EPOCH=${CUR_EPOCH:-0}

        # 动态 MIN_EPOCHS：短预测30ep，中预测60ep，长预测100ep
        if [ "$PRED" -ge 864 ]; then
            MIN_EP=100
        elif [ "$PRED" -ge 288 ]; then
            MIN_EP=60
        else
            MIN_EP=30
        fi

        if [ -n "$CUR_BEST" ]; then
            if [ "$CUR_EPOCH" -lt "$MIN_EP" ]; then
                log "Monitor[$DS pred=$PRED tag=$TAG]: ep=$CUR_EPOCH mae=${CUR_MAE:-?} best_MAE=$CUR_BEST div=${CUR_DIV:-?} vsp=${CUR_VSP:-?} acc15=${CUR_ACC15:-?} (target=$TARGET, wait for ep>=$MIN_EP) | tasks=${#RUNNING[@]}"
            else
                ROUND_TARGET_HIT=$(awk -v mae="${CUR_MAE:-999}" -v best="${CUR_BEST:-999}" -v tgt="$TARGET" '
                    function r2(x) { return sprintf("%.2f", x + 0) + 0 }
                    BEGIN { print ((r2(mae) <= r2(tgt) || r2(best) <= r2(tgt)) ? 1 : 0) }
                ')
                STRICT_CANDIDATE=$(awk -v hit="$ROUND_TARGET_HIT" -v div="${CUR_DIV:-0}" -v vsp="${CUR_VSP:-999}" '
                    BEGIN { print (hit+0 == 1 && div+0 > 0.5 && vsp+0 < 0.95) ? 1 : 0 }
                ')
                if [ "$STRICT_CANDIDATE" = "1" ]; then
                    freeze_cell_if_missing "$(default_frozen_file)" "$(default_frozen_lock_file)" "$DS" "$PRED" "$TAG" >/dev/null
                    log "VAL_REACHED: $DS pred=$PRED ep=$CUR_EPOCH mae=${CUR_MAE:-?} best_MAE=$CUR_BEST div=${CUR_DIV:-?} vsp=${CUR_VSP:-?} acc15=${CUR_ACC15:-?} — killing PID=$PID"
                    kill "$PID" 2>/dev/null
                    sleep 5; kill -9 "$PID" 2>/dev/null
                    unset RUNNING[$PID]
                    unset RUNNING_TASK[$PID]
                elif [ "$ROUND_TARGET_HIT" = "1" ]; then
                    log "WARN: $DS pred=$PRED ep=$CUR_EPOCH rounded(MAE/best_MAE)<=rounded($TARGET) but div=${CUR_DIV:-?} vsp=${CUR_VSP:-?} acc15=${CUR_ACC15:-?} — suspicious / not true reached"
                else
                    log "Monitor[$DS pred=$PRED tag=$TAG]: ep=$CUR_EPOCH mae=${CUR_MAE:-?} best_MAE=$CUR_BEST div=${CUR_DIV:-?} vsp=${CUR_VSP:-?} acc15=${CUR_ACC15:-?} (target=$TARGET) | tasks=${#RUNNING[@]}"
                fi
            fi
        fi
    done

    self_backfill_queue

    # 3. 尝试启动新任务（只要显存够就继续拿任务）
    while true; do
        FREE=$(free_vram)
        [ -z "$FREE" ] && break

        TASK=$(find_and_pop_fitting_task "$FREE")
        if [ -z "$TASK" ]; then
            # 队列空或没有合适任务
            if [ "${#RUNNING[@]}" -eq 0 ]; then
                # 再确认一次队列是否真的空
                QLEN=$(count_queue_effective_tasks "$QUEUE"); QLEN=${QLEN:-0}
                if [ "$QLEN" -eq 0 ]; then
                    log "Queue empty and no running tasks. Waiting for pools/watchdog refill..."
                    break
                fi
                log "No fitting task for GPU${GPU_ID} (free=${FREE}MiB), waiting for tasks to complete..."
            fi
            break
        fi

        if ! start_task "$TASK"; then
            log "START_FAILED: requeue task for retry: $TASK"
            append_task_if_missing "$QUEUE" "$LOCKFILE" "$TASK" >/dev/null
            break
        fi
        # 等待新进程分配显存后再判断是否能再启动
        # 增加到 150 秒以避免并发 CUDA 初始化失败导致 CPU fallback
        sleep 150
    done

    sleep 60
done
