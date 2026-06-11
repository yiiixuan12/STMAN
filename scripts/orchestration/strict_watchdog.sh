#!/bin/bash
# strict_watchdog.sh - 每隔固定时间巡检训练状态，记录报告并在明显无效时停掉任务。
# 用法:
#   bash strict_watchdog.sh --once
#   bash strict_watchdog.sh --loop 2400

set -u

WORKDIR=/mnt/nfs/yxwang/code/multifract
LOGDIR=$WORKDIR/logs
REPORT_LOG=$LOGDIR/strict_watchdog.log
ACTION_LOG=$LOGDIR/strict_watchdog_actions.log
PYTHON=/mnt/nfs/yxwang/env/fract_env/bin/python

cd "$WORKDIR" || exit 1
. "$WORKDIR/strict_watchdog_lib.sh"
. "$WORKDIR/worker_requeue_lib.sh"

QUEUE=$WORKDIR/task_queue.txt
# Match dynamic_worker.sh: NFS-hosted flock can leave queue maintenance stuck.
LOCKFILE=${MULTIFRACT_QUEUE_LOCK:-/tmp/yxwang_multifract_task_queue.lock}
FROZEN_FILE=$(default_frozen_file)
FROZEN_LOCK=$(default_frozen_lock_file)
BLOCKED_FILE=$(default_blocked_file)
BLOCKED_LOCK=$(default_blocked_lock_file)
HOPEFUL_POOL=$WORKDIR/watchdog_hopeful_pool.txt
FALLBACK_POOL=$WORKDIR/watchdog_fallback_pool.txt
MANAGED_GPUS="${MULTIFRACT_MANAGED_GPUS:-0 1 3}"
QUEUE_FILL_HOPEFUL_TARGET=${WATCHDOG_QUEUE_FILL_HOPEFUL_TARGET:-4}
QUEUE_FILL_TARGET=${WATCHDOG_QUEUE_FILL_TARGET:-8}
WORKER_START_MIN_FREE_MIB=${WATCHDOG_WORKER_START_MIN_FREE_MIB:-3000}

MODE="once"
INTERVAL=2400
if [ "${1:-}" = "--loop" ]; then
    MODE="loop"
    INTERVAL="${2:-2400}"
elif [ "${1:-}" = "--once" ]; then
    MODE="once"
fi

log_line() {
    local msg="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$REPORT_LOG"
}

action_line() {
    local msg="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$ACTION_LOG" | tee -a "$REPORT_LOG" >/dev/null
}

reason_should_block_task() {
    local reason="$1"
    case "$reason" in
        far_above_target_after_review|collapsed_or_worse_than_persistence_and_stalled|plateau_far_from_target|false_convergence_and_stalled|model_warmstart_plateau)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

extract_arg() {
    local args="$1"
    local key="$2"
    sed -n "s/.*--$key \\([^ ]*\\).*/\\1/p" <<< "$args" | head -1
}

extract_metric() {
    local line="$1"
    local key="$2"
    awk -v k="$key" '{
        for (i = 1; i <= NF; ++i) {
            if ($i ~ ("^" k "=")) {
                split($i, a, "=")
                print a[2]
                exit
            }
        }
    }' <<< "$line"
}

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

min_epoch() {
    local pred=$1
    if [ "$pred" -ge 864 ]; then
        echo 100
    elif [ "$pred" -ge 288 ]; then
        echo 60
    else
        echo 30
    fi
}

recommendation_for() {
    local ds="$1"
    local pred="$2"
    case "$ds" in
        PEMS07)
            if [ "$pred" -ge 288 ]; then
                echo "revisit_gcn_khop_or_add_vmg_qml"
            else
                echo "revisit_gcn_or_batch"
            fi
            ;;
        METR-LA|PEMS-BAY)
            echo "keep_qml_family_or_try_no_residual_delta"
            ;;
        PEMS03|PEMS08)
            echo "revisit_long_horizon_head_or_diversity"
            ;;
        *)
            echo "manual_review"
            ;;
    esac
}

train_stdout_log() {
    local ds="$1"
    local pred="$2"
    local tag="$3"
    local ds_lower
    ds_lower=$(echo "$ds" | tr '[:upper:]' '[:lower:]' | tr -d '-')
    echo "$LOGDIR/${ds_lower}_${pred}_${tag}.log"
}

fallback_progress_from_train_log() {
    local ds="$1"
    local pred="$2"
    local tag="$3"
    local args="$4"
    local train_log seed_best parsed

    train_log=$(train_stdout_log "$ds" "$pred" "$tag")
    [ -f "$train_log" ] || return 1

    parsed=$(awk '
        /\[Epoch [0-9]+\]/ {
            if (match($0, /\[Epoch[[:space:]]*([0-9]+)\]/, a)) epoch=a[1]
            if (match($0, /MAE=([0-9.]+)/, a)) mae=a[1]
            if (match($0, /Div=([0-9.]+)/, a)) div=a[1]
            if (match($0, /vsP=([0-9.]+)/, a)) vsp=a[1]
            if (match($0, /Acc15=([0-9.]+)%/, a)) acc15=a[1]
            if (mae != "") {
                if (best == "" || mae + 0 < best + 0) best = mae
                last_epoch = epoch
                last_mae = mae
                last_div = div
                last_vsp = vsp
                last_acc15 = acc15
            }
        }
        END {
            if (last_epoch != "") {
                printf "epoch=%s MAE=%s best_MAE=%s div=%s vsp=%s acc15=%s", last_epoch, last_mae, best, last_div, last_vsp, last_acc15
            }
        }
    ' "$train_log")

    [ -n "$parsed" ] || return 1

    seed_best=$(extract_arg "$args" "seed_best_mae")
    if [ -n "$seed_best" ]; then
        parsed=$(awk -v line="$parsed" -v seed="$seed_best" '
            BEGIN {
                n = split(line, a, " ")
                for (i = 1; i <= n; ++i) {
                    if (a[i] ~ /^best_MAE=/) {
                        split(a[i], kv, "=")
                        best = kv[2] + 0
                        if (seed + 0 < best) a[i] = "best_MAE=" seed
                    }
                }
                for (i = 1; i <= n; ++i) {
                    printf "%s%s", a[i], (i == n ? "" : " ")
                }
            }
        ')
    fi

    echo "$parsed"
}

stdout_progress_points() {
    local ds="$1"
    local pred="$2"
    local tag="$3"
    local train_log

    train_log=$(train_stdout_log "$ds" "$pred" "$tag")
    [ -f "$train_log" ] || { echo "0"; return; }
    awk '/\[Epoch[[:space:]]+[0-9]+\]/ { n++ } END { print n + 0 }' "$train_log"
}

stdout_recent_best_improve() {
    local ds="$1"
    local pred="$2"
    local tag="$3"
    local args="$4"
    local train_log seed_best

    train_log=$(train_stdout_log "$ds" "$pred" "$tag")
    [ -f "$train_log" ] || { echo "0"; return; }
    seed_best=$(extract_arg "$args" "seed_best_mae")

    awk -v seed="$seed_best" '
        /\[Epoch[[:space:]]+[0-9]+\]/ {
            mae = ""
            if (match($0, /MAE=([0-9.]+)/, a)) {
                mae = a[1] + 0
            }
            if (mae != "") {
                if (best == "" || mae < best) best = mae
                if (seed != "" && (best == "" || seed + 0 < best)) best = seed + 0
                vals[++n] = best
            }
        }
        END {
            if (n < 2 || vals[1] <= 0) {
                print "0"
            } else {
                improve = (vals[1] - vals[n]) / vals[1]
                if (improve < 0) improve = 0
                printf "%.4f", improve
            }
        }
    ' "$train_log"
}

resolve_progress_line() {
    local ds="$1"
    local pred="$2"
    local tag="$3"
    local plog="$4"
    local args="$5"
    local last_line=""

    if [ -n "$plog" ] && [ -f "$plog" ]; then
        last_line=$(tail -1 "$plog" 2>/dev/null)
    fi

    if [ -n "$last_line" ]; then
        echo "progress|$last_line"
        return 0
    fi

    last_line=$(fallback_progress_from_train_log "$ds" "$pred" "$tag" "$args" || true)
    if [ -n "$last_line" ]; then
        echo "stdout|$last_line"
        return 0
    fi

    return 1
}

strict_eval_checkpoint() {
    local ds="$1"
    local pred="$2"
    local tag="$3"
    local ckpt="$WORKDIR/checkpoints/${ds}_${pred}_${tag}_best.pt"
    local ds_lower cache ckpt_mtime cached_line cached_mtime out
    ds_lower=$(echo "$ds" | tr '[:upper:]' '[:lower:]' | tr -d '-')
    cache="$LOGDIR/strict_eval_${ds_lower}_${pred}_${tag}.log"

    if [ ! -f "$ckpt" ]; then
        echo "status=ERROR error=no_checkpoint"
        return
    fi

    ckpt_mtime=$(stat -c %Y "$ckpt" 2>/dev/null || echo 0)
    if [ -f "$cache" ]; then
        cached_line=$(tail -1 "$cache" 2>/dev/null)
        cached_mtime=$(extract_metric "$cached_line" "ckpt_mtime")
        if [ -n "$cached_mtime" ] && [ "$cached_mtime" = "$ckpt_mtime" ]; then
            echo "$cached_line"
            return
        fi
    fi

    out=$(PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" "$PYTHON" "$WORKDIR/batch_eval.py" \
        --gpu -1 --max_batches 0 \
        --dataset "$ds" --pred_len "$pred" --tag "$tag" 2>/dev/null | tail -1)
    if [ -z "$out" ]; then
        out="status=ERROR error=empty_eval_output"
    fi
    echo "$out ckpt_mtime=$ckpt_mtime" > "$cache"
    echo "$out ckpt_mtime=$ckpt_mtime"
}

kill_task() {
    local pid="$1"
    local pgid
    pgid=$(ps -p "$pid" -o pgid= 2>/dev/null | tr -d ' ')
    if [ -n "$pgid" ] && [ "$pgid" != "1" ]; then
        kill -- "-$pgid" 2>/dev/null || true
        sleep 5
        kill -9 -- "-$pgid" 2>/dev/null || true
        return
    fi
    kill "$pid" 2>/dev/null || true
    sleep 5
    kill -9 "$pid" 2>/dev/null || true
}

queue_effective_len() {
    count_queue_effective_tasks "$QUEUE"
}

gpu_free_vram() {
    local gpu=$1
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' '
}

worker_running_for_gpu() {
    local gpu=$1
    ps -eo args= 2>/dev/null | awk -v gpu="$gpu" '
        /dynamic_worker\.sh/ {
            n = split($0, a, /[[:space:]]+/)
            if (a[n] == gpu) {
                found = 1
                exit
            }
        }
        END { exit(found ? 0 : 1) }
    '
}

start_worker_for_gpu() {
    local gpu=$1
    local log_file="$LOGDIR/queue_worker_gpu${gpu}_managed.log"

    setsid -f bash -lc "cd '$WORKDIR' && PYTHONNOUSERSITE=1 bash '$WORKDIR/dynamic_worker.sh' '$gpu' >> '$log_file' 2>&1"
    action_line "START_WORKER gpu=$gpu pid=detached log=$(basename "$log_file")"
}

ensure_workers_running() {
    local queue_len free
    queue_len=$(queue_effective_len)
    [ "${queue_len:-0}" -gt 0 ] || return 0

    local gpu
    for gpu in $MANAGED_GPUS; do
        if worker_running_for_gpu "$gpu"; then
            continue
        fi
        free=$(gpu_free_vram "$gpu")
        free=${free:-0}
        if [ "$free" -lt "$WORKER_START_MIN_FREE_MIB" ]; then
            log_line "SKIP worker start gpu=$gpu free_vram=${free}MiB below threshold=${WORKER_START_MIN_FREE_MIB}MiB"
            continue
        fi
        start_worker_for_gpu "$gpu"
    done
}

sanitize_queue_file() {
    [ -f "$QUEUE" ] || return 0

    (
        flock -x 9
        local tmp="${QUEUE}.tmp"
        local line
        : > "$tmp"
        while IFS= read -r line || [ -n "$line" ]; do
            if [ -z "$line" ]; then
                continue
            fi
            if [[ "$line" =~ ^[[:space:]]*# ]]; then
                printf '%s\n' "$line" >> "$tmp"
                continue
            fi
            if task_line_is_valid "$line"; then
                if is_blocked_task_line "$line" "$BLOCKED_FILE"; then
                    action_line "QUEUE_DROP reason=blocked line=$line"
                else
                    printf '%s\n' "$line" >> "$tmp"
                fi
            else
                action_line "QUEUE_DROP reason=malformed line=$line"
            fi
        done < "$QUEUE"
        mv "$tmp" "$QUEUE"
    ) 9>"$LOCKFILE"
}

fill_queue_from_pool() {
    local pool_file=$1
    local target=$2
    local label=$3
    local current line ds pred tag added

    [ -f "$pool_file" ] || return 0
    current=$(queue_effective_len)
    [ "${current:-0}" -lt "$target" ] || return 0

    while IFS= read -r line || [ -n "$line" ]; do
        current=$(queue_effective_len)
        [ "${current:-0}" -lt "$target" ] || break

        if ! task_line_is_valid "$line"; then
            [[ "$line" =~ ^[[:space:]]*$ ]] && continue
            [[ "$line" =~ ^[[:space:]]*# ]] || log_line "SKIP malformed pool line from $(basename "$pool_file"): $line"
            continue
        fi

        ds=$(awk '{print $1}' <<< "$line")
        pred=$(awk '{print $2}' <<< "$line")
        tag=$(awk '{print $6}' <<< "$line")

        if frozen_contains_cell "$FROZEN_FILE" "$ds" "$pred"; then
            continue
        fi
        if blocked_contains_triplet "$BLOCKED_FILE" "$ds" "$pred" "$tag"; then
            continue
        fi
        if task_signature_running_global "$ds" "$pred" "$tag"; then
            continue
        fi

        added=$(append_task_if_missing "$QUEUE" "$LOCKFILE" "$line")
        if [ "$added" = "1" ]; then
            action_line "QUEUE_ADD source=$label task=$ds/$pred/$tag"
        fi
    done < "$pool_file"
}

ensure_queue_backfill() {
    sanitize_queue_file
    fill_queue_from_pool "$HOPEFUL_POOL" "$QUEUE_FILL_HOPEFUL_TARGET" "hopeful"
    fill_queue_from_pool "$FALLBACK_POOL" "$QUEUE_FILL_TARGET" "fallback"
}

report_once() {
    log_line "===== strict watchdog pass ====="

    local active=0
    while IFS=$'\t' read -r pid ppid etimes args; do
        [ -z "${pid:-}" ] && continue
        active=$((active + 1))

	        local ds pred tag plog target min_ep review_ep hard_ep
	        ds=$(extract_arg "$args" "dataset")
	        pred=$(extract_arg "$args" "pred_len")
	        tag=$(extract_arg "$args" "tag")
	        plog=$(extract_arg "$args" "progress_log")
        target=$(get_target "$ds" "$pred")
        min_ep=$(min_epoch "$pred")
        review_ep=$((min_ep + 20))
        hard_ep=$((min_ep * 2))

        if frozen_contains_cell "$FROZEN_FILE" "$ds" "$pred"; then
            action_line "KILL pid=$pid $ds/$pred/$tag reason=frozen_cell"
            kill_task "$pid"
            continue
        fi
        if blocked_contains_triplet "$BLOCKED_FILE" "$ds" "$pred" "$tag"; then
            action_line "KILL pid=$pid $ds/$pred/$tag reason=blocked_task"
            kill_task "$pid"
            continue
        fi

	        local resolved last_line source epoch mae best div vsp acc15
	        resolved=$(resolve_progress_line "$ds" "$pred" "$tag" "$plog" "$args" || true)
        last_line=${resolved#*|}
        source=${resolved%%|*}
        if [ -z "$resolved" ] || [ -z "$last_line" ]; then
            if [ "$etimes" -ge 7200 ]; then
                action_line "KILL pid=$pid $ds/$pred/$tag reason=no_progress_signal_after_${etimes}s rec=$(recommendation_for "$ds" "$pred")"
                kill_task "$pid"
            else
                log_line "WAIT pid=$pid $ds/$pred/$tag no_progress_signal yet et=${etimes}s"
            fi
            continue
        fi

        epoch=$(extract_metric "$last_line" "epoch")
        mae=$(extract_metric "$last_line" "MAE")
        best=$(extract_metric "$last_line" "best_MAE")
        div=$(extract_metric "$last_line" "div")
        vsp=$(extract_metric "$last_line" "vsp")
        acc15=$(extract_metric "$last_line" "acc15")

        epoch=${epoch:-0}
        mae=${mae:-999}
        best=${best:-999}
        div=${div:-0}
        vsp=${vsp:-999}
	        acc15=${acc15:-0}

	        local resume_mode seed_best warmstart_grace_seconds warmstart_grace progress_points
	        resume_mode=$(extract_arg "$args" "resume_mode")
	        seed_best=$(extract_arg "$args" "seed_best_mae")
	        warmstart_grace_seconds=0
	        warmstart_grace=0
	        progress_points=0
	        if [ -n "$plog" ] && [ -f "$plog" ]; then
	            progress_points=$(grep -c '^epoch=' "$plog" 2>/dev/null)
	            progress_points=${progress_points:-0}
	        fi
	        if [ "${progress_points:-0}" -lt 2 ]; then
	            local stdout_points
	            stdout_points=$(stdout_progress_points "$ds" "$pred" "$tag")
	            stdout_points=${stdout_points:-0}
	            if [ "$stdout_points" -gt "$progress_points" ]; then
	                progress_points="$stdout_points"
	            fi
	        fi
	        if [ "$resume_mode" = "model" ]; then
	            if [ "$pred" -ge 864 ]; then
	                warmstart_grace_seconds=7200
	            elif [ "$pred" -ge 288 ]; then
	                warmstart_grace_seconds=3600
	            else
	                warmstart_grace_seconds=1800
	            fi
	            if [ "$etimes" -lt "$warmstart_grace_seconds" ]; then
	                warmstart_grace=1
	            fi
	        fi

	        local recent_improve="0"
        local best_series
        best_series=$(tail -3 "$plog" 2>/dev/null | awk '{
            for (i = 1; i <= NF; ++i) {
                if ($i ~ /^best_MAE=/) {
                    split($i, a, "=")
                    print a[2]
                    break
                }
            }
        }')
        if [ "$(wc -l <<< "$best_series" | tr -d ' ')" -ge 2 ]; then
            recent_improve=$(awk 'NR==1{first=$1} {last=$1} END{
                if (first > 0) printf "%.4f", (first-last)/first;
                else print "0";
            }' <<< "$best_series")
        else
            recent_improve=$(stdout_recent_best_improve "$ds" "$pred" "$tag" "$args")
            recent_improve=${recent_improve:-0}
        fi

        local seed_plateau
        seed_plateau=$(model_warmstart_seed_plateau \
            "$resume_mode" "$seed_best" "$best" "$recent_improve" "$progress_points" "$etimes")

        local verdict="KEEP"
        local reason="early_or_improving"

	        if [ "$resume_mode" = "model" ] && [ "$seed_plateau" = "1" ]; then
	            verdict="KILL"
	            reason="model_warmstart_plateau"
	        elif [ "$epoch" -lt "$min_ep" ]; then
	            verdict="KEEP"
	            reason="before_review_window"
	        elif [ "$warmstart_grace" = "1" ]; then
	            verdict="KEEP"
	            reason="model_warmstart_grace"
	        else
            local hopeless far_from_target stall collapse false_good proxy_true rounded_hit regressed_resume
            hopeless=$(awk -v b="$best" -v t="$target" 'BEGIN{print (b > t * 1.40) ? 1 : 0}')
            far_from_target=$(awk -v b="$best" -v t="$target" 'BEGIN{print (b > t * 1.20) ? 1 : 0}')
            stall=$(awk -v r="$recent_improve" 'BEGIN{print (r < 0.02) ? 1 : 0}')
            collapse=$(awk -v d="$div" -v v="$vsp" 'BEGIN{print (d < 0.40 || v > 1.05) ? 1 : 0}')
            regressed_resume=$(awk -v m="$mae" -v b="$best" 'BEGIN{print (b > 0 && m > b * 1.10) ? 1 : 0}')
            rounded_hit=$(awk -v m="$mae" -v b="$best" -v t="$target" '
                function r2(x) { return sprintf("%.2f", x + 0) + 0 }
                BEGIN { print ((r2(m) <= r2(t) || r2(b) <= r2(t)) ? 1 : 0) }
            ')
            false_good=$(awk -v hit="$rounded_hit" -v d="$div" -v v="$vsp" 'BEGIN{print (hit+0 == 1 && (d <= 0.50 || v >= 0.95)) ? 1 : 0}')
            proxy_true=$(awk -v hit="$rounded_hit" -v d="$div" -v v="$vsp" 'BEGIN{print (hit+0 == 1 && d > 0.50 && v < 0.95) ? 1 : 0}')

            if [ "$epoch" -ge "$min_ep" ] && [ "$proxy_true" = "1" ]; then
                verdict="KILL"
                reason="val_target_reached"
            elif [ "$epoch" -ge "$review_ep" ] && [ "$hopeless" = "1" ]; then
                verdict="KILL"
                reason="far_above_target_after_review"
            elif [ "$epoch" -ge "$review_ep" ] && [ "$collapse" = "1" ] && [ "$stall" = "1" ]; then
                verdict="KILL"
                reason="collapsed_or_worse_than_persistence_and_stalled"
            elif [ "$epoch" -ge "$hard_ep" ] && [ "$far_from_target" = "1" ] && [ "$stall" = "1" ]; then
                verdict="KILL"
                reason="plateau_far_from_target"
            elif [ "$epoch" -ge "$review_ep" ] && [ "$false_good" = "1" ] && [ "$stall" = "1" ]; then
                verdict="KILL"
                reason="false_convergence_and_stalled"
            elif [ "$epoch" -ge "$review_ep" ] && [ "$regressed_resume" = "1" ] && [ "$stall" = "1" ]; then
                verdict="KILL"
                reason="resume_regressed_from_best"
            elif [ "$collapse" = "1" ]; then
                verdict="WATCH"
                reason="bad_quality_but_too_early_to_kill"
            elif [ "$far_from_target" = "1" ]; then
                verdict="WATCH"
                reason="far_from_target_but_still_running"
            else
                verdict="KEEP"
                reason="quality_or_trend_ok"
            fi
        fi

        if [ "$verdict" = "KILL" ]; then
            if [ "$reason" = "val_target_reached" ]; then
                freeze_cell_if_missing "$FROZEN_FILE" "$FROZEN_LOCK" "$ds" "$pred" "$tag" >/dev/null
            elif reason_should_block_task "$reason"; then
                block_task_if_missing "$BLOCKED_FILE" "$BLOCKED_LOCK" "$ds" "$pred" "$tag" "$reason" >/dev/null
            fi
            action_line "KILL pid=$pid $ds/$pred/$tag ep=$epoch mae=$mae best=$best div=$div vsp=$vsp acc15=$acc15 target=$target improve=$recent_improve points=$progress_points source=$source reason=$reason rec=$(recommendation_for "$ds" "$pred")"
            kill_task "$pid"
        else
            log_line "$verdict pid=$pid $ds/$pred/$tag ep=$epoch mae=$mae best=$best div=$div vsp=$vsp acc15=$acc15 target=$target improve=$recent_improve points=$progress_points source=$source reason=$reason"
        fi
    done < <(list_master_trainers_live)

    ensure_queue_backfill
    ensure_workers_running

    if [ "$active" -eq 0 ]; then
        log_line "no active master training tasks found"
    fi
}

if [ "$MODE" = "once" ]; then
    report_once
    exit 0
fi

while true; do
    report_once
    sleep "$INTERVAL"
done
