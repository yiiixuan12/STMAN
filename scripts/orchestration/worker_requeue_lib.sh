#!/usr/bin/env bash

task_requeue_key() {
    local ds=$1
    local pred=$2
    local tag=$3
    echo "${ds}__${pred}__${tag}"
}

is_known_dataset() {
    local ds=$1
    case "$ds" in
        METR-LA|PEMS-BAY|PEMS03|PEMS07|PEMS08) return 0 ;;
        *) return 1 ;;
    esac
}

is_known_pred_len() {
    local pred=$1
    case "$pred" in
        12|48|96|288|864|2016) return 0 ;;
        *) return 1 ;;
    esac
}

is_nonneg_integer() {
    local val=$1
    [[ "$val" =~ ^[0-9]+$ ]]
}

task_line_is_valid() {
    local task_line=$1
    local -a fields
    local ds pred d_model patience tag batch seq_len direct enc_layers epochs k_hop

    [ -n "$task_line" ] || return 1
    [[ "$task_line" =~ ^[[:space:]]*# ]] && return 1

    read -r -a fields <<< "$task_line"
    [ "${#fields[@]}" -ge 19 ] || return 1

    ds=${fields[0]:-}
    pred=${fields[1]:-}
    d_model=${fields[2]:-}
    patience=${fields[4]:-}
    tag=${fields[5]:-}
    batch=${fields[7]:-}
    seq_len=${fields[8]:-}
    direct=${fields[9]:-}
    k_hop=${fields[11]:-}
    enc_layers=${fields[12]:-}
    epochs=${fields[13]:-}

    is_known_dataset "$ds" || return 1
    is_known_pred_len "$pred" || return 1
    is_nonneg_integer "$d_model" || return 1
    is_nonneg_integer "$patience" || return 1
    is_nonneg_integer "$batch" || return 1
    is_nonneg_integer "$seq_len" || return 1
    is_nonneg_integer "$direct" || return 1
    is_nonneg_integer "$k_hop" || return 1
    is_nonneg_integer "$enc_layers" || return 1
    is_nonneg_integer "$epochs" || return 1
    [ -n "$tag" ] || return 1
    [[ "$tag" =~ ^[A-Za-z0-9._-]+$ ]] || return 1
    return 0
}

count_queue_effective_tasks() {
    local queue_file=$1
    [ -f "$queue_file" ] || { echo "0"; return; }

    local count=0 line
    while IFS= read -r line || [ -n "$line" ]; do
        if task_line_is_valid "$line"; then
            count=$((count + 1))
        fi
    done < "$queue_file"
    echo "$count"
}

backfill_queue_from_pool() {
    local queue_file=$1
    local lock_file=$2
    local pool_file=$3
    local target=$4

    [ -f "$pool_file" ] || { echo "0"; return; }

    local current added_total line ds pred tag added
    current=$(count_queue_effective_tasks "$queue_file")
    current=${current:-0}
    [ "$current" -lt "$target" ] || { echo "0"; return; }

    added_total=0
    while IFS= read -r line || [ -n "$line" ]; do
        current=$(count_queue_effective_tasks "$queue_file")
        current=${current:-0}
        [ "$current" -lt "$target" ] || break

        if ! task_line_is_valid "$line"; then
            continue
        fi

        ds=$(awk '{print $1}' <<< "$line")
        pred=$(awk '{print $2}' <<< "$line")
        tag=$(awk '{print $6}' <<< "$line")
        if task_signature_running_global "$ds" "$pred" "$tag"; then
            continue
        fi

        added=$(append_task_if_missing "$queue_file" "$lock_file" "$line")
        if [ "$added" = "1" ]; then
            added_total=$((added_total + 1))
        fi
    done < "$pool_file"

    echo "$added_total"
}

backfill_queue_from_pools() {
    local queue_file=$1
    local lock_file=$2
    local hopeful_pool=$3
    local hopeful_target=$4
    local fallback_pool=$5
    local total_target=$6

    local hopeful_added fallback_added
    hopeful_added=$(backfill_queue_from_pool "$queue_file" "$lock_file" "$hopeful_pool" "$hopeful_target")
    fallback_added=$(backfill_queue_from_pool "$queue_file" "$lock_file" "$fallback_pool" "$total_target")
    echo $(( ${hopeful_added:-0} + ${fallback_added:-0} ))
}

default_frozen_file() {
    local root
    root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    echo "${MULTIFRACT_FROZEN_FILE:-$root/checkpoints/FROZEN.txt}"
}

default_frozen_lock_file() {
    echo "${MULTIFRACT_FROZEN_LOCK_FILE:-/tmp/yxwang_multifract_FROZEN.lock}"
}

default_blocked_file() {
    local root
    root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    echo "${MULTIFRACT_BLOCKED_FILE:-$root/checkpoints/BLOCKED.txt}"
}

default_blocked_lock_file() {
    echo "${MULTIFRACT_BLOCKED_LOCK_FILE:-/tmp/yxwang_multifract_BLOCKED.lock}"
}

frozen_contains_cell() {
    local frozen_file=$1
    local ds=$2
    local pred=$3
    [ -f "$frozen_file" ] || return 1
    awk -v ds="$ds" -v pred="$pred" '
        NF >= 2 && $1 !~ /^#/ && $1 == ds && $2 == pred { found=1 }
        END { exit(found ? 0 : 1) }
    ' "$frozen_file"
}

frozen_contains_triplet() {
    local frozen_file=$1
    local ds=$2
    local pred=$3
    local tag=$4
    [ -f "$frozen_file" ] || return 1
    awk -v ds="$ds" -v pred="$pred" -v tag="$tag" '
        NF >= 3 && $1 !~ /^#/ && $1 == ds && $2 == pred && $3 == tag { found=1 }
        END { exit(found ? 0 : 1) }
    ' "$frozen_file"
}

freeze_cell_if_missing() {
    local frozen_file=$1
    local lock_file=$2
    local ds=$3
    local pred=$4
    local tag=$5

    [ -n "$frozen_file" ] || return 1
    [ -n "$lock_file" ] || lock_file="${frozen_file}.lock"
    mkdir -p "$(dirname "$frozen_file")"

    (
        flock -x 9
        touch "$frozen_file"
        if frozen_contains_cell "$frozen_file" "$ds" "$pred"; then
            echo "0"
            return
        fi
        printf '%s %s %s\n' "$ds" "$pred" "$tag" >> "$frozen_file"
        echo "1"
    ) 9>"$lock_file"
}

is_frozen_task_line() {
    local task_line=$1
    local frozen_file=${2:-$(default_frozen_file)}
    local ds pred
    ds=$(awk '{print $1}' <<< "$task_line")
    pred=$(awk '{print $2}' <<< "$task_line")
    [ -n "$ds" ] || return 1
    [ -n "$pred" ] || return 1
    frozen_contains_cell "$frozen_file" "$ds" "$pred"
}

blocked_contains_triplet() {
    local blocked_file=$1
    local ds=$2
    local pred=$3
    local tag=$4
    [ -f "$blocked_file" ] || return 1
    awk -v ds="$ds" -v pred="$pred" -v tag="$tag" '
        NF >= 3 && $1 !~ /^#/ && $1 == ds && $2 == pred && $3 == tag { found=1 }
        END { exit(found ? 0 : 1) }
    ' "$blocked_file"
}

block_task_if_missing() {
    local blocked_file=$1
    local lock_file=$2
    local ds=$3
    local pred=$4
    local tag=$5
    local reason=${6:-manual}

    [ -n "$blocked_file" ] || return 1
    [ -n "$lock_file" ] || lock_file="${blocked_file}.lock"
    mkdir -p "$(dirname "$blocked_file")"

    (
        flock -x 9
        touch "$blocked_file"
        if blocked_contains_triplet "$blocked_file" "$ds" "$pred" "$tag"; then
            echo "0"
            return
        fi
        printf '%s %s %s %s\n' "$ds" "$pred" "$tag" "$reason" >> "$blocked_file"
        echo "1"
    ) 9>"$lock_file"
}

is_blocked_task_line() {
    local task_line=$1
    local blocked_file=${2:-$(default_blocked_file)}
    local ds pred tag
    ds=$(awk '{print $1}' <<< "$task_line")
    pred=$(awk '{print $2}' <<< "$task_line")
    tag=$(awk '{print $6}' <<< "$task_line")
    [ -n "$ds" ] || return 1
    [ -n "$pred" ] || return 1
    [ -n "$tag" ] || return 1
    blocked_contains_triplet "$blocked_file" "$ds" "$pred" "$tag"
}

extract_progress_metric() {
    local line=$1
    local key=$2
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

extract_cli_arg() {
    local args=$1
    local key=$2
    sed -n "s/.*--$key \\([^ ]*\\).*/\\1/p" <<< "$args" | head -1
}

has_cli_flag() {
    local args=$1
    local key=$2
    grep -q -- "--$key" <<< "$args"
}

progress_metric_from_log() {
    local progress_log=$1
    local key=$2
    local last_line
    last_line=$(tail -n 1 "$progress_log" 2>/dev/null)
    extract_progress_metric "$last_line" "$key"
}

task_line_from_args() {
    local args=$1
    local ds pred d_model lr patience tag scaler batch seq_len enc_layers epochs weight_decay t_mult
    local div_weight vmg_weight qml_weight direct spatial_mode k_hop
    local scheduler resume_ckpt resume_mode finetune_lr seed_best_mae split_rate val_ratio
    local direct_head_mode direct_step_refine monitor_mode strict_diversity_gate strict_vs_persistence_gate
    local ema_decay ema_start_epoch ema_eval_interval
    local grad_accum_steps head_warmup_epochs head_lr_mult
    local plateau_patience dropout ff_multiplier
    local -a extras=()

    ds=$(extract_cli_arg "$args" "dataset")
    pred=$(extract_cli_arg "$args" "pred_len")
    d_model=$(extract_cli_arg "$args" "d_model")
    lr=$(extract_cli_arg "$args" "lr")
    patience=$(extract_cli_arg "$args" "patience")
    tag=$(extract_cli_arg "$args" "tag")
    scaler=$(extract_cli_arg "$args" "scaler")
    batch=$(extract_cli_arg "$args" "batch_size")
    seq_len=$(extract_cli_arg "$args" "seq_len")
    enc_layers=$(extract_cli_arg "$args" "enc_layers")
    epochs=$(extract_cli_arg "$args" "epochs")
    weight_decay=$(extract_cli_arg "$args" "weight_decay")
    t_mult=$(extract_cli_arg "$args" "cosine_T_mult")
    div_weight=$(extract_cli_arg "$args" "diversity_weight")
    vmg_weight=$(extract_cli_arg "$args" "vmg_weight")
    qml_weight=$(extract_cli_arg "$args" "qml_weight")
    spatial_mode=$(extract_cli_arg "$args" "spatial_mode")
    k_hop=$(extract_cli_arg "$args" "k_hop")
    scheduler=$(extract_cli_arg "$args" "scheduler")
    split_rate=$(extract_cli_arg "$args" "split_rate")
    val_ratio=$(extract_cli_arg "$args" "val_ratio")
    plateau_patience=$(extract_cli_arg "$args" "plateau_patience")
    dropout=$(extract_cli_arg "$args" "dropout")
    ff_multiplier=$(extract_cli_arg "$args" "ff_multiplier")
    resume_ckpt=$(extract_cli_arg "$args" "resume_ckpt")
    resume_mode=$(extract_cli_arg "$args" "resume_mode")
    finetune_lr=$(extract_cli_arg "$args" "finetune_lr")
    seed_best_mae=$(extract_cli_arg "$args" "seed_best_mae")
    direct_head_mode=$(extract_cli_arg "$args" "direct_head_mode")
    direct_step_refine=$(extract_cli_arg "$args" "direct_step_refine")
    monitor_mode=$(extract_cli_arg "$args" "monitor_mode")
    strict_diversity_gate=$(extract_cli_arg "$args" "strict_diversity_gate")
    strict_vs_persistence_gate=$(extract_cli_arg "$args" "strict_vs_persistence_gate")
    ema_decay=$(extract_cli_arg "$args" "ema_decay")
    ema_start_epoch=$(extract_cli_arg "$args" "ema_start_epoch")
    ema_eval_interval=$(extract_cli_arg "$args" "ema_eval_interval")
    grad_accum_steps=$(extract_cli_arg "$args" "grad_accum_steps")
    head_warmup_epochs=$(extract_cli_arg "$args" "head_warmup_epochs")
    head_lr_mult=$(extract_cli_arg "$args" "head_lr_mult")

    direct=0
    has_cli_flag "$args" "use_direct_pred" && direct=1

    spatial_mode=${spatial_mode:-attention}
    k_hop=${k_hop:-8}

    [ -n "$ds" ] || return 1
    [ -n "$pred" ] || return 1
    [ -n "$tag" ] || return 1

    if [ -n "$scheduler" ]; then
        extras+=("scheduler=$scheduler")
    fi
    if [ -n "$split_rate" ] && [ "$split_rate" != "0.6" ] && [ "$split_rate" != "0.60" ]; then
        extras+=("split_rate=$split_rate")
    fi
    if [ -n "$val_ratio" ] && [ "$val_ratio" != "0.2" ] && [ "$val_ratio" != "0.20" ]; then
        extras+=("val_ratio=$val_ratio")
    fi
    if [ -n "$plateau_patience" ] && [ "$plateau_patience" != "8" ]; then
        extras+=("plateau_patience=$plateau_patience")
    fi
    if [ -n "$dropout" ] && [ "$dropout" != "0.1" ] && [ "$dropout" != "0.10" ]; then
        extras+=("dropout=$dropout")
    fi
    if [ -n "$ff_multiplier" ] && [ "$ff_multiplier" != "4" ]; then
        extras+=("ff_multiplier=$ff_multiplier")
    fi
    if [ -n "$resume_ckpt" ]; then
        extras+=("resume_ckpt=$resume_ckpt")
    fi
    if [ -n "$resume_mode" ]; then
        extras+=("resume_mode=$resume_mode")
    fi
    if [ -n "$finetune_lr" ]; then
        extras+=("finetune_lr=$finetune_lr")
    fi
    if [ -n "$seed_best_mae" ]; then
        extras+=("seed_best_mae=$seed_best_mae")
    fi
    if [ -n "$monitor_mode" ]; then
        extras+=("monitor_mode=$monitor_mode")
    fi
    if [ -n "$direct_head_mode" ]; then
        extras+=("direct_head_mode=$direct_head_mode")
    fi
    if [ -n "$direct_step_refine" ]; then
        extras+=("direct_step_refine=$direct_step_refine")
    fi
    if [ -n "$strict_diversity_gate" ]; then
        extras+=("strict_diversity_gate=$strict_diversity_gate")
    fi
    if [ -n "$strict_vs_persistence_gate" ]; then
        extras+=("strict_vs_persistence_gate=$strict_vs_persistence_gate")
    fi
    if [ -n "$ema_decay" ] && [ "$ema_decay" != "0.0" ] && [ "$ema_decay" != "0" ]; then
        extras+=("ema_decay=$ema_decay")
    fi
    if [ -n "$ema_start_epoch" ] && [ "$ema_start_epoch" != "0" ]; then
        extras+=("ema_start_epoch=$ema_start_epoch")
    fi
    if [ -n "$ema_eval_interval" ] && [ "$ema_eval_interval" != "1" ]; then
        extras+=("ema_eval_interval=$ema_eval_interval")
    fi
    if [ -n "$grad_accum_steps" ] && [ "$grad_accum_steps" != "1" ]; then
        extras+=("grad_accum_steps=$grad_accum_steps")
    fi
    if [ -n "$head_warmup_epochs" ] && [ "$head_warmup_epochs" != "0" ]; then
        extras+=("head_warmup_epochs=$head_warmup_epochs")
    fi
    if [ -n "$head_lr_mult" ] && [ "$head_lr_mult" != "1" ] && [ "$head_lr_mult" != "1.0" ]; then
        extras+=("head_lr_mult=$head_lr_mult")
    fi

    printf '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s' \
        "$ds" "${pred:-12}" "${d_model:-64}" "${lr:-1e-3}" "${patience:-30}" "$tag" "${scaler:-standard}" \
        "${batch:-32}" "${seq_len:-12}" "$direct" "$spatial_mode" "$k_hop" "${enc_layers:-3}" \
        "${epochs:-200}" "${weight_decay:-1e-5}" "${t_mult:-2}" "${div_weight:-0.0}" "${vmg_weight:-0.0}" "${qml_weight:-0.0}"
    if [ "${#extras[@]}" -gt 0 ]; then
        printf ' %s' "${extras[@]}"
    fi
    printf '\n'
}

normalize_task_resume_line() {
    local task_line=$1
    local root_dir=${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
    local checkpoints_dir="$root_dir/checkpoints"
    local ds pred tag same_tag_last
    local -a fields extras normalized_extras
    local token has_explicit_resume fresh_guard

    task_line_is_valid "$task_line" || return 1

    read -r -a fields <<< "$task_line"
    ds=${fields[0]}
    pred=${fields[1]}
    tag=${fields[5]}
    same_tag_last="$checkpoints_dir/${ds}_${pred}_${tag}_last.pt"

    if [ ! -f "$same_tag_last" ]; then
        printf '%s\n' "$task_line"
        return 0
    fi

    extras=("${fields[@]:19}")
    has_explicit_resume=0
    fresh_guard=0
    for token in "${extras[@]}"; do
        case "$token" in
            fresh=1|fresh=true|fresh=yes|fresh=on|no_auto_resume=1|no_auto_resume=true|no_auto_resume=yes|no_auto_resume=on)
                fresh_guard=1
                ;;
            resume_ckpt=*|resume_mode=*|finetune_lr=*|seed_best_mae=*)
                has_explicit_resume=1
                ;;
        esac
    done

    if [ "$fresh_guard" = "1" ]; then
        printf '%s\n' "$task_line"
        return 0
    fi

    for token in "${extras[@]}"; do
        case "$token" in
            resume_ckpt=*|resume_mode=*|finetune_lr=*)
                continue
                ;;
            *)
                normalized_extras+=("$token")
                ;;
        esac
    done
    normalized_extras+=("resume_ckpt=$same_tag_last" "resume_mode=states")

    printf '%s' "${fields[0]}"
    local idx
    for ((idx = 1; idx < 19 && idx < ${#fields[@]}; ++idx)); do
        printf ' %s' "${fields[$idx]}"
    done
    for token in "${normalized_extras[@]}"; do
        printf ' %s' "$token"
    done
    printf '\n'
}

auto_requeue_min_epoch() {
    local pred=$1
    if [ "$pred" -ge 864 ]; then
        echo 100
    elif [ "$pred" -ge 288 ]; then
        echo 60
    else
        echo 30
    fi
}

calc_recent_best_improve() {
    local progress_log=$1
    local window=${2:-5}
    local best_series
    best_series=$(tail -n "$window" "$progress_log" 2>/dev/null | awk '{
        for (i = 1; i <= NF; ++i) {
            if ($i ~ /^best_MAE=/) {
                split($i, a, "=")
                print a[2]
                break
            }
        }
    }')

    if [ "$(printf '%s\n' "$best_series" | sed '/^$/d' | wc -l | tr -d ' ')" -lt 2 ]; then
        echo "0"
        return
    fi

    awk 'NR==1{first=$1} {last=$1} END{
        if (first <= 0) {
            print "0"
        } else {
            improve=(first-last)/first
            if (improve < 0) improve=0
            printf "%.4f", improve
        }
    }' <<< "$best_series"
}

auto_requeue_decision() {
    local progress_log=$1
    local target=$2
    local pred=$3
    local attempts=${4:-0}

    local close_ratio=${AUTO_REQUEUE_CLOSE_RATIO:-0.10}
    local min_improve=${AUTO_REQUEUE_MIN_IMPROVE:-0.003}
    local max_attempts=${AUTO_REQUEUE_MAX_ATTEMPTS:-2}
    local window=${AUTO_REQUEUE_WINDOW:-5}

    local last_line epoch best div vsp recent_improve min_ep close_ok quality_ok improve_ok
    last_line=$(tail -n 1 "$progress_log" 2>/dev/null)
    epoch=$(extract_progress_metric "$last_line" "epoch")
    best=$(extract_progress_metric "$last_line" "best_MAE")
    div=$(extract_progress_metric "$last_line" "div")
    vsp=$(extract_progress_metric "$last_line" "vsp")

    epoch=${epoch:-0}
    best=${best:-999}
    div=${div:-0}
    vsp=${vsp:-999}
    min_ep=$(auto_requeue_min_epoch "$pred")
    recent_improve=$(calc_recent_best_improve "$progress_log" "$window")

    if [ "$attempts" -ge "$max_attempts" ]; then
        echo "0|budget_exhausted|epoch=$epoch|best=$best|recent=$recent_improve|attempts=$attempts"
        return
    fi

    if [ "$epoch" -lt "$min_ep" ]; then
        echo "0|before_min_epoch|epoch=$epoch|best=$best|recent=$recent_improve|attempts=$attempts"
        return
    fi

    close_ok=$(awk -v best="$best" -v target="$target" -v ratio="$close_ratio" '
        BEGIN { print (best + 0 <= target * (1 + ratio)) ? 1 : 0 }
    ')
    if [ "$close_ok" != "1" ]; then
        echo "0|not_close_enough|epoch=$epoch|best=$best|recent=$recent_improve|attempts=$attempts"
        return
    fi

    quality_ok=$(awk -v div="$div" -v vsp="$vsp" '
        BEGIN { print (div + 0 > 0.5 && vsp + 0 < 0.95) ? 1 : 0 }
    ')
    if [ "$quality_ok" != "1" ]; then
        echo "0|quality_gate_failed|epoch=$epoch|best=$best|recent=$recent_improve|attempts=$attempts"
        return
    fi

    improve_ok=$(awk -v improve="$recent_improve" -v threshold="$min_improve" '
        BEGIN { print (improve + 0 >= threshold + 0) ? 1 : 0 }
    ')
    if [ "$improve_ok" != "1" ]; then
        echo "0|not_improving|epoch=$epoch|best=$best|recent=$recent_improve|attempts=$attempts"
        return
    fi

    echo "1|eligible|epoch=$epoch|best=$best|recent=$recent_improve|attempts=$attempts"
}

requeue_state_get_attempts() {
    local state_file=$1
    local key=$2
    if [ ! -f "$state_file" ]; then
        echo "0"
        return
    fi
    awk -F'\t' -v key="$key" '$1 == key { val=$2 } END { print (val == "" ? 0 : val) }' "$state_file"
}

requeue_state_increment() {
    local state_file=$1
    local lock_file=$2
    local key=$3
    local ds=$4
    local pred=$5
    local tag=$6
    local best=$7
    local reason=$8

    mkdir -p "$(dirname "$state_file")"

    (
        flock -x 9
        local tmp="${state_file}.tmp"
        local now current
        now=$(date '+%Y-%m-%d %H:%M:%S')
        current=0

        if [ -f "$state_file" ]; then
            awk -F'\t' -v key="$key" '$1 == key { val=$2 } END { print (val == "" ? 0 : val) }' "$state_file" > "${tmp}.count"
            current=$(cat "${tmp}.count" 2>/dev/null)
            rm -f "${tmp}.count"
            awk -F'\t' -v key="$key" '$1 != key' "$state_file" > "$tmp"
        else
            : > "$tmp"
        fi

        current=$((current + 1))
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$key" "$current" "$now" "$ds" "$pred" "$tag" "$best" "$reason" >> "$tmp"
        mv "$tmp" "$state_file"
        echo "$current"
    ) 9>"$lock_file"
}

queue_contains_task_signature() {
    local queue_file=$1
    local ds=$2
    local pred=$3
    local tag=$4
    [ -f "$queue_file" ] || return 1
    awk -v ds="$ds" -v pred="$pred" -v tag="$tag" '
        NF && $1 !~ /^#/ && $1 == ds && $2 == pred && $6 == tag { found=1 }
        END { exit(found ? 0 : 1) }
    ' "$queue_file"
}

task_signature_running_global() {
    local ds=$1
    local pred=$2
    local tag=$3

    ps -eo args= 2>/dev/null | grep -F "train_metrla_optimized.py" | \
        grep -F -- "--dataset $ds" | \
        grep -F -- "--pred_len $pred" | \
        grep -F -- "--tag $tag" | \
        grep -v grep >/dev/null
}

append_task_if_missing() {
    local queue_file=$1
    local lock_file=$2
    local task_line=$3

    local ds pred tag frozen_file blocked_file
    task_line_is_valid "$task_line" || { echo "0"; return; }
    task_line=$(normalize_task_resume_line "$task_line")
    ds=$(awk '{print $1}' <<< "$task_line")
    pred=$(awk '{print $2}' <<< "$task_line")
    tag=$(awk '{print $6}' <<< "$task_line")
    frozen_file=$(default_frozen_file)
    blocked_file=$(default_blocked_file)

    (
        flock -x 9
        if frozen_contains_cell "$frozen_file" "$ds" "$pred"; then
            echo "0"
            return
        fi
        if blocked_contains_triplet "$blocked_file" "$ds" "$pred" "$tag"; then
            echo "0"
            return
        fi
        if queue_contains_task_signature "$queue_file" "$ds" "$pred" "$tag"; then
            echo "0"
            return
        fi
        printf '%s\n' "$task_line" >> "$queue_file"
        echo "1"
    ) 9>"$lock_file"
}
