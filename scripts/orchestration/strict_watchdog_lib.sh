#!/usr/bin/env bash

is_train_master_record() {
    local args=$1
    local parent_args=$2
    local cmd0 parent_cmd0

    [[ "$args" == *"train_metrla_optimized.py"* ]] || return 1
    cmd0=${args%% *}
    [[ "${cmd0##*/}" == python* ]] || return 1
    parent_cmd0=${parent_args%% *}
    if [[ "${parent_cmd0##*/}" == python* && "$parent_args" == *"train_metrla_optimized.py"* ]]; then
        return 1
    fi
    return 0
}

list_master_trainers_live() {
    local ps_snapshot parent_snapshot
    ps_snapshot=$(mktemp)
    parent_snapshot=$(mktemp)
    ps -eo pid=,ppid=,etimes=,args= > "$ps_snapshot"
    awk '{ pid=$1; $1=""; $2=""; $3=""; sub(/^[ \t]+/, ""); print pid "\t" $0 }' "$ps_snapshot" > "$parent_snapshot"

    while IFS= read -r line; do
        [ -z "$line" ] && continue

        local pid ppid etimes args parent_args
        read -r pid ppid etimes args <<< "$line"
        parent_args=$(awk -F'\t' -v key="$ppid" '$1 == key { print $2; exit }' "$parent_snapshot")

        if is_train_master_record "$args" "$parent_args"; then
            printf '%s\t%s\t%s\t%s\n' "$pid" "$ppid" "$etimes" "$args"
        fi
    done < "$ps_snapshot"

    rm -f "$ps_snapshot" "$parent_snapshot"
}

list_master_trainers_from_fixtures() {
    local ps_fixture=$1
    local parent_fixture=$2

    while IFS= read -r line; do
        [ -z "$line" ] && continue

        local pid ppid etimes args parent_args
        read -r pid ppid etimes args <<< "$line"
        parent_args=$(awk -F'\t' -v key="$ppid" '$1 == key { print $2; exit }' "$parent_fixture")
        if [ -z "$parent_args" ]; then
            parent_args=$(awk -v key="$ppid" '$1 == key { $1=""; $2=""; $3=""; sub(/^[ \t]+/, ""); print; exit }' "$ps_fixture")
        fi

        if is_train_master_record "$args" "$parent_args"; then
            printf '%s\t%s\t%s\t%s\n' "$pid" "$ppid" "$etimes" "$args"
        fi
    done < "$ps_fixture"
}

model_warmstart_seed_plateau() {
    local resume_mode=$1
    local seed_best=$2
    local best=$3
    local recent_improve=$4
    local progress_points=$5
    local etimes=$6

    if [ "$resume_mode" != "model" ] || [ -z "$seed_best" ]; then
        echo 0
        return 0
    fi
    awk -v s="$seed_best" -v b="$best" -v r="$recent_improve" \
        -v n="$progress_points" -v et="$etimes" '
        BEGIN {
            margin = (s + 0 < 10) ? 0.001 : (s + 0) * 0.0001
            no_seed_gain = (b + 0 >= s - margin) ? 1 : 0
            mature_plateau = (n + 0 >= 6) ? 1 : 0
            enough_points = (n + 0 >= 2) ? 1 : 0
            enough_elapsed = (et + 0 >= 1200) ? 1 : 0
            stalled = (r + 0 < 0.0005) ? 1 : 0
            print (enough_points && enough_elapsed && stalled && (no_seed_gain || mature_plateau)) ? 1 : 0
        }
    '
}
