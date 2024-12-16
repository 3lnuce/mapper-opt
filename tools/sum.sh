filename=$1
grep -oP '^\S+: \K\d+(\.\d+)?' $filename | awk '{sum += $1} END {print sum}'

