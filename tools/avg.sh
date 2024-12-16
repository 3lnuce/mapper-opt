filename=$1
pattern=$2

# grep -oP '^\S+: \K\d+(\.\d+)?' $filename | awk '{sum += $1; count++} END {if (count > 0) print sum / count; else print "No values to average"}'

grep $pattern $filename | awk -F': ' '{sum += $2} END {if (NR > 0) print sum / NR; else print "No matching lines"}'
