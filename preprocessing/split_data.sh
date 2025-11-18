#!/bin/bash
input="data/train_clean_v1.jsonl"
out="data/cross-validation"
k=10
mkdir -p "$out"
n=$(wc -l < "$input")
s=$((n / k))
for i in $(seq 1 $k); do
  a=$(( (i-1)*s+1 ))
  b=$(( i==k ? n : i*s ))
  sed -n "${a},${b}p" "$input" > "$out/valid_$i.jsonl"
  { head -n $((a-1)) "$input"; tail -n $((n-b)) "$input"; } > "$out/train_$i.jsonl"
done
