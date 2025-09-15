#!/usr/bin/env bash

# Run all embedding evaluations

models=(
    "../llm-models/Qwen3-Embedding-8B"
    "../llm-models/Qwen3-Embedding-4B"
    "../llm-models/SFR-Embedding-Mistral"
    "../llm-models/granite-embedding-30m-english"
    "../llm-models/granite-embedding-small-english-r2"
    "text-embedding-3-small"
    "text-embedding-3-large"
)

for model in "${models[@]}"; do
    # If model is a filesystem path, use only its basename for the results file
    if [[ "$model" == *"/"* ]]; then
        model_basename="${model##*/}"
    else
        model_basename="$model"
    fi
    python embedding_eval.py \
        --model "$model" \
        --para_dataset stsb \
        --qa_dataset trivia_qa \
        --para_size 1000 \
        --qa_size 1000 \
        --balance \
        --eta_targets 0.20,0.15,0.10 \
        --delta_target 1e-2 \
        --unit_delta \
        --hyp 0.85 \
        --seed 68 \
        --max_pairs 500 \
        --window_size 120 \
        --op_mode robust \
        --answer_repr window \
        --delta_space whiten \
        --out "results_${model_basename}.json"
done
