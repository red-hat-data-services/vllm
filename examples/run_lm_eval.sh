MODEL=neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
lm_eval --model vllm --model_args "pretrained=$MODEL" --tasks gsm8k --batch_size "auto"
