# LSLlama: Fine-Tuned LLaMA for Lexical Simplification

Set-Up:
Add tuned-llama-7b from https://huggingface.co/acbaez/LSLlama/tree/main to ./LSLlama/

LSBert baseline:
RUN: run_LSBert2_en.sh
EVALUATE: python get_metrics_NNSeval_BenchLS.py + python get_metrics_lexmturk.py

UniHD baseline:
RUN: python context_predictor.py
EVALUATE: python get_metrics.py (need to select dataset in code)

LSLlama:
TRAIN: run_train_model.sh
RUN/EVALUATE: python generate.py (need to select dataset in code)
