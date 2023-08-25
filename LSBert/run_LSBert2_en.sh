export TSAR_LSBERT_PATH=/notebooks/FRONTIERS_TSAR2022/TSAR-LSBert/
export TSAR_LSBERT_CODE_PATH=$TSAR_LSBERT_PATH/code/
export RESULTS_PATH=$TSAR_LSBERT_CODE_PATH/results/en


if [ ! -d $Result_DIR ] 
then
    mkdir $Result_DIR    
fi
# DATASET_FILE=../datasets/lex.mturk.txt
# DATASET_FILE=../datasets/BenchLS.txt
# DATASET_FILE=../datasets/NNSeval.txt
DATASET_FILE=../datasets/tsar2022_en_test_gold.tsv
HUGGINGFACE_BERT_BASED_MODEL=bert-large-uncased-whole-word-masking
TOP_K_SUBSTITUTION_GENERATION=50
TOP_K_OUTPUT_RESULTS=3
PROBABILITY_MASKING_RATIO=0.5
WORD_EMBEDDINGS_FILE=resources/en/fasttext/crawl-300d-2M-subword.vec
WORD_FREQUENCY_FILE=resources/en/SUBTLEX/SUBTLEX_frequency.xlsx 
PPDB_FILE=resources/en/PPDB/ppdb-2.0-tldr


python3 LSBert2_en.py \
  --do_lower_case \
  --num_selections_SG $TOP_K_SUBSTITUTION_GENERATION \
  --num_selections_output $TOP_K_OUTPUT_RESULTS \
  --prob_mask $PROBABILITY_MASKING_RATIO \
  --eval_dir $DATASET_FILE \
  --bert_model $HUGGINGFACE_BERT_BASED_MODEL \
  --max_seq_length 350 \
  --word_embeddings $WORD_EMBEDDINGS_FILE  \
  --word_frequency $WORD_FREQUENCY_FILE \
  --output_SR_file results/en/tsar_results.txt \
  --ppdb $PPDB_FILE
#   --do_eval True \
  #--no_cuda \  

  

 

 
