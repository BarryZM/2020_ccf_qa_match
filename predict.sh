#!/bin/sh
PWD_DIR=$(cd `dirname $0`; pwd)
export BERT_BASE_DIR=/home/syzong/nlp_deeplearning/chinese_L-12_H-768_A-12
export GLUE_DIR=$PWD_DIR/data
export TRAINED_CLASSIFIER=$PWD_DIR/output
export EXP_NAME=model

python -u run.py \
  --do_predict=true \
  --data_dir=$GLUE_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER/$EXP_NAME \
  --max_seq_length=128 \
  --output_dir=$TRAINED_CLASSIFIER
