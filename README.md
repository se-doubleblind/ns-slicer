# NS-Slicer: A Learning-Based Approach to Static Program Slicing
Traditional program slicing approaches are very useful in debugging and early bug detection. However, they do not work for incomplete code snippets. As a result, code snippets containing vulnerabilities can migrate from developer forums to applications. In this work, we introduce **NS-Slicer**, a neural network-based program slicing approach for any variable at any location in both complete and partial programs. We leverage the learning of a pre-trained language model (PLM) on source code, which inherently possesses an understanding of the fine-grained dependencies among the variables within a statement and the other statements. We design distinct multi-layer perceptron (MLP) heads on top PLM, such that, given a variable at a specific location and a statement in a code snippet, NS-Slicer predicts whether the statement belongs to the set of statements in the backward slice or forward slice, respectively.

We conducted several experiments to evaluate NS-Slicer. On complete code, it predicts the backward and forward slices with an F1-score of 97.41% and 95.82%, respectively, while recording an overall F1-score of 96.77%. Moreover, in 85.20% of the cases, the static program slices predicted by NS-Slicer match exactly entire slices in the oracle. For partial programs, it records an F1-score of 96.77%–97.49% for backward slicing, 92.14%–95.40% for forward slicing, and an overall F1-score of 94.66%–96.62%. To illustrate NS-Slicer’s usefulness, we integrated the slices from NS-Slicer into VulDeePecker, a vulnerability detection tool, which originally does not work on incomplete code. The results show that VulDeePecker with our slices even achieved a high overall F1-score of 73.38%.


## Dataset Links
Here is the link for the dataset used in this paper: [link](https://drive.google.com/drive/folders/1qj0KWtqQS3E5stqjax-t_6XI04BvXPqf?usp=share_link)


## Pre-Trained Model/Tokenizer Asset Links
Here are the links for NS-Slicer with: 
1. GraphCodeBERT [link](https://drive.google.com/drive/folders/1zq0NUt7WFXfu4Q5b3oLrHLq_iffv-r5M?usp=share_link)
2. CodeBERT [link](https://drive.google.com/drive/folders/1wxyL6pRESee4WSFMuX0EmCEsRYlxSwvD?usp=share_link)

## Getting Started with NS-Slicer
### Run Instructions
  
```
$ python run.py --help     
usage: run.py [-h] [--data_dir DATA_DIR] [--dataset_key {Project_CodeNet_Java250}] [--model_key {microsoft/codebert-base,microsoft/graphcodebert-base,roberta-base}]
              [--output_dir OUTPUT_DIR] [--max_tokens MAX_TOKENS] [--use_statement_ids] [--pooling_strategy {mean,max}] [--pct PCT] [--load_model_path LOAD_MODEL_PATH]
              [--do_train] [--do_eval] [--do_eval_partial] [--do_eval_aliasing] [--do_predict] [--pretrain] [--save_predictions] [--train_batch_size TRAIN_BATCH_SIZE]
              [--eval_batch_size EVAL_BATCH_SIZE] [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--adam_epsilon ADAM_EPSILON]
              [--num_train_epochs NUM_TRAIN_EPOCHS] [--seed SEED]

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Path to datasets directory.
  --dataset_key {Project_CodeNet_Java250}
                        Dataset type string.
  --model_key {microsoft/codebert-base,microsoft/graphcodebert-base,roberta-base}
                        Model string.
  --output_dir OUTPUT_DIR
                        Dataset type string.
  --max_tokens MAX_TOKENS
                        Maximum number of tokens in a statement
  --use_statement_ids   Use statement ids in input embeddings
  --pooling_strategy {mean,max}
                        Pooling strategy.
  --pct PCT             Percentage of code to strip to simulate partial code.
  --load_model_path LOAD_MODEL_PATH
                        Path to trained model: Should contain the .bin files
  --do_train            Whether to run training.
  --do_eval             Whether to run eval on the dev set.
  --do_eval_partial     Whether to run eval on the partial snippets in dev set.
  --do_eval_aliasing    Whether to run variable aliasing on dev set.
  --do_predict          Whether to predict on given dataset.
  --pretrain            Use xBERT model off-the-shelf
  --save_predictions    Cache model predictions during evaluation.
  --train_batch_size TRAIN_BATCH_SIZE
                        Batch size per GPU/CPU for training.
  --eval_batch_size EVAL_BATCH_SIZE
                        Batch size per GPU/CPU for evaluation.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --weight_decay WEIGHT_DECAY
                        Weight decay for Adam optimizer.
  --adam_epsilon ADAM_EPSILON
                        Epsilon for Adam optimizer.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --seed SEED           random seed for initialization

``` 

### Sample Commands for Replicating Experiments:
1. Training
```bash
$ python run.py --data_dir ../dataset/slice/Project_CodeNet_Java250 --output_dir ../dataset/outputs/gcb_ft/ --do_train
```

2. Inference
```bash
$ python run.py --data_dir ../dataset/slice/Project_CodeNet_Java250 --load_model_path ../dataset/outputs/gcb_ft/Epoch_4/model.ckpt --do_eval
```
