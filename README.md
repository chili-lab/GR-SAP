# GR-SAP

This is repo for the paper ***GR-SAP: Generative Replay for Safety Alignment Preservation during Fine-Tuning***

## Data Processing

Install required dependencies

```
pip install -r requirements.txt
```

Preprocess safety datasets:

```
python extract_aegis.py/extract_beavertails.py/extract_tulu_subdatasets.py
```

Extract alignment data from an instruction-tuned model:

```
python extract_data.py --model $model
```

Preprocessed and extracted data are saved as csv files

Postprocess extracted data:

```
python process.py --file_name $file_name
```

Measure the semantic similarity using ``target_dataset.csv`` as reference:

```
python measure_similarity.py --file_name target_dataset.csv
```

## Downstream Finetuning

Finetune model on a downstream task with DeepSpeed:

```
deepspeed --num_nodes=$num_nodes --num_gpus=$num_gpus train.py \
            --deepspeed ds_config.json \
            --model $model \
            --alignment_data_file $alignment_data \
            --dataset $downstream_dataset \
            --ratio $mixing_ratio
```

The default folder for saving checkpoints is ``./ckpt_output``

## Evaluation

Evaluate the safety (characterized by ratio of harmful response) and downstream performance of a single checkpoint:

```
python score.py --model $checkpoint
```

To evaluate the whole training dynamic (multiple checkpoints):

```
python score_dir.py --checkpoint_dir $checkpoint_dir
```

