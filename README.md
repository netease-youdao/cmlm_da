# cmlm_da

Code for the COLING2022 paper "**Semantically Consistent Data Augmentation for Neural Machine Translation via Conditional Masked Language Model**".

Our repo is forked from [fairseq v0.8.0](https://github.com/facebookresearch/fairseq/tree/v0.8.0). 

## Usage
### IWSLT14 DE-EN

Train a cmlm model. 

```
export CUDA_VISIBLE_DEVICES=0
# The training data is generated as the same as previous works.
data_dir=iwslt14.tokenized.de-en/dump
# we modified the config of bert-base-multilingual-cased, see our paper for details
cmlm_config=cmlm_config/bert_config.json
# The following code shows how to train a cmlm for source side.
# Unset `--source-da` option if you want a cmlm for target side.
src_cmlm_dir=iwslt14.tokenized.de-en/src_cmlm_save
python train.py $data_dir \
                --task cmlm \
                -s de -t en \
                --cmlm-config $cmlm_config \
                --source-da \
                --clip-norm 0.1 \
                --dropout 0.1 \
                --max-tokens 16000 \
                --num-workers 4 \
                --optimizer adafactor \
                --criterion masked_lm \
                --weight-decay 0.0 \
                --lr 0.0009 \
                --lr-scheduler inverse_sqrt \
                --warmup-init-lr 1e-07 \
                --warmup-updates 4000 \
                --save-dir ${src_cmlm_dir} \
                --tensorboard-logdir ${src_cmlm_dir}/tensorboard \
                --ddp-backend=no_c10d \
                --arch lightconv \
```

Train nmt model with cmlm data augmentation.

```
export CUDA_VISIBLE_DEVICES=0
data_dir=iwslt14.tokenized.de-en/dump
src_cmlm_dir=iwslt14.tokenized.de-en/src_cmlm_save
tgt_cmlm_dir=iwslt14.tokenized.de-en/tgt_cmlm_save
cmlm-src-ckpt=${src_cmlm_dir}/checkpoint_best.pt
cmlm-src-config=cmlm_config/bert_config.json
cmlm-tgt-ckpt=${tgt_cmlm_dir}/checkpoint_best.pt
cmlm-tgt-config=cmlm_config/bert_config.json
cmlm_alpha=0.2
cmlm_temper=2.0
python train.py $data_dir \
                --cmlm-src-ckpt $cmlm_src_ckpt \
                --cmlm-src-config $cmlm_src_config \
                --cmlm-tgt-ckpt $cmlm_tgt_ckpt \
                --cmlm-tgt-config $cmlm_tgt_config \
                --cmlm-alpha $cmlm_alpha \
                --cmlm-temper $cmlm_temper \
                --clip-norm 0.0 \
                --dropout 0.3 \
                --max-tokens 4096 \
                --num-workers 4 \
                --optimizer adam \
                --adam-betas '(0.9, 0.98)' \
                --share-decoder-input-output-embed \
                --criterion label_smoothed_cross_entropy \
                --label-smoothing 0.1 \
                --weight-decay 0.0001 \
                --lr 0.0005 \
                --lr-scheduler inverse_sqrt \
                --warmup-init-lr 1e-07 \
                --warmup-updates 4000 \
                --task translation \
                --arch transformer_iwslt_de_en \
                --save-dir $model_dir \
                --max_source_positions 150 \
                --max_target_positions 150 \
                --tensorboard-logdir ${model_dir}/tensorboard \
                --ddp-backend=no_c10d \
                --left-pad-source False \
                --left-pad-target False \
```


## Citation

If you find the resources in this repository helpful, please cite as:
```
@inproceedings{cheng-etal-2022-semantically,
    title = "Semantically Consistent Data Augmentation for Neural Machine Translation via Conditional Masked Language Model",
    author = "Cheng, Qiao  and Huang, Jin  and Duan, Yitao",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.457",
    pages = "5148--5157",
}
```
