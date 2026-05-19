
CUDA_VISIBLE_DEVICES=2 accelerate launch train.py       --config configs/train/train_stage3_terediff.yaml \
                                                        --config_testr testr/configs/TESTR/TESTR_R_50_Polygon.yaml \
