script_file='run_rcganu.sh'
checkpoint_dir='rcganu'
trial=0

[ -d ${checkpoint_dir} ] || mkdir ${checkpoint_dir}

alpha=0.3
device=0
epoch=100

rcganu(){
  CUDA_VISIBLE_DEVICES=${2} python -u main.py \
    --algorithm "rcgan" --alpha ${1} --disc_type "projection" \
    --estimate_confuse --aux_classifier \
    --noadd_noise --noconcat_y \
    --spectral_norm --max_norm \
    --checkpoint_dir $checkpoint_dir --script_file ${script_file} \
    --epoch $epoch |& tee -a ${checkpoint_dir}/rcganu_alpha${1}_epoch${epoch}_${trial}.txt
}

rcganu ${alpha} ${device}