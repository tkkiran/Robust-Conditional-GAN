script_file='run_unbiased.sh'
checkpoint_dir='unbiased'
trial=0

[ -d ${checkpoint_dir} ] || mkdir ${checkpoint_dir}

alpha=0.6
device=0
epoch=100

unbiased(){
  CUDA_VISIBLE_DEVICES=${2} python -u main.py \
    --algorithm "unbiased" --alpha ${1} --disc_type "projection" \
    --noestimate_confuse --noaux_classifier \
    --noadd_noise --noconcat_y \
    --spectral_norm --max_norm \
    --checkpoint_dir $checkpoint_dir --script_file ${script_file} \
    --epoch $epoch |& tee -a ${checkpoint_dir}/unbiased_alpha${1}_epoch${epoch}_${trial}.txt
}

unbiased ${alpha} ${device}