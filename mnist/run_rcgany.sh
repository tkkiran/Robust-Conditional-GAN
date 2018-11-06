script_file='run_rcgany.sh'
checkpoint_dir='rcgany'
trial=0

[ -d ${checkpoint_dir} ] || mkdir ${checkpoint_dir}

alpha=0.125
device=0
epoch=100
concat_y_layers='1'
noise_alpha=0.3
noise_start=30
noise_end=80

rcgany(){
  CUDA_VISIBLE_DEVICES=${2} python -u main.py \
    --algorithm "rcgan" --alpha ${1} --disc_type "projection" \
    --noestimate_confuse --noaux_classifier \
    --add_noise --noise_alpha ${noise_alpha} \
    --noise_start ${noise_start} --noise_end ${noise_end} \
    --concat_y --concat_y_layers ${concat_y_layers} \
    --spectral_norm --max_norm \
    --checkpoint_dir $checkpoint_dir --script_file ${script_file} \
    --epoch $epoch |& tee -a ${checkpoint_dir}/rcgany_alpha${1}_epoch${epoch}_${trial}.txt
}

rcgany ${alpha} ${device}