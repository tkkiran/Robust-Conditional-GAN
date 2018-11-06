script_file='run_biased.sh'
checkpoint_dir='biased'
trial=0

[ -d ${checkpoint_dir} ] || mkdir ${checkpoint_dir}

alpha=0.6
device=0
epoch=100

biased(){
  CUDA_VISIBLE_DEVICES=${2} python -u main.py \
    --algorithm "biased" --alpha ${1} --disc_type "vanilla" \
    --loss_fn "ce" --real_match \
    --noestimate_confuse --noaux_classifier \
    --noadd_noise --noconcat_y \
    --nospectral_norm --nomax_norm \
    --checkpoint_dir $checkpoint_dir --script_file ${script_file} \
    --epoch $epoch |& tee -a ${checkpoint_dir}/biased_alpha${1}_epoch${epoch}_${trial}.txt
}

biased ${alpha} ${device}