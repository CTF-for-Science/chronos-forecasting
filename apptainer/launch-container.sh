export repo='/mmfs1/home/alexeyy/storage/CTF-for-Science/models/chronos-forecasting'
apptainer run --nv --cwd "/app/code" --overlay "$repo"/apptainer/overlay.img --no-home --contain --bind "$repo":"/app/code" "$repo"/apptainer/gpu.sif # python -u /app/code/train.py --gpu 0 --dataset heat2d --use-normalizer unit  --normalize_x unit --component all --comment rel2  --loss-name rel2 --epochs 500 --batch-size 4 --model-name CGPT --optimizer AdamW --weight-decay 0.00005   --lr 0.001 --lr-method cycle  --grad-clip 1000.0   --n-hidden 128 --n-layers 3  --use-tb 0

# fuser -v overlay.img
# kill -9 <pid>
# fsck.ext3 overlay.img
