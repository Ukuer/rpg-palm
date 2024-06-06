set -ex

TOTAL_ID=40
SAMPLES=10
N_PROC=10
DATASETS=./datasets/bezier
NAME=palm_full_2

OUTPUT=./bezier_rpg

# important augments. split datasets into n parts.
# then process each part in parallel
SPLIT=2


# generate bezier palm first!
python get_bezier.py \
    --num_ids $TOTAL_ID \
    --samples $SAMPLES \
    --nproc $N_PROC \
    --output $DATASETS/test/ \

python script/split.py \
    --path $DATASETS \
    --split $SPLIT


for((i=0; i<$SPLIT; i++))
do
  CUDA_VISIBLE_DEVICES=0 python test.py \
    --name ${NAME} \
    --dataroot ${DATASETS}_${i}/test \
    --load_size 256 \
    --crop_size 256 \
    --input_nc 1 \
    --n_samples 1 \
    --dataset_mode single \
    --no_flip \
    --results_dir ./results/${NAME}_${i} \
    --use_dropout &
done
wait

python script/transform.py \
    --split $SPLIT \
    --output $OUTPUT \
    --path ./results/${NAME}


echo "results is saved in $OUTPUT"