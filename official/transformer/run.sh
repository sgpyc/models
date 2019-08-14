#!/bin/bash

OUTPUT_DIR="/data/20190812_DynamicLossScale_FusedBatchNorm"
NVPROF_FLAG="--profile-child-processes"
TRAIN_FLAGS=""
TRAIN_FLAGS+=" --data_dir=/data/wmt32k-en2de-official"
TRAIN_FLAGS+=" --vocab_file=/data/wmt32k-en2de-official/vocab.ende.32768"
#TRAIN_FLAGS+=" --bleu_ref=/data/newstest2014/newstest2014.de"
#TRAIN_FLAGS+=" --bleu_source=/data/newstest2014/newstest2014.en"
TRAIN_FLAGS+=" --param_set=big"
TRAIN_FLAGS+=" --clean"
TRAIN_FLAGS+=" --log_steps=1"
TRAIN_FLAGS+=" --steps_between_evals=20000"
SCRIPT="/workspace/Projects/tf-models/official/transformer/v2/transformer_main.py"
#NUM_STEPS="10"
#NUM_GPUS="1"

export PYTHONPATH="/workspace/Projects/tf-models"

for batch in "static" "dynamic"; do
for num_steps in 10; do # 100; do
for num_gpus in 8 1; do
for xla in "enable" "disable"; do
for dtype in "fp16" "fp32"; do
    PER_GPU_BATCH_SIZE=""
    #if [ "${dtype}" == "fp32" ]; then
        PER_GPU_BATCH_SIZE="3072"
    #else
    #    PER_GPU_BATCH_SIZE="5120"
    #fi
    BATCH_SIZE=$(($PER_GPU_BATCH_SIZE * $num_gpus))

    OUTPUT_PATH=${OUTPUT_DIR}/${xla}xla_${batch}batch_${dtype}_${PER_GPU_BATCH_SIZE}x${num_gpus}x${num_steps}
    #export XLA_FLAGS=--xla_dump_to=${OUTPUT_PATH}
    mkdir -p ${OUTPUT_PATH}

    for profiler in "noprof" "nvprof" "tfprof"; do
	if [ "${profiler}" != "noprof" ] && [ "${num_steps}" -ge "100" ]; then
	  continue
        fi

        FLAGS=${TRAIN_FLAGS}
        FLAGS="${FLAGS} --model_dir=${OUTPUT_PATH}"
        FLAGS="${FLAGS} --log_dir=${OUTPUT_PATH}"
        FLAGS="${FLAGS} --train_steps=${num_steps}"
        FLAGS="${FLAGS} --num_gpus=${num_gpus}"
        FLAGS="${FLAGS} --dtype=${dtype}"
        FLAGS="${FLAGS} --batch_size=${BATCH_SIZE}"

        if [ "${xla}" == "enable" ]; then
	    FLAGS="${FLAGS} --enable_xla=true"
        fi
    
        if [ "${batch}" == "dynamic" ]; then
	    FLAGS="${FLAGS} --static_batch=false"
        else 
	    FLAGS="${FLAGS} --static_batch=true --max_length=64"
        fi

        EXEC="python3"
        if [ "${profiler}" == "nvprof" ]; then
	    EXEC="nvprof -o ${OUTPUT_PATH}/%p.nvprof ${NVPROF_FLAG} ${EXEC}"
        fi
        if [ "${profiler}" == "tfprof" ]; then
	    FLAGS="${FLAGS} --profile_steps=2,5"
        fi
        OUTPUT="${OUTPUT_PATH}/${profiler}_log.txt"

        echo "${EXEC} ${SCRIPT} ${FLAGS}" > ${OUTPUT}
        echo "${EXEC} ${SCRIPT} ${FLAGS} > ${OUTPUT} 2>&1"
	      ${EXEC} ${SCRIPT} ${FLAGS} >> ${OUTPUT} 2>&1
        rm ${OUTPUT_PATH}/*ckpt*
    done
    #unset XLA_FLAGS
done; done; done; done; done

