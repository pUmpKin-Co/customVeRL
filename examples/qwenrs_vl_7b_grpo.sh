set -x

MODEL_PATH=Qwen/Qwen2-VL-2B-Instruct  # replace it with your local file path
REWARD_MODEL_PATH=PumpkinCat/ScoreRS
REWARD_MODEL_DEVICE_ID=1
FORMAT_WEIGHT=0.3
TENSOR_PARALLEL_SIZE=1

python3 -m customVeRL.verl.trainer.main \
    config=/home/aiscuser/customVeRL/examples/config.yaml \
    data.train_files=/home/aiscuser/scorers_data \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwenrs_vl_2b_grpo_geogrpo \
    trainer.project_name=VeRL \
    trainer.save_checkpoint_path=/home/aiscuser/qwenrs_vl_2b_grpo_geogrpo \
    trainer.n_gpus_per_node=1 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    worker.reward.reward_function_kwargs.reward_model_name=${REWARD_MODEL_PATH} \
    worker.reward.reward_function_kwargs.reward_model_device_id=${REWARD_MODEL_DEVICE_ID} \
    worker.reward.reward_function_kwargs.format_weight=${FORMAT_WEIGHT} \
    worker.rollout.tensor_parallel_size=${TENSOR_PARALLEL_SIZE}