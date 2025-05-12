set -x

MODEL_PATH=Qwen/Qwen2-VL-2B-Instruct  # replace it with your local file path
REWARD_MODEL_PATH=PumpkinCat/ScoreRS
REWARD_MODEL_SERVER_URL=http://localhost:8000/v1/rewards
FORMAT_WEIGHT=0.3
TENSOR_PARALLEL_SIZE=1
FORMAT_PROMPT=/home/aiscuser/customVeRL/examples/format_prompt/scorers_format.json

MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=2048

python3 -m customVeRL.verl.trainer.main \
    config=/home/aiscuser/customVeRL/examples/config.yaml \
    data.train_files=/home/aiscuser/scorers_data \
    data.val_files=/home/aiscuser/scorers_test \
    data.image_key=image \
    data.format_prompt=${FORMAT_PROMPT} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwenrs_vl_2b_grpo_geogrpo \
    trainer.project_name=VeRL \
    trainer.save_checkpoint_path=/home/aiscuser/qwenrs_vl_2b_grpo_geogrpo \
    trainer.n_gpus_per_node=1 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    worker.reward.reward_function_kwargs.reward_model_name=${REWARD_MODEL_PATH} \
    worker.reward.reward_function_kwargs.reward_server_url=${REWARD_MODEL_SERVER_URL} \
    worker.reward.reward_function_kwargs.format_weight=${FORMAT_WEIGHT} \
    worker.rollout.tensor_parallel_size=${TENSOR_PARALLEL_SIZE}