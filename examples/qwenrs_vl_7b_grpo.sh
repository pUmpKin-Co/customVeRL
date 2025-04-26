set -x

MODEL_PATH=PumpkinCat/Qwen2VL-7B-RS  # replace it with your local file path

python3 -m customVeRL.verl.trainer.main \
    config=/home/aiscuser/customVeRL/examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train,/home/aiscuser/scorers_data \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwenrs_vl_7b_grpo_geogrpo \
    trainer.project_name=VeRL \
    trainer.save_checkpoint_path=/blob/qwenrs_vl_7b_grpo_geogrpo \
    trainer.n_gpus_per_node=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \