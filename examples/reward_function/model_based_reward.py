import torch
from typing import Dict, Optional, Any
from transformers import AutoProcessor, Qwen2VLConfig
from .qwen_reward import Qwen2Reward
from qwen_vl_utils import process_vision_info

_model = None
_tokenizer = None


BOX_START = "<|box_start|>"
BOX_END = "<|box_end|>"
BOX_PATTERN = re.compile(
    r"\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\),\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\)"
)
BRACKET_BOX_PATTERN = re.compile(
    r"\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]"
)


def load_scoring_model(
    model_name: str = "microsoft/deberta-v3-base",
    reward_model_device_id: str = "0",
) -> None:
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        try:
            _tokenizer = AutoProcessor.from_pretrained(model_name)
            _tokenizer.tokenizer.padding_side = "right"
            model_config = Qwen2VLConfig.from_pretrained(model_name)
            model_config.ranked_candidate_num = 1
            model_config.pad_token_id = _tokenizer.tokenizer.pad_token_id
            _model = Qwen2Reward(
                model_name,
                config=model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=f"cuda:{reward_model_device_id}",
            )
            _model.eval()
            for param in _model.parameters():
                param.requires_grad = False
            print(f"Successfully loaded model: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")


def get_model_score(
    text1: str, text2: str, problem: str, images: List[Image.Image]
) -> float:
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_scoring_model first.")

    predict_match = re.search(r"<answer>(.*?)</answer>", predict_str)
    predict_answer = (
        predict_match.group(1).strip() if predict_match else predict_str.strip()
    )
    ground_truth = text2

    problem = problem.replace("<image>", "")
    if isinstance(images, list):
        image = images[0]
    else:
        image = images

    def build_messages(question: str, answer: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image", "image": image},
                ],
            },
            {"role": "assistant", "content": answer},
        ]

    predict_message = build_messages(problem, predict_answer)
    ground_truth_message = build_messages(problem, ground_truth)

    messages = [predict_message, ground_truth_message]
    texts = [
        _tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(_model.device)

    with torch.inference_mode() and torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        outputs = _model(**inputs, return_dict=True, return_loss=False)

    score = outputs.values.flatten().to(torch.float32)
    predict_score, ground_truth_score = score
    predict_score = predict_score.item()
    ground_truth_score = ground_truth_score.item()

    if predict_score > ground_truth_score:
        diff = predict_score - ground_truth_score
        reward = 1.0 - np.exp(-diff * 0.2)
    else:
        reward = 0

    return reward


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def accuracy_reward(predict_str: str, ground_truth: str) -> float:
    predict_match = re.search(r"<answer>(.*?)</answer>", predict_str)
    predict_answer = (
        predict_match.group(1).strip() if predict_match else predict_str.strip()
    )

    reward = 0.0
    if BOX_START in ground_truth and BOX_END in ground_truth:
        answer_coords = BOX_PATTERN.findall(ground_truth)
        assert len(answer_coords) > 0, f"No answer coordinates found in {ground_truth}"
        answer_coords = [list(map(float, coord)) for coord in answer_coords]
        predict_coords = BOX_PATTERN.findall(predict_answer)
        if len(predict_coords) == 0:
            predict_coords = BRACKET_BOX_PATTERN.findall(predict_answer)
            if len(predict_coords) == 0:
                parse_success = False
            else:
                predict_coords = [list(map(float, coord)) for coord in predict_coords]
                predict_coords = [
                    [1000 * coord for coord in coords] for coords in predict_coords
                ]
                parse_success = True
        else:
            predict_coords = [list(map(float, coord)) for coord in predict_coords]
            parse_success = True

        if parse_success:
            if len(predict_coords) != len(answer_coords):
                reward = 0.0
            else:
                reward_for_per_correct_box = 0.5 / len(answer_coords)
                for pred_coord, ans_coord in zip(predict_coords, answer_coords):
                    iou = calculate_iou(pred_coord, ans_coord)
                    if iou > 0.5:
                        reward += reward_for_per_correct_box

                    if iou > 0.8:
                        reward += 0.5
                    elif iou > 0.7:
                        reward += 0.2
                    elif iou > 0.6:
                        reward += 0.1
    else:
        processed_predict = predict_answer.lower().replace(" ", "").replace(".", "")
        while " " in processed_predict:
            processed_predict = processed_predict.replace(" ", "")

        processed_solution = ground_truth.lower().replace(" ", "").replace(".", "")
        while " " in processed_solution:
            processed_solution = processed_solution.replace(" ", "")

        if processed_predict == processed_solution:
            reward = 1.0
        else:
            reward = 0.0

    return reward


def compute_score(
    predict_str: str,
    ground_truth: str,
    problem: str = "",
    images: List[Image.Image] = [],
    open_ended: bool = False,
    reward_model_name: str = "microsoft/deberta-v3-base",
    reward_model_device_id: str = "0",
    format_weight: float = 0.3,
    threshold: float = 0.5,
) -> Dict[str, float]:
    if _model is None:
        load_scoring_model(reward_model_name, reward_model_device_id)

    format_score = format_reward(predict_str)
    if format_score == 0.0:
        return {"overall": 0.0, "format": format_score, "accuracy": 0.0}
    else:
        if not open_ended:
            accuracy_score = accuracy_reward(predict_str, ground_truth)
        else:
            score = get_model_score(predict_str, ground_truth, problem, images)

    # Binary accuracy based on threshold
    accuracy = 1.0 if score >= threshold else 0.0

    # The model score itself serves as a continuous reward
    return {"overall": score, "model_score": score, "accuracy": accuracy}
