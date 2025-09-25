from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) 
    
    dist.init_process_group(backend="nccl")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    print(f"Process {rank}/{world_size} initialized on cuda:{local_rank}")
    return local_rank, world_size, rank

local_rank, world_size, rank = setup_distributed()
device = f"cuda:{local_rank}"


steps = 300
MODEL_PATH=f"/training/Anonymous/LLaMA-Factory/saves/Qwen2.5-VL-3b-VRSBench-grounding-train260/full/sft/checkpoint-{steps}"
OUTPUT_PATH="./logs_260_10shot_3B/rec_results_{DATASET}_qwen2_5vl_3b_instruct_sft260_{STEPS}.json"

#BSZ=64
BSZ=8
DATA_ROOT = "/training/Anonymous/VLM-R1/data/VRSBench"

TEST_DATASETS = ['VRSBench_EVAL_referring_detailed']
# TEST_DATASETS = ['VRSBench_EVAL_referring_non_dior']
# TEST_DATASETS = ['VRSBench_EVAL_referring_dior_only']
# TEST_DATASETS = ['test']
IMAGE_ROOT = "/training/Anonymous/grsm/VRSBench_data/Images_val"
# IMAGE_ROOT = "/training/Anonymous/VLM-R1/data"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": local_rank}, 
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

def extract_bbox_answer(content):
    bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    # bbox_pattern = r'\[(-?\d*\.?\d+),\s*(-?\d*\.?\d+),\s*(-?\d*\.?\d+),\s*(-?\d*\.?\d+)\]'
    bbox_match = re.search(bbox_pattern, content)

    if bbox_match:
        bbox = [float(bbox_match.group(1)), float(bbox_match.group(2)), float(bbox_match.group(3)), float(bbox_match.group(4))]
        return bbox
    return [0, 0, 0, 0]

def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    if union <= 0:
        return 0.0
    return float(inter)/union

num_samples = 20000
for ds in TEST_DATASETS:
    if rank == 0:
        print(f"Processing {ds}...")
    ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
    data = json.load(open(ds_path, "r"))
    random.seed(42)
    random.shuffle(data)
    data = data[:num_samples]

    QUESTION_TEMPLATE = "{Question} Output the detected target's bbox coordinates in JSON format. The format of the bbox coordinates is:\n```json\n[\n\t{{\"bbox_2d\": [x1, y1, x2, y2]}}\n]\n```\nIf there are no such targets in the image, simply respond with None."
    # print(f"The Question Template is: {QUESTION_TEMPLATE}")
    
    per_rank_data = len(data) // world_size
    start_idx = rank * per_rank_data
    end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
    rank_data = data[start_idx:end_idx]
    
    messages = []

    for x in rank_data:
        image_path = os.path.join(IMAGE_ROOT, x['image'])
        message = [
            {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": f"file://{image_path}"
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=x['problem'])
                }
            ]
        }]
        messages.append(message)

    rank_outputs = []
    all_outputs = []

    for i in tqdm(range(0, len(messages), BSZ), disable=rank != 0):
        batch_messages = messages[i:i + BSZ]
    
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        rank_outputs.extend(batch_output_text)

    print(f"Rank {rank} has finished processing {len(rank_outputs)} examples")

    all_outputs = [None] * len(data)
    rank_results = [(start_idx + i, output) for i, output in enumerate(rank_outputs)]

    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, rank_results)
    
    assert gathered_results[-1][-1][0] == len(data) - 1

    if rank == 0:
        for results in gathered_results:
            for idx, output in results:
                assert idx < len(all_outputs)
                all_outputs[idx] = output
        assert all_outputs[-1] is not None

        final_output = []
        correct_number = 0

        for input_example, model_output in zip(data, all_outputs):
            original_output = model_output
            ground_truth = input_example['solution']
            model_answer = extract_bbox_answer(original_output)
            
            correct = 0
            if model_answer is not None:
                if iou(model_answer, ground_truth) > 0.5:
                    correct = 1
            correct_number += correct
            
            result = {
                'image': input_example['image'],
                'question': input_example['problem'],
                'ground_truth': ground_truth,
                'model_output': original_output,
                'extracted_answer': model_answer,
                'correct': correct
            }
            final_output.append(result)

        accuracy = correct_number / len(data) * 100
        print(f"\nAccuracy of {ds}: {accuracy:.2f}%")

        output_path = OUTPUT_PATH.format(DATASET=ds, STEPS=steps)
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, "w") as f:
            json.dump({
                'accuracy': accuracy,
                'results': final_output
            }, f, indent=2)

        print(f"Results saved to {output_path}")
        print("-"*100)
    
    dist.barrier()


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 test_rec_baseline.py
