import os
import json
import re
import argparse
import contextlib
import io
import itertools
import warnings
from tqdm import tqdm
from tabulate import tabulate
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig, Qwen2Config
from qwen_vl_utils import process_vision_info

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import logging

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


PROBLEM_TEMPLATE = "Please carefully check the image and detect the following objects: {category}. " \
    "Output each detected target's bbox coordinates in JSON format. The format of the bbox coordinates is:\n" \
    "```json\n" \
    "[\n" \
    '    {{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"target name\"}},\n' \
    '    {{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"target name\"}}\n' \
    "]\n" \
    "```\n" \
    "If there are no such targets in the image, simply respond with None."

QUESTION_TEMPLATE = "{problem} Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."


def extract_bboxes_from_response(response_text: str):
    match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
    if not match:
        # 如果没有找到代码块，可能是直接输出了列表
        match = re.search(r'(\[[\s\S]*\])', response_text)
        if not match:
            return []

    json_str = match.group(1).strip()
    
    if json_str.lower() == 'none':
        return []

    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            bboxes = [item['bbox_2d'] for item in data if 'bbox_2d' in item]
            return bboxes
    except json.JSONDecodeError:
        return []
    return []

def extract_bboxes_from_response(response_text: str):
    match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
    if not match:
        match = re.search(r'(\[[\s\S]*\])', response_text)
        if not match:
            return []

    json_str = match.group(1).strip()
    
    if json_str.lower() == 'none':
        return []

    try:
        data = json.loads(json_str)
        if not isinstance(data, list):
            return []

        bboxes = []
        for item in data:
            if 'bbox_2d' in item and isinstance(item['bbox_2d'], list) and len(item['bbox_2d']) == 4:
                try:
                    numeric_bbox = [int(coord) for coord in item['bbox_2d']]
                    bboxes.append(numeric_bbox)
                except (ValueError, TypeError):
                    continue
        
        return bboxes

    except json.JSONDecodeError:
        return []

def xyxy2xywh(bbox):
    return [
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    ]

# --- Distributed Setup ---

def setup_distributed():
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355" # or any free port
    
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    if int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
    
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    return local_rank, world_size, rank

# --- Inference Logic ---

def run_inference(args, device, world_size, rank):
    if rank == 0:
        print("--- Starting Inference Phase ---")
        print(f"Loading model from: {args.model_path}")


    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        # config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": device},
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    if rank == 0:
        print(f"Loading annotations from: {args.annotation_path}")
    with open(args.annotation_path, 'r') as f:
        coco_data = json.load(f)

    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    category_name_to_id = {v: k for k, v in categories.items()}
    
    with open(args.exist_cat_path, 'r') as f:
        exist_cat = json.load(f)

    images = coco_data['images']
    images_per_rank = len(images) // world_size
    start_idx = rank * images_per_rank
    end_idx = start_idx + images_per_rank if rank < world_size - 1 else len(images)
    rank_images = images[start_idx:end_idx]

    if rank == 0:
        print(f"Total images: {len(images)}. Rank {rank} processing {len(rank_images)} images.")

    bbox_count = 0
    rank_predictions = []
    for image_info in tqdm(rank_images, desc=f"Rank {rank} Inference"):
        image_id = image_info['id']
        file_name = image_info['file_name']
        image_path = os.path.join(args.image_root, file_name)

        if str(image_id) not in exist_cat:
            continue

        for category_name in exist_cat[str(image_id)]:
            if category_name not in category_name_to_id:
                continue

            category_id = category_name_to_id[category_name]

            problem = PROBLEM_TEMPLATE.format(category=category_name)
            # TODO: if you want to use the QUESTION_TEMPLATE, uncomment the next line
            # prompt = QUESTION_TEMPLATE.format(problem=problem)
            prompt = problem

            print(f"category:  {category_name} image_id:  {image_id}")
            print(f"prompt: {prompt}")

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ],
            }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            generated_ids = model.generate(**inputs, max_new_tokens=1024, use_cache=True, do_sample=False)
            generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
            response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
            print(f"Rank {rank} - Image ID: {image_id}, Category: {category_name}, Response: {response}")
            
            bboxes_xyxy = extract_bboxes_from_response(response)

            for bbox in bboxes_xyxy:
                bbox_xywh = xyxy2xywh(bbox)
                pred = {
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': bbox_xywh,
                    'score': 1.0
                }
                rank_predictions.append(pred)
                bbox_count += 1
                print(f'bbox_count: {bbox_count}')

    if world_size > 1:
        all_predictions = [None] * world_size
        dist.all_gather_object(all_predictions, rank_predictions)
    else:
        all_predictions = [rank_predictions]

    if rank == 0:
        final_results = [item for sublist in all_predictions for item in sublist]
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        prediction_file = os.path.join(args.output_dir, "predictions.json")
        with open(prediction_file, 'w') as f:
            json.dump(final_results, f)
        
        print(f"\nInference complete. Found {len(final_results)} bounding boxes.")
        print(f"Prediction results saved to: {prediction_file}")
        return prediction_file
    
    return None

# --- Evaluation Logic ---

def run_evaluation(args, prediction_file):
    print("\n--- Starting Evaluation Phase ---")
    
    coco_gt = COCO(args.annotation_path)
    
    with open(prediction_file, 'r') as f:
        results_json = json.load(f)
    if not results_json:
        print("Warning: Prediction file is empty. Evaluation skipped.")
        return

    coco_dt = coco_gt.loadRes(prediction_file)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()

    redirect_string = io.StringIO()
    with contextlib.redirect_stdout(redirect_string):
        coco_eval.summarize()
    print("\nCOCO Evaluation Summary:")
    print(redirect_string.getvalue())

    print("Per-class AP:")
    headers = ["class", "AP50", "mAP"]
    class_names = [cat['name'] for cat in coco_gt.loadCats(sorted(coco_gt.getCatIds()))]
    precisions = coco_eval.eval["precision"]

    per_class_ap50s = []
    per_class_maps = []
    
    for idx, name in enumerate(class_names):
        # AP@.50
        precision_50 = precisions[0, :, idx, 0, -1]
        precision_50 = precision_50[precision_50 > -1]
        ap50 = np.mean(precision_50) if precision_50.size else 0.0
        per_class_ap50s.append(float(ap50 * 100))

        # mAP (AP@[.50:.05:.95])
        precision_map = precisions[:, :, idx, 0, -1]
        precision_map = precision_map[precision_map > -1]
        ap = np.mean(precision_map) if precision_map.size else 0.0
        per_class_maps.append(float(ap * 100))

    num_cols = 3
    flatten_results = []
    for name, ap50, mAP in zip(class_names, per_class_ap50s, per_class_maps):
        flatten_results += [name, f"{ap50:.1f}", f"{mAP:.1f}"]
    
    row_pair = itertools.zip_longest(*[flatten_results[i::num_cols*2] for i in range(num_cols*2)])
    table_headers = headers * 2
    table = tabulate(
        row_pair,
        tablefmt="pipe",
        headers=table_headers,
        numalign="left",
    )
    print(table)
    
    print("\n--- Evaluation Finished ---")

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Run inference and evaluation for Qwen2.5-VL on NWPU dataset.")
    
    # --- 请在这里修改你的默认路径 ---
    parser.add_argument('--model_path', type=str, default="/training/Anonymous/VLM-R1/checkpoints/rl/Qwen2.5-VL-3B-Instruct-rlrs-nwpu/checkpoint", help="Path to the trained model checkpoint.")
    parser.add_argument('--annotation_path', type=str, default="/training/Anonymous/dataset/NWPU-new/annotations_test_set.json", help="Path to the COCO-style test annotation file.")
    parser.add_argument('--image_root', type=str, default="/training/Anonymous/dataset/NWPU-10/positive_image_set", help="Root directory of the test images.")
    parser.add_argument('--exist_cat_path', type=str, default="/training/Anonymous/dataset/NWPU-10/nwpu_exist_cat.json", help="Path to the JSON file mapping image_id to existing categories.")
    parser.add_argument('--output_dir', type=str, default="./qwen25_eval_results", help="Directory to save prediction JSON file.")
    # ------------------------------------
    
    args = parser.parse_args()

    local_rank, world_size, rank = setup_distributed()
    device = f"cuda:{local_rank}"

    prediction_file = run_inference(args, device, world_size, rank)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        if prediction_file and os.path.exists(prediction_file):
            run_evaluation(args, prediction_file)
        else:
            print("Prediction file not found. Skipping evaluation.")

if __name__ == "__main__":
    main()

'''
torchrun --nproc_per_node=8 test_ovd_nwpu_eval.py \
    --model_path /training/Anonymous/LLaMA-Factory/saves/Qwen2.5-VL-3b-nwpu-4class-10shot/full/sft/checkpoint \
    --annotation_path /training/Anonymous/VLM-R1/data/NWPU/new_split/annotations_test_set_new.json \
    --image_root /training/Anonymous/dataset/NWPU-10/positive_image_set \
    --exist_cat_path /training/Anonymous/VLM-R1/data/NWPU/new_split/nwpu_exist_cat.json \
    --output_dir ./qwen25_eval_results-4class-sft \
    2>&1 | tee -a qwen25_eval_results-4class-sft.txt
'''
