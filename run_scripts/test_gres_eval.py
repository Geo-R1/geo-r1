import os
import sys
import json
import re
import argparse
import warnings
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from tabulate import tabulate

import torch
import torch.distributed as dist
from PIL import Image
from datetime import timedelta

# 确保这些依赖在您的环境中可用
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 引入 SAM2 相关模块
try:
    # 确保此路径对您的环境是正确的
    sys.path.insert(0, "/training/Anonymous/VLM-R1/seg/sam2") 
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("⚠️ 警告: sam2 未能成功导入或安装。")
    SAM2ImagePredictor = None

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
BATCH_SIZE = 4 # you can adjust based on your GPU memory

SYSTEM_PROMPT = (
    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
FORMAT_PROMPT = (
    "Please carefully check the image and answer: {Question}. Based on your answer, detect all relevant objects in the image."
    "Output each detected target's bbox coordinates in JSON format. "
    "The format of the bbox coordinates is:\n```json\n[\n    {{\"bbox_2d\": [x1, y1, x2, y2], \"keypoint1\": [x3, y3], \"keypoint2\": [x4, y4]}},\n    "
    "{{\"bbox_2d\": [x1, y1, x2, y2], \"keypoint1\": [x3, y3], \"keypoint2\": [x4, y4]}}\n]\n```\n"
    "If there are no such targets in the image, simply respond with None."
)
FULL_PROMPT_TEMPLATE = SYSTEM_PROMPT + '\n' + FORMAT_PROMPT

SAM2_PREDICTOR = None
def initialize_sam2_predictor(device="cuda"):
    global SAM2_PREDICTOR
    if SAM2_PREDICTOR is None:
        if SAM2ImagePredictor is None: raise ImportError("SAM2 模块未能成功导入。")
        print(f"\n[SAM2 Initializing on {device}] 正在加载 SAM2 模型...")
        checkpoint = "/training/Anonymous/vlm-r1seg/sam2/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam_model = build_sam2(model_cfg, checkpoint)
        sam_model.to(device=device).eval()
        SAM2_PREDICTOR = SAM2ImagePredictor(sam_model)
        print("[SAM2 Initialized] SAM2 predictor 已就绪。\n")
    return SAM2_PREDICTOR


def clip_mask_to_bbox(mask, bbox):
    clipped_mask = np.zeros_like(mask, dtype=bool)
    
    x_min, y_min, x_max, y_max = [int(c) for c in bbox]
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, mask.shape[1])
    y_max = min(y_max, mask.shape[0])

    if x_min < x_max and y_min < y_max:
        clipped_mask[y_min:y_max, x_min:x_max] = mask[y_min:y_max, x_min:x_max]
    
    return clipped_mask


def setup_distributed():
    if "RANK" not in os.environ:
        os.environ["RANK"], os.environ["LOCAL_RANK"], os.environ["WORLD_SIZE"] = "0", "0", "1"
        os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", "12355"
    local_rank = int(os.environ.get("LOCAL_RANK"))
    torch.cuda.set_device(local_rank) 
    if int(os.environ["WORLD_SIZE"]) > 1: dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    return local_rank, dist.get_world_size(), dist.get_rank()

def parse_vlm_predictions_for_sam(content: str) -> list:
    json_str_to_parse = None
    match_strict = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if match_strict: json_str_to_parse = match_strict.group(1).strip()
    else:
        match_answer = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if match_answer:
            answer_content = match_answer.group(1).strip()
            match_array = re.search(r"(\[.*\])", answer_content, re.DOTALL)
            json_str_to_parse = match_array.group(1) if match_array else answer_content
    if not json_str_to_parse: return []
    try:
        data = json.loads(json_str_to_parse)
        if not isinstance(data, list): return []
        parsed_preds = []
        for item in data:
            bbox, kp1, kp2 = item.get("bbox_2d"), item.get("keypoint1"), item.get("keypoint2")
            if (isinstance(bbox, list) and len(bbox) == 4 and isinstance(kp1, list) and len(kp1) == 2 and isinstance(kp2, list) and len(kp2) == 2):
                parsed_preds.append({"box": bbox, "points": [kp1, kp2]})
        return parsed_preds
    except json.JSONDecodeError: return []

def run_inference_and_calc_iou(args, all_data, device, world_size, rank):
    if rank == 0: print(f"\n--- 开始推理, 共 {len(all_data)} 个样本, Batch Size: {BATCH_SIZE} ---")
    
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map={"": device})
    processor = AutoProcessor.from_pretrained(args.model_path)
    sam_predictor = initialize_sam2_predictor(device=device)

    data_per_rank = len(all_data) // world_size
    start_idx = rank * data_per_rank
    end_idx = start_idx + data_per_rank if rank < world_size - 1 else len(all_data)
    rank_data = all_data[start_idx:end_idx]
    if rank == 0: print(f"Rank {rank} 将处理 {len(rank_data)} 个样本.")

    rank_results = []
    for i in tqdm(range(0, len(rank_data), BATCH_SIZE), desc=f"Rank {rank} 推理中", disable=(rank!=0)):
        batch_data = rank_data[i:i + BATCH_SIZE]
        batch_messages = []
        
        for item in batch_data:
            prompt_text = FULL_PROMPT_TEMPLATE.format(Question=item['problem'])
            image_path = item['image']
            batch_messages.append([{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_text}]}])
        
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        image_inputs, _ = process_vision_info(batch_messages)
        inputs = processor(text=texts, images=image_inputs, padding=True, padding_side="left", return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = vlm_model.generate(**inputs, max_new_tokens=1024, use_cache=True, do_sample=False)
        
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        batch_responses = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

        for item, response in zip(batch_data, batch_responses):
            try:
                print(f"[Rank {rank}] Processing item: {item['id']}") 
                pred_prompts_for_sam = parse_vlm_predictions_for_sam(response)
                gt_mask = np.array(Image.open(item['mask']).convert("L")) > 0
                intersection_val = 0
                union_val = int(np.sum(gt_mask))
                
                result_dict = {'id': item['id'], 'category': item['category'], 'intersection': 0, 'union': int(np.sum(gt_mask))}

                if pred_prompts_for_sam:
                    image = np.array(Image.open(item['image']).convert("RGB"))
                    sam_predictor.set_image(image)
                    
                    combined_pred_mask = np.zeros(image.shape[:2], dtype=bool)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        for prompt in pred_prompts_for_sam:
                            masks, _, _ = sam_predictor.predict(point_coords=np.array(prompt['points']), point_labels=np.array([1, 1]), box=np.array(prompt['box']), multimask_output=False)

                            clipped_mask = clip_mask_to_bbox(masks[0], prompt['box'])
                            combined_pred_mask = np.logical_or(combined_pred_mask, clipped_mask)

                    if gt_mask.shape != combined_pred_mask.shape:
                        combined_pred_mask = np.array(Image.fromarray(combined_pred_mask).resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST))
                    
                    # result_dict['intersection'] = int(np.sum(np.logical_and(combined_pred_mask, gt_mask)))
                    # result_dict['union'] = int(np.sum(np.logical_or(combined_pred_mask, gt_mask)))
                    intersection_val = int(np.sum(np.logical_and(combined_pred_mask, gt_mask)))
                    union_val = int(np.sum(np.logical_or(combined_pred_mask, gt_mask)))

                result_dict = {
                    'id': item['id'], 'category': item['category'], 'image': item['image'],
                    'question': item['problem'], 'ground_truth': item.get('bbox_list'),
                    'model_output': response, 'extracted_answer': pred_prompts_for_sam,
                    'intersection': intersection_val, 'union': union_val
                }
                rank_results.append(result_dict)
            except Exception as e:
                print(f"处理样本 {item['id']} 时发生错误: {e}")
                gt_mask_area = int(np.sum(np.array(Image.open(item['mask']).convert("L")) > 0))
                # rank_results.append({'id': item['id'], 'category': item['category'], 'intersection': 0, 'union': gt_mask_area})
                rank_results.append({
                    'id': item['id'], 'category': item['category'], 'image': item['image'],
                    'question': item['problem'], 'ground_truth': item.get('bbox_list'),
                    'model_output': 'ERROR_DURING_INFERENCE', 'extracted_answer': [],
                    'intersection': 0, 'union': gt_mask_area
                })

    all_results = []
    if world_size > 1:
        gathered_objects = [None] * world_size
        dist.all_gather_object(gathered_objects, rank_results)
        if rank == 0:
            for sublist in gathered_objects: all_results.extend(sublist)
    else:
        all_results = rank_results
    return all_results


def run_inference_and_calc_iou_new(args, all_data, device, world_size, rank):
    if rank == 0: print(f"\n--- 开始推理, 共 {len(all_data)} 个样本, Batch Size: {BATCH_SIZE} ---")
    
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map={"": device})
    processor = AutoProcessor.from_pretrained(args.model_path)
    sam_predictor = initialize_sam2_predictor(device=device)

    data_per_rank = len(all_data) // world_size
    start_idx = rank * data_per_rank
    end_idx = start_idx + data_per_rank if rank < world_size - 1 else len(all_data)
    rank_data = all_data[start_idx:end_idx]
    if rank == 0: print(f"Rank {rank} 将处理 {len(rank_data)} 个样本.")

    rank_results = []
    for i in tqdm(range(0, len(rank_data), BATCH_SIZE), desc=f"Rank {rank} 推理中", disable=(rank!=0)):
        batch_data = rank_data[i:i + BATCH_SIZE]
        batch_messages = []
        
        for item in batch_data:
            prompt_text = FULL_PROMPT_TEMPLATE.format(Question=item['problem'])
            image_path = item['image']
            batch_messages.append([{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_text}]}])
        
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        image_inputs, _ = process_vision_info(batch_messages)
        inputs = processor(text=texts, images=image_inputs, padding=True, padding_side="left", return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = vlm_model.generate(**inputs, max_new_tokens=1024, use_cache=True, do_sample=False)
        
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        batch_responses = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

        for item, response in zip(batch_data, batch_responses):
            try:
                pred_prompts_for_sam = parse_vlm_predictions_for_sam(response)
                gt_mask = np.array(Image.open(item['mask']).convert("L")) > 0
                
                result_dict = {'id': item['id'], 'category': item['category'], 'intersection': 0, 'union': np.sum(gt_mask)}

                if pred_prompts_for_sam:
                    image = np.array(Image.open(item['image']).convert("RGB"))
                    sam_predictor.set_image(image)
                    
                    batch_boxes = np.array([p['box'] for p in pred_prompts_for_sam])
                    batch_points = np.array([p['points'] for p in pred_prompts_for_sam])
                    batch_point_labels = np.ones_like(batch_points[:, :, 0]) # 所有点都是前景点(1)

                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        all_masks, _, _ = sam_predictor.predict(
                            point_coords=batch_points,
                            point_labels=batch_point_labels,
                            box=batch_boxes,
                            multimask_output=False,
                        )
                    
                    combined_pred_mask = np.any(all_masks, axis=0)

                    if gt_mask.shape != combined_pred_mask.shape:
                        combined_pred_mask = np.array(Image.fromarray(combined_pred_mask).resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST))
                    
                    result_dict['intersection'] = np.sum(np.logical_and(combined_pred_mask, gt_mask))
                    result_dict['union'] = np.sum(np.logical_or(combined_pred_mask, gt_mask))

                rank_results.append(result_dict)
            except Exception as e:
                print(f"处理样本 {item['id']} 时发生错误: {e}")
                gt_mask_area = np.sum(np.array(Image.open(item['mask']).convert("L")) > 0)
                rank_results.append({'id': item['id'], 'category': item['category'], 'intersection': 0, 'union': gt_mask_area})

    all_results = []
    if world_size > 1:
        gathered_objects = [None] * world_size
        dist.all_gather_object(gathered_objects, rank_results)
        if rank == 0:
            for sublist in gathered_objects: all_results.extend(sublist)
    else:
        all_results = rank_results
    return all_results


def run_final_evaluation(results):
    if not results:
        print("警告: 没有可供评估的结果。")
        return

    print("\n--- 开始最终评估 ---")
    
    print("\n" + "="*60)
    print("           SAM-OD 任务总体评估结果")
    print("="*60)
    print("评估指标说明:")
    print("  - gIoU (mean IoU): 逐个计算每个样本的IoU，然后取平均值。反映模型在所有样本上的平均表现。")
    print("  - cIoU (cumulative IoU): 将所有样本的交集之和除以并集之和。更能反映大目标的表现。")
    print("-" * 60)
    
    all_ious = [(item['intersection'] / item['union'] if item['union'] > 0 else 0) for item in results]
    total_intersection = sum(item['intersection'] for item in results)
    total_union = sum(item['union'] for item in results)
    overall_gIoU = np.mean(all_ious)
    overall_cIoU = total_intersection / total_union if total_union > 0 else 0
    
    print(f"总样本数: {len(results)}")
    print(f"** 总体 gIoU: {overall_gIoU:.4f} **")
    print(f"** 总体 cIoU: {overall_cIoU:.4f} **")
    print("="*60)

    print("\n" + "="*60)
    print("           各类别细分评估结果")
    print("="*60)
    
    results_by_cat = defaultdict(list)
    for item in results:
        results_by_cat[item['category']].append(item)
    
    table_data = []
    category_metrics_report = {}
    for category, cat_results in sorted(results_by_cat.items()):
        cat_ious = [(item['intersection'] / item['union'] if item['union'] > 0 else 0) for item in cat_results]
        cat_total_intersection = sum(item['intersection'] for item in cat_results)
        cat_total_union = sum(item['union'] for item in cat_results)
        
        cat_gIoU = np.mean(cat_ious)
        cat_cIoU = cat_total_intersection / cat_total_union if cat_total_union > 0 else 0
        
        table_data.append([category, len(cat_results), f"{cat_gIoU:.4f}", f"{cat_cIoU:.4f}"])
        category_metrics_report[category] = {
            "count": len(cat_results),
            "gIoU": cat_gIoU,
            "cIoU": cat_cIoU
        }
        
    table_data.sort(key=lambda x: float(x[2]), reverse=True)
    headers = ["类别 (Category)", "样本数", "gIoU", "cIoU"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("="*60)

    return category_metrics_report

def main():
    parser = argparse.ArgumentParser(description="为SAM-OD任务运行评估。")
    parser.add_argument('--model_path', type=str, required=True, help="要评估的模型checkpoint路径。")
    parser.add_argument('--data_path', type=str, required=True, help="源JSONL文件路径。")
    parser.add_argument('--output_dir', type=str, default="./sam_od_eval_results", help="保存中间结果的目录。")
    args = parser.parse_args()

    local_rank, world_size, rank = setup_distributed()
    
    all_data = []
    if rank == 0:
        with open(args.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                all_data.append(json.loads(line.strip()))
    
    if world_size > 1:
        data_to_bcast = [all_data]
        dist.broadcast_object_list(data_to_bcast, src=0)
        all_data = data_to_bcast[0]

    if not all_data:
        if rank == 0: print("数据加载失败。")
        return

    results = run_inference_and_calc_iou(args, all_data, f"cuda:{local_rank}", world_size, rank)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        model_name = os.path.basename(os.path.normpath(args.model_path))
        
        category_report = run_final_evaluation(results)
        
        all_ious = [(item['intersection'] / item['union'] if item['union'] > 0 else 0) for item in results]
        final_results_list_to_save = []
        for i, res in enumerate(results):
            clean_res = {
                "id": res.get('id'),
                "image": res.get('image'),
                "question": res.get('question'),
                "ground_truth_bbox_list": res.get('ground_truth'),
                "model_output": res.get('model_output'),
                "extracted_answer": res.get('extracted_answer'),
                "iou_score": all_ious[i]
            }
            final_results_list_to_save.append(clean_res)

        overall_gIoU = np.mean(all_ious) if all_ious else 0
        total_intersection = sum(item['intersection'] for item in results)
        total_union = sum(item['union'] for item in results)
        overall_cIoU = total_intersection / total_union if total_union > 0 else 0

        final_output_json = {
            "model_path": args.model_path,
            "overall_gIoU": overall_gIoU,
            "overall_cIoU": overall_cIoU,
            "per_category_metrics": category_report,
            "results": final_results_list_to_save
        }

        results_file_name = f"sam_od_results_{model_name}_detailed.json"
        results_file_path = os.path.join(args.output_dir, results_file_name)
        with open(results_file_path, 'w') as f:
            json.dump(final_output_json, f, indent=2)
        print(f"\n✅ 包含分类别分数的详细结果已保存到: {results_file_path}")

if __name__ == "__main__":
    main()


'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 test_res_sam_eval.py \
    --model_path /training/Anonymous/VLM-R1/checkpoints/rl/Qwen2.5-VL-3B-Instruct-rlrs-sam-od-earthreason-10shot/checkpoint \
    --data_path /training/Anonymous/VLM-R1/data/EarthReason/test_earthreason_final.jsonl \
    --output_dir ./eval_results/earthreason_test
'''
