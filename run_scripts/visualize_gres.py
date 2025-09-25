import os
import sys
import json
import random
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import textwrap
import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    sys.path.insert(0, "/training/Anonymous/vlm-r1seg/sam2") # 确保路径正确
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("⚠️ 警告: sam2 未能成功导入或安装。")
    SAM2ImagePredictor = None

SAM2_PREDICTOR = None

def initialize_sam2_predictor(cfg_path, ckpt_path, device="cuda"):
    global SAM2_PREDICTOR
    if SAM2_PREDICTOR is None:
        if SAM2ImagePredictor is None: raise ImportError("SAM2 模块未能成功导入。")
        print(f"\n[SAM2 Initializing on {device}] 正在加载 SAM2 模型...")
        sam_model = build_sam2(cfg_path, ckpt_path)
        sam_model.to(device=device).eval()
        SAM2_PREDICTOR = SAM2ImagePredictor(sam_model)
        print("[SAM2 Initialized] SAM2 predictor 已就绪。\n")
    return SAM2_PREDICTOR

def generate_pred_mask(sam_predictor, image_np, extracted_answer):
    if not extracted_answer:
        return np.zeros(image_np.shape[:2], dtype=bool)

    sam_predictor.set_image(image_np)
    
    valid_prompts = [p for p in extracted_answer if p.get('box') and p.get('points')]
    if not valid_prompts:
        return np.zeros(image_np.shape[:2], dtype=bool)

    batch_boxes = np.array([p['box'] for p in valid_prompts])
    batch_points = np.array([p['points'] for p in valid_prompts])
    batch_point_labels = np.ones_like(batch_points[:, :, 0])

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        all_masks, _, _ = sam_predictor.predict(
            point_coords=batch_points,
            point_labels=batch_point_labels,
            box=batch_boxes,
            multimask_output=False,
        )
    
    combined_pred_mask = np.any(all_masks, axis=0)
    return combined_pred_mask

def draw_bbox_with_label(draw, bbox, color, label, font, align='left', other_label_bbox=None):
    draw.rectangle(bbox, outline=color, width=4)
    try:
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    except AttributeError:
        text_width, text_height = draw.textsize(label, font=font)

    padding = 5
    if align == 'left':
        bg_pos = [bbox[0], bbox[1] - text_height - padding*2, bbox[0] + text_width + padding*2, bbox[1]]
    else:
        bg_pos = [bbox[2] - text_width - padding*2, bbox[1] - text_height - padding*2, bbox[2], bbox[1]]

    if bg_pos[1] < 0:
        if align == 'left':
            bg_pos = [bbox[0], bbox[1], bbox[0] + text_width + padding*2, bbox[1] + text_height + padding*2]
        else:
            bg_pos = [bbox[2] - text_width - padding*2, bbox[1], bbox[2], bbox[1] + text_height + padding*2]
    
    if other_label_bbox and not (bg_pos[2] < other_label_bbox[0] or bg_pos[0] > other_label_bbox[2] or bg_pos[3] < other_label_bbox[1] or bg_pos[1] > other_label_bbox[3]):
        bg_pos = [bbox[2] - text_width - padding*2, bbox[3] - text_height - padding*2, bbox[2], bbox[3]]

    text_pos = (bg_pos[0] + padding, bg_pos[1] + padding)
    
    draw.rectangle(bg_pos, fill=(*color, 150))
    draw.text(text_pos, label, fill="white", font=font)
    
    return bg_pos

def main():
    # =========================  配置区  =========================
    RESULTS_JSON_PATH = '/training/Anonymous/VLM-R1/src/eval/eval_results/earthreason_test_20shot_maskonly/sam_od_results.json'
    ORIGINAL_DATASET_JSONL_PATH = '/training/Anonymous/VLM-R1/data/EarthReason/test_earthreason_new_mask_bbox_first_only.jsonl'
    SAM_CHECKPOINT_PATH = "/training/Anonymous/VLM-R1/seg/sam2/checkpoints/sam2.1_hiera_large.pt"
    SAM_CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_l.yaml"
    OUTPUT_DIR = './final_visualizations_report'
    NUM_TO_VISUALIZE = 20
    MUST_SEE_IDS = [
        "EarthReason_0826_0",
		"EarthReason_0824_0"
    ]
    # ==========================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_predictor = initialize_sam2_predictor(SAM_CONFIG_PATH, SAM_CHECKPOINT_PATH, device)

    print(f"🔍 正在从 '{os.path.basename(ORIGINAL_DATASET_JSONL_PATH)}' 构建GT Mask路径查找表...")
    mask_map = {json.loads(line)['id']: json.loads(line)['mask'] for line in open(ORIGINAL_DATASET_JSONL_PATH, 'r') if line.strip()}
    print(f"查找表构建完成，包含 {len(mask_map)} 条记录。")

    print(f"🔍 正在从 '{RESULTS_JSON_PATH}' 加载结果数据...")
    with open(RESULTS_JSON_PATH, 'r', encoding='utf-8') as f:
        all_results = json.load(f).get('results', [])
    
    must_see_records, other_records = [], []
    must_see_set = set(MUST_SEE_IDS)
    for r in all_results:
        if r.get('id') in must_see_set: must_see_records.append(r)
        else: other_records.append(r)
            
    random.shuffle(other_records)
    num_random_needed = max(0, NUM_TO_VISUALIZE - len(must_see_records))
    results_to_process = must_see_records + other_records[:num_random_needed]
    
    print(f"数据加载完成。将处理 {len(must_see_records)} 个指定样本和 {num_random_needed} 个随机样本。")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for result in tqdm(results_to_process, desc="生成可视化图片"):
        try:
            record_id = result.get('id')
            image_path = result.get('image')
            mask_path = mask_map.get(record_id)
            gt_bboxes = result.get('ground_truth_bbox_list', [])
            pred_prompts = result.get('extracted_answer', [])

            if not image_path or not os.path.exists(image_path): continue

            base_image = Image.open(image_path).convert("RGB")
            gt_mask = np.array(Image.open(mask_path).convert("L")) > 0 if mask_path and os.path.exists(mask_path) else None
            pred_mask = generate_pred_mask(sam_predictor, np.array(base_image), pred_prompts)

            fig, axes = plt.subplots(1, 3, figsize=(24, 12))
            fig.patch.set_facecolor('white')

            bbox_image = base_image.copy()
            draw = ImageDraw.Draw(bbox_image, 'RGBA')
            color_gt = (0, 114, 178)
            color_pred = (227, 26, 28)
            
            try:
                font_size = max(12, int(base_image.height / 50))
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
            
            gt_label_bbox = None
            if gt_bboxes:
                gt_label_bbox = draw_bbox_with_label(draw, gt_bboxes[0], color_gt, "Ground Truth", font, align='left')
                for bbox in gt_bboxes[1:]:
                    draw.rectangle(bbox, outline=color_gt, width=4)

            if pred_prompts:
                first_pred = pred_prompts[0]
                if first_pred.get('box'):
                    draw_bbox_with_label(draw, first_pred['box'], color_pred, "Prediction", font, align='right', other_label_bbox=gt_label_bbox)
                for p in pred_prompts:
                    if p.get('points'):
                        for kp in p['points']:
                            r = max(5, int(min(base_image.size) * 0.005))
                            draw.ellipse([kp[0]-r, kp[1]-r, kp[0]+r, kp[1]+r], fill="yellow", outline="black")

            axes[0].imshow(bbox_image)
            axes[0].set_title('BBox & Keypoint Comparison')
            axes[0].axis('off')

            axes[1].set_title('Ground Truth Mask')
            axes[1].axis('off')
            if gt_mask is not None:
                axes[1].imshow(gt_mask, cmap='gray')
            
            pred_mask_image = base_image.copy()
            if pred_mask is not None:
                overlay = np.zeros((*pred_mask.shape, 4), dtype=np.uint8)
                overlay[pred_mask] = [*color_pred, 100]
                overlay_pil = Image.fromarray(overlay, 'RGBA').resize(pred_mask_image.size, Image.NEAREST)
                pred_mask_image = Image.alpha_composite(pred_mask_image.convert("RGBA"), overlay_pil)
            axes[2].imshow(pred_mask_image)
            axes[2].set_title('Predicted Mask (from SAM)')
            axes[2].axis('off')
            
            plt.subplots_adjust(bottom=0.35)
            info_text = (
                f"ID: {record_id}\nCategory: {result.get('category', 'N/A')}\n"
                f"GT BBox List: {gt_bboxes}\n\n"
                f"Question:\n{'='*20}\n{textwrap.fill(result.get('question', 'N/A'), width=120)}\n\n"
                f"Model Raw Output:\n{'='*20}\n{textwrap.fill(result.get('model_output', 'N/A'), width=120)}"
            )
            fig.text(0.05, 0.02, info_text, fontsize=10, va="bottom", ha="left", wrap=True,
                     bbox=dict(boxstyle="round,pad=0.5", fc="ivory", ec="black", lw=1))
            
            save_path = os.path.join(OUTPUT_DIR, f"{record_id}.png")
            plt.savefig(save_path)
            plt.close(fig)

        except Exception as e:
            print(f"\n处理记录 {result.get('id', 'N/A')} 时发生错误: {e}")
            
    print("\n🎉 所有可视化图片生成完毕！")

if __name__ == "__main__":
    main()

