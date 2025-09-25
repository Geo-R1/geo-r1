import os
import json
from tqdm import tqdm
from collections import defaultdict

def iou(box1, box2):
    if not (isinstance(box1, list) and len(box1) == 4 and
            isinstance(box2, list) and len(box2) == 4):
        return 0.0
    box1 = [float(c) for c in box1]
    box2 = [float(c) for c in box2]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2] - 1, box2[2] - 1)
    inter_y2 = min(box1[3] - 1, box2[3] - 1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    if union <= 0:
        return 0.0
    return float(inter) / union

def calculate_accuracies_at_thresholds(results_list, iou_thresholds):
    if not results_list:
        return {thresh: (0.0, 0) for thresh in iou_thresholds}, 0

    total_samples = len(results_list)
    correct_counts = {threshold: 0 for threshold in iou_thresholds}

    for result in results_list:
        gt_box = result.get('ground_truth')
        pred_box = result.get('extracted_answer')
        current_iou = iou(pred_box, gt_box)
        
        for threshold in iou_thresholds:
            if current_iou > threshold:
                correct_counts[threshold] += 1
    
    report = {}
    for threshold, count in correct_counts.items():
        accuracy = (count / total_samples * 100) if total_samples > 0 else 0
        report[threshold] = (accuracy, count)
        
    return report, total_samples

def analyze_by_uniqueness_multi_iou(results_path, original_test_set_path, iou_thresholds):
    print(f"步骤 1/3: 正在从 '{os.path.basename(original_test_set_path)}' 加载 is_unique 信息...")
    uniqueness_map = {}
    try:
        with open(original_test_set_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        for item in original_data:
            key = (item.get('image'), item.get('problem'))
            uniqueness_map[key] = item.get('is_unique', False)
        print(f"成功构建 is_unique 查找表，包含 {len(uniqueness_map)} 条记录。")
    except Exception as e:
        print(f"❌ 读取或解析原始测试集时出错: {e}")
        return

    print(f"\n步骤 2/3: 正在从 '{os.path.basename(results_path)}' 加载并分析推理结果...")
    with open(results_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f).get('results', [])

    unique_results = []
    non_unique_results = []
    for result in tqdm(results_data, desc="正在分组"):
        key = (result.get('image'), result.get('question'))
        is_unique_flag = uniqueness_map.get(key)
        if is_unique_flag is None:
            continue
        if is_unique_flag:
            unique_results.append(result)
        else:
            non_unique_results.append(result)

    print("\n步骤 3/3: 正在计算统计数据并生成报告...")
    
    unique_report, total_unique = calculate_accuracies_at_thresholds(unique_results, iou_thresholds)
    non_unique_report, total_non_unique = calculate_accuracies_at_thresholds(non_unique_results, iou_thresholds)
    
    print("\n" + "="*70)
    print("📊 按目标唯一性 (is_unique) 划分的多IoU阈值准确率分析报告")
    print("="*70)
    
    print(f"✅ **唯一目标 (is_unique = True)** 的情况 (共 {total_unique} 例):")
    for threshold in sorted(iou_thresholds):
        accuracy, correct_count = unique_report[threshold]
        print(f"   - Acc @ IoU > {threshold:.2f} :  {accuracy:.2f}%  ({correct_count} / {total_unique})")
    
    print(f"\n❌ **非唯一目标 (is_unique = False)** 的情况 (共 {total_non_unique} 例):")
    for threshold in sorted(iou_thresholds):
        accuracy, correct_count = non_unique_report[threshold]
        print(f"   - Acc @ IoU > {threshold:.2f} :  {accuracy:.2f}%  ({correct_count} / {total_non_unique})")
        
    print("="*70)


def recalculate_accuracy_at_thresholds(results_path, iou_thresholds):
    if not os.path.exists(results_path):
        print(f"❌ 错误：结果文件未找到: '{results_path}'")
        return

    print(f"🔍 正在加载结果文件: {os.path.basename(results_path)}")
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_results = data.get('results', [])

    if not all_results:
        print("⚠️ 警告：文件中没有找到 'results' 列表，或列表为空。")
        return

    correct_counts = {threshold: 0 for threshold in iou_thresholds}
    total_samples = len(all_results)

    print(f"🔄 正在为 {total_samples} 条记录重新计算 IoU 并统计...")
    for result in all_results:
        gt_box = result.get('ground_truth')
        pred_box = result.get('extracted_answer')

        # 计算当前样本的IoU
        current_iou = iou(pred_box, gt_box)
        
        # 与每个阈值进行比较
        for threshold in iou_thresholds:
            if current_iou > threshold:
                correct_counts[threshold] += 1

    print("\n" + "="*50)
    print("📊 多 IoU 阈值准确率 (Acc@IoU) 分析报告")
    print("="*50)
    
    for threshold in sorted(iou_thresholds):
        correct = correct_counts[threshold]
        accuracy = (correct / total_samples * 100) if total_samples > 0 else 0
        print(f"   - Acc @ IoU > {threshold:.2f} :  {accuracy:.2f}%  ({correct} / {total_samples})")
        
    print("="*50)


if __name__ == "__main__":
    # recalculate_acc_unique
    RESULTS_JSON_PATH = '/training/Anonymous/VLM-R1/src/eval/logs_vrsbench_dapo/rec_results_qwen2_5_vl_7b_vrsbench260_grpo.json'
    ORIGINAL_TEST_SET_PATH = '/training/Anonymous/VLM-R1/data/VRSBench/VRSBench_EVAL_referring_detailed_with_unique.json'
    
    IOU_THRESHOLDS_TO_TEST = [0.5, 0.7]

    analyze_by_uniqueness_multi_iou(
        RESULTS_JSON_PATH, 
        ORIGINAL_TEST_SET_PATH,
        IOU_THRESHOLDS_TO_TEST
    )

    # recalculate_acc_multi_thres
    RESULTS_JSON_PATH = '/training/Anonymous/VLM-R1/src/eval/logs_vrsbench_dapo/rec_results_qwen2_5_vl_7b_vrsbench260_grpo.json'
    IOU_THRESHOLDS_TO_TEST = [0.5, 0.7]

    recalculate_accuracy_at_thresholds(RESULTS_JSON_PATH, IOU_THRESHOLDS_TO_TEST)