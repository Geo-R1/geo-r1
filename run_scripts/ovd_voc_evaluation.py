#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def load_coco_annotations(ann_path):
    data = json.load(open(ann_path, 'r', encoding='utf-8'))
    images = {img['id']: img for img in data['images']}
    anns   = defaultdict(list)
    for ann in data['annotations']:
        anns[ann['image_id']].append({
            'category_id': ann['category_id'],
            'bbox': ann['bbox']  # [x,y,w,h]
        })
    cats = {cat['id']: cat['name'] for cat in data['categories']}
    return images, anns, cats

def load_predictions(pred_path):
    data = json.load(open(pred_path, 'r', encoding='utf-8'))
    preds = defaultdict(list)
    for det in data:
        preds[det['image_id']].append({
            'category_id': det['category_id'],
            'bbox': det['bbox'],  # [x,y,w,h]
            'score': det['score']
        })
    return preds

def iou_xywh(b1, b2):
    x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1+w1, x2+w2)
    yb = min(y1+h1, y2+h2)
    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter = inter_w * inter_h
    union = w1*h1 + w2*h2 - inter
    return inter / union if union>0 else 0

def compute_precision_recall(gt_annos, pred_annos, iou_thresh):
    npos = sum(len(v) for v in gt_annos.values())
    if npos == 0:
        return [0.0], [0.0]

    preds = []
    for img_id, dets in pred_annos.items():
        for det in dets:
            preds.append((img_id, det['bbox'], det['score']))
    preds.sort(key=lambda x: x[2], reverse=True)

    matched = {img_id: np.zeros(len(gt_annos.get(img_id, [])), dtype=bool)
               for img_id in gt_annos}

    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    for idx, (img_id, bbox, score) in enumerate(preds):
        gts = gt_annos.get(img_id, [])
        best_iou = 0
        best_j = -1
        for j, gt in enumerate(gts):
            iou = iou_xywh(bbox, gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thresh:
            if not matched[img_id][best_j]:
                tp[idx] = 1
                matched[img_id][best_j] = True
            else:
                fp[idx] = 1
        else:
            fp[idx] = 1

    fp_cum = np.cumsum(fp)
    tp_cum = np.cumsum(tp)
    recalls = tp_cum / npos
    precisions = tp_cum / (tp_cum + fp_cum + 1e-10)
    return recalls.tolist(), precisions.tolist()

def voc_ap(recalls, precisions):
    ap = 0.
    for thr in np.linspace(0,1,11):
        p = [p for r,p in zip(recalls, precisions) if r>=thr]
        ap += max(p) if p else 0
    return ap / 11

def evaluate_voc_map(gt_path, pred_path, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.1,1.0,10)

    images, gt_dict, cat_map = load_coco_annotations(gt_path)
    pred_dict = load_predictions(pred_path)
    results = {}

    for cat_id, cat_name in cat_map.items():
        gt_annos = {img_id: [ann for ann in lst if ann['category_id']==cat_id]
                    for img_id,lst in gt_dict.items()}
        pred_annos = {img_id: [ann for ann in lst if ann['category_id']==cat_id]
                      for img_id,lst in pred_dict.items()}

        ap_per_thresh = []
        for thr in iou_thresholds:
            rec, prec = compute_precision_recall(gt_annos, pred_annos, thr)
            ap = voc_ap(rec, prec)
            ap_per_thresh.append(ap)

        results[cat_name] = {
            'APs': ap_per_thresh,
            'mAP': float(np.mean(ap_per_thresh))
        }

    # overall mAP
    all_maps = [v['mAP'] for v in results.values()]
    results['mAP@VOC'] = float(np.mean(all_maps))

    return results

def main():
    gt_json = "/training/Anonymous/VLM-R1/data/NWPU/new_split/annotations_test_set_new.json"
    pred_json = '/training/Anonymous/VLM-R1/src/eval/qwen25_eval_results-4class-sft/predictions.json'
    
    out_path = "voc_map_results.json"

    voc_results = evaluate_voc_map(gt_json, pred_json)

    print("VOC-style mAP Evaluation:\n")
    for cls, vals in voc_results.items():
        if cls != 'mAP@VOC':
            print(f"Class {cls}: mAP = {vals['mAP']*100:.2f}%")
            for i, ap in enumerate(vals['APs']):
                print(f"  IoU={0.1*(i+1):.1f}: AP={ap*100:.2f}%")
        else:
            print(f"\nOverall {cls} = {vals*100:.2f}%")
    with open(out_path, 'w') as f:
        json.dump(voc_results, f, indent=2)
    print(f"\nSaved results to {out_path}")

if __name__=="__main__":
    main()
