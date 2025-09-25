import contextlib
import copy
import io
import itertools
import json
import logging
import os
import warnings

import numpy as np
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import cv2
import torch
from pycocotools.coco import COCO

logger = logging.getLogger("coco")


def xyxy2xywh(bbox):
    """
    change bbox to coco format
    :param bbox: [x1, y1, x2, y2]
    :return: [x, y, w, h]
    """
    return [
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    ]


class CocoDetectionEvaluator:
    def __init__(self, ann_path):
        
        self.coco_api = COCO(ann_path)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.class_names = [cat["name"] for cat in self.cats]
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        
        # assert hasattr(dataset, "coco_api")
        # self.class_names = dataset.class_names
        # self.coco_api = dataset.coco_api
        # self.cat_ids = dataset.cat_ids
        self.metric_names = ["mAP", "AP_50", "AP_75", "AP_small", "AP_m", "AP_l"]

    def results2json(self, results):
        """
        results: {image_id: {label: [bboxes...] } }
        :return coco json format: {image_id:
                                   category_id:
                                   bbox:
                                   score: }
        """
        json_results = []
        for image_id, dets in results.items():
            for label, bboxes in dets.items():
                category_id = self.cat_ids[label]
                for bbox in bboxes:
                    score = float(bbox[4])
                    detection = dict(
                        image_id=int(image_id),
                        category_id=int(category_id),
                        bbox=xyxy2xywh(bbox),
                        score=score,
                    )
                    json_results.append(detection)
        return json_results

    def evaluate(self, results, save_dir, rank=-1):
        ### Original
        # results_json = self.results2json(results)
        ### lzy modified
        with open(results, 'r') as json_file:
            results_json = json.load(json_file)
        if len(results_json) == 0:
            warnings.warn(
                "Detection result is empty! Please check whether "
                "training set is too small (need to increase val_interval "
                "in config and train more epochs). Or check annotation "
                "correctness."
            )
            empty_eval_results = {}
            for key in self.metric_names:
                empty_eval_results[key] = 0
            return empty_eval_results
        json_path = os.path.join(save_dir, "results{}.json".format(rank))
        json.dump(results_json, open(json_path, "w"))
        coco_dets = self.coco_api.loadRes(json_path)
        coco_eval = COCOeval(
            copy.deepcopy(self.coco_api), copy.deepcopy(coco_dets), "bbox"
        )
        coco_eval.evaluate()
        coco_eval.accumulate()

        # use logger to log coco eval results
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            coco_eval.summarize()
        logger.info("\n" + redirect_string.getvalue())

        # print per class AP
        headers = ["class", "AP50", "mAP"]
        colums = 6
        per_class_ap50s = []
        per_class_maps = []
        precisions = coco_eval.eval["precision"]
        # dimension of precisions: [TxRxKxAxM]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(self.class_names) == precisions.shape[2]

        ### lzy modified
        per_class_results = []
        for idx, name in enumerate(self.class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision_50 = precisions[0, :, idx, 0, -1]
            precision_50 = precision_50[precision_50 > -1]
            ap50 = np.mean(precision_50) if precision_50.size else float("nan")
            per_class_ap50s.append(float(ap50 * 100))

            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            per_class_maps.append(float(ap * 100))
            per_class_results.append({name:float(ap * 100)})

        num_cols = min(colums, len(self.class_names) * len(headers))
        flatten_results = []
        for name, ap50, mAP in zip(self.class_names, per_class_ap50s, per_class_maps):
            flatten_results += [name, ap50, mAP]

        row_pair = itertools.zip_longest(
            *[flatten_results[i::num_cols] for i in range(num_cols)]
        )
        table_headers = headers * (num_cols // len(headers))
        table = tabulate(
            row_pair,
            tablefmt="pipe",
            floatfmt=".1f",
            headers=table_headers,
            numalign="left",
        )
        logger.info("\n" + table)

        aps = coco_eval.stats[:6]
        eval_results = {}
        for k, v in zip(self.metric_names, aps):
            eval_results[k] = v
        return eval_results, per_class_results
        
        
# evaluation.ipynb contents：
from ovd_coco_evaluation import CocoDetectionEvaluator

evaluator = CocoDetectionEvaluator('/training/Anonymous/VLM-R1/data/NWPU/new_split/annotations_test_set_new.json')
results_path = '/training/Anonymous/VLM-R1/src/eval/qwen25_eval_results-4class-grpo/predictions.json'

save_dir = './results'
results, per_class_results = evaluator.evaluate(results_path, save_dir)

### mAP and AP for all categories
print(results, per_class_results)

### mAP and AP for selected categories
selected_cate = [
        'airplane', 'ship', 'storage_tank', 'baseball_diamond',
        'tennis_court', 'basketball_court', 'ground_track_field',
        'harbor', 'bridge', 'vehicle']


# selected_cate = ['airplane', 'baseball_diamond', 'basketball_court']
# selected_cate = ['mouse', 'fork', 'hot dog', 'cat', 'airplane', 'suitcase', 'parking meter', 'sandwich', 'train', 'hair drier', 'toilet', 'toaster', 'snowboard', 'frisbee', 'bear']
results, per_class_results
AP_sum = 0
for item in per_class_results:
    for key, value in item.items():
        if key in selected_cate:
            print(f"Key: {key}, Value: {value}")
            AP_sum += value
print("mAP for selected categories: ", (AP_sum)/(len(selected_cate)))
