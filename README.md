# Geo-R1

## Model
* Load or download model from [Geo-R1 huggingface page](https://huggingface.co/Geo-R1)
  
* Naming Convention "Geo-R1-3B-GRPO-GRES-10shot"
  * **Geo-R1**: Our model trained with RL-based post-training paradigm 
  * **3B**: Model size
  * **GRPO**: RL algorithm
  * **GRES**: General Referring Expression Task
  * **10shot**: Number of few-shot samples


## Dataset
Geo-R1 utilizes several public remote sensing datasets. We provide a series of preprocessing scripts to convert them into a unified format suitable for our training and evaluation pipelines, also you can download our processed dataset from repo and use it directly.

| Dataset Name | Task Type | Evaluation Script |
| :--- | :--- | :--- |
| **EarthReason** | **FS-GRES** | `test_gres_eval.py` |
| **RRSIS-D** | **FS-GRES (Cross-Dataset Eval)** | `test_gres_eval.py` |
| **NWPU-VHR-10** | **FS-OVD** | `test_ovd_nwpu_eval.py` |
| **VRSBench** | **FS-REC** | `test_rec_r1_eval.py` |


## Inference
### FS-REC Task
This task is a simplified version of GRES where the output is a single bounding box. The evaluation is based on **IoU@0.5** and **IoU@0.7**. This is primarily tested on the **VRSBench** dataset. The evaluation results are typically generated in a rich JSON format for detailed analysis.

**Example Evaluation Command (on VRSBench dataset):**
```bash
# you need to change the model path in file
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 test_rec_r1_eval.py
# analyze
python recal_rec_acc_unique.py
```


### FS-OVD Task
This task evaluates the model's ability to detect all objects of a given class in an image. We use standard **COCO mAP** metrics for evaluation. This is primarily tested on the **NWPU-VHR-10** dataset.

**Example Evaluation Command (on NWPU dataset):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 test_ovd_nwpu_eval.py \
    --model_path /path/to/your/Geo-R1-3B-GRPO-OVD-10shot/checkpoint \
    --annotation_path /path/to/your/nwpu/annotations_test_set_new.json \
    --image_root /path/to/your/nwpu/positive_image_set \
    --exist_cat_path /path/to/your/nwpu/nwpu_exist_cat.json \
    --output_dir ./eval_results/nwpu_test_result
```

### FS-GRES Task

This task evaluates the model's ability to produce a segmentation mask for a given textual description, using SAM as a proxy. We use **gIoU** (mean IoU) and **cIoU** (cumulative IoU) as the primary metrics.

**Example Evaluation Command (on EarthReason or RRSIS-D dataset):**
```bash
# Set CUDA devices, then run the evaluation script
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 test_gres_eval.py \
    --model_path /path/to/your/Geo-R1-3B-GRPO-GRES-10shot/checkpoint \
    --data_path /path/to/your/test_earthreason_final.jsonl \
    --output_dir ./eval_results/earthreason_test_result
```




<!--
**Geo-R1/geo-r1** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
