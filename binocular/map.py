import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def calculate_iou(box1, box2):
    """计算两个边界框的IoU值"""
    # 转换为边界框的坐标(x1, y1, x2, y2)
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    
    # 计算交集区域
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    
    # 如果交集区域不存在，IoU为0
    if x2 <= x1 or y2 <= y1:
        return 0
    
    # 计算交集和并集的面积
    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union = box1_area + box2_area - intersection
    
    # 计算IoU
    iou = intersection / union
    return iou

def read_yolo_format(file_path, conf_threshold, is_pred=False):
    """读取YOLO格式的文件，返回边界框列表"""
    boxes = []
    if not os.path.exists(file_path):
        return boxes
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
            
        class_id = int(parts[0])
        # 归一化的中心点坐标和宽高
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        if is_pred:
            # 预测框包含置信度
            if len(parts) < 6:
                score = 1.0
            else:
                score = float(parts[5])
            if score >= conf_threshold:
                boxes.append((class_id, x_center, y_center, width, height, score))
        else:
            # 标注框不包含置信度
            boxes.append((class_id, x_center, y_center, width, height))
    
    return boxes

def evaluate_image(pred_boxes, gt_boxes, iou_threshold=0.5):
    """评估单张图像的预测结果，返回每个类别的TP、FP和置信度"""
    # 按类别组织预测框和标注框
    pred_boxes_by_class = defaultdict(list)
    gt_boxes_by_class = defaultdict(list)
    
    for box in pred_boxes:
        class_id = box[0]
        pred_boxes_by_class[class_id].append(box)
    
    for box in gt_boxes:
        class_id = box[0]
        gt_boxes_by_class[class_id].append(box)
    
    # 按置信度对预测框排序
    for class_id in pred_boxes_by_class:
        pred_boxes_by_class[class_id].sort(key=lambda x: x[5], reverse=True)
    
    results = {}
    for class_id in pred_boxes_by_class:
        pred_boxes_class = pred_boxes_by_class[class_id]
        gt_boxes_class = gt_boxes_by_class.get(class_id, [])
        
        # 标记已匹配的标注框
        gt_matched = [False] * len(gt_boxes_class)
        
        # 初始化TP和FP列表
        tp = [0] * len(pred_boxes_class)
        fp = [0] * len(pred_boxes_class)
        scores = [box[5] for box in pred_boxes_class]
        
        # 遍历每个预测框
        for pred_idx, pred_box in enumerate(pred_boxes_class):
            max_iou = 0
            max_gt_idx = -1
            
            # 提取预测框的坐标部分（去掉类别和置信度）
            pred_coords = pred_box[1:5]
            
            # 找到与当前预测框IoU最大的标注框
            for gt_idx, gt_box in enumerate(gt_boxes_class):
                # 提取标注框的坐标部分（去掉类别）
                gt_coords = gt_box[1:5]
                iou = calculate_iou(pred_coords, gt_coords)
                
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            # 判断是TP还是FP
            if max_iou >= iou_threshold and not gt_matched[max_gt_idx]:
                tp[pred_idx] = 1
                gt_matched[max_gt_idx] = True
            else:
                fp[pred_idx] = 1
        
        results[class_id] = {
            'tp': tp,
            'fp': fp,
            'scores': scores,
            'num_gt': len(gt_boxes_class)
        }
    
    return results

def calculate_ap(recall, precision):
    ap = 0
    last_p = 0
    last_t = 0
    precisions = []
    for t in np.arange(0, 1.01, 0.01):
        if np.sum(recall >= t) > 0:
            last_p = np.max(precision[recall >= t])
            last_t = t
    last_ranges = 1.0 - last_t
    
    for t in np.arange(0, 1.01, 0.01):
        if np.sum(recall >= t) == 0:
            p = (1.0 - t) / last_ranges * last_p
        else:
            p = np.max(precision[recall >= t])
        ap += p / 101.0
        precisions.append(p)
    return ap, precisions, np.arange(0, 1.01, 0.01)

def calculate_map(all_results, iou_threshold=0.5, draw_plot=False):
    """计算mAP值"""
    # 收集所有类别的结果
    class_results = defaultdict(lambda: {'tp': [], 'fp': [], 'scores': [], 'num_gt': 0})
    
    for img_results in all_results:
        for class_id, res in img_results.items():
            class_results[class_id]['tp'].extend(res['tp'])
            class_results[class_id]['fp'].extend(res['fp'])
            class_results[class_id]['scores'].extend(res['scores'])
            class_results[class_id]['num_gt'] += res['num_gt']
    
    # 计算每个类别的AP
    aps = []
    for class_id in range(5):
        res = class_results[class_id]
        if res['num_gt'] == 0:
            continue
            
        # 按置信度排序
        scores = np.array(res['scores'])
        indices = np.argsort(-scores)
        tp = np.array(res['tp'])[indices]
        fp = np.array(res['fp'])[indices]
        
        # 计算累积TP和FP
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        
        # 计算精确率和召回率
        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / res['num_gt']
        
        # 计算AP
        ap, precs, recalls = calculate_ap(recall, precision)
        aps.append(ap)
        
        if draw_plot == True:
            # 绘制PR曲线
            plt.figure()
            plt.plot(recalls, np.array(precs), 'b-', linewidth=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Class {class_id} PR Curve (AP = {ap:.4f})')
            plt.grid(True)
            plt.axis([0, 1, 0, 1.05])
            plt.savefig(f'pr_curve_class_{class_id}.png')
            plt.close()
    
    # 计算mAP
    if not aps:
        return 0
    return aps

def evaluate_detection(pred_dir, gt_dir, iou_threshold=0.5, conf_threshold=0.25, draw_plot=False):
    """评估整个检测结果集"""
    all_results = []
    
    # 获取所有图像ID（假设文件名相同，扩展名可能不同）
    pred_files = os.listdir(pred_dir)
    image_ids = set()
    
    for f in pred_files:
        if f.endswith('.txt'):
            image_id = os.path.splitext(f)[0]
            image_ids.add(image_id)
    
    # 评估每张图像
    for image_id in image_ids:
        pred_file = os.path.join(pred_dir, f'{image_id}.txt')
        gt_file = os.path.join(gt_dir, f'{image_id}.txt')
        
        # 读取预测和标注
        pred_boxes = read_yolo_format(pred_file, conf_threshold, is_pred=True)
        gt_boxes = read_yolo_format(gt_file, conf_threshold, is_pred=False)
        
        # 评估单张图像
        img_results = evaluate_image(pred_boxes, gt_boxes, iou_threshold)
        all_results.append(img_results)
    
    # 计算mAP
    mAP = calculate_map(all_results, iou_threshold, draw_plot)
    return mAP

if __name__ == "__main__":
    # 使用示例
    gt_dir = "/root/autodl-tmp/train/labels/val/"
    pred_dir = "/root/yolov12_rgbt/runs/detect/val_yolov12n_rgbt_2208_608_best_restore_1/"
    iou_threshold = 0.5
    draw_plot = False
    
    best_map = [0.0, 0.0, 0.0, 0.0, 0.0]
    best_conf = [0.0, 0.0, 0.0, 0.0, 0.0]
    for conf_threshold in np.arange(0.01, 0.50, 0.01):
        aps = evaluate_detection(pred_dir, gt_dir, iou_threshold, conf_threshold, draw_plot)
        mAP = np.mean(aps)
        print(f"conf: {conf_threshold:.2f}, mAP50: {mAP:.4f}, cls0: {aps[0]:.4f}, cls1: {aps[1]:.4f}, cls2: {aps[2]:.4f}, cls3: {aps[3]:.4f}, cls4: {aps[4]:.4f}")
        for i in range(5):
            if aps[i] > best_map[i]:
                best_map[i] = aps[i]
                best_conf[i] = conf_threshold
                
    best_mAP = np.mean(best_map)
    print(f"\nbest_conf          cls0: {best_conf[0]:.4f}, cls1: {best_conf[1]:.4f}, cls2: {best_conf[2]:.4f}, cls3: {best_conf[3]:.4f}, cls4: {best_conf[4]:.4f}")
    print(f"best_mAP: {best_mAP:.4f}   cls0: {best_map[0]:.4f}, cls1: {best_map[1]:.4f}, cls2: {best_map[2]:.4f}, cls3: {best_map[3]:.4f}, cls4: {best_map[4]:.4f}")