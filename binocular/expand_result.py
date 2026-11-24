import os
import math

def expand_dataset(input_label_dir, output_label_dir, top_crop, bottom_crop):
    orin_height = 720
    crop_height = orin_height - top_crop - bottom_crop

    # 创建输出目录
    os.makedirs(output_label_dir, exist_ok=True)
    offset_cnt = 0
    total_cnt = 0

    # 遍历所有图片
    for index, label_name in enumerate(os.listdir(input_label_dir)):
        label_path = os.path.join(input_label_dir, label_name)

        # 处理标签（如果有）
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()

            labels = []
            for line in lines:
                parts = line.strip().split()

                class_id = parts[0]
                norm_x = float(parts[1])
                norm_y = float(parts[2])
                norm_w = float(parts[3])
                norm_h = float(parts[4])           
                
                if len(parts) < 6:
                    score = 1.0
                else:
                    score = float(parts[5])

                norm_y = (norm_y * crop_height + top_crop) / orin_height
                norm_h = norm_h * crop_height / orin_height
                
                labels.append(f"{class_id} {norm_x:.8f} {norm_y:.8f} {norm_w:.8f} {norm_h:.8f} {score:.8f}")
                total_cnt = total_cnt + 1
            if labels:
                with open(os.path.join(output_label_dir, label_name), 'w') as f:
                    f.write("\n".join(labels))

if __name__ == "__main__":
    # 配置路径
    input_label_dir = "/root/yolov12_rgbt/runs/detect/val_yolov12n_rgbt_2208_608_best_1/labels/"  # 原始标签目录
    output_label_dir = "/root/yolov12_rgbt/runs/detect/val_yolov12n_rgbt_2208_608_best_restore_1/"  # 裁剪后标签输出目录

    # 处理整个数据集
    top_crop = 200
    bottom_crop = 168
    expand_dataset(input_label_dir, output_label_dir, top_crop, bottom_crop)