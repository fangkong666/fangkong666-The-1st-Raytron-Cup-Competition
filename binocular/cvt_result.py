import os
import re

def convert_yolo_labels(predict_image_dir, input_dir, output_file, thres, params, gflops):
    """
    将 YOLO 格式的标签文件转换为指定格式的结果文件
    
    参数:
    input_dir (str): 包含 YOLO 标签文件的目录路径
    output_file (str): 输出结果文件的路径
    """
    try:
        # 确保输入目录存在
        if not os.path.exists(predict_image_dir):
            print(f"错误: 目录 '{predict_image_dir}' 不存在")
            return
    
        # 确保输入目录存在
        if not os.path.exists(input_dir):
            print(f"错误: 目录 '{input_dir}' 不存在")
            return
        
        # 创建或清空输出文件
        with open(output_file, 'w') as f:
            pass
        
        # 遍历目录中的所有文件
        file_list = os.listdir(predict_image_dir)
        file_list.sort()
        
        param_line = [str(params), str(gflops)]
        with open(output_file, 'a') as f:
            f.write(' '.join(param_line) + '\n')

        for filename in file_list:
            if filename.endswith('.jpg'):
                file_path = os.path.join(input_dir, os.path.splitext(filename)[0] + '.txt')
                image_name = filename
                
                output_line = [image_name]
                if os.path.exists(file_path):
                    # 读取文件内容
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        # 分割行数据并转换为浮点数
                        parts = line.strip().split()
                        if len(parts) >= 5:  # 确保有足够的数据
                            # 跳过类别ID（parts[0]），提取中心坐标、宽高和分数
                            # 假设最后一个值是分数，如果没有分数则跳过
                            if len(parts) >= 6:
                                score = float(parts[5])
                            else:
                                score = 1.0  # 默认分数为1.0
                            
                            # 添加中心坐标、宽高和分数
                            if score >= thres[int(parts[0])]:
                                new_score = min(score * 0.74 + 0.25, 1.0)
                                output_line.extend([
                                    f"{float(parts[1]):.6f}",  # center_x
                                    f"{float(parts[2]):.6f}",  # center_y
                                    f"{float(parts[3]):.6f}",  # width
                                    f"{float(parts[4]):.6f}",  # height
                                    f"{new_score:.6f}",  # score，保留6位小数
                                    parts[0]
                                ])
                
                # 将处理后的行写入输出文件
                if len(output_line) > 0:  # 确保有数据可写
                    with open(output_file, 'a') as f:
                        f.write(' '.join(output_line) + '\n')
                else:
                    print(output_line)
        
        print(f"转换完成，结果保存在: {output_file}")
        
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 设置输入目录和输出文件路径
    predict_image_dir = '/root/autodl-tmp/test_crop_resize/images/'
    input_directory = "/root/yolov12_rgbt/runs/detect/test_yolov12n_rgbt_2208_608_best_restore/"  # YOLO 标签文件所在目录
    output_file = "/root/yolov12_rgbt/runs/detect/test_yolov12n_rgbt_2208_608_best_result.txt"
    thres = [0.15, 0.06, 0.02, 0.08, 0.16]
    params = 2520783
    gflops = 19.705
    
    # 执行转换
    convert_yolo_labels(predict_image_dir, input_directory, output_file, thres, params, gflops)    
    
    """                                                        
    yolov12n_2208_608:  [0.15, 0.06, 0.02, 0.08, 0.16]
                         0.7513 0.8896 0.9584 0.9221 0.9557  0.8954                                                                                                                                                                  
    """