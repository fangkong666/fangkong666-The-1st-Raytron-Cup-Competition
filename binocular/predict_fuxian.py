from ultralytics import YOLO
from datetime import datetime
import torch.nn as nn
import thop
import torch
from copy import deepcopy
from expand_result import expand_dataset
from cvt_result import convert_yolo_labels

# 与官网要求的实现完全相同
def calculate_params_flops(model, imgsz1=640,imgsz2=640):
    """计算模型参数量和GFLOPs"""
    # 获取实际的PyTorch模型
    if hasattr(model, 'model'):
        pytorch_model = model.model
    else:
        pytorch_model = model
    
    # 参数量计算
    n_p = sum(x.numel() for x in pytorch_model.parameters())  # number parameters
    n_g = sum(x.numel() for x in pytorch_model.parameters() if x.requires_grad)  # number gradients
    
    # GFLOPs计算
    try:
        im = torch.empty((1, 4, imgsz1, imgsz2))  # input image in BCHW format
        flops = thop.profile(deepcopy(pytorch_model), inputs=(im,), verbose=False)[0] / 1E9 * 2  # GFLOPs
        gflops = flops
    except Exception as e:
        print(f"GFLOPs计算错误: {e}")
        gflops = 0
    
    print(f"Parameters: {n_p:,}")
    print(f"GFLOPs: {gflops:.3f}")
    
    return n_p, gflops

def main():
    pth_path = 'runs/detect/yolov12n_rgbt_2208_608/weights/best.pt'
    image_path = "/input/test_crop_resize/images/"
    result_name = "predict_fuxian"
    expand_path = "runs/detect/predict_fuxian_restore/"
    output_file = "/output/result_fuxian.txt"
    
    imgsz = [608, 2208]
    top_crop = 200
    bottom_crop = 168
    thres = [0.15, 0.06, 0.02, 0.08, 0.16]
    
    # 加载与参数算力计算
    model = YOLO(pth_path)
    params, gflops = calculate_params_flops(model, imgsz1=imgsz[1],imgsz2=imgsz[0])
    print(" #1 ----- 模型加载完成")

    # 模型推理
    model.predict(source=image_path, workers=4, name=result_name,
              batch=8, imgsz=imgsz, device=[0], resume=False, augment=False, save_conf=True, save_txt=True, iou=0.45, conf=0.02)
    print(" #2 ----- 模型推理完成")

    # 裁剪还原
    expand_dataset("runs/detect/{}/labels/".format(result_name), expand_path, top_crop, bottom_crop)
    print(" #3 ----- 推理结果还原完成")
    
    # 根据阈值
    convert_yolo_labels(image_path, expand_path, output_file, thres, params, gflops) 
    print(" #4 ----- 结果转换完成")  
    
    # 打印分辨率，参数量和计算量
    print(f"image size:{imgsz}, {params:,} parameters, {gflops:.3f} GFLOPs")

if __name__ == "__main__":
    main()