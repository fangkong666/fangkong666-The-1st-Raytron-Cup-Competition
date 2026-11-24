from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
from datetime import datetime
import torch

def get_gpu_count():
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        return 0
    # 返回GPU数量
    return torch.cuda.device_count()

if __name__ == "__main__":
    gpu_count = get_gpu_count()
    print(f"可用GPU数量: {gpu_count}")

    DEFAULT_CFG.save_dir= "runs/detect/yolov12n_rgbt_2208_608"

    model = YOLO("ultralytics/cfg/models/v12/yolov12n.yaml")
    model.load("yolov12n.pt")
    model.info()
    
    if gpu_count >=2:
        device_list = [i for i in range(gpu_count)]
        batch_calu = int(gpu_count) * 16
        print("device_list: {}, batch_calu: {}".format(device_list, batch_calu))
        model.train(data="ultralytics/cfg/datasets/binocular_rgbt_crop.yaml", close_mosaic=1,
                    mixup=0.1, copy_paste=0.1, scale=0.5, 
                    epochs=80, cache=False, workers=16, batch=batch_calu, imgsz=2208, device=device_list, resume=False, val=True)
    else:
        model.train(data="ultralytics/cfg/datasets/binocular_rgbt_crop.yaml", close_mosaic=1,
                    mixup=0.1, copy_paste=0.1, scale=0.5, 
                    epochs=5, cache=False, workers=8, batch=2, imgsz=2208, device=[0], resume=False, val=True)        

# nohup python binocular/train.py > log_v12_train.log 2>&1 &