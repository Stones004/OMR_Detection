import os
import shutil

run_dir = r"G:\YOLO-NAS_Detection\runs\detect\finrag_vision\yolov5l_custom_v1"
folders = ['Metrics', 'Batches', 'Weights_Backup']

for f in folders:
    os.makedirs(os.path.join(run_dir, f), exist_ok=True)

for file in os.listdir(run_dir):
    if file.endswith('.png') or file.endswith('.csv'):
        shutil.move(os.path.join(run_dir, file), os.path.join(run_dir, 'Metrics'))
    elif file.startswith(('train_batch', 'val_batch')):
        shutil.move(os.path.join(run_dir, file), os.path.join(run_dir, 'Batches'))