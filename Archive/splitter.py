import os
import shutil
import random

def split_dataset(source_img, source_lab, dest_root, split_ratio=0.8):
    # 1. Setup Destination Folders
    for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(dest_root, folder), exist_ok=True)

    # 2. Get list of all images
    images = [f for f in os.listdir(source_img) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)

    # 3. Calculate split index
    split_idx = int(len(images) * split_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    def move_files(files, split_type):
        for f in files:
            # Move Image
            shutil.copy2(os.path.join(source_img, f), os.path.join(dest_root, 'images', split_type, f))
            
            # Move corresponding Label (.txt)
            label_file = os.path.splitext(f)[0] + '.txt'
            src_label = os.path.join(source_lab, label_file)
            if os.path.exists(src_label):
                shutil.copy2(src_label, os.path.join(dest_root, 'labels', split_type, label_file))

    # 4. Execute the move
    move_files(train_imgs, 'train')
    move_files(val_imgs, 'val')
    print(f"Done! Moved {len(train_imgs)} to train and {len(val_imgs)} to val.")

# --- RUN THE SCRIPT ---
split_dataset(
    source_img=r"G:/final_dataset/images", 
    source_lab=r"G:/final_dataset/labels", 
    dest_root=r"G:/YOLO-NAS_Detection\dataset"
)