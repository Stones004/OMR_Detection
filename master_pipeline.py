import os
import glob
from ultralytics import YOLO

# -----------------------------------------
# 1. CORE POST-PROCESSING LOGIC
# -----------------------------------------

def compute_iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) for overlapping boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def deduplicate_boxes(boxes, iou_thresh=0.6):
    """Removes overlapping duplicates by keeping the highest confidence box."""
    # Sort by confidence descending
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    final = []

    while boxes:
        best = boxes.pop(0)
        final.append(best)
        # Keep only boxes that do not overlap significantly with the best box
        boxes = [b for b in boxes if compute_iou(best, b) < iou_thresh]

    return final

def enforce_constraints(bands, bubbles):
    """Enforces: 1 band -> max 1 filled bubble per column (A, B, C)."""
    final_answers = []

    for band in bands:
        bx, by, bw, bh, _ = band
        # Find bubbles strictly inside this horizontal band
        band_bubbles = [b for b in bubbles if by < b[1] < by + bh]

        # Divide the band into 3 columns (A, B, C)
        col_w = bw / 3
        cols = {0: [], 1: [], 2: []}

        for b in band_bubbles:
            x, y, w, h, conf = b
            col_idx = int((x - bx) / col_w)
            col_idx = min(max(col_idx, 0), 2) # Clamp between 0-2
            cols[col_idx].append(b)

        # Pick the highest confidence bubble per column (if any exists)
        row_answers = []
        for col_idx, col in cols.items():
            if col:
                best = max(col, key=lambda x: x[4])
                row_answers.append({
                    "column": ["A", "B", "C"][col_idx],
                    "confidence": best[4]
                })
        
        if row_answers:
            final_answers.append({"band_y": by, "answers": row_answers})

    # Sort answers top-to-bottom
    return sorted(final_answers, key=lambda x: x["band_y"])


# -----------------------------------------
# 2. MASTER AUTOMATION WORKFLOW
# -----------------------------------------

def process_student_exams():
    # Setup Paths
    raw_scripts_dir = r"G:\raw_ans_scripts"
    models_dir = r"G:\YOLO-NAS_Detection\models"
    
    # Grab the most recent weights
    latest_weights = max(glob.glob(os.path.join(models_dir, '*.pt')), key=os.path.getctime)
    print(f"Loading Model: {latest_weights}")
    model = YOLO(latest_weights)

    # Loop through each student folder (e.g., "Student_001")
    for student_folder in os.listdir(raw_scripts_dir):
        student_path = os.path.join(raw_scripts_dir, student_folder)
        if not os.path.isdir(student_path): continue
        
        print(f"\n--- Processing {student_folder} ---")
        student_results = []
        
        # Loop through every page/image for that student
        for page_img in glob.glob(os.path.join(student_path, '*.*')):
            # Note: In a full pipeline, you would call your `pre-processing-new.py` function here first
            
            # Run YOLO Inference (in memory, no need to save messy .txt files everywhere)
            results = model.predict(source=page_img, conf=0.60, save=False)
            
            for result in results:
                boxes = result.boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, class]
                
                # Format to [x, y, w, h, conf]
                formatted_bands = []
                formatted_bubbles = []
                
                for b in boxes:
                    x, y, w, h = b[0], b[1], b[2]-b[0], b[3]-b[1]
                    conf, cls = b[4], int(b[5])
                    
                    if cls == 1: # Band
                        formatted_bands.append([x, y, w, h, conf])
                    elif cls == 2: # Filled Bubble
                        formatted_bubbles.append([x, y, w, h, conf])

                # Run Post-Processing
                clean_bands = deduplicate_boxes(formatted_bands)
                clean_bubbles = deduplicate_boxes(formatted_bubbles)
                page_answers = enforce_constraints(clean_bands, clean_bubbles)
                
                student_results.append({
                    "page": os.path.basename(page_img),
                    "answers": page_answers
                })

        # Output final JSON or print for the student
        print(f"Final Grades for {student_folder}:")
        for res in student_results:
            print(f"  {res['page']}: {res['answers']}")

def run_retraining_pipeline():
    print("\n--- 🚀 OPTION A: AUTOMATED RETRAINING ---")
    zip_path = input("Drag and drop your Label Studio ZIP file here (or paste path): ").strip().strip('"')
    
    if not os.path.exists(zip_path):
        print("❌ Error: Could not find that ZIP file.")
        return
        
    import sys
    sys.path.append(r"G:\YOLO-NAS_Detection")
    
    try:
        from import_ls_export import import_ls_export
        from train_yolo import train_custom_model
        
        # 1. Extract and Merge Data
        target_dataset = r"G:\YOLO-NAS_Detection\dataset"
        import_ls_export(zip_path, target_dataset)
        
        # 2. Retrain the Model
        print("\n⏳ Starting YOLO Fine-tuning...")
        # Change current working directory temporarily so train_yolo finds data.yaml
        original_cwd = os.getcwd()
        os.chdir(r"G:\YOLO-NAS_Detection")
        train_custom_model()
        os.chdir(original_cwd)
        
    except Exception as e:
        print(f"❌ Error during retraining: {e}")

if __name__ == "__main__":
    print("========================================")
    print("   🎓 OMR MASTER PIPELINE MANAGER     ")
    print("========================================")
    print("1. Grade Exams (Inference + Post-Processing)")
    print("2. Retrain Model (Import Label Studio ZIP & Fine-Tune)")
    print("========================================")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == '1':
        process_student_exams()
    elif choice == '2':
        run_retraining_pipeline()
    else:
        print("Invalid choice. Exiting.")
