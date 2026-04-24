import cv2
import numpy as np

def process_half_page(image_path):
    # Load grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    # Define the Top Half
    half_h = h // 2
    top_half = img[0:half_h, :]
    
    # 1. Enhance Contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(top_half)
    
    # 2. Bilateral Filtering (Preserve edges of bubbles)
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # 3. FIXED Adaptive Threshold (Higher block size to keep shading)
    # We increase 11 -> 31 so it sees the whole bubble context
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 71, 5)
    
    # Recombine with original bottom half for comparison
    final_img = img.copy()
    # Convert threshold back to BGR if you want to see it in a color window
    final_img[0:half_h, :] = thresh
    
    cv2.imwrite('half_page_fixed.png', final_img)
    print("✅ Fixed half-page image saved.")

# Run the comparison
process_half_page(r'G:/OMR_Detection/processed_imgs/img_000038.png')