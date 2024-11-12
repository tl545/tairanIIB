import cv2
import numpy as np
import os

image_files = ['/Users/lit./Desktop/iibproject/Fractal Images/Fractal_Black.png', 
               '/Users/lit./Desktop/iibproject/Fractal Images/Fractal_Blue.png',
               '/Users/lit./Desktop/iibproject/Fractal Images/Fractal_Green.png', 
               '/Users/lit./Desktop/iibproject/Fractal Images/Fractal_Red.png',
                '/Users/lit./Desktop/iibproject/Fractal Images/Fractal_Yellow.png']

first_img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
height, width = first_img.shape  

processed_images = []

for i, file in enumerate(image_files):

    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (width, height)) 

    img = cv2.bitwise_not(img) # inverse the black and white so that the black part can be recognised as patterns 

    _, binary_img = cv2.threshold(img, np.mean(img), 255, cv2.THRESH_BINARY)
    #_, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

    # M1: Find the label of the two largest components
    #sorted = np.argsort(stats[1:, cv2.CC_STAT_AREA])[-2:]  
    #largest_label = sorted[1] + 1  
    #second_label = sorted[0] + 1
    #mask = np.where((labels == largest_label) | (labels == second_label), 255, 0).astype(np.uint8)
    #processed_img = cv2.bitwise_and(binary_img, mask)


    # M2: select the minimum component size to keep
    min_size = 60  
    processed_img = np.zeros_like(binary_img)
    for j in range(1, num_labels):  
        if stats[j, cv2.CC_STAT_AREA] >= min_size:
            processed_img[labels == j] = 255



    # M3: opening operation
    #kernel = np.ones((5, 5), np.uint8) 
    #processed_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    

    processed_img = cv2.bitwise_not(processed_img)  # inverse back

    output_filename = f'denoised_image_{i+1}.png'
    cv2.imwrite(output_filename, processed_img)
            
    processed_images.append(processed_img)


intersection = processed_images[0]
for img in processed_images[1:]:
    intersection = cv2.bitwise_and(intersection, img)

total_area = cv2.countNonZero(processed_images[0])  # The first image is used as reference for total area
overlap_area = cv2.countNonZero(intersection)

overlap_percentage = (overlap_area / total_area) * 100

print(f"Overlap Percentage: {overlap_percentage:.2f}%")


union = processed_images[0]
for img in processed_images[1:]:
    union = cv2.bitwise_or(union, img)

non_overlapping_area =  cv2.bitwise_not(cv2.bitwise_and(union, cv2.bitwise_not(intersection)))
cv2.imwrite('d_non_overlapping_area.png', non_overlapping_area)