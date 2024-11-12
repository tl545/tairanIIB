import cv2
import numpy as np
import os
import random

def load_samples(directory, samples = 10):
    digit_images = []
    for label in range(10):
        label_dir = os.path.join(directory, str(label))
        
        if not os.path.exists(label_dir):
            print(f"Directory {label_dir} does not exist")
            continue

        all_images = os.listdir(label_dir)
        selected = random.sample(all_images, samples)
        
        digit_images_for_label = []
        for image_file in selected:
            image_path = os.path.join(label_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            digit_images_for_label.append(image)
        
        digit_images.append(digit_images_for_label)
    
    return digit_images

digit_samples = load_samples("/Users/lit./Desktop/iibproject/mnist_png/train")

digit_size = (50, 50)  
samples = 10  
num_digits = 10  
height = 640
width = 1080

digit_colors = [
    (255, 0, 0),    # 0: Red
    (0, 255, 0),    # 1: Green
    (0, 0, 255),    # 2: Blue
    (255, 255, 0),  # 3: Cyan
    (255, 0, 255),  # 4: Magenta
    (0, 255, 255),  # 5: Yellow
    (128, 0, 128),  # 6: Purple
    (255, 165, 0),  # 7: Orange
    (0, 128, 128),  # 8: Teal
    (128, 128, 0),  # 9: Olive
]

canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

x_spacing = width // samples
y_spacing = height // num_digits


for i, digit_images in enumerate(digit_samples): 
    for j, img in enumerate(digit_images):  

        resized_img = cv2.resize(img, digit_size, interpolation=cv2.INTER_LINEAR)
        
        _, mask = cv2.threshold(resized_img, 128, 255, cv2.THRESH_BINARY_INV)

        rgb_img = np.ones((digit_size[1], digit_size[0], 3), dtype=np.uint8) * 255
        color = digit_colors[i]  
        
        for c in range(3):  
            rgb_img[:, :, c] = np.where(mask == 0, color[c], rgb_img[:, :, c])
    
        x = j * x_spacing + (x_spacing - digit_size[0]) // 2
        y = i * y_spacing + (y_spacing - digit_size[1]) // 2
        
        canvas[y:y + digit_size[1], x:x + digit_size[0]] = rgb_img

output_image = cv2.resize(canvas, (1080, 640), interpolation=cv2.INTER_LINEAR)

cv2.imwrite("mnist_colored_digits_grid.png", output_image)
print("Image saved as mnist_colored_digits_grid.png")