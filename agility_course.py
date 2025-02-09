import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyautogui
import pygetwindow as gw
import os
from PIL import Image

def capture_window_screenshot(window_title):
    # Get the window by title
    window = None
    for win in gw.getAllTitles():
        if window_title.lower() in win.lower():
            window = gw.getWindowsWithTitle(win)[0]
            break

    if not window:
        print(f"Window titled '{window_title}' not found.")
        return None

    # Screenshot the window's region
    x, y, width, height = window.left, window.top, window.width, window.height
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return screenshot

def get_average_color(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    avg_color = np.mean(image, axis=(0, 1))
    return avg_color.astype(int)

def extract_color_regions(image, target_color, threshold=30):
    # Convert target color to numpy array
    target_color = np.array(target_color, dtype=np.uint8)
    
    # Define color range
    lower_bound = np.clip(target_color - threshold, 0, 255)
    upper_bound = np.clip(target_color + threshold, 0, 255)
    
    # Create a mask for the target color
    mask = cv2.inRange(image, lower_bound, upper_bound)
    
    # Extract the regions of interest
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def detect_colored_regions_from_screenshot(window_title, sample_image_path):
    # Capture the screenshot
    screenshot = capture_window_screenshot(window_title)
    if screenshot is None:
        return
    
    # Convert PIL image to OpenCV format
    image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # Get the average color from the sample image
    target_color = get_average_color(sample_image_path)
    
    # Extract regions matching the target color
    extracted_regions = extract_color_regions(image, target_color)
    
    # Show the result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(extracted_regions, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# Example usage: specify a sample image to determine color
detect_colored_regions_from_screenshot("RuneLite", "sample/roof_sample.png")