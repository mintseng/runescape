import pygetwindow as gw
import pyautogui
import cv2
import numpy as np
import time
import pyautogui
import time
import random
import math
import os
from PIL import ImageGrab

from humancursor import SystemCursor
cursor = SystemCursor()

def capture_window_screenshot(window_title, output_path):
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
    # screenshot.save("runescape/" + output_path)
    # print(f"Screenshot saved to {output_path}")
    # return output_path
    return screenshot

def check_if_screenshot_contains(screenshot, target_image_path, threshold=0.8):
    # Convert the screenshot to a format compatible with OpenCV
    screenshot_np = np.array(screenshot)
    screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2GRAY)

    # Load the target image
    target = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    if target is None:
        print(f"Could not load target image from {target_image_path}.")
        return False

    # Match the target image in the screenshot
    result = cv2.matchTemplate(screenshot_gray, target, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Check if the match is above the threshold
    if max_val >= threshold:
        print("Found " + target_image_path + "!")
        return True
    else:
        print("Did not find" + target_image_path + " ):")
        return False

def find_and_click_in_window(window_title, target_image_path, threshold=0.8):
    # Get the window by title
    window = None
    for win in gw.getAllTitles():
        if window_title.lower() in win.lower():
            window = gw.getWindowsWithTitle(win)[0]
            break

    if not window:
        print(f"Window titled '{window_title}' not found.")
        return False

    # Get the window's region
    x, y, width, height = window.left, window.top, window.width, window.height

    # Take a screenshot of the window region
    screenshot = pyautogui.screenshot(region=(x, y, width, height))

    # Convert the screenshot to a format compatible with OpenCV
    screenshot_np = np.array(screenshot)
    screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2GRAY)

    # Load the target image
    target = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    if target is None:
        print(f"Could not load target image from {target_image_path}.")
        return False

    # Match the target image in the screenshot
    result = cv2.matchTemplate(screenshot_gray, target, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Check if the match is above the threshold
    if max_val >= threshold:
        # Get the center of the matched area
        target_height, target_width = target.shape
        center_x = x + max_loc[0] + target_width // 2
        center_y = y + max_loc[1] + target_height // 2

        # Click the center of the matched area
        pyautogui.click(x=center_x, y=center_y)
        print(f"Clicked at ({center_x}, {center_y}). Match confidence: {max_val:.2f}")
        return True
    else:
        print("No match found.")
        return False

def click(window_title, target_image_path, threshold=0.8):
    window = None
    for win in gw.getAllTitles():
        if window_title.lower() in win.lower():
            window = gw.getWindowsWithTitle(win)[0]
            break

    if not window:
        print(f"Window titled '{window_title}' not found.")
        return False

    # Get the window's region
    x, y, width, height = window.left, window.top, window.width, window.height

    # Take a screenshot of the window region
    screenshot = pyautogui.screenshot(region=(x, y, width, height))

    # Convert the screenshot to a format compatible with OpenCV
    screenshot_np = np.array(screenshot)
    screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2GRAY)

    # Load the target image
    target = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    if target is None:
        print(f"Could not load target image from {target_image_path}.")
        return False

    # Match the target image in the screenshot
    result = cv2.matchTemplate(screenshot_gray, target, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Check if the match is above the threshold
    if max_val >= threshold:
        # Get the center of the matched area
        target_height, target_width = target.shape
        center_x = x + max_loc[0] + target_width // 2
        center_y = y + max_loc[1] + target_height // 2

        # Click the center of the matched area
        # smooth_move_to(center_x, center_y, 0.3, 15)
        cursor.move_to([center_x, center_y])
        pyautogui.click(x=center_x, y=center_y)
        print(f"Clicked at ({center_x}, {center_y}). Match confidence: {max_val:.2f}")
        return True
    else:
        print("No match found.")
        return False

def click_below(window_title, target_image_path, threshold=0.8):
    window = None
    for win in gw.getAllTitles():
        if window_title.lower() in win.lower():
            window = gw.getWindowsWithTitle(win)[0]
            break

    if not window:
        print(f"Window titled '{window_title}' not found.")
        return False

    # Get the window's region
    x, y, width, height = window.left, window.top, window.width, window.height

    # Take a screenshot of the window region
    screenshot = pyautogui.screenshot(region=(x, y, width, height))

    # Convert the screenshot to a format compatible with OpenCV
    screenshot_np = np.array(screenshot)
    screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2GRAY)

    # Load the target image
    target = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    if target is None:
        print(f"Could not load target image from {target_image_path}.")
        return False

    # Match the target image in the screenshot
    result = cv2.matchTemplate(screenshot_gray, target, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Check if the match is above the threshold
    if max_val >= threshold:
        # Get the center of the matched area
        target_height, target_width = target.shape
        center_x = x + max_loc[0] + target_width // 2
        center_y = y + max_loc[1] + target_height // 2

        # Click the center of the matched area
        # smooth_move_to(center_x, center_y, 0.3, 15)
        cursor.move_to([center_x, center_y])
        pyautogui.click(x=center_x, y=center_y)
        print(f"Clicked at ({center_x}, {center_y}). Match confidence: {max_val:.2f}")
        return True
    else:
        print("No match found.")
        return False


def click_faster(window_title, target_image_path, threshold=0.8):
    window = None
    for win in gw.getAllTitles():
        if window_title.lower() in win.lower():
            window = gw.getWindowsWithTitle(win)[0]
            break

    if not window:
        print(f"Window titled '{window_title}' not found.")
        return False

    # Get the window's region
    x, y, width, height = window.left, window.top, window.width, window.height

    # Take a screenshot of the window region
    screenshot = pyautogui.screenshot(region=(x, y, width, height))

    # Convert the screenshot to a format compatible with OpenCV
    screenshot_np = np.array(screenshot)
    screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2GRAY)

    # Load the target image
    target = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    if target is None:
        print(f"Could not load target image from {target_image_path}.")
        return False

    # Match the target image in the screenshot
    result = cv2.matchTemplate(screenshot_gray, target, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Check if the match is above the threshold
    if max_val >= threshold:
        # Get the center of the matched area
        target_height, target_width = target.shape
        center_x = x + max_loc[0] + target_width // 2
        center_y = y + max_loc[1] + target_height // 2

        # Click the center of the matched area
        # smooth_move_to(center_x, center_y, 0.3, 15)
        cursor.move_to([center_x, center_y], 0.1)
        pyautogui.click(x=center_x, y=center_y)
        print(f"Clicked at ({center_x}, {center_y}). Match confidence: {max_val:.2f}")
        return True
    else:
        print("No match found.")
        return False


def click_right(window_title, target_image_path, threshold=0.8):
    window = None
    for win in gw.getAllTitles():
        if window_title.lower() in win.lower():
            window = gw.getWindowsWithTitle(win)[0]
            break

    if not window:
        print(f"Window titled '{window_title}' not found.")
        return False

    # Get the window's region
    x, y, width, height = window.left, window.top, window.width, window.height

    # Take a screenshot of the window region
    screenshot = pyautogui.screenshot(region=(x, y, width, height))

    # Convert the screenshot to a format compatible with OpenCV
    screenshot_np = np.array(screenshot)
    screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2GRAY)

    # Load the target image
    target = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    if target is None:
        print(f"Could not load target image from {target_image_path}.")
        return False

    # Match the target image in the screenshot
    result = cv2.matchTemplate(screenshot_gray, target, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Check if the match is above the threshold
    if max_val >= threshold:
        # Get the center of the matched area
        target_height, target_width = target.shape
        center_x = x + max_loc[0] + target_width // 2
        center_y = y + max_loc[1] + target_height // 2

        # Click the center of the matched area
        # smooth_move_to(center_x, center_y, 0.3, 15)
        cursor.move_to([center_x, center_y])
        pyautogui.click(x=center_x, y=center_y)
        print(f"Clicked at ({center_x}, {center_y}). Match confidence: {max_val:.2f}")
        return True
    else:
        print("No match found.")
        return False

def ease_in_out(t):
    """Cubic easing in-out function."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - math.pow(-2 * t + 2, 3) / 2

def smooth_move_to(x, y, duration=0.5, step=10):
    """
    Moves the mouse smoothly and naturally to the target position.

    Args:
        x (int): Target x-coordinate.
        y (int): Target y-coordinate.
        duration (float): Total time for the movement in seconds.
        step (int): Number of steps for the movement.
    """
    current_x, current_y = pyautogui.position()  # Get current mouse position

    # Calculate the distance to move in each step
    delta_x = x - current_x
    delta_y = y - current_y
    total_distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
    steps = max(1, step)

    # Calculate time per step
    time_per_step = duration / steps

    for i in range(1, steps + 1):
        # Interpolate position
        move_x = current_x + (delta_x * i / steps)
        move_y = current_y + (delta_y * i / steps)
        
        # Add a bit of randomness for natural movement
        jitter_x = random.uniform(-1, 1) if i < steps else 0
        jitter_y = random.uniform(-1, 1) if i < steps else 0

        options = [pyautogui.easeOutQuad, pyautogui.easeOutBack, pyautogui.easeInOutQuad]

        # pyautogui.moveTo(move_x + jitter_x, move_y + jitter_y, np.random.uniform(0.6, 2.7), options[0])
        pyautogui.moveTo(move_x + jitter_x, move_y + jitter_y)

        time.sleep(time_per_step)

def fish():
    while True:
        time.sleep(5)
        screenshot = capture_window_screenshot("RuneLite", "screenshot.png")
        
        if not check_if_screenshot_contains(screenshot, "runescape/fishing.png"):
            click("RuneLite", "runescape/lobster.png")

def fish_shark():
    while True:
        time.sleep(5)
        screenshot = capture_window_screenshot("RuneLite", "screenshot.png")
        
        if not check_if_screenshot_contains(screenshot, "runescape/fishing.png"):
            click("RuneLite", "runescape/shark_small.png")

def attack():
    # for path in get_image_paths("runescape"):
    #     print(path)
    
    while True:
        time.sleep(5)
        screenshot = capture_window_screenshot("RuneLite", "screenshot.png")
        threshold = 0.70

        if check_if_screenshot_contains(screenshot, "runescape/continue.png", threshold):
            click("RuneLite", "runescape/continue.png", threshold)
            time.sleep(3)

        for path in get_image_paths("runescape/lesser_demon"):
            if check_if_screenshot_contains(screenshot, path, threshold):
                click_below("RuneLite", path, threshold)
                break



def get_image_paths(directory, extensions=(".png", ".jpg", ".jpeg")):
    """
    Returns a list of all image file paths in the specified directory.
    """
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.lower().endswith(extensions)
    ]

def build():
    while True:
        # time.sleep(1)
        screenshot = capture_window_screenshot("RuneLite", "screenshot.png")
        threshold = 0.70

        sleep_time = 0.5

        if check_if_screenshot_contains(screenshot, "runescape/building/ghost_build.png", threshold):
            click_faster("RuneLite", "runescape/building/ghost_build.png", threshold)
            time.sleep(sleep_time)

        if check_if_screenshot_contains(screenshot, "runescape/building/magical_balance_2.png", threshold):
            click_faster("RuneLite", "runescape/building/magical_balance_2.png", threshold)
            time.sleep(sleep_time)
        
        if check_if_screenshot_contains(screenshot, "runescape/building/build2.png", threshold):
            click_faster("RuneLite", "runescape/building/build2.png", threshold)
            time.sleep(sleep_time)

        screenshot = capture_window_screenshot("RuneLite", "screenshot.png")
        
        if check_if_screenshot_contains(screenshot, "runescape/building/yes.png", threshold):
            click_faster("RuneLite", "runescape/building/yes.png", threshold)
            time.sleep(sleep_time)

def click_polygon(screenshot=None):
    if screenshot is None:
        screenshot = capture_window_screenshot("RuneLite", "screenshot.png")
    
    screenshot_np = np.array(screenshot)

    # Convert RGB (Pillow) to BGR (OpenCV)
    screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    # Convert to HSV
    hsv_image = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2HSV)

    # Define red color range
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = mask1 | mask2

    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour (if there's more than one polygon)
    contour = max(contours, key=cv2.contourArea)

    # TESTING CODE
    # cv2.drawContours(screenshot_np, [contour], -1, (0, 255, 0), 3)
    # cv2.imshow('Detected Polygon', screenshot_np)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])  # x-coordinate of centroid
        cy = int(M['m01'] / M['m00'])  # y-coordinate of centroid
    else:
        return


    window = None
    for win in gw.getAllTitles():
        if "RuneLite".lower() in win.lower():
            window = gw.getWindowsWithTitle(win)[0]
            break
    
    screen_x = window.left + cx
    screen_y = window.top + cy

    cursor.move_to([screen_x, screen_y], 0.1)
    pyautogui.click(x=screen_x, y=screen_y)

    time.sleep(5)

# attack()
fish_shark()
# build()

# time.sleep(1)
# while True:
#     time.sleep(1)
#     screenshot = capture_window_screenshot("RuneLite", "screenshot.png")

#     monster = "cow"
#     png1 = f"runescape/{monster}/{monster}.png"
#     png2 = f"runescape/{monster}/{monster}_dead.png"

    
#     if not check_if_screenshot_contains(screenshot, png1) or check_if_screenshot_contains(screenshot, png2, .95) :
#         click_polygon()