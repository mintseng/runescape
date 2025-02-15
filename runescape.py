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
import json
import keyboard
import mouse
from PIL import ImageGrab

from humancursor import SystemCursor
cursor = SystemCursor()


def print_runtime(print_interval=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            print(f"Starting {func.__name__}...")
            
            def print_elapsed():
                while True:
                    elapsed = time.time() - start_time
                    hours = int(elapsed // 3600)
                    minutes = int((elapsed % 3600) // 60)
                    seconds = int(elapsed % 60)
                    print(f"\r{func.__name__} running for: {hours:02d}:{minutes:02d}:{seconds:02d}", end="")
                    time.sleep(print_interval)
            
            # Start timer thread
            import threading
            timer_thread = threading.Thread(target=print_elapsed, daemon=True)
            timer_thread.start()
            
            # Run the actual function
            return func(*args, **kwargs)
        return wrapper
    
    # Handle both @print_runtime and @print_runtime(interval) syntax
    if callable(print_interval):
        return decorator(print_interval)
    return decorator

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

def check_if_screenshot_contains(screenshot, target_image_path, threshold=0.85):
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

def click(window_title, target_image_path, threshold=0.8, skip_move=False):
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

        if not skip_move:
            cursor.move_to([center_x, center_y], duration=0.5)
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

    cv2.imshow("Red Mask", red_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour (if there's more than one polygon)
    contour = max(contours, key=cv2.contourArea)

    # TESTING CODE
    cv2.drawContours(screenshot_np, [contour], -1, (0, 255, 0), 3)
    cv2.imshow('Detected Polygon', screenshot_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
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

    time.sleep(2)

# attack()
# fish_shark()
# build()

def monster(mob_name):
    time.sleep(1)
    while True:
        time.sleep(1)
        screenshot = capture_window_screenshot("RuneLite", "screenshot.png")

        monster = mob_name
        png1 = f"runescape/{monster}/{monster}.png"
        png2 = f"runescape/{monster}/{monster}_dead.png"

        
        if not check_if_screenshot_contains(screenshot, png1) or check_if_screenshot_contains(screenshot, png2, .95) :
            click_polygon()

# monster("buffalo")

def cosmic_rune():
    while True:
        # time.sleep(5)
        screenshot = capture_window_screenshot("RuneLite", "screenshot.png")
        
        if check_if_screenshot_contains(screenshot, "runescape/cosmic_rune/essence.png"):
            click("RuneLite", "runescape/cosmic_rune/essence.png", skip_move=True)
            time.sleep(0.5)

            if check_if_screenshot_contains(screenshot, "runescape/cosmic_rune/banker_note.png"):
                click("RuneLite", "runescape/cosmic_rune/banker_note.png", skip_move=True)
                time.sleep(0.5)

                if check_if_screenshot_contains(screenshot, "runescape/cosmic_rune/cosmic_rune.png"):
                    click("RuneLite", "runescape/cosmic_rune/cosmic_rune.png")
                    # time.sleep(1)


def build_portal():
    while True:
        # time.sleep(5)
        screenshot = capture_window_screenshot("RuneLite", "screenshot.png")
        
        if check_if_screenshot_contains(screenshot, "runescape/build_portal/build_fence_1.png"):
            click("RuneLite", "runescape/build_portal/build_fence_1.png")
            time.sleep(2)

            screenshot = capture_window_screenshot("RuneLite", "screenshot.png")

            if check_if_screenshot_contains(screenshot, "runescape/build_portal/build_fence_2.png"):
                click("RuneLite", "runescape/build_portal/build_fence_2.png")
                time.sleep(3)

                screenshot = capture_window_screenshot("RuneLite", "screenshot.png")

                if check_if_screenshot_contains(screenshot, "runescape/build_portal/build_fence_1.png"):
                    click("RuneLite", "runescape/build_portal/build_fence_1.png")
                    time.sleep(2)

                    screenshot = capture_window_screenshot("RuneLite", "screenshot.png")

                    if check_if_screenshot_contains(screenshot, "runescape/build_portal/build_fence_yes.png"):
                        click("RuneLite", "runescape/build_portal/build_fence_yes.png")
                        time.sleep(1.5)

                        screenshot = capture_window_screenshot("RuneLite", "screenshot.png")

                        if check_if_screenshot_contains(screenshot, "runescape/build_portal/build_fence_note.png"):
                            click("RuneLite", "runescape/build_portal/build_fence_note.png")
                            time.sleep(1.5)

# build_portal()

# cosmic_rune()



def record_actions():
    print("Recording actions. Press 'Esc' to stop.")
    actions = []
    start_time = time.time()

    def record_mouse_event(event):
        timestamp = time.time() - start_time
        if hasattr(event, 'button') and hasattr(event, 'event_type'):
            x, y = mouse.get_position()  # Get the current mouse position
            actions.append({
                "type": "mouse_click",
                "button": event.button,
                "event_type": event.event_type,
                "x": x,
                "y": y,
                "time": timestamp
            })

    def record_keyboard_event(event):
        timestamp = time.time() - start_time
        actions.append({
            "type": "key_event",
            "key": event.name,
            "event_type": event.event_type,
            "time": timestamp
        })

    mouse.hook(record_mouse_event)
    keyboard.hook(record_keyboard_event)

    try:
        while True:
            if keyboard.is_pressed('Esc'):
                print("Recording stopped.")
                break
            time.sleep(0.01)  # Reduce CPU usage
    finally:
        mouse.unhook(record_mouse_event)
        keyboard.unhook(record_keyboard_event)

    # Save actions to a file
    with open("recorded_actions.json", "w") as file:
        json.dump(actions, file)

    return actions

@print_runtime(60)
def replay_actions(record_files=["recorded_actions.json"], actions=None, sleep_time=1):
    # if not actions:
    #     with open("recorded_actions.json", "r") as file:
    #         actions = json.load(file)

    print("Replaying actions...")

    while True:
        record_file = random.choice(record_files)
        with open(record_file, "r") as file:
            actions = json.load(file)

        start_time = time.time()

        for action in actions:
            if keyboard.is_pressed('Esc'):  # Check if 'Esc' is pressed to stop replay
                print("Replay stopped.")
                break
            action_time = action["time"]
            elapsed_time = time.time() - start_time
            wait_time = action_time - elapsed_time

            if wait_time > 0:
                time.sleep(wait_time)

            if action["type"] == "mouse_click":
                if action["event_type"] == "up":
                    continue
                cursor.move_to([action["x"], action["y"]], duration=0.5)
                pyautogui.click(x=action["x"], y=action["y"], button=action["button"])
            elif action["type"] == "key_event":
                if action["event_type"] == "down":
                    keyboard.press(action["key"])
                elif action["event_type"] == "up":
                    keyboard.release(action["key"])
        
        time.sleep(sleep_time)

    print("Replay completed.")


# record_actions()
# replay_actions()

def fish_k():
    while True:
        time.sleep(5)
        screenshot = capture_window_screenshot("RuneLite", "screenshot.png")
        
        if not check_if_screenshot_contains(screenshot, "runescape/fishing.png"):
            click("RuneLite", "runescape/k.png")

# fish_k()


# pc_varlamore_course_files = [
#     # "varlamore_course_1.json",
#     # "varlamore_course_2.json",
#     # "varlamore_course_3.json",
#     # "pc_varlamore_course_1.json",
#     "pc_varlamore_course_2.json",
# ]

# mac_varlamore_course_files = [
#     # "mac_varlamore_course_1.json",
# ]

# # record_actions()
# replay_actions(record_files=pc_varlamore_course_files)

def get_numbered_images(directory, extension=".png"):
    # Get all files in directory that match the pattern
    import os
    import re
    
    # Get all png files that start with a number
    files = [f for f in os.listdir(directory) if f.endswith(extension) and f[0].isdigit()]
    
    # Sort files based on the numeric part
    files.sort(key=lambda f: int(re.search(r'\d+', f).group()))
    
    # Return full paths
    return [os.path.join(directory, f) for f in files]


def varlamore_agility():
    images = get_numbered_images("runescape/varlamore_agility")
    sleep_timer = [
        5,
        21,
        # 13,
        3,
        7,
        20,
        # 12,
        # 4,
        11,
    ]
    
    while True:
        i = 0
        # for i in range(len(images)):
        while i < len(images):
            if click("RuneLite", images[i]):
                time.sleep(sleep_timer[i])
                i += 1
            else:
                # if click("RuneLite", "runescape/varlamore_agility/failsafe_1.png"):
                #     time.sleep(4)
                #     i = 5
                # elif click("RuneLite", "runescape/varlamore_agility/failsafe_2.png"):
                #     time.sleep(13)
                #     i = 3
                # elif click("RuneLite", "runescape/varlamore_agility/failsafe_3.png"):
                #     time.sleep(11)
                #     i = 0
                # elif click("RuneLite", "runescape/varlamore_agility/failsafe_4.png"):
                #     time.sleep(11)
                #     i = 0
                # elif click("RuneLite", "runescape/varlamore_agility/failsafe_5.png"):
                #     time.sleep(4)
                #     i = 5
                # elif click("RuneLite", "runescape/varlamore_agility/failsafe_6.png"):
                #     time.sleep(11)
                #     i = 0
                if click("RuneLite", "runescape/varlamore_agility/failsafe_7.png"):
                    time.sleep(20)
                    i = 5
                elif click("RuneLite", "runescape/varlamore_agility/failsafe_8.png"):
                    time.sleep(3)
                    i = 3
                elif click("RuneLite", "runescape/varlamore_agility/failsafe_9.png"):
                    time.sleep(11)
                    i = 0
# varlamore_agility()

# images = get_numbered_images("runescape/varlamore_agility")
# print(images)
# click("RuneLite", images[0])

# time.sleep(2)
# while True:
#     varlamore_agility()

# print(get_numbered_images("runescape/varlamore_agility"))

# record_actions()

# time.sleep(500)
# replay_actions()
# replay_actions()

# monster("cyclops")
# click_polygon()

import cv2
import numpy as np
import pyautogui  # For simulating clicks
from colorsys import rgb_to_hsv

def get_hsv_range(color_rgb, hue_variation=10, sat_variation=50, val_variation=50):
    """Convert RGB to HSV and create a range of close colors."""
    r, g, b = color_rgb

    # Normalize RGB to [0,1] for conversion
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    h, s, v = rgb_to_hsv(r, g, b)

    # Convert H to OpenCV scale (0-180 instead of 0-360)
    h = int(h * 180)
    s = int(s * 255)
    v = int(v * 255)

    # Define the lower and upper bounds
    lower_bound = np.array([max(h - hue_variation, 0), max(s - sat_variation, 50), max(v - val_variation, 50)])
    upper_bound = np.array([min(h + hue_variation, 180), min(s + sat_variation, 255), min(v + val_variation, 255)])

    return lower_bound, upper_bound

import matplotlib.pyplot as plt

def click_closest_polygon(color_rgb, min_area=100):
    """Find and click the closest polygon matching the given color."""
    screenshot = capture_window_screenshot("RuneLite", "screenshot.png")

    screenshot_np = np.array(screenshot)
    screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    hsv_image = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2HSV)

    # Get color range based on input
    lower_color, upper_color = get_hsv_range(color_rgb)

    # Create mask and find contours
    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    if not contours:
        print("No polygons detected.")
        return

    # Get image center
    height, width = screenshot_bgr.shape[:2]
    center_x, center_y = width // 2, height // 2

    # Find closest contour
    closest_contour = None
    min_distance = float("inf")

    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        distance = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)

        if distance < min_distance:
            min_distance = distance
            closest_contour = cnt

    if closest_contour is None:
        print("No valid contour found.")
        return

    # Draw all detected polygons in blue
    cv2.drawContours(screenshot_bgr, contours, -1, (255, 0, 0), 2)  # Blue outline

    # Highlight closest polygon in red
    if closest_contour is not None:
        cv2.drawContours(screenshot_bgr, [closest_contour], -1, (0, 0, 255), 3)  # Red outline


    # Show the result using matplotlib
    # plt.figure(figsize=(10, 6))
    # plt.imshow(cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.show()

    # Get centroid of closest polygon
    M = cv2.moments(closest_contour)
    final_cx, final_cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

    # Simulate a mouse click
    cursor.move_to([final_cx, final_cy], duration=0.1)
    pyautogui.click(final_cx, final_cy)
    print(f"Clicked at ({final_cx}, {final_cy})")

lower_green = np.array([10, 250, 10])  # Lower bound (darker/muted green)
upper_green = np.array([70, 255, 255])  # Upper bound (bright green)
# click_closest_polygon(lower_green, upper_green)
color_rgb = (10, 250, 10)
# while True:
#     if keyboard.is_pressed("esc"):
#         print("Escape key pressed, exiting loop.")
#         break

#     click_closest_polygon(color_rgb)
#     time.sleep(2)



# import cv2
# import numpy as np
# import pytesseract

# import easyocr
# import logging

# Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Initialize the OCR reader
# reader = easyocr.Reader(['en'])

# def extract_number_from_icon(region):
#     """Extract number from a specific region next to an icon."""
#     # Convert to grayscale
#     gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     gray = clahe.apply(gray)

#     gray = cv2.bilateralFilter(gray, 9, 75, 75)
    

#     # Threshold to enhance the numbers
#     # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

#     # kernel = np.ones((3, 3), np.uint8)
#     # # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)




#     cv2.imshow("Processed Region", thresh)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Clean up the image with morphological operations (if necessary)
#     # kernel = np.ones((2, 2), np.uint8)
#     # processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


#     # cv2.imshow("Processed Region", processed)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

#     result = reader.readtext(thresh)
#         # logging.info(f"OCR result: {result}")
        
#     if not result:
#         logging.warning("No text detected in the image")
#         return ""
    
#     text = ' '.join([entry[1] for entry in result])
#     logging.info(f"Text extracted. Length: {len(text)} characters")


#     # Use Tesseract to extract numbers
#     # config = "--psm 7 -c tessedit_char_whitelist=0123456789"
#     config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'

#     extracted_text = pytesseract.image_to_string(thresh, config=config).strip()
#     print(extracted_text)

#     return extracted_text


# def detect_icons_and_extract_numbers(screenshot, icon_templates):
#     """Detect icons in the screenshot and extract corresponding numbers."""
    
#     results = {}
#     screenshot_np = np.array(screenshot)
#     screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2GRAY)

#     for icon_name, template in icon_templates.items():
#         # Match the icon in the screenshot using template matching
#         res = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
#         threshold = 0.8  # Adjust threshold for better matching
#         loc = np.where(res >= threshold)

#         for pt in zip(*loc[::-1]):  # Reverse the order for (x, y) points
#             x, y = pt
#             w, h = template.shape[::-1]  # Width and height of the template

#             left_region_length = 30
#             region = screenshot_np[y:y + h, x - left_region_length:x + w]  # Get the region around the icon
            
#             # You can visualize or log the region if needed
#             cv2.imshow("Icon Region", region)  # Optional: visualize the detected icon region
#             # Wait indefinitely until a key is pressed
#             cv2.waitKey(0)

#             # Close all OpenCV windows
#             cv2.destroyAllWindows()

#             # Define the area to the left of the icon to look for the number
#             number_region = screenshot_np[y:y + h, x - 80:x]  # Adjust width based on the distance between the icon and number

#             result = reader.readtext(number_region)
#                 # logging.info(f"OCR result: {result}")
                
#             if not result:
#                 logging.warning("No text detected in the image")
#                 # return ""
            
#             text = ' '.join([entry[1] for entry in result])
#             logging.info(f"Text extracted. Length: {len(text)} characters")

#             number = extract_number_from_icon(number_region)
            
#             if number:  # Only store the number if it's detected
#                 results[icon_name] = number
    
#     return results


# Example Usage
# screenshot = capture_window_screenshot("RuneLite", "screenshot.png")

# # Load icon templates
# icon_templates = {
#     "health_icon": cv2.imread("runescape/icons/health.png", cv2.IMREAD_GRAYSCALE),  # Example icon for health
#     "prayer_icon": cv2.imread("runescape/icons/prayer.png", cv2.IMREAD_GRAYSCALE),  # Example icon for mana
#     # Add other icons here...
# }

# # Detect icons and extract numbers
# numbers = detect_icons_and_extract_numbers(screenshot, icon_templates)

# print("Detected numbers:", numbers)

def barbarian_fishing():
    status = "start"

    while True:
        screenshot = capture_window_screenshot("RuneLite", "screenshot.png")

        if status == "start":
            print("Fishing...")
            if not check_if_screenshot_contains(screenshot, "runescape/barbarian_fishing/fishing_text.png"):
                click("RuneLite", "runescape/barbarian_fishing/fishing.png", threshold=0.8) or click("RuneLite", "runescape/barbarian_fishing/fishing_2.png", threshold=0.8)
                time.sleep(5)

            
            while not check_if_screenshot_contains(screenshot, "runescape/barbarian_fishing/fishing_text.png") or not check_if_screenshot_contains(screenshot, "runescape/barbarian_fishing/fishing_text_2.png"):
                time.sleep(5)
                screenshot = capture_window_screenshot("RuneLite", "screenshot.png")

                if check_if_screenshot_contains(screenshot, "runescape/barbarian_fishing/full_fish.png"):
                    status = "cook"
                    break
                elif check_if_screenshot_contains(screenshot, "runescape/barbarian_fishing/not_fishing.png", threshold=0.65):
                    break
            
            time.sleep(2)
        elif status == "cook":
            print("Cooking...")
            if (
                click("RuneLite", "runescape/barbarian_fishing/fire.png", threshold=0.7) or
                click("RuneLite", "runescape/barbarian_fishing/fire_2.png", threshold=0.7) or
                click("RuneLite", "runescape/barbarian_fishing/fire_3.png", threshold=0.7) or
                click("RuneLite", "runescape/barbarian_fishing/fire_4.png", threshold=0.7)
            ):
                time.sleep(5)

                if click("RuneLite", "runescape/barbarian_fishing/raw_salmon.png"):
                    time.sleep(3)
                    while check_if_screenshot_contains(screenshot, "runescape/barbarian_fishing/cooking.png") or check_if_screenshot_contains(screenshot, "runescape/barbarian_fishing/cooking_2.png"):
                        time.sleep(1)
                        screenshot = capture_window_screenshot("RuneLite", "screenshot.png")
                elif click("RuneLite", "runescape/barbarian_fishing/raw_trout.png"):
                    time.sleep(3)
                    while check_if_screenshot_contains(screenshot, "runescape/barbarian_fishing/cooking.png") or check_if_screenshot_contains(screenshot, "runescape/barbarian_fishing/cooking_2.png"):
                        time.sleep(1)
                        screenshot = capture_window_screenshot("RuneLite", "screenshot.png")
                else:
                    status = "drop"
        elif status == "drop":
            print("Dropping...")
            while click("RuneLite", "runescape/barbarian_fishing/drop_salmon.png"):
                pass
                
            while click("RuneLite", "runescape/barbarian_fishing/drop_burnt_fish.png"):
                pass
                
            while click("RuneLite", "runescape/barbarian_fishing/drop_trout.png"):
                pass
            
            time.sleep(2)
            status = "start"


    

# barbarian_fishing()

def mistrock_fishing():
    status = "start"
    folder = "mistrock_fishing"

    while True:
        screenshot = capture_window_screenshot("RuneLite", "screenshot.png")

        if status == "start":
            print("Fishing...")
            if not check_if_screenshot_contains(screenshot, "runescape/" + folder + "/fishing_text.png"):
                (
                    click("RuneLite", "runescape/" + folder + "/fishing_1.png", threshold=0.75)
                    or click("RuneLite", "runescape/" + folder + "/fishing_2.png", threshold=0.75)
                    or click("RuneLite", "runescape/" + folder + "/fishing_3.png", threshold=0.75)
                )
                time.sleep(5)

            
            while (
                not check_if_screenshot_contains(screenshot, "runescape/" + folder + "/fishing_text.png")
                or not check_if_screenshot_contains(screenshot, "runescape/" + folder + "/fishing_text_2.png")
            ):
                time.sleep(5)
                screenshot = capture_window_screenshot("RuneLite", "screenshot.png")

                if check_if_screenshot_contains(screenshot, "runescape/" + folder + "/full_fish.png"):
                    status = "bank"
                    break
                elif check_if_screenshot_contains(screenshot, "runescape/" + folder + "/not_fishing.png", threshold=0.65):
                    break
                else:
                    (
                        click("RuneLite", "runescape/" + folder + "/fishing_1.png", threshold=0.75)
                        or click("RuneLite", "runescape/" + folder + "/fishing_2.png", threshold=0.75)
                        or click("RuneLite", "runescape/" + folder + "/fishing_3.png", threshold=0.75)
                    )

                    time.sleep(3)
            
            time.sleep(2)
        elif status == "bank":
            print("Banking...")
            bank_walk_options = [
                "runescape/" + folder + "/bank_walk_1.png",
                "runescape/" + folder + "/bank_walk_2.png",
                "runescape/" + folder + "/bank_walk_3.png",
                "runescape/" + folder + "/bank_walk_4.png",
            ]

            shuffled_bank_walk_options = random.sample(bank_walk_options, len(bank_walk_options))
            
            found = False

            while not found:
                for bank_walk_option in shuffled_bank_walk_options:
                    if click("RuneLite", bank_walk_option):
                        found = True
                        break
            
            time.sleep(7)

            banker_options = [
                # "runescape/" + folder + "/banker_1.png",
                # "runescape/" + folder + "/banker_2.png",
                "runescape/" + folder + "/banker_3.png",
                "runescape/" + folder + "/banker_4.png",
            ]

            shuffled_banker_options = random.sample(banker_options, len(banker_options))

            found = False
            while not found:
                for banker_option in shuffled_banker_options:
                    if click("RuneLite", banker_option, threshold=0.75):
                        found = True
                        break
            
            time.sleep(11)

            click("RuneLite", "runescape/" + folder + "/bank_tuna.png")
            time.sleep(0.5)
            click("RuneLite", "runescape/" + folder + "/bank_swordfish.png")
            
            time.sleep(0.5)
            click("RuneLite", "runescape/" + folder + "/close.png")

            time.sleep(0.5)

            status = "walk_back"
        elif status == "walk_back":
            print("Walking back...")
            bank_walk_options = [
                "runescape/" + folder + "/bank_walk_1.png",
                "runescape/" + folder + "/bank_walk_2.png",
                "runescape/" + folder + "/bank_walk_3.png",
                "runescape/" + folder + "/bank_walk_4.png",
                "runescape/" + folder + "/bank_walk_5.png",
                "runescape/" + folder + "/bank_walk_6.png",
            ]

            shuffled_bank_walk_options = random.sample(bank_walk_options, len(bank_walk_options))
            
            found_path_back = False
            while not found_path_back:
                for bank_walk_option in shuffled_bank_walk_options:
                    if click("RuneLite", bank_walk_option, threshold=0.75):
                        found_path_back = True
                        break
            
            time.sleep(11)
            status = "start"

# mistrock_fishing()

# click("RuneLite", "runescape/barbarian_fishing/fire.png", threshold=0.8)

def varlamore_thieving():
    folder = "varlamore_thieving"
    thieving_options = [
        "runescape/" + folder + "/thieving.png",
    ]

    thieving_png = "runescape/" + folder + "/thieving.png"

    while True:
        screenshot = capture_window_screenshot("RuneLite", "screenshot.png")
        if check_if_screenshot_contains(screenshot, thieving_png):
            time.sleep(2)

            while True:
                if click("RuneLite", thieving_png):
                    break
            time.sleep(20)

            click("RuneLite", "runescape/" + folder + "/coin.png")
            time.sleep(50)
        time.sleep(0.5)

varlamore_thieving()