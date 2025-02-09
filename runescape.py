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
def replay_actions(record_files=["recorded_actions.json"], actions=None):
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
        
        time.sleep(1)

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


varlamore_course_files = [
    # "varlamore_course_1.json",
    # "varlamore_course_2.json",
    # "varlamore_course_3.json",
    "recorded_actions.json",
]

# record_actions()
replay_actions(record_files=varlamore_course_files)