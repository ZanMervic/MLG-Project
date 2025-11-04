from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import os
import unicodedata

import re

#masters 2019
def normalize_text(name):
    import unicodedata
    # remove accents and convert to lowercase
    return ''.join(
        c for c in unicodedata.normalize('NFD', name)
        if unicodedata.category(c) != 'Mn'
    ).lower()


options = UiAutomator2Options()
options.platform_name = "Android"
options.platform_version = "14"
options.device_name = "GalaxyA54"
options.automation_name = "UiAutomator2"
options.app_package = "com.trainingboard.moon"
options.app_activity = ".MainActivity" 
options.no_reset = True
options.unicode_keyboard = True
options.reset_keyboard = True


driver = webdriver.Remote("http://127.0.0.1:4723/wd/hub", options=options)
time.sleep(2)  # let app load

with open("problem_names.json", "r", encoding="utf-8") as f:
    names_list = json.load(f)
os.makedirs("screenshots", exist_ok=True)


start_name = "SACRÉ BLEU"
skip = bool(start_name)

error_list = []

for name in names_list:
    if skip:
        if name == start_name:
            skip = False
        else:
            continue
    try:
        #find and type in search bar
        search_bar = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((AppiumBy.CLASS_NAME, "android.widget.EditText"))
        )
        search_bar.click()
        search_bar.clear()
        search_bar.send_keys(name.lower())
        print(f"Typed: {name}")
        #time.sleep(2)  # wait for search results

        #find element by content-desc and click
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((AppiumBy.ANDROID_UIAUTOMATOR,f'new UiSelector().descriptionContains("{normalize_text(name)}")'))
        )
        element.click()
        print(f"Clicked on element for: {name}")
        #time.sleep(2)  # wait for element to load

        #take screenshot
        screenshot_path = f"screenshots/{name.replace(' ', '_')}_screenshot.png"
        driver.save_screenshot(screenshot_path)
        print(f"Screenshot saved: {screenshot_path}")

        #Go back to search screen
        driver.back()
        #time.sleep(1)

    except Exception as e:
        print(f"Error for '{name}': {e}")
        error_list.append(name)
        
        
print(f"Error list: {error_list}")
with open("error_list.json", "w", encoding="utf-8") as f:
    json.dump(error_list, f, ensure_ascii=False, indent=4)