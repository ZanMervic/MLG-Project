from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
import time
import os
import json



#spremenita glede na svoj telefon/emulator
options = UiAutomator2Options()
options.platform_name = "Android"
options.platform_version = "14"
options.device_name = "GalaxyA54"
options.automation_name = "UiAutomator2"
options.app_package = "com.trainingboard.moon"
options.app_activity = ".MainActivity" 
options.no_reset = True


driver = webdriver.Remote("http://127.0.0.1:4723/wd/hub", options=options)#povezava z appium serverjem
time.sleep(2)

with open("problem_names.json", "r", encoding="utf-8") as f:
    names_list = json.load(f)
    
os.makedirs("screenshots", exist_ok=True)


for name in names_list:
    try:
        #find search bar and type name
        search_bar = driver.find_element(AppiumBy.CLASS_NAME, "android.widget.EditText")
        search_bar.clear()
        search_bar.send_keys(name)
        time.sleep(1)

        #find the element matching the name (content-desc / accessibility id)
        element = driver.find_element(
            AppiumBy.ANDROID_UIAUTOMATOR,
            f'new UiSelector().descriptionContains("{name}")'
        )
        element.click()
        time.sleep(2)#wait to load... lahkon izboljšamo

        #take full screen screenshot
        screenshot_path = f"screenshots/{name.replace(' ', '_')}.png"
        driver.save_screenshot(screenshot_path)
        print(f"Screenshot saved: {screenshot_path}")

        #go back to search screen
        driver.back()
        time.sleep(1)

    except Exception as e:
        print(f"Error for '{name}': {e}")

driver.quit()
print("All done!")
