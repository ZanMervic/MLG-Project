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



##### Last: YOU’RE MINES STILL

driver = webdriver.Remote("http://127.0.0.1:4723/wd/hub", options=options)

time.sleep(2)  # let app load

driver.execute_script("mobile: shell", {
    "command": "settings",
    "args": ["put", "global", "stay_on_while_plugged_in", "3"]
})

#3 že
#2 še
with open("error_list_2.json", "r", encoding="utf-8") as f:#problem_names
    names_list = json.load(f)
os.makedirs("screenshots", exist_ok=True)

# Read existing error and missing lists if they exist
try:
    with open("error_list_new.json", "r", encoding="utf-8") as f:
        error_list = json.load(f)
    print(f"Loaded existing error list with {len(error_list)} items")
except FileNotFoundError:
    error_list = []
    print("No existing error list found, starting fresh")

try:
    with open("missing_list.json", "r", encoding="utf-8") as f:
        missing_list = json.load(f)
    print(f"Loaded existing missing list with {len(missing_list)} items")
except FileNotFoundError:
    missing_list = []
    print("No existing missing list found, starting fresh")

start_name = ""#fixed
skip = bool(start_name)

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
    
    
        # Check search results count using the "Problems\nX" element
        try:
            # Wait a moment for search results to load
            time.sleep(1)
            
            # Find the "Problems\nX" element to get exact count
            problems_element = driver.find_element(AppiumBy.XPATH, "//android.view.View[@content-desc and starts-with(@content-desc, 'Problems')]")
            problems_text = problems_element.get_attribute("content-desc")
            
            # Extract the number from "Problems\n22" format
            match = re.search(r'Problems\n(\d+)', problems_text)
            if match:
                result_count = int(match.group(1))
                print(f"Found {result_count} search results for: {name}")
            else:
                print(f"Could not parse result count from: {problems_text}")
                result_count = 0
            
            if result_count == 0:
                print(f"No search results found for: {name}")
                missing_list.append(name)
                # Write missing list to file immediately
                with open("missing_list.json", "w", encoding="utf-8") as f:
                    json.dump(missing_list, f, ensure_ascii=False, indent=4)
                continue
            elif result_count == 1:
                # If exactly 1 result, click it directly without name matching
                print(f"Found exactly 1 search result for: {name} - clicking it directly")
                problem_elements = driver.find_elements(AppiumBy.XPATH, "//android.view.View[@content-desc and contains(@content-desc, 'Set by')]")
                if problem_elements:
                    problem_elements[0].click()
                    print(f"Clicked on single result for: {name}")
                else:
                    print(f"Error: Count shows 1 but no problem elements found for: {name}")
                    error_list.append(name)
                    with open("error_list_3.json", "w", encoding="utf-8") as f:
                        json.dump(error_list, f, ensure_ascii=False, indent=4)
                    continue
            else:
                # Multiple results - try to find exact match first
                print(f"Found {result_count} search results for: {name}")
                
                # Get all problem elements to check for exact matches
                # First, scroll to ensure all elements are loaded
                scroll_view = driver.find_element(AppiumBy.XPATH, "//android.widget.ScrollView")
                
                # Scroll to the bottom to load all elements using swipe
                size = scroll_view.size
                location = scroll_view.location
                start_x = location['x'] + size['width'] // 2
                start_y = location['y'] + size['height'] - 100
                end_y = location['y'] + 100
                
                driver.swipe(start_x, start_y, start_x, end_y, 1000)
                time.sleep(0.5)  # Wait for elements to load
                
                # Now get all problem elements from the ScrollView
                problem_elements = driver.find_elements(AppiumBy.XPATH, "//android.widget.ScrollView//android.view.View[@clickable='true' and @content-desc]")
                
                # Filter for elements that look like problems (contain grade info, repeats, or "Set by")
                problem_elements = [elem for elem in problem_elements if 'Set by' in elem.get_attribute("content-desc") or 'repeats' in elem.get_attribute("content-desc") or 'Grade:' in elem.get_attribute("content-desc") or 'climb this problem' in elem.get_attribute("content-desc")]
                
                print(f"Found {len(problem_elements)} problem elements after scrolling")
                
                # If we still don't get enough elements, try alternative approaches
                if len(problem_elements) < result_count:
                    print(f"Still only {len(problem_elements)} elements, trying broader search...")
                    # Try finding all View elements with content-desc in ScrollView
                    all_elements = driver.find_elements(AppiumBy.XPATH, "//android.widget.ScrollView//android.view.View[@content-desc]")
                    problem_elements = [elem for elem in all_elements if 'Set by' in elem.get_attribute("content-desc") or 'repeats' in elem.get_attribute("content-desc") or 'Grade:' in elem.get_attribute("content-desc") or 'climb this problem' in elem.get_attribute("content-desc")]
                    print(f"Found {len(problem_elements)} elements with broader search")
                
                # Look for exact match by checking if the problem name appears at the start of the content-desc
                exact_match_found = False
                normalized_name = normalize_text(name).lower()
                print(f"Looking for exact match of: '{normalized_name}'")
                
                for element in problem_elements:
                    content_desc = element.get_attribute("content-desc")
                    # Get the problem name part (before the first \n)
                    problem_name = content_desc.split('\n')[0].lower()
                    print(f"  Checking problem: '{problem_name}'")
                    
                    # Check for exact match - the problem name must be exactly the same
                    if problem_name == normalized_name:
                        element.click()
                        print(f"Clicked on exact match for: {name}")
                        exact_match_found = True
                        break
                
                if not exact_match_found:
                    # If no exact match, find the closest match
                    print(f"No exact match found for: {name}, looking for closest match")
                    if problem_elements:
                        best_match = None
                        best_score = 0
                        
                        for element in problem_elements:
                            content_desc = element.get_attribute("content-desc")
                            problem_name = content_desc.split('\n')[0].lower()
                            
                            # Calculate similarity score with better matching logic
                            search_term = normalized_name
                            score = 0
                            
                            # Check if search term is at the beginning of problem name
                            if problem_name.startswith(search_term):
                                # Check if it's a complete word (next char is space, end, or special char)
                                if len(problem_name) == len(search_term) or problem_name[len(search_term)] in ' \n\t.,!?':
                                    score = 1.0  # Perfect match
                                else:
                                    score = 0.8  # Starts with but not complete word
                            # Check if search term is contained as complete word
                            elif f" {search_term} " in f" {problem_name} " or problem_name.endswith(f" {search_term}"):
                                score = 0.9  # Complete word match
                            # Check if search term is contained anywhere
                            elif search_term in problem_name:
                                score = 0.5  # Partial match
                            
                            if score > best_score:
                                best_score = score
                                best_match = element
                        
                        if best_match:
                            best_match.click()
                            print(f"Clicked on closest match for: {name} (score: {best_score:.2f})")
                        else:
                            # If no good match found, click first result
                            problem_elements[0].click()
                            print(f"No good match found for: {name}, clicking first result")
                    else:
                        print(f"Error: No problem elements found for: {name}")
                        error_list.append(name)
                        with open("error_list_3.json", "w", encoding="utf-8") as f:
                            json.dump(error_list, f, ensure_ascii=False, indent=4)
                        continue
        except Exception as e:
            print(f"Error checking search results for '{name}': {e}")
            # If we can't check results, treat as error
            error_list.append(name)
            with open("error_list_3.json", "w", encoding="utf-8") as f:
                json.dump(error_list, f, ensure_ascii=False, indent=4)
            continue
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
        # Write error list to file immediately after each error
        with open("error_list_3.json", "w", encoding="utf-8") as f:
            json.dump(error_list, f, ensure_ascii=False, indent=4)
        
        
print(f"Error list: {error_list}")
print(f"Missing list: {missing_list}")