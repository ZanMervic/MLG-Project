import json
import os
import time
import ctypes
import re
import sys
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException


def extract_number(text):
    match = re.search(r"\d+", text or "")
    return int(match.group()) if match else None


def count_stars(ul_el):
    try:
        # Count imgs that are NOT the empty star; keeps behavior while being a bit future-proof
        return len(
            ul_el.find_elements(By.XPATH, ".//img[not(contains(@src, 'starempty'))]")
        )
    except Exception:
        return 0


def process_entry(entry, uid, date):
    try:
        # --- Problem basic info ---
        a = entry.find_element(By.XPATH, ".//h3/a")
        problem_name = a.text.strip()
        # setter is the first <p> after the <h3>, per the sample HTML
        p_tags = entry.find_elements(By.XPATH, "./p")
        setter = p_tags[0].text.strip() if len(p_tags) > 0 else None

        # Grade line looks like: "7A. You graded this problem 7A."
        grade_text = ""
        for p in p_tags:
            t = p.text.strip()
            if "graded this problem" in t:
                grade_text = t
                break
        official_grade = grade_text.split(".", 1)[0].strip() if grade_text else None

        # Holds (e.g., "Any marked holds") – keep as-is
        holds_text = None
        for p in p_tags:
            t = p.text.strip()
            if "holds" in t.lower():
                holds_text = t
                break

        # Ratings: first <ul> is average problem rating, second <ul> is user's rating
        uls = entry.find_elements(By.XPATH, "./ul")
        avg_problem_rating = count_stars(uls[0]) if len(uls) >= 1 else None
        user_rating = count_stars(uls[1]) if len(uls) >= 2 else None

        # Attempts: bold <p> like "Flashed (1)" – extract number in parentheses
        attempts = None
        bold_ps = entry.find_elements(By.XPATH, ".//p[contains(@class, 'bold')]")
        for bp in bold_ps:
            m = re.search(r"\((\d+)\)", bp.text or "")
            if m:
                attempts = int(m.group(1))
                break
        if attempts is None:
            attempts = 0

        # Comment: grab a non-bold <p> that isn't the setter/grade/holds lines (often last)
        comment = None
        skip_texts = {setter, grade_text, holds_text}
        for p in reversed(p_tags):  # prefer later paragraphs
            t = (p.text or "").strip()
            if (
                t
                and "graded this problem" not in t
                and t not in skip_texts
                and "moonboard" not in t.lower()
            ):
                comment = t
                break

        # --- Update user dict ---
        users.setdefault(uid, {})
        users[uid].setdefault("problems", {})
        users[uid]["problems"][problem_name] = {
            "grade": official_grade,
            "rating": (
                float(user_rating) if user_rating is not None else None
            ),  # user's own rating
            "date": date.strftime("%Y-%m-%d"),
            "attempts": attempts,
            "comment": comment,
        }

        # --- Update global problems dict ---
        if problem_name not in problems:
            problems[problem_name] = {
                "grade": official_grade,
                "rating": (
                    float(avg_problem_rating)
                    if avg_problem_rating is not None
                    else None
                ),  # average rating (left stars)
                "num_sends": 1,
                "setter": setter,
                "holds": holds_text,
            }
        else:
            # Only increment sends; do not modify other fields
            problems[problem_name]["num_sends"] = (
                problems[problem_name].get("num_sends") or 0
            ) + 1

        # print(f"Added entry for '{problem_name}' on {date.strftime('%Y-%m-%d')}")
    except Exception as e:
        pass
        # print(f"Error parsing entry for date {date.strftime('%Y-%m-%d')}: {e}")


def process_profile(href):
    try:
        if not href:
            return
        # Navigate to the profile page (safe even if already on this URL in the new tab)
        driver.get(href)
        # print(f"Processing profile: {href}")

        # Wait for the logbook section to load
        select = Select(
            wait.until(EC.presence_of_element_located((By.ID, "Holdsetup")))
        )
        select.select_by_visible_text("MoonBoard Masters 2019")

        time.sleep(0.5)

        # Extract the information about the climber:
        uid = href.rstrip("/").split("/")[-1]
        users.setdefault(uid, {})
        bio_section = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".user-bio.jumbotron"))
        )
        bio = bio_section.find_element(
            By.XPATH, ".//label[normalize-space()='Bio:']/following-sibling::span"
        ).text.strip()
        ranking_text = bio_section.find_element(By.ID, "spnRank").text.strip()
        height_text = bio_section.find_element(
            By.XPATH, ".//label[@for='User_Height']/following-sibling::span"
        ).text.strip()
        highest_grade_text = bio_section.find_element(By.ID, "hhighgrade").text.strip()
        weight_text = bio_section.find_element(
            By.XPATH, ".//label[@for='User_Weight']/following-sibling::span"
        ).text.strip()
        total_problems_text = bio_section.find_element(
            By.ID, "htotalclimbed"
        ).text.strip()

        if (
            extract_number(total_problems_text) == 0
            or extract_number(ranking_text) == 0
        ):
            # print(f"User {uid} has sent 0 problems; skipping logbook")
            users.pop(uid, None)
            return

        users[uid].update(
            {
                "bio": bio or None,
                "ranking": extract_number(ranking_text),
                "highest_grade": highest_grade_text.split(":", 1)[-1].strip() or None,
                "height": extract_number(height_text),
                "weight": extract_number(weight_text),
                "problems_sent": extract_number(total_problems_text) or 0,
            }
        )

        # Loop through each page of the logbook
        while True:
            # Wait for the logbook table OR the "no logbook" message
            table = wait.until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "k-master-row"))
            )

            if not table:
                # Remove this user if they have no logbook entries
                users.pop(uid, None)
                # print(f"User {uid} has no logbook entries")
                break

            # Process each row in the table
            for row in table:
                try:
                    # Get date
                    header = row.find_element(By.CSS_SELECTOR, ".logbook-grid-header")
                    date_str = header.text.splitlines()[0].strip()
                    date = datetime.strptime(date_str, "%d %b %y")

                    # Expand the row (tolerant to icon state)
                    toggle = row.find_element(
                        By.CSS_SELECTOR, ".k-i-expand, .k-i-collapse"
                    )
                    if "k-i-expand" in (toggle.get_attribute("class") or ""):
                        toggle.click()

                    # Scope the expanded detail row to this row only
                    expanded_row = wait.until(
                        lambda d: row.find_element(
                            By.XPATH,
                            "following-sibling::tr[contains(@class,'k-detail-row')][1][not(contains(@style,'display: none'))]",
                        )
                    )

                    entries = wait.until(
                        lambda d: expanded_row.find_elements(
                            By.XPATH,
                            ".//div[@class='logbookentry']//div[@class='entry']//div[@class='entry']",
                        )
                    )
                    # print(
                    #     f"Found {len(entries)} entries for date {date.strftime('%Y-%m-%d')}"
                    # )
                    # Collect data from each entry
                    for entry in entries:
                        process_entry(entry, uid, date)

                    # Collapse if currently expanded
                    toggle = row.find_element(
                        By.CSS_SELECTOR, ".k-i-expand, .k-i-collapse"
                    )
                    if "k-i-collapse" in (toggle.get_attribute("class") or ""):
                        toggle.click()

                except Exception as e:
                    # print(f"Error processing row in profile {href}: {e}")
                    continue

            # Go to the next page of the logbook if available
            try:
                next_button = wait.until(
                    EC.presence_of_element_located(
                        (
                            By.CSS_SELECTOR,
                            "a.k-link.k-pager-nav[aria-label='Go to the next page']",
                        )
                    )
                )
                classes = (next_button.get_attribute("class") or "").split()
                if "k-state-disabled" in classes:
                    # print(f"No more logbook pages for user {uid}")
                    return
                next_button.click()
                # Wait for the current table to be replaced
                wait.until(EC.staleness_of(table[0]))
                # print("Navigated to next page of logbook")
            except TimeoutException:
                # print(f"No next button found; finished logbook for user {uid}")
                return

    except Exception as e:
        # TODO: Handle more gracefully if the user simply has no logbook
        print(f"Error processing profile {href}: {e}")


def open_in_new_tab_and_process(href, fn):
    """Open href in a new tab, run fn() while focused on that tab, then close and switch back."""
    list_handle = driver.current_window_handle
    driver.execute_script("window.open(arguments[0], '_blank');", href)
    wait.until(lambda d: len(d.window_handles) > 1)
    new_handle = [h for h in driver.window_handles if h != list_handle][0]
    driver.switch_to.window(new_handle)
    try:
        fn()
    finally:
        driver.close()
        driver.switch_to.window(list_handle)


def navigate_to_page(target_page):
    """Navigate to the target page using the pager."""
    while True:
        # Get current page
        current_page_el = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "span.k-state-selected"))
        )
        current_page = int(current_page_el.text.strip())

        if current_page == target_page:
            break

        # Get total pages
        last_page_btn = driver.find_element(By.CSS_SELECTOR, "a.k-pager-last")
        total_pages = int(last_page_btn.get_attribute("data-page"))

        # Decide direction
        if target_page > total_pages // 2 and current_page < total_pages // 2:
            # Jump to last page first
            last_page_btn.click()
            wait.until(EC.staleness_of(current_page_el))
            continue

        # Try to find target page link
        page_links = driver.find_elements(
            By.CSS_SELECTOR, ".k-pager-numbers a.k-link[data-page]"
        )

        target_found = False
        for link in page_links:
            page_num = int(link.get_attribute("data-page"))
            if page_num == target_page:
                link.click()
                wait.until(EC.staleness_of(current_page_el))
                target_found = True
                break

        if target_found:
            continue

        # Click ellipsis (...) to skip forward or backward
        ellipsis_links = [
            l for l in page_links if l.get_attribute("title") == "More pages"
        ]

        if target_page > current_page and len(ellipsis_links) > 0:
            # Click forward ellipsis (last one)
            ellipsis_links[-1].click()
            wait.until(EC.staleness_of(current_page_el))
        elif target_page < current_page and len(ellipsis_links) > 0:
            # Click backward ellipsis (first one)
            ellipsis_links[0].click()
            wait.until(EC.staleness_of(current_page_el))
        else:
            # No ellipsis available, use next/prev buttons
            if target_page > current_page:
                next_btn = driver.find_element(
                    By.CSS_SELECTOR, "a[aria-label='Go to the next page']"
                )
                next_btn.click()
            else:
                prev_btn = driver.find_element(
                    By.CSS_SELECTOR, "a[aria-label='Go to the previous page']"
                )
                prev_btn.click()
            wait.until(EC.staleness_of(current_page_el))


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(
            "To run this script you need to input 4 arguments (in this order):\nMoonboard account email\nMoonboard account password\nPage number from which you want to start\nPage number at which you want to end"
        )
        print("e.g. python moonboard.py user@gmail.com geslo123 100 200")
        sys.exit()

    email = sys.argv[1]
    password = sys.argv[2]
    from_page = int(sys.argv[3])
    to_page = int(sys.argv[4])

    if to_page < from_page:
        print("The Start page number should be smaller than the End page number")
        sys.exit()

    # Prevent system sleep (Windows) -------------------------------------
    # Flags
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_AWAYMODE_REQUIRED = 0x00000040  # optional (media PCs)

    class stay_awake:
        def __enter__(self):
            # Prevent system sleep while this context is active
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED  # | ES_AWAYMODE_REQUIRED
            )

        def __exit__(self, exc_type, exc, tb):
            # Clear the requirement so normal sleep can resume
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

    BASE_URL = "https://moonboard.com/Account/Login"

    options = Options()
    # options.add_argument("--headless=new")  # or "--headless" if needed
    # options.add_argument("--disable-gpu")
    # options.add_argument("--no-sandbox")
    # options.add_argument("--disable-dev-shm-usage")
    # options.add_argument("--disable-extensions")
    options.add_argument("--window-size=1280,800")
    # options.page_load_strategy = "eager"

    # Dictionary with the following structure:
    # {
    #   "uid": {
    #       "height": int,
    #       "weight": int,
    #       "ranking": int,
    #       "highest_grade": str,
    #       "problems_sent": int,
    #       "problems": {
    #           "problem_name": {
    #               "grade": str,
    #               "rating": float,
    #               "date": str,
    #               "attempts": int,
    #               "comment": str
    #           }
    #       }
    #   }
    # }
    users = {}

    # Dictionary with the following structure:
    # {
    #   "problem_name": {
    #       "grade": str,
    #       "rating": float,
    #       "num_sends": int,
    #       "setter": str,
    #       "holds": str,
    #   }
    # }
    problems = {}

    page = from_page

    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, 10)
    driver.get(BASE_URL)

    driver.find_element(By.ID, "Login_Username").send_keys(email)
    driver.find_element(By.ID, "Login_Password").send_keys(password)
    driver.find_element(By.NAME, "send").click()

    with stay_awake():
        try:
            print("Navigating to Users tab...")
            # Go to the Users tab
            users_button = wait.until(
                EC.presence_of_element_located((By.ID, "m-users"))
            )
            users_button.click()

            # Go to the Start User page number
            print(f"Navigating to page {from_page} of user list...")
            if from_page > 1:
                navigate_to_page(from_page)

            # Main loop
            while page <= to_page:
                print(f"Processing page {page}/{to_page}")

                # Get all profile links on the current page
                profile_links = wait.until(
                    EC.presence_of_all_elements_located(
                        (By.XPATH, "//a[starts-with(@href, '/Account/Profile/')]")
                    )
                )
                hrefs = [
                    link.get_attribute("href")
                    for link in profile_links
                    if link.get_attribute("href")
                ]

                # Process each profile link in a new tab (list page stays open)
                for href in hrefs:

                    def _process():
                        process_profile(href)

                    open_in_new_tab_and_process(href, _process)

                # Save to JSON
                with open(
                    f"users_{from_page}_{page - 1}.json", "w", encoding="utf-8"
                ) as file:
                    json.dump(users, file, ensure_ascii=False, indent=4)

                with open(
                    f"problems_{from_page}_{page - 1}.json", "w", encoding="utf-8"
                ) as file:
                    json.dump(problems, file, ensure_ascii=False, indent=4)

                os.rename(
                    f"users_{from_page}_{page - 1}.json",
                    f"users_{from_page}_{page}.json",
                )
                os.rename(
                    f"problems_{from_page}_{page - 1}.json",
                    f"problems_{from_page}_{page}.json",
                )

                # Click to the next page on the list (still on the list tab)
                next_button = wait.until(
                    EC.presence_of_element_located(
                        (
                            By.CSS_SELECTOR,
                            "a.k-link.k-pager-nav[aria-label='Go to the next page']",
                        )
                    )
                )
                classes = (next_button.get_attribute("class") or "").split()
                if "k-state-disabled" in classes:
                    break  # no more pages
                next_button.click()
                page += 1
                wait.until(EC.staleness_of(profile_links[0]))
                # print("Navigated to next page of logbook")

        except TimeoutException:
            print("Users button not found or clickable")
            driver.quit()
            exit(1)
