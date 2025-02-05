import sys
import time
from selenium import webdriver
import chromedriver_autoinstaller

def load_selenium(url):
    sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless') 
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chromedriver_autoinstaller.install()

    driver = webdriver.Chrome(options=chrome_options)

    driver.get(url)
    driver.implicitly_wait(10)

    print("Starting the wait...")
    time.sleep(15)
    print("15 seconds are over. Continuing execution...")

    return driver

def extract_time_taken(value):
        time_strings = value.split('(min) ')
        return [float(time) for time in time_strings if time.isdigit()]