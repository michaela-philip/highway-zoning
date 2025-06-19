# should eventually turn this into a function that takes city, state as input 

import pandas as pd
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from selenium.webdriver.support.ui import Select

### FUNCTION TO SCRAPE STREET NAMES FROM STEVE MORSE ###
def scrape_streets(url):
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    driver.get(url)
    wait = WebDriverWait(driver, 10)

    # choose the frame that contains the dropdown for census year
    census = wait.until(EC.presence_of_element_located((By.NAME, 'aFrame')))
    driver.switch_to.frame(census)

    # choose 1940 census
    year = wait.until(EC.element_to_be_clickable((By.NAME, 'yr')))
    select = Select(year)
    select.select_by_visible_text('1940')
    time.sleep(2)

    # go back to main frame then choose state frame
    driver.switch_to.default_content()
    stateframe = wait.until(EC.presence_of_element_located((By.NAME, 'stateFrame')))
    driver.switch_to.frame(stateframe)

    # choose Georgia
    state = wait.until(EC.element_to_be_clickable((By.NAME, 'states')))
    state.click()
    state.find_element(By.XPATH, './option[@value="GA"]').click()

    # go back to main frame then choose city frame
    driver.switch_to.default_content()
    cityframe = wait.until(EC.presence_of_element_located((By.NAME, 'cityFrame')))
    driver.switch_to.frame(cityframe)

    # choose Atlanta
    city = wait.until(EC.element_to_be_clickable((By.NAME, 'cities')))
    city.click()
    city.find_element(By.XPATH, './option[@value="AT"]').click()

    # go back to main frame then choose enumeration district frame
    driver.switch_to.default_content()
    edframe = wait.until(EC.presence_of_element_located((By.NAME, 'streetFrame')))
    driver.switch_to.frame(edframe)

    # identify ed dropdown menu and get a list of enumeration districts 
    ed_dropdown = wait.until(EC.presence_of_element_located((By.NAME, 'streets')))
    select = Select(ed_dropdown)
    eds = select.options[1:]

    street_list = []
    i = 1

    for ed in eds:
        ed_name = ed.text
        ed_dropdown.click()
        ed.click()

        # switch back to default content and then to the streets frame
        driver.switch_to.default_content()
        streetsframe = wait.until(EC.presence_of_element_located((By.NAME, 'edFrame')))
        driver.switch_to.frame(streetsframe)

        # pull street names
        streets = driver.find_element(By.TAG_NAME, 'body').text
        street_names = [s.strip() for s in streets.split('\n') if s.strip()]
        for street in street_names:
            street_list.append({'ed_name': ed_name, 'street': street})    
        driver.switch_to.default_content()
        time.sleep(1)
        # print statement to verify - keep in til including states and cities in output, then can remove
        print(f"Scraped ED: {ed_name}, Streets: {', '.join(street_names)}")

        # go back to default then enumeration district frame
        driver.switch_to.default_content()
        driver.switch_to.frame(edframe)

    driver.quit()
    street_list = pd.DataFrame(street_list)
    return street_list 

### FUNCTION TO SCRAPE LIST OF ATL STREET CHANGES ###
def scrape_street_changes(url):
    street_changes = []
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    driver.get(url)

    main_ul = driver.find_element(By.TAG_NAME, 'ul')
    top_lis = main_ul.find_elements(By.TAG_NAME, 'li')
    for li in top_lis:
        new_name = li.text.split('\n')[0]
        sub_uls = li.find_elements(By.TAG_NAME, 'ul')
        if sub_uls:
            old_lis = sub_uls[0].find_elements(By.TAG_NAME, 'li')
            for old_li in old_lis:
                old_name = old_li.text
                if old_name:
                    street_changes.append({
                    'new_name': new_name,
                    'old_name': old_name,
                })

    driver.quit()
    return street_changes

### FUNCTION TO FORMAT STREET NAMES THE SAME WAY AS IN CENSUS/STEVE MORSE ###
def format_street_changes(df):
    df = (df
        .str.lower()
        .str.replace(r'\([^()]*\)', '', regex=True)
        .str.replace(r'\bavenue\b', 'ave', regex=True)
        .str.replace(r'\bstreet\b', '', regex=True)
        .str.replace(r'\broad\b', ' rd', regex=True)
        .str.replace(r'\bdrive\b', 'dr', regex=True)
        .str.replace(r'\bplace\b', 'pl', regex=True)
        .str.replace(r'\bcourt\b', ' ct', regex=True)
        .str.replace(r'\b(nw|ne|sw|se)\b', '', regex=True)
        .str.replace(r'-\s*', '', regex=True)
        .str.strip()
    )
    return df

####################################################################################################

url = 'https://stevemorse.org/census/index.html?ed2street=1'
street_list = scrape_streets(url)
street_list.to_csv('data/output/ga_streets.csv', index=False)
print('street_list csv created!')

url = 'http://jolomo.net/atlanta/streets.html'
atl_street_changes = pd.DataFrame(scrape_street_changes(url))
atl_street_changes['new_name'] = format_street_changes(atl_street_changes['new_name'])
atl_street_changes['old_name'] = format_street_changes(atl_street_changes['old_name'])

atl_street_changes.to_csv('data/output/atl_street_changes.csv', index=False)
print('atl street_change csv created!')