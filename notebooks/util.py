from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import requests
import sys
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup, Tag
import os

show_tables = "SELECT name FROM sqlite_master WHERE type='table';"
show_columns = "SELECT name FROM PRAGMA_TABLE_INFO('%s');"

def get_url(url, dom=None, retries=float('inf')):
    if dom:
        url = dom + url
    try:
        return requests.get(url)
    except:
        if retries == 0:
            print('timed out')
            return url
        return get_url(url, retries=retries-1)



def multiprocess(input_list, func):
    """Simple ThreadPoolExecutor Wrapper.

        Input_list: list to be mapped over by the executor.
        func: function to be mapped.

        returns output of the function over the list.

        Code:
            with ThreadPoolExecutor(max_workers=2) as executor:
                for output in executor.map(func, input_list):
                    yield output
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        for output in executor.map(func, input_list):
            yield output


import numpy as np
emails = [
    "alex2awesome@gmail.com",
    "alvivianspangher@aol.com", 
    "alex3awesome@gmail.com",
    "alex4awesome@gmail.com",
    "alex5awesome@gmail.com"
]
def unpaywall(doi, retry=0, pdfonly=False):
    email = np.random.choice(emails)
    try:
        r = requests.get("https://api.unpaywall.org/v2/{}".format(doi), params={"email":email})
    except:
        if retry < 3:
            return unpaywall(doi, retry+1)
        else:
            print("Retried 3 times and failed. Giving up")
            return None

    ## handle unpaywall API 
    if r.status_code == 404:
        print("Invalid/unknown DOI {}".format(doi))
        return None
    if r.status_code == 500:
        print("Unpaywall API failed. Try: {}/3".format(retry+1))
        if retry < 3:
            return unpaywall(doi, retry+1)
        else:
            print("Retried 3 times and failed. Giving up")
            return None

    try:
        results = r.json()
    except json.decoder.JSONDecodeError:
        print("Response was not json")
        return r.text
   
    return results


def selenium_get(link, headless=True):
    """Scrape webpage using selenium Chrome.
    
    Returns: 
        link, html
    """
    try:
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("headless")
        if 'win' in sys.platform:
            driver = webdriver.Chrome('C:/Users/alex2/chromedriver.exe', options=options)
        else:
            driver = webdriver.Chrome("/mnt/c/Users/alex2/chromedriver.exe", options=options)
            
        driver.implicitly_wait(1)
        driver.get(link)
        return link, driver.page_source
    except:
        return link, ""
    

def clean_me(html):
    if not isinstance(html, (Tag, BeautifulSoup)):
        html = BeautifulSoup(html, 'lxml')
    for s in html(['script', 'style']):#, 'meta', 'noscript']):
        s.decompose()
    return ' '.join(html.stripped_strings)