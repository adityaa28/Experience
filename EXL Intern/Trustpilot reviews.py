#!/usr/bin/env python
# coding: utf-8

# In[65]:


from selenium import webdriver      
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from bs4 import BeautifulSoup as bs
import time


# In[43]:


'''
#browser = webdriver.Firefox()#Chrome('./chromedriver.exe')
df = pd.DataFrame(columns=['Title','Content','Time and tag']) # creates master dataframe 
page_url1='https://www.trustpilot.com/review/lendingclub.com?page=2'
PATIENCE_TIME = 60
#LOAD_MORE_BUTTON_XPATH = '//*[@id="bvTrackingContainer"]/div[4]/a'
#title_xpath="reviews__item-title"
#content_xpath="reviews__quote"
time_xpath="styles_datesWrapper__RCEKH"
driver = webdriver.Safari()
driver.get(page_url1)
tme=driver.find_elements(by=By.CLASS_NAME, value=time_xpath)
for i in range(len(tme)):
    print(tme[i].text)
i=0

while (True | i!=2000):
    try:
        loadMoreButton=WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="bvTrackingContainer"]/div[4]/a')))
        #loadMoreButton=WebDriverWait(driver, 10).until(EC.presence_of_element_located((by=By.XPATH, value=LOAD_MORE_BUTTON_XPATH)))
        #loadMoreButton = driver.find_element(by=By.XPATH, value=LOAD_MORE_BUTTON_XPATH)
        time.sleep(0.5)
        loadMoreButton.click()
        i+=1
        time.sleep(4)
    except Exception as e:
        print (e)
        break

title = driver.find_elements(by=By.CLASS_NAME, value=title_xpath)
content = driver.find_elements(by=By.CLASS_NAME, value=content_xpath)
time_tag=driver.find_elements(by=By.CLASS_NAME, value=time_xpath)
print(len(title))
title_list=[]
content_list=[]
time_tag_list=[]
for p in range(len(title)):
        title_list.append(title[p].text)
        content_list.append(content[p].text)
        time_tag_list.append(time_tag[p].text)
time.sleep(10)
driver.quit()
temp_df = pd.DataFrame(list(zip(title_list[1:],content_list[1:],time_tag_list[1:])), columns=['Title','Content','Time and tag'])
df = df.append(temp_df)
df.to_csv('Downloads/file.csv')
'''


# In[66]:


df = pd.DataFrame(columns=['Title','Content','Date'])
data=[]
for pages in range(1,172):
    page_url1="https://www.trustpilot.com/review/lendingclub.com?page="+str(pages)
    PATIENCE_TIME = 60
    driver = webdriver.Safari()
    driver.get(page_url1)
    all_items = driver.find_elements(by=By.CSS_SELECTOR, value='.paper_paper__1PY90.paper_square__lJX8a.card_card__lQWDv.card_noPadding__D8PcU.styles_cardWrapper__LcCPA.styles_show__HUXRb.styles_reviewCard__9HxJJ')

    for item in all_items:
        try:
            title=item.find_element(by=By.CSS_SELECTOR, value=".typography_typography__QgicV.typography_h4__E971J.typography_color-black__5LYEn.typography_weight-regular__TWEnf.typography_fontstyle-normal__kHyN3.styles_reviewTitle__04VGJ").text
        except Exception as ex:
            #print('[Exception] name:', ex)
            title = ''

        try: 

            content = item.find_element(by=By.CSS_SELECTOR, value=".typography_typography__QgicV.typography_body__9UBeQ.typography_color-black__5LYEn.typography_weight-regular__TWEnf.typography_fontstyle-normal__kHyN3").text
        except Exception as ex:
            #print('[Exception] price:', ex)
            content = ''

        try: 

            date = item.find_element(by=By.CSS_SELECTOR, value=".typography_typography__QgicV.typography_bodysmall__irytL.typography_color-gray-6__TogX2.typography_weight-regular__TWEnf.typography_fontstyle-normal__kHyN3.styles_datesWrapper__RCEKH").text
        except Exception as ex:
            #print('[Exception] other:', ex)
            date = 'NAN'
        data.append([title,content,date])
    time.sleep(1.5)
    driver.close()

df=pd.DataFrame(data)
df.to_csv('Downloads/tpilot.csv')


# In[48]:


'''
rating=driver.find_elements(by=By.CSS_SELECTOR, value=".star-rating_starRating__4rrcf.star-rating_medium__iN6Ty")
for i in range(len(rating)):
    print(rating[i].text)
'''

