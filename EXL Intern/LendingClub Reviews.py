#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import time
from bs4 import BeautifulSoup as bs
#https://www.lendingclub.com/company/lendingclub-reviews


# In[34]:


driver = webdriver.Safari()
#driver.get("https://www.trustpilot.com/review/lendingclub.com?page=2")
page_url1='https://www.lendingclub.com/company/lendingclub-reviews'
#for page_number in range(1,170):
 #   page_url2=f'https://www.trustpilot.com/review/lendingclub.com?page={page_number}'


# In[38]:


driver.get(page_url1)
#h1 = driver.find_element_by_class_name('reviews__item-title text-static--lg bold')


# In[20]:


#Scroll to the end of the page
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(5)#sleep_between_interactions


# In[1]:


#h1=driver.find_element(by=class_name, value='reviews__item-title text-static--lg bold')


# In[93]:


from selenium import webdriver      
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time


# In[112]:


#browser = webdriver.Firefox()#Chrome('./chromedriver.exe')
df = pd.DataFrame(columns=['Title','Content','Time and tag']) # creates master dataframe 
page_url1='https://www.lendingclub.com/company/lendingclub-reviews'
PATIENCE_TIME = 60
LOAD_MORE_BUTTON_XPATH = '//*[@id="bvTrackingContainer"]/div[4]/a'
title_xpath="reviews__item-title"
content_xpath="reviews__quote"
time_xpath="text-static--sm"
driver = webdriver.Safari()
driver.get(page_url1)
#print(driver.find_elements(by=By.CLASS_NAME, value=title_xpath))
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


# In[99]:





# In[2]:


#driver.find_elements(by=By.CLASS_NAME, value="reviews__item-title")

