import os
import datetime as dt

from src.GoogleTrendsScraper_playwright import GoogleTrendsScraper


output_folder_in = 'out'

gts = GoogleTrendsScraper(sleep=5,
                          # path_driver='ASSETS/BRAVE_Chromedriver/chromedriver-win64/chromedriver-win64/chromedriver.exe',
                          output_folder=output_folder_in)

# gts = GoogleTrendsScraper(sleep=5, headless=False)

startdate, enddate = '2024-01-01', '2024-01-02'
keyword_in = 'chatgpt'
region_in = 'US'

data = gts.get_trends(keyword_in, start=startdate, end=enddate, region=region_in)

creation_date = dt.datetime.now().strftime("%m_%d_%Y_%H_%M")

try:
    data.to_csv(f'{output_folder_in}/{keyword_in}_{region_in}_{creation_date}.csv', index=False)
except Exception as datasave_error:
    print(datasave_error)

del gts


# add logging, add progress bar
# add iter over iso2