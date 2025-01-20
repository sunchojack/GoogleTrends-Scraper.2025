import os
import datetime as dt
import pycountry

from src.GoogleTrendsScraper import GoogleTrendsScraper


class ISO2CodeGetter:
    def __init__(self):
        self.iso2_list = []

    def get_iso2_codes(self):
        """Populate and return the list of ISO2 country codes."""
        self.iso2_list = [country.alpha_2 for country in pycountry.countries]
        return self.iso2_list


output_folder_in = 'out'

# gts = GoogleTrendsScraper(sleep=5, headless=False)

startdate, enddate = '2024-01-01', '2024-01-15'
keywords = ['chatgpt']
default_region_in = 'US'

iso_getter = ISO2CodeGetter()
iso2_codes = iso_getter.get_iso2_codes()

for keyword_in in keywords:
    for region_in in iso2_codes:
        try:
            gts = GoogleTrendsScraper(sleep=5,
                                      path_driver='ASSETS/BRAVE_Chromedriver/chromedriver-win64/chromedriver-win64/chromedriver.exe',
                                      output_folder=output_folder_in)

            # data = gts.get_trends(keyword_in, start=startdate, end=enddate, region=region_in)
            data = gts.get_trends(keyword_in, start=startdate, end=enddate, region='US')

            creation_date = dt.datetime.now().strftime("%m_%d_%Y_%H_%M")

            try:
                data.to_csv(f'{output_folder_in}/{keyword_in}_{region_in}_{creation_date}.csv', index=False)
            except Exception as datasave_error:
                print(datasave_error)

            del gts

        except Exception as loop_error:
            print(loop_error)
            del gts
            pass


# add logging, add progress bar