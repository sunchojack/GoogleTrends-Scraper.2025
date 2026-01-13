import math
import os
import re
import tempfile
import time
from datetime import datetime, timedelta
from functools import reduce

import numpy as np
import pandas as pd
from selenium import webdriver

from selenium.common import exceptions
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import datetime as dt

# Name of the download file created by Google Trends
# NAME_DOWNLOAD_FILE = f'multiTimeline_{dt.datetime.now().strftime("%Y-%m-%d_%H-%M")}.csv'
NAME_DOWNLOAD_FILE = f'multiTimeline.csv'
# Max number of consecutive daily observations scraped in one go
MAX_NUMBER_DAILY_OBS = 100
# Max number of simultaneous keywords scraped
MAX_KEYWORDS = 5


def scale_trend(data_daily, data_all, frequency):
    """
    Function that rescales the data at daily frequency using data at lower frequency
    Args:
        data_daily: pandas.DataFrame of daily trend data
        data_all: pandas.DataFrame of the trend data over the entire sample at a lower frequency
        frequency: frequency of 'data_all'

    Returns: pandas.DataFrame

    """
    # Ensure that all data is numeric:
    data_daily = data_daily.replace('<1', 0.5).astype('float')
    data_all = data_all.replace('<1', 0.5).astype('float')
    # Compute factor by distinguishing between the frequency of the data at the lower frequency (weekly or monthly)
    if frequency == "Weekly":
        factor_dates = pd.date_range(next_weekday(data_daily.index[0], 0),
                                     previous_weekday(data_daily.index[-1], 6), freq='D')
        data_daily_weekly = data_daily.loc[factor_dates].resample('W').sum()
        factor = data_all.loc[data_daily_weekly.index] / data_daily_weekly
    elif frequency == "Monthly":
        factor_dates = pd.date_range(ceil_start_month(data_daily.index[0]),
                                     floor_end_month(data_daily.index[-1]), freq='D')
        data_daily_monthly = data_daily.loc[factor_dates].resample(
            'M', label="left", loffset=timedelta(1)).sum()
        factor = data_all.loc[data_daily_monthly.index] / data_daily_monthly
    # Transform the factor from a pandas.DataFrame to a flat numpy.array
    factor = np.array(factor).flatten()
    # Remove all factor entries for which either of the series is zero
    factor = factor[factor != 0]
    factor = factor[np.isfinite(factor)]
    # Rescale and return the daily trends
    return data_daily * np.median(factor)


def concat_data(data_list, data_all, keywords, frequency):
    """
    Function that merge the DataFrames obtained from different scrapes of GoogleTrends. The DataFrames are collected in
    a list (ordered chronologically), with the last and first observation of two consecutive DataFrames being from the
    same day.
    Args:
        data_list: list of pandas DataFrame objects
        data_all: pandas.DataFrame of trend data over the entire period for the same keywords
        keywords: list of the keywords for which GoogleTrends has been scraped
        frequency:

    Returns: pandas DataFrame with a 'Date' column and a column for each keyword in 'keywords'

    """
    # Remove trend subperiods for which no data has been found
    data_list = [data for data in data_list if data.shape[0] != 0]
    # Rescale the daily trends based on the data at the lower frequency
    data_list = [scale_trend(x, data_all, frequency) for x in data_list]
    # Combine the single trends that were scraped:
    # data = reduce((lambda x, y: x.combine_first(y)), data_list)

    if data_list:  # Check if the list is not empty
        data = reduce((lambda x, y: x.combine_first(y)), data_list)
    else:
        print("Data list is empty, cannot combine trends")
        data = pd.DataFrame()
        # Find the maximal value across keywords and time
    max_value = data.max().max()
    # Rescale the trends by the maximal value, i.e. such that the largest value across keywords and time is 100
    data = 100 * data / max_value
    if data.empty:
        print("Data is empty, skipping column renaming.")
    else:
        data.columns = keywords

    return data


def merge_two_keyword_chunks(data_first, data_second):
    """
    Given two data frame objects with same index and one overlapping column (keyword), a scaling factor
    is determined and the data frames are merged, where the second data frame is rescaled to match the
    scale of the first data set.

    Args:
        data_first: pandas.DataFrame obtained from the csv-file created by GoogleTrends
        data_second: pandas.DataFrame obtained from the csv-file created by GoogleTrends

    Returns: pandas.DataFrame of the merge and re-scaled input pandas.DataFrame

    """
    common_keyword = data_first.columns.intersection(data_second.columns)[0]
    scaling_factor = np.nanmedian(
        data_first[common_keyword] / data_second[common_keyword])
    data_second = data_second.apply(lambda x: x * scaling_factor)
    data = pd.merge(data_first, data_second.drop(
        common_keyword, axis=1), left_index=True, right_index=True)
    return data


def merge_keyword_chunks(data_list):
    """
    Merge a list of pandas.DataFrame objects with the same index and one overlapping column by appropriately
    re-scaling.

    Args:
        data_list: list of pandas.DataFrame objects to be merged

    Returns: pandas.DataFrame objects of the merged data sets contained in the input list

    """
    # Iteratively merge the DataFrame objects in the list of data
    data = reduce((lambda x, y: merge_two_keyword_chunks(x, y)), data_list)
    # Find the maximal value across keywords and time
    max_value = data.max().max()
    # Rescale the trends by the maximal value, i.e. such that the largest value across keywords and time is 100
    data = 100 * data / max_value
    return data


def adjust_date_format(date, format_in, format_out):
    """
    Converts a date-string from one format to another
    Args:
        date: datetime as a string
        format_in: format of 'date'
        format_out: format to which 'date' should be converted

    Returns: date as a string in the new format

    """
    return datetime.strptime(date, format_in).strftime(format_out)


def get_chunks(list_object, chunk_size):
    """
    Generator that divides a list into chunks. If the list is divided in two or more chunks, two consecutive chunks
    have one common element.

    Args:
        list_object: iterable
        chunk_size: size of each chunk as an integer

    Returns: iterable list in chunks with one overlapping element

    """
    size = len(list_object)
    if size <= chunk_size:
        yield list_object
    else:
        chunks_nb = math.ceil(size / chunk_size)
        iter_ints = range(0, chunks_nb)
        for i in iter_ints:
            j = i * chunk_size
            if i + 1 < chunks_nb:
                k = j + chunk_size
                yield list_object[max(j - 1, 0):k]
            else:
                yield list_object[max(j - 1, 0):]


def previous_weekday(date, weekday):
    """
    Function that rounds a date down to the previous date of the desired weekday
    Args:
        date: a datetime.date or datetime.datetime object
        weekday: the desired week day as integer (Monday = 0, ..., Sunday = 6)

    Returns: datetime.date or datetime.datetime object

    """
    delta = date.weekday() - weekday
    if delta < 0:
        delta += 7
    return date + timedelta(days=-int(delta))


def next_weekday(date, weekday):
    """
    Function that rounds a date up to the next date of the desired weekday
    Args:
        date: a datetime.date or datetime.datetime object
        weekday: the desired week day as integer (Monday = 0, ..., Sunday = 6)

    Returns: datetime.date or datetime.datetime object

    """
    delta = weekday - date.weekday()
    if delta < 0:
        delta += 7
    return date + timedelta(days=int(delta))


def ceil_start_month(date):
    """
    Ceil date to the start date of the next month
    Args:
        date: datetime.datetime object

    Returns: datetime.datetime object

    """
    if date.month == 12:
        date = datetime(date.year + 1, 1, 1)
    else:
        date = datetime(date.year, date.month + 1, 1)
    return date


def floor_end_month(date):
    """
    Floor date to the end of the previous month
    Args:
        date: datetime.datetime object

    Returns: datetime.datetime object

    """
    return datetime(date.year, date.month, 1) + timedelta(days=-1)


class GoogleTrendsScraper:
    def __init__(self, sleep=1,
                 path_driver=None,
                 headless=True, date_format='%Y-%m-%d',
                 output_folder="out"):
        """
        Constructor of the Google-Scraper-Class
        Args:
            sleep: integer number of seconds where the scraping waits (avoids getting blocked and gives the code time
                    to download the data
            path_driver: path as string to where the chrome driver is located
            headless: boolean indicating whether the browser should be displayed or not
            date_format: format in which the date-strings are passed to the object
            n_overlap: integer number of overlapping observations used to rescale multiple sub-period trends
        """
        # Current directory
        self.dir = os.getcwd()
        # Define download folder for browser:
        folder_path = output_folder if output_folder else os.path.join(self.dir, self.output_folder)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        else:
            print(f"Using existing folder: {folder_path}")

        self.download_path = folder_path
        self.download_path = os.path.abspath(self.download_path)  # Convert to absolute path
        print(f"Download path (absolute): {self.download_path}")

        # Define the path to the downloaded csv-files (this is where the trends are saved)
        self.filename = os.path.join(self.download_path, NAME_DOWNLOAD_FILE)
        # Whether the browser should be opened in headless mode
        self.headless = headless
        # Path to the driver of Google Chrome
        self.path_driver = path_driver
        # self.binary_path = binary_path
        # Initialize the browser variable
        self.browser = None
        # Sleep time used during the scraping procedure
        self.sleep = sleep
        # Maximal number of consecutive days scraped
        self.max_days = MAX_NUMBER_DAILY_OBS
        # Format of the date-strings
        self.date_format = date_format
        # Format of dates used by google
        self._google_date_format = '%Y-%m-%d'
        # Lunch the browser
        self.start_browser()

    def start_browser(self):
        if self.browser is not None:
            print('Browser already running')
            return

        if not os.path.exists(self.download_path):
            print(f"Download path does not exist: {self.download_path}")
            exit(1)
        else:
            print(f"Using download path: {self.download_path}")

        chrome_options = webdriver.ChromeOptions()

        print("Current Chrome options:", chrome_options.arguments)
        print("Chrome options:", chrome_options.to_capabilities())

        if self.headless == False:
            print('>>> Non-headless will not work, switching to headless! <<<')
            self.headless = True

        if self.headless == True:  # safeguard
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('window-size=1920x1080')

        # Set preferences for language and download settings in a single block
        chrome_options.add_experimental_option('prefs', {
            'intl.accept_languages': 'en,en_US',
            'download.default_directory': str(self.download_path),  # Set the download path correctly
            'download.prompt_for_download': False,  # Disable the download prompt
            'download.directory_upgrade': True  # Allow download directory changes
        })

        if self.path_driver is None:
            print('ChromeDriver path not found, exiting...')
            exit(1)

        service = Service(self.path_driver)
        self.browser = webdriver.Chrome(service=service, options=chrome_options)

        print("Browser started successfully")
        print("Temp Chrome options:", chrome_options.arguments)
        print("Chrome options:", chrome_options.to_capabilities())

    def quit_browser(self):
        """
        Method that closes the existing browser

        """
        if self.browser is not None:
            self.browser.quit()
            self.browser = None

    def get_trends(self, keywords, start, end, region=None, category=None):
        """
        Function that starts the scraping procedure and returns the Google Trend data.
        Args:
            keywords: list or string of keyword(s)
            region: string indicating the region for which the trends are computed, default is None (Worldwide trends)
            start: start date as a string
            end: end date as a string
            category: integer indicating the category (e.g. 7 is the category "Finance")

        Returns: pandas DataFrame with a 'Date' column and a column containing the trend for each keyword in 'keywords'

        """
        # If only a single keyword is given, i.e. as a string and not as a list, put the single string into a list
        if not isinstance(keywords, list):
            keywords = [keywords]
        # Convert the date strings to Google's format:
        start = adjust_date_format(
            start, self.date_format, self._google_date_format)
        end = adjust_date_format(
            end, self.date_format, self._google_date_format)
        # Create datetime objects from the date-strings:
        start_datetime = datetime.strptime(start, self._google_date_format)
        end_datetime = datetime.strptime(end, self._google_date_format)
        data_keywords_list = []
        for keywords_i in get_chunks(keywords, MAX_KEYWORDS):
            # Get the trends over the entire sample:
            url_all_i = self.create_url(keywords_i,
                                        previous_weekday(start_datetime, 0), next_weekday(
                    end_datetime, 6),
                                        region, category)
            data_all_i, frequency_i = self.get_data(url_all_i)
            # If the data for the entire sample is already at the daily frequency we are done. Otherwise we need to
            # get the trends for sub-periods
            if frequency_i == 'Daily':
                data_i = data_all_i
            else:
                # Iterate over the URLs of the sub-periods and retrieve the Google Trend data for each
                data_time_list = []
                for url in self.create_urls_subperiods(keywords_i, start_datetime, end_datetime, region, category):
                    data_time_list.append(self.get_data(url)[0])
                # Concatenate the so obtained set of DataFrames to a single DataFrame
                data_i = concat_data(
                    data_time_list, data_all_i, keywords_i, frequency_i)
            # Add the data for the current list of keywords to a list
            data_keywords_list.append(data_i)
        # Merge the multiple keyword chunks
        data = merge_keyword_chunks(data_keywords_list)
        # Cut data to return only the desired period:
        data = data.loc[data.index.isin(pd.date_range(
            start_datetime, end_datetime, freq='D'))]
        return data

    def create_urls_subperiods(self, keywords, start, end, region=None, category=None):
        """
        Generator that creates an iterable of URLs that open the Google Trends for a series of sub periods
        Args:
            keywords: list of string keywords
            region: string indicating the region for which the trends are computed, default is None (Worldwide trends)
            start: start date as a string
            end: end date as a string
            category: integer indicating the category (e.g. 7 is the category "Finance")

        Returns: iterable of URLs for Google Trends for sub periods of the entire period defined by 'start' and 'end'

        """
        # Calculate number of sub-periods and their respective length:
        num_subperiods = np.ceil(((end - start).days + 1) / self.max_days)
        num_days_in_subperiod = np.ceil(
            ((end - start).days + 1) / num_subperiods)
        for x in range(int(num_subperiods)):
            start_sub = start + timedelta(days=x * num_days_in_subperiod)
            end_sub = start + \
                      timedelta(days=(x + 1) * num_days_in_subperiod - 1)
            if end_sub > end:
                end_sub = end
            if start_sub < end:
                yield self.create_url(keywords, start_sub, end_sub, region=region, category=category)

    def create_url(self, keywords, start, end, region=None, category=None):
        """
        Creates a URL for Google Trends
        Args:
            keywords: list of string keywords
            start: start date as a string
            end: end date as a string
            region: string indicating the region for which the trends are computed, default is None (Worldwide trends)
            category: integer indicating the category (e.g. 7 is the category "Finance")

        Returns: string of the URL for Google Trends of the given keywords over the time period from 'start' to 'end'

        """
        # Replace the '+' symbol in a keyword with '%2B'
        keywords = [re.sub(r'[+]', '%2B', kw) for kw in keywords]
        # Replace white spaces in a keyword with '%20'
        keywords = [re.sub(r'\s', '%20', kw) for kw in keywords]
        # Define main components of the URL
        base = "https://trends.google.com/trends/explore"
        geo = f"geo={region}&" if region is not None else ""
        query = f"q={','.join(keywords)}"
        cat = f"cat={category}&" if category is not None else ""
        # Format the datetime objects to strings in the format used by google
        start_string = datetime.strftime(start, self._google_date_format)
        end_string = datetime.strftime(end, self._google_date_format)
        # Define the date-range component for the URL
        date = f"date={start_string}%20{end_string}"
        # Construct the URL
        url = f"{base}?{cat}{date}&{geo}{query}"
        return url

    def get_data(self, url):
        """
        Method that retrieves for a specific URL the Google Trend data. Note that this is done by downloading a csv-file
        which is then loaded and stored as a pandas.DataFrame object
        Args:
            url: URL for the trend to be scraped as a string

        Returns: a pandas.DataFrame object containing the trends for the given URL

        """
        # Initialize the button that needs to be pressed to get download the data
        button = None
        # While this button is of type 'None' we reload the browser
        while button is None:
            try:
                # Navigate to the URL
                self.go_to_url(url)
                # Sleep the code by the defined time plus a random number of seconds between 0s and 2s. This should
                # reduce the likelihood that Google detects us as a scraper
                time.sleep(self.sleep * (1 + np.random.rand()))
                # Try to find the button and click it
                line_chart = self.browser.find_element(By.CSS_SELECTOR,
                                                       "widget[type='fe_line_chart']")
                button = line_chart.find_element(By.CSS_SELECTOR,
                                                 '.widget-actions-item.export')
                button.click()
            except exceptions.NoSuchElementException:
                # If the button cannot be found, try again (load page, ...)
                pass
        # After downloading, wait again to allow the file to be downloaded
        time.sleep(self.sleep * (1 + np.random.rand()))
        # Load the data from the csv-file as pandas.DataFrame object
        data = pd.read_csv(self.filename, skiprows=1)
        # Set date as index:
        if 'Day' in data.columns:
            data.Day = pd.to_datetime(data.Day)
            data = data.set_index("Day")
            frequency = 'Daily'
        elif 'Week' in data.columns:
            data.Week = pd.to_datetime(data.Week)
            data = data.set_index("Week")
            frequency = 'Weekly'
        else:
            data.Month = pd.to_datetime(data.Month)
            data = data.set_index("Month")
            frequency = 'Monthly'
        # Sleep again
        time.sleep(self.sleep * (1 + np.random.rand()))

        # Delete the intermediary file
        while os.path.exists(self.filename):
            try:
                os.remove(self.filename)
            except:
                pass

        return data, frequency

    def go_to_url(self, url):
        """
        Method that navigates in the browser to the given URL
        Args:
            url: URL to which we want to navigate as a string

        """
        if self.browser is not None:
            self.browser.get(url)
        else:
            print('Browser is not running')

    def __del__(self):
        """
        When deleting an instance of this class, delete the temporary file folder and close the browser

        """
        # self.download_path.cleanup()
        self.quit_browser()
