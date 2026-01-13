import math
import os
import re
import time
from datetime import datetime, timedelta
from functools import reduce

import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright
# dont forget:
# pip install playwright
# and:
# playwright install chromium

import datetime as dt

NAME_DOWNLOAD_FILE = 'multiTimeline.csv'
MAX_NUMBER_DAILY_OBS = 100
MAX_KEYWORDS = 5


class PlaywrightBrowser:
    """wrapper for playwright browser"""

    def __init__(self, headless=True, download_path=None):
        self.headless = headless
        self.download_path = download_path
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    def start(self):
        """start browser"""
        if self.page is not None:
            return

        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox'
            ]
        )

        context_options = {
            'locale': 'en-US',
            'timezone_id': 'America/New_York',
            'accept_downloads': True,
            'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'viewport': {'width': 1920, 'height': 1080}
        }

        self.context = self.browser.new_context(**context_options)

        # hide automation signals
        self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        """)

        self.page = self.context.new_page()

    def goto(self, url):
        """navigate to url"""
        self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
        self.page.wait_for_load_state('networkidle', timeout=30000)

    def click_download_button(self):
        """click export and download csv"""
        try:
            # wait longer and check if page loaded
            self.page.wait_for_selector("widget[type='fe_line_chart']", timeout=30000, state='visible')

            # extra wait for dynamic content
            self.page.wait_for_timeout(2000)

            export_button = self.page.locator("widget[type='fe_line_chart'] .widget-actions-item.export")
            export_button.wait_for(state='visible', timeout=10000)

            with self.page.expect_download(timeout=30000) as download_info:
                export_button.click()

            download = download_info.value
            filepath = os.path.join(self.download_path, NAME_DOWNLOAD_FILE)
            download.save_as(filepath)
            return filepath

        except Exception as e:
            # save screenshot for debugging
            screenshot_path = os.path.join(self.download_path, 'error_screenshot.png')
            self.page.screenshot(path=screenshot_path)
            print(f'screenshot saved to {screenshot_path}')
            raise Exception(f'download failed: {e}')

    def quit(self):
        """close browser"""
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        self.page = None
        self.context = None
        self.browser = None
        self.playwright = None


def scale_trend(data_daily, data_all, frequency):
    """rescale daily data using lower frequency data"""
    data_daily = data_daily.replace('<1', 0.5).astype('float')
    data_all = data_all.replace('<1', 0.5).astype('float')

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

    factor = np.array(factor).flatten()
    factor = factor[factor != 0]
    factor = factor[np.isfinite(factor)]

    return data_daily * np.median(factor)


def concat_data(data_list, data_all, keywords, frequency):
    """merge dataframes from different scrapes"""
    data_list = [data for data in data_list if data.shape[0] != 0]
    data_list = [scale_trend(x, data_all, frequency) for x in data_list]

    if data_list:
        data = reduce((lambda x, y: x.combine_first(y)), data_list)
    else:
        return pd.DataFrame()

    max_value = data.max().max()
    data = 100 * data / max_value

    if not data.empty:
        data.columns = keywords

    return data


def merge_two_keyword_chunks(data_first, data_second):
    """merge two dataframes with overlapping keyword"""
    common_keyword = data_first.columns.intersection(data_second.columns)[0]
    scaling_factor = np.nanmedian(data_first[common_keyword] / data_second[common_keyword])
    data_second = data_second.apply(lambda x: x * scaling_factor)
    data = pd.merge(data_first, data_second.drop(common_keyword, axis=1),
                    left_index=True, right_index=True)
    return data


def merge_keyword_chunks(data_list):
    """merge list of dataframes with rescaling"""
    data = reduce((lambda x, y: merge_two_keyword_chunks(x, y)), data_list)
    max_value = data.max().max()
    data = 100 * data / max_value
    return data


def adjust_date_format(date, format_in, format_out):
    """convert date string format"""
    return datetime.strptime(date, format_in).strftime(format_out)


def get_chunks(list_object, chunk_size):
    """divide list into overlapping chunks"""
    size = len(list_object)
    if size <= chunk_size:
        yield list_object
    else:
        chunks_nb = math.ceil(size / chunk_size)
        for i in range(chunks_nb):
            j = i * chunk_size
            if i + 1 < chunks_nb:
                k = j + chunk_size
                yield list_object[max(j - 1, 0):k]
            else:
                yield list_object[max(j - 1, 0):]


def previous_weekday(date, weekday):
    """round date down to previous weekday"""
    delta = date.weekday() - weekday
    if delta < 0:
        delta += 7
    return date + timedelta(days=-int(delta))


def next_weekday(date, weekday):
    """round date up to next weekday"""
    delta = weekday - date.weekday()
    if delta < 0:
        delta += 7
    return date + timedelta(days=int(delta))


def ceil_start_month(date):
    """ceil to start of next month"""
    if date.month == 12:
        date = datetime(date.year + 1, 1, 1)
    else:
        date = datetime(date.year, date.month + 1, 1)
    return date


def floor_end_month(date):
    """floor to end of previous month"""
    return datetime(date.year, date.month, 1) + timedelta(days=-1)


class GoogleTrendsScraper:
    def __init__(self, sleep=5, headless=False, date_format='%Y-%m-%d', output_folder="out"):
        """
        Args:
            sleep: seconds between requests (min 5 to avoid rate limits)
            headless: run without ui (set False to debug)
            date_format: format for date strings
            output_folder: csv save location
        """
        self.dir = os.getcwd()
        folder_path = output_folder if output_folder else os.path.join(self.dir, "out")
        os.makedirs(folder_path, exist_ok=True)

        self.download_path = os.path.abspath(folder_path)
        self.filename = os.path.join(self.download_path, NAME_DOWNLOAD_FILE)
        self.headless = headless
        self.browser = None
        self.sleep = max(sleep, 5)  # enforce minimum 5s
        self.max_days = MAX_NUMBER_DAILY_OBS
        self.date_format = date_format
        self._google_date_format = '%Y-%m-%d'

        self.start_browser()

    def start_browser(self):
        """initialize playwright browser"""
        if self.browser is not None:
            return

        self.browser = PlaywrightBrowser(headless=self.headless, download_path=self.download_path)
        self.browser.start()

    def quit_browser(self):
        """close browser"""
        if self.browser is not None:
            self.browser.quit()
            self.browser = None

    def get_trends(self, keywords, start, end, region=None, category=None):
        """
        scrape google trends
        Args:
            keywords: string or list of keywords
            start: start date string
            end: end date string
            region: region code (e.g. 'US')
            category: category id (e.g. 7 for finance)
        Returns:
            pandas dataframe with trends
        """
        if not isinstance(keywords, list):
            keywords = [keywords]

        start = adjust_date_format(start, self.date_format, self._google_date_format)
        end = adjust_date_format(end, self.date_format, self._google_date_format)
        start_datetime = datetime.strptime(start, self._google_date_format)
        end_datetime = datetime.strptime(end, self._google_date_format)

        data_keywords_list = []
        for keywords_i in get_chunks(keywords, MAX_KEYWORDS):
            url_all_i = self.create_url(keywords_i, previous_weekday(start_datetime, 0),
                                        next_weekday(end_datetime, 6), region, category)
            data_all_i, frequency_i = self.get_data(url_all_i)

            if frequency_i == 'Daily':
                data_i = data_all_i
            else:
                data_time_list = []
                for url in self.create_urls_subperiods(keywords_i, start_datetime, end_datetime, region, category):
                    data_time_list.append(self.get_data(url)[0])
                data_i = concat_data(data_time_list, data_all_i, keywords_i, frequency_i)

            data_keywords_list.append(data_i)

        data = merge_keyword_chunks(data_keywords_list)
        data = data.loc[data.index.isin(pd.date_range(start_datetime, end_datetime, freq='D'))]
        return data

    def create_urls_subperiods(self, keywords, start, end, region=None, category=None):
        """generate urls for subperiods"""
        num_subperiods = np.ceil(((end - start).days + 1) / self.max_days)
        num_days_in_subperiod = np.ceil(((end - start).days + 1) / num_subperiods)

        for x in range(int(num_subperiods)):
            start_sub = start + timedelta(days=x * num_days_in_subperiod)
            end_sub = start + timedelta(days=(x + 1) * num_days_in_subperiod - 1)
            if end_sub > end:
                end_sub = end
            if start_sub < end:
                yield self.create_url(keywords, start_sub, end_sub, region=region, category=category)

    def create_url(self, keywords, start, end, region=None, category=None):
        """build google trends url"""
        keywords = [re.sub(r'[+]', '%2B', kw) for kw in keywords]
        keywords = [re.sub(r'\s', '%20', kw) for kw in keywords]

        base = "https://trends.google.com/trends/explore"
        geo = f"geo={region}&" if region is not None else ""
        query = f"q={','.join(keywords)}"
        cat = f"cat={category}&" if category is not None else ""
        start_string = datetime.strftime(start, self._google_date_format)
        end_string = datetime.strftime(end, self._google_date_format)
        date = f"date={start_string}%20{end_string}"

        return f"{base}?{cat}{date}&{geo}{query}"

    def get_data(self, url):
        """scrape data from url"""
        self.go_to_url(url)
        time.sleep(self.sleep * (1 + np.random.rand()))

        try:
            self.browser.click_download_button()
        except Exception as e:
            print(f'download failed for url: {url}')
            print(f'error: {e}')
            return pd.DataFrame(), 'Daily'

        time.sleep(self.sleep * (1 + np.random.rand()))

        if not os.path.exists(self.filename):
            print(f'file not found: {self.filename}')
            return pd.DataFrame(), 'Daily'

        data = pd.read_csv(self.filename, skiprows=1)

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

        time.sleep(self.sleep * (1 + np.random.rand()))

        if os.path.exists(self.filename):
            try:
                os.remove(self.filename)
            except:
                pass

        return data, frequency

    def go_to_url(self, url):
        """navigate to url"""
        if self.browser is not None:
            self.browser.goto(url)

    def __del__(self):
        """cleanup"""
        self.quit_browser()