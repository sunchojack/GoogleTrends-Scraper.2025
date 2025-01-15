import os

from src.GoogleTrendsScraper import GoogleTrendsScraper

# gts = GoogleTrendsScraper(sleep=5, path_driver=os.environ['CHROMEDRIVER'], headless=True)
gts = GoogleTrendsScraper(sleep=5, headless=False,
                          binary_path='ASSETS/BraveSoftware/Brave-Browser/Application/brave-browser',
                          path_driver='ASSETS/BRAVE_Chromedriver/chromedriver-win64/chromedriver-win64/chromedriver.exe')

# gts = GoogleTrendsScraper(sleep=5, headless=False)

data = gts.get_trends('foo', '2018-07-02', '2019-04-02', 'US')

print(data)

del gts
