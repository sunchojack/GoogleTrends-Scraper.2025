# Google Trend scraper
## Introduction
A wraper for the [GoogleTrends](https://trends.google.com/trends/?geo=US) scraper by 
[@dballinari](https://github.com/dballinari/GoogleTrends-Scraper):

"For a given keyword, time range and region, the scraping tool collects the GoogleTrend data. This
data is scaled ranging from 0 to 100, where the day with the highest number of searches is set to 
100\. Note that Google estimates the volume of searches based on a random sample of search-queries".
  
### On Windows:
Should be safe to use the chromedriver from the Brave Browser that is included with the repository.
 
In the `example_script.py`:

1. Uncomment the path to the chromedriver
2. replace 
```python
from src.GoogleTrendsScraper_playwright import GoogleTrendsScraper
```
with
```python
 from src.GoogleTrendsScraper_selenium import GoogleTrendsScraper
```

### On Mac:
1. Keep the playwright version
2. Be patient about the rate limits. Google will sometimes require a 30 min
cooldown period. Alternatively, try downloading a chromedriver version like in the Windows approach.
The script was initially made and used with Windows and a preselected Brave Browser's chromedriver.exe.