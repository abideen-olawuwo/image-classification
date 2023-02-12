import urllib.request
import panda as pd

url = "https://en.wikipedia.org/wiki/List_of_publicly_listed_software_companies_of_India"

with urllib.request.urlopen(url) as i:
    html = i.read()

data = pd.read_html(html)[0]
print(data.head())