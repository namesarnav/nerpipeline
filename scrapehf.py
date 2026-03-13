import requests
from bs4 import BeautifulSoup

# Fetch the content from the URL

timex = []
eventx = []

for page in range(2):
    print(page)
    url = f'https://huggingface.co/mdg-nlp/datasets?sort=alphabetical&p={page}'

    response = requests.get(url)

    if response.status_code == 200:

        soup = BeautifulSoup(response.content, 'html.parser')

    print(soup.title.string)


    for link in soup.find_all('a'):
        href = link.get('href', '')
        if 'event' in href:
            print(f"Event: {href}")
            eventx.append(href)
        if 'timex' in href or 'time' in href:
            print(f"Timex/Time: {href}")
            timex.append(href)

eventx = list(set(eventx))
timex = list(set(timex))

print("Unique EventX Datasets:")
for dataset in eventx:
    print(f'"{dataset[10:]}"')


print(f"Total unique EventX datasets: {len(eventx)}\n")

print("Unique Timex Datasets:")
for dataset in timex:
    print(f'"{dataset[10:]}"')
print(f"Total unique Timex datasets: {len(timex)}\n")