"""
1. Generate a list of 150 cities, 100 train, reserve 50 for future use.
2. Download wikipedia pages for all 250 cities
3. Chunk all pages
4. Generate a trivia question, answer and the passage
5. Split into train, dev and test set
"""

import csv
import json

import dspy  # type: ignore
import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

lm = dspy.LM(model="openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)


assert load_dotenv("../../.env")


# 1. Generate a list of 150 cities, 100 train, reserve 50 for future use.
class City(BaseModel):
    city: str = Field(description="Name of a city")
    contitent: str = Field(description="Name of the continent the city is in")


class CityList(dspy.Signature):
    """Given user's question, answer with JSON readable python list"""

    question: str = dspy.InputField(desc="User's question")
    answer: list[City] = dspy.OutputField(
        desc="list of cities and the continents they are in"
    )


predict = dspy.TypedChainOfThought(CityList)

regions = [
    "North America",
    "Europe",
    "South America",
    "Africa",
    "India",
    "China",
    "Pakistan",
    "Bangladesh",
    "Philippines",
    "Malaysia",
    "Japan",
    "Indonesia",
    "Asia",
]

cities = []
for region in regions:
    prediction = predict(
        question=f"Give me a list of ALL the cities in {region} with at least 1 million population"
    )
    cities.append(prediction.answer)

cities = [city for city_list in cities for city in city_list]

seen_cities = []
unique_cities = []
for city in cities:
    if city.city not in seen_cities:
        seen_cities.append(city.city)
        unique_cities.append(city)


with open("data/cities.csv", mode="w") as file:
    writer = csv.writer(file, lineterminator="\n")
    writer.writerow(["city", "continent"])
    for city in unique_cities:
        writer.writerow([city.city, city.contitent])

# 2. Download wikipedia pages for all 150 cities


def download_wikipedia_page(title: str) -> str:
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": True,
            "titles": title,
        }
        response = requests.get(url, params=params)
        data = response.json()
        page = next(iter(data["query"]["pages"].values()))
        return page.get("extract", "")
    except Exception as e:
        print(f"Error downloading page {title}: {e}")
        return ""


cities = pd.read_csv("data/cities.csv")
pages = {}
for city in cities.itertuples():
    city_name = city[1]
    city_data = download_wikipedia_page(city_name)
    print(f"{city_name} {len(city_data)}")
    pages[city_name] = city_data

cities_without_pages = [k for k, v in pages.items() if len(v) <= 10000]

# Manually fixing some cities that didnt get data
pages["San Jose"] = download_wikipedia_page("San Jose, California")
pages["Austin"] = download_wikipedia_page("Austin, Texas")
pages["Jacksonville"] = download_wikipedia_page("Jacksonville, Florida")
pages["Fort Worth"] = download_wikipedia_page("Fort Worth, Texas")
pages["Salvador"] = download_wikipedia_page("Salvador, Bahia")
pages["Kano"] = download_wikipedia_page("Kano (city)")
pages["George Town"] = download_wikipedia_page("George Town, Penang")
pages["Kawasaki"] = download_wikipedia_page("Kawasaki, Kanagawa")

cities_without_pages = [k for k, v in pages.items() if len(v) <= 10000]

try:
    with open("data/pages.json", "w") as json_file:
        json.dump(pages, json_file)
except Exception as e:
    print(f"Error writing to JSON file: {e}")


# create text files for each city
def read_json_to_dict(filepath: str) -> dict[str, str]:
    try:
        with open(filepath, "r") as file:
            data = json.load(file)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading JSON file: {e}")
        return {}


cities = read_json_to_dict("data/pages.json")

for k, v in cities.items():
    filepath = f"data/text/{k.replace(" ", "_")}.txt"
    try:
        with open(filepath, "w") as file:
            file.write(v)
    except IOError as e:
        print(f"Error writing to file: {e}")


# Shorten the pages to first
SHORTEN_BY = 0.5
# shorten each city pages by top 50% of the paragraphs and save it to short text location.
short_city = {}
for city in cities.keys():
    city_data = cities[city].split("\n\n")
    len_paragraphs = int(len(city_data) * SHORTEN_BY)
    shortened = "\n\n".join(city_data[:len_paragraphs])
    short_city[city] = shortened
