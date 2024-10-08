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
