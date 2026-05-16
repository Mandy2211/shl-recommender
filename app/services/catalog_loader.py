import json


def load_catalog():
    with open("app/data/shl_catalog.json", "r", encoding="utf-8") as f:
        return json.load(f)