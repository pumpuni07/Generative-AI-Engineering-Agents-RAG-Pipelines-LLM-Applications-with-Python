"""
Lab 1: Building AI Agents from Scratch with Python
====================================================
Topic: AI Chatbot with Routing Agents for a Restaurant (The Daily Dish)

This lab demonstrates:
  - Query routing between specialized agents (WeatherAgent, DailyDishAgent)
  - Text preprocessing: lowercasing, cleaning, synonym expansion
  - Agent-based answer retrieval with fallback handling
  - An interactive chat loop

Author notes:
  Based on IBM Skills Network lab material.
  Adapted and extended with full implementations by Jack Pumpuni Frimpong-Manso.

Prerequisites:
  pip install requests python-dotenv

Environment Variables (optional for live weather):
  OPENWEATHERMAP_API_KEY=<your_key>   # Free tier at openweathermap.org
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import re
import os
import json
import random
import requests
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
RESTAURANT_NAME = "The Daily Dish"
RESTAURANT_CITY = "New York"          # Change to your restaurant's city
WEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")  # Optional live API
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE  (The Daily Dish menu + FAQ)
# ─────────────────────────────────────────────────────────────────────────────
MENU = {
    "starters": [
        {"name": "Garden Soup",        "price": 6.99,  "description": "Fresh vegetables in a light broth."},
        {"name": "Crispy Calamari",    "price": 9.99,  "description": "Lightly breaded, served with marinara."},
        {"name": "Caesar Salad",       "price": 8.49,  "description": "Romaine, croutons, parmesan, caesar dressing."},
    ],
    "mains": [
        {"name": "Grilled Salmon",     "price": 18.99, "description": "Atlantic salmon with lemon butter sauce."},
        {"name": "Pasta Primavera",    "price": 13.99, "description": "Penne with seasonal vegetables in tomato sauce."},
        {"name": "Beef Burger",        "price": 14.49, "description": "Half-pound patty, cheddar, lettuce, tomato, pickles."},
        {"name": "Chicken Stir Fry",   "price": 15.99, "description": "Wok-fried chicken breast with mixed vegetables."},
    ],
    "desserts": [
        {"name": "Chocolate Lava Cake","price": 7.49,  "description": "Warm chocolate cake with vanilla ice cream."},
        {"name": "Cheesecake",         "price": 6.99,  "description": "New York-style with fresh berry compote."},
    ],
    "drinks": [
        {"name": "Fresh Lemonade",     "price": 3.49,  "description": "Squeezed daily."},
        {"name": "Iced Tea",           "price": 2.99,  "description": "Sweetened or unsweetened."},
        {"name": "House Wine",         "price": 8.99,  "description": "Red or white, glass."},
    ],
}

FAQS = {
    "hours": "We are open Monday–Friday 11 AM – 10 PM, and Saturday–Sunday 10 AM – 11 PM.",
    "reservation": "You can make a reservation by calling +1-555-0199 or visiting our website.",
    "location": f"We are located at 42 Maple Street, {RESTAURANT_CITY}. Parking available on-site.",
    "wifi": "Yes, we offer free Wi-Fi. Ask your server for the password.",
    "dietary": "We offer vegan, vegetarian, and gluten-free options. Please inform your server of any allergies.",
    "payment": "We accept cash, all major credit cards, and contactless payments.",
    "takeout": "Yes, we offer takeout and delivery via our website or the DoorDash app.",
    "kids": "We have a kids' menu available for children under 12.",
    "parking": "Free parking is available in our lot on the east side of the building.",
    "alcohol": "We are fully licensed. Happy hour runs 4–6 PM on weekdays.",
}


# ─────────────────────────────────────────────────────────────────────────────
# SYNONYM MAP  (expands user language → canonical keywords)
# ─────────────────────────────────────────────────────────────────────────────
SYNONYM_MAP = {
    # weather synonyms
    "temperature": "weather", "temp": "weather", "hot": "weather",
    "cold": "weather", "rain": "weather", "raining": "weather",
    "sunny": "weather", "forecast": "weather", "climate": "weather",
    "outside": "weather", "umbrella": "weather",
    # menu synonyms
    "food": "menu", "eat": "menu", "dish": "menu", "dishes": "menu",
    "meal": "menu", "meals": "menu", "order": "menu", "ordering": "menu",
    "appetizer": "starters", "appetizers": "starters", "starter": "starters",
    "entree": "mains", "entrees": "mains", "main course": "mains",
    "drink": "drinks", "beverage": "beverages", "cocktail": "drinks",
    "sweet": "desserts", "sweets": "desserts", "dessert": "desserts",
    # hours synonyms
    "open": "hours", "close": "hours", "closing": "hours",
    "opening": "hours", "schedule": "hours", "timing": "hours",
    # reservation synonyms
    "book": "reservation", "booking": "reservation", "reserve": "reservation",
    "table": "reservation",
}

# Weather-trigger keywords (used by the router)
WEATHER_KEYWORDS = {
    "weather", "temperature", "temp", "hot", "cold", "rain", "raining",
    "sunny", "forecast", "climate", "outside", "umbrella", "windy", "wind",
    "humidity", "snow", "snowing", "storm",
}


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 1: QueryProcessor
# ─────────────────────────────────────────────────────────────────────────────
class QueryProcessor:
    """
    Preprocesses raw user text:
      1. Lowercase
      2. Strip punctuation / extra whitespace
      3. Expand synonyms to canonical keywords
    """

    def __init__(self, synonym_map: dict):
        self.synonym_map = synonym_map

    def _lowercase(self, text: str) -> str:
        return text.lower()

    def _clean(self, text: str) -> str:
        # Remove non-alphanumeric characters except spaces
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        # Collapse multiple spaces
        return re.sub(r"\s+", " ", text).strip()

    def _expand_synonyms(self, text: str) -> str:
        words = text.split()
        expanded = []
        i = 0
        while i < len(words):
            # Try two-word phrases first
            if i + 1 < len(words):
                bigram = words[i] + " " + words[i + 1]
                if bigram in self.synonym_map:
                    expanded.append(self.synonym_map[bigram])
                    i += 2
                    continue
            # Single word
            word = words[i]
            expanded.append(self.synonym_map.get(word, word))
            i += 1
        return " ".join(expanded)

    def process(self, raw_text: str) -> str:
        text = self._lowercase(raw_text)
        text = self._clean(text)
        text = self._expand_synonyms(text)
        return text


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 2: Router
# ─────────────────────────────────────────────────────────────────────────────
def route_query(user_question: str) -> str:
    """
    Determines which agent should handle the query.

    Returns:
        "weather"    → WeatherAgent
        "restaurant" → DailyDishAgent
    """
    lower_q = user_question.lower()
    tokens = set(re.findall(r"\w+", lower_q))
    if tokens & WEATHER_KEYWORDS:
        return "weather"
    return "restaurant"


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 3: WeatherAgent
# ─────────────────────────────────────────────────────────────────────────────
class WeatherAgent:
    """
    Fetches current weather for the restaurant's city.
    Falls back to a simulated response if no API key is set.
    """

    def __init__(self, api_key: str, api_url: str):
        self.api_key = api_key
        self.api_url = api_url

    def _fetch_live(self, city: str) -> Optional[dict]:
        """Calls OpenWeatherMap API. Returns parsed JSON or None on error."""
        try:
            params = {"q": city, "appid": self.api_key, "units": "metric"}
            resp = requests.get(self.api_url, params=params, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    def _simulate(self, city: str) -> str:
        """Returns a simulated weather response when API key is absent."""
        conditions = ["sunny", "partly cloudy", "overcast", "light rain", "clear skies"]
        temp = random.randint(10, 30)
        condition = random.choice(conditions)
        return (
            f"🌤️ Current weather in {city}: {temp}°C, {condition}. "
            f"(Simulated — set OPENWEATHERMAP_API_KEY for live data.)"
        )

    def answer(self, city: str) -> str:
        if self.api_key:
            data = self._fetch_live(city)
            if data:
                temp = data["main"]["temp"]
                feels = data["main"]["feels_like"]
                desc = data["weather"][0]["description"].capitalize()
                humidity = data["main"]["humidity"]
                return (
                    f"🌤️ Weather in {city} right now: {desc}, {temp:.1f}°C "
                    f"(feels like {feels:.1f}°C), humidity {humidity}%."
                )
        return self._simulate(city)


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 4: DailyDishAgent
# ─────────────────────────────────────────────────────────────────────────────
class DailyDishAgent:
    """
    Answers restaurant-related queries using the knowledge base (MENU + FAQS).
    Uses keyword matching on processed query tokens.
    """

    def __init__(self, menu: dict, faqs: dict, restaurant_name: str):
        self.menu = menu
        self.faqs = faqs
        self.restaurant_name = restaurant_name

    # ── Menu helpers ──────────────────────────────────────────────────────────
    def _format_category(self, category: str) -> str:
        items = self.menu.get(category, [])
        if not items:
            return f"No {category} available."
        lines = [f"📋 **{category.capitalize()}**:"]
        for item in items:
            lines.append(f"  • {item['name']} — ${item['price']:.2f}: {item['description']}")
        return "\n".join(lines)

    def _full_menu(self) -> str:
        sections = [self._format_category(cat) for cat in self.menu]
        return f"🍽️ Here is the full menu for {self.restaurant_name}:\n\n" + "\n\n".join(sections)

    def _search_menu_item(self, query_tokens: set) -> Optional[str]:
        """Search for a specific dish by name tokens."""
        results = []
        for category, items in self.menu.items():
            for item in items:
                item_tokens = set(item["name"].lower().split())
                if query_tokens & item_tokens:
                    results.append(
                        f"  • {item['name']} (${item['price']:.2f}): {item['description']}"
                    )
        if results:
            return "Here's what I found:\n" + "\n".join(results)
        return None

    # ── FAQ helpers ───────────────────────────────────────────────────────────
    def _check_faqs(self, query_tokens: set) -> Optional[str]:
        """Match processed tokens against FAQ keys."""
        for key, answer in self.faqs.items():
            if key in query_tokens:
                return answer
        return None

    # ── Main answer method ────────────────────────────────────────────────────
    def answer(self, processed_query: str) -> Optional[str]:
        tokens = set(processed_query.split())

        # 1. Full menu request
        if "menu" in tokens or "dishes" in tokens:
            return self._full_menu()

        # 2. Category-specific menu
        for category in self.menu:
            if category in tokens or category.rstrip("s") in tokens:
                return self._format_category(category)

        # 3. Specific item search
        item_result = self._search_menu_item(tokens)
        if item_result:
            return item_result

        # 4. FAQ lookup
        faq_result = self._check_faqs(tokens)
        if faq_result:
            return faq_result

        # 5. Greeting detection
        greetings = {"hi", "hello", "hey", "howdy", "greetings"}
        if tokens & greetings:
            return (
                f"👋 Welcome to {self.restaurant_name}! "
                "I can help with our menu, hours, reservations, location, and more. "
                "What would you like to know?"
            )

        return None  # Signals the chatbot to use its fallback message


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 5: Main Chatbot Function
# ─────────────────────────────────────────────────────────────────────────────

# Instantiate components (done once at module load)
query_processor = QueryProcessor(SYNONYM_MAP)
weather_agent   = WeatherAgent(WEATHER_API_KEY, WEATHER_API_URL)
daily_dish_agent = DailyDishAgent(MENU, FAQS, RESTAURANT_NAME)


def chatbot(user_question: str) -> str:
    """
    Main chatbot function.

    Pipeline:
      1. Route the query to the appropriate agent.
      2. Preprocess the query text.
      3. Dispatch to WeatherAgent or DailyDishAgent.
      4. Return the answer, or a fallback message.

    Args:
        user_question (str): Raw text input from the user.

    Returns:
        str: The chatbot's response.
    """
    # Step 1 — Route
    route = route_query(user_question)

    # Step 2 — Preprocess
    processed = query_processor.process(user_question)

    # Step 3 — Dispatch
    if route == "weather":
        return weather_agent.answer(RESTAURANT_CITY)

    # Step 4 — DailyDish agent
    answer = daily_dish_agent.answer(processed)
    if answer:
        return answer

    # Step 5 — Fallback
    return (
        f"I'm not sure about that. "
        f"Please ask a question related to {RESTAURANT_NAME} — "
        "such as our menu, hours, location, or reservations!"
    )


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE CHAT LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_chat():
    """
    Launches the interactive terminal chat loop.
    Type 'exit', 'quit', or 'bye' to end the session.
    """
    print(f"🍽️  Welcome to {RESTAURANT_NAME} Chatbot!")
    print("Ask me about our menu, hours, location, reservations, or today's weather.")
    print("Type 'exit' to end the conversation.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\nChatbot: 👋 Thanks for visiting {RESTAURANT_NAME}! See you soon.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "bye"}:
            print(f"Chatbot: 👋 Thanks for visiting {RESTAURANT_NAME}! See you soon.")
            break

        response = chatbot(user_input)
        print(f"Chatbot: {response}\n")


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS  (run with: python lab1_ai_agents_chatbot.py --test)
# ─────────────────────────────────────────────────────────────────────────────
def run_tests():
    """Basic smoke tests for each component."""
    print("=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)

    # QueryProcessor
    qp = QueryProcessor(SYNONYM_MAP)
    assert "weather" in qp.process("What's the temperature outside?"), "Synonym expansion failed"
    assert "menu" in qp.process("What food do you serve?"), "Synonym expansion failed"
    print("✅ QueryProcessor: PASSED")

    # Router
    assert route_query("Is it raining?") == "weather", "Weather routing failed"
    assert route_query("Do you have pasta?") == "restaurant", "Restaurant routing failed"
    print("✅ Router: PASSED")

    # DailyDishAgent
    agent = DailyDishAgent(MENU, FAQS, RESTAURANT_NAME)
    menu_resp = agent.answer("menu")
    assert RESTAURANT_NAME in menu_resp, "Menu response missing restaurant name"
    hours_resp = agent.answer("hours")
    assert "open" in hours_resp.lower(), "Hours FAQ failed"
    none_resp = agent.answer("quantum physics lecture")
    assert none_resp is None, "Should return None for unrecognized query"
    print("✅ DailyDishAgent: PASSED")

    # Chatbot fallback
    fb = chatbot("What is the speed of light?")
    assert "not sure" in fb.lower(), "Fallback message failed"
    print("✅ Chatbot fallback: PASSED")

    print("\nAll tests passed! ✅")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        run_tests()
    else:
        run_chat()
