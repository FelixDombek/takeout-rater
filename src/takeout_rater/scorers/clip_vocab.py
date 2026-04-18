"""Curated vocabulary of visual concepts for zero-shot CLIP image interrogation.

Each entry is a short phrase that CLIP's text encoder can meaningfully compare
against an image embedding.  The vocabulary is organised by category so it can
be extended or filtered in the future.

Usage::

    from takeout_rater.scorers.clip_vocab import CLIP_VOCAB_TERMS

    # CLIP_VOCAB_TERMS is a flat list[str] ready for tokenisation.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Category buckets (kept separate so callers can filter by category later)
# ---------------------------------------------------------------------------

_SCENES: list[str] = [
    # Natural outdoor
    "beach",
    "ocean",
    "lake",
    "river",
    "waterfall",
    "mountain",
    "cliff",
    "canyon",
    "valley",
    "meadow",
    "field",
    "forest",
    "jungle",
    "desert",
    "snow",
    "glacier",
    "cave",
    "island",
    "volcano",
    "sunrise",
    "sunset",
    "starry night sky",
    "cloudy sky",
    "rainbow",
    "fog",
    "storm",
    # Built / urban
    "city skyline",
    "street",
    "alley",
    "bridge",
    "building",
    "skyscraper",
    "tower",
    "castle",
    "church",
    "temple",
    "mosque",
    "ruins",
    "museum",
    "library",
    "stadium",
    "market",
    "shopping mall",
    "train station",
    "airport",
    "harbour",
    "lighthouse",
    "farm",
    "barn",
    "vineyard",
    # Indoor
    "living room",
    "bedroom",
    "kitchen",
    "bathroom",
    "office",
    "restaurant",
    "cafe",
    "bar",
    "gym",
    "hospital",
    "school",
    "classroom",
    "laboratory",
    "studio",
    "stage",
    "concert hall",
    "cinema",
]

_OBJECTS: list[str] = [
    # Nature
    "flower",
    "tree",
    "grass",
    "leaf",
    "mushroom",
    "cactus",
    "coral",
    "snow crystal",
    # Animals
    "dog",
    "cat",
    "bird",
    "horse",
    "cow",
    "sheep",
    "pig",
    "deer",
    "rabbit",
    "fox",
    "wolf",
    "bear",
    "lion",
    "tiger",
    "elephant",
    "giraffe",
    "monkey",
    "fish",
    "shark",
    "whale",
    "dolphin",
    "turtle",
    "frog",
    "butterfly",
    "bee",
    # People & body
    "person",
    "child",
    "baby",
    "face",
    "hand",
    "crowd",
    "couple",
    # Food & drink
    "food",
    "pizza",
    "burger",
    "sushi",
    "cake",
    "fruit",
    "vegetable",
    "salad",
    "bread",
    "coffee",
    "wine",
    "beer",
    "cocktail",
    # Vehicles
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "train",
    "airplane",
    "helicopter",
    "boat",
    "sailboat",
    "ship",
    # Tech & everyday
    "book",
    "laptop",
    "phone",
    "camera",
    "guitar",
    "piano",
    "furniture",
    "candle",
    "lamp",
    "clock",
    "painting",
    "sculpture",
    "toy",
    "ball",
    # Architecture details
    "door",
    "window",
    "staircase",
    "arch",
    "column",
    "roof",
    "fence",
    "wall",
    "pavement",
]

_STYLES: list[str] = [
    # Photographic styles
    "bokeh",
    "long exposure",
    "macro photography",
    "aerial photography",
    "underwater photography",
    "black and white",
    "silhouette",
    "high contrast",
    "low contrast",
    "high key lighting",
    "low key lighting",
    "film grain",
    "double exposure",
    "panoramic",
    "fisheye lens",
    "tilt shift",
    # Aesthetic styles
    "minimalist",
    "vintage",
    "retro",
    "modern",
    "rustic",
    "industrial",
    "neon lights",
    "golden hour",
    "blue hour",
    "symmetrical composition",
    "leading lines",
    "rule of thirds",
    "close-up",
    "wide angle",
    # Art styles
    "watercolour",
    "oil painting",
    "sketch",
    "abstract",
    "graffiti",
]

_MOODS: list[str] = [
    "joyful",
    "happy",
    "playful",
    "romantic",
    "intimate",
    "peaceful",
    "serene",
    "calm",
    "meditative",
    "cozy",
    "warm",
    "nostalgic",
    "melancholic",
    "lonely",
    "mysterious",
    "eerie",
    "dramatic",
    "intense",
    "energetic",
    "vibrant",
    "festive",
    "celebratory",
    "cheerful",
    "gloomy",
    "dark",
    "moody",
    "ethereal",
    "dreamy",
    "surreal",
]

_COLORS: list[str] = [
    "warm colours",
    "cool colours",
    "monochrome",
    "colourful",
    "pastel colours",
    "earth tones",
    "neon colours",
    "muted colours",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "white",
    "black",
    "grey",
    "golden",
    "teal",
    "turquoise",
    "coral",
    "lavender",
]

_ACTIVITIES: list[str] = [
    # Sport & movement
    "running",
    "cycling",
    "swimming",
    "surfing",
    "skiing",
    "snowboarding",
    "climbing",
    "hiking",
    "yoga",
    "dancing",
    "playing football",
    "playing basketball",
    "tennis",
    "golf",
    "boxing",
    "skateboarding",
    "skating",
    # Social & lifestyle
    "eating",
    "cooking",
    "reading",
    "working",
    "studying",
    "sleeping",
    "laughing",
    "hugging",
    "kissing",
    "walking",
    "sitting",
    "standing",
    "jumping",
    "waving",
    # Events
    "wedding",
    "birthday party",
    "graduation",
    "concert",
    "festival",
    "protest",
    "parade",
    "picnic",
    "camping",
    "fireworks",
    "Christmas",
    "Halloween",
    # Travel
    "travel",
    "tourist",
    "sightseeing",
    "road trip",
    "backpacking",
]

_PHOTOGRAPHY_SUBJECTS: list[str] = [
    "portrait",
    "selfie",
    "group photo",
    "landscape",
    "cityscape",
    "street photography",
    "wildlife",
    "sports",
    "architecture",
    "food photography",
    "product photography",
    "fashion",
    "documentary",
    "abstract",
    "nature",
    "astrophotography",
    "wedding photography",
    "event photography",
]

_LIGHTING: list[str] = [
    "sunlight",
    "shadow",
    "backlit",
    "side lit",
    "candle light",
    "artificial light",
    "neon light",
    "flash photography",
    "overcast light",
    "dappled light",
    "reflections",
]

_WEATHER_TIME: list[str] = [
    "daytime",
    "night",
    "dawn",
    "dusk",
    "sunny",
    "rainy",
    "snowy",
    "windy",
    "misty",
    "foggy",
    "stormy",
    "cloudy",
    "clear sky",
    "spring",
    "summer",
    "autumn",
    "winter",
]

# ---------------------------------------------------------------------------
# Public API — flat list of all terms
# ---------------------------------------------------------------------------

CLIP_VOCAB_TERMS: list[str] = (
    _SCENES
    + _OBJECTS
    + _STYLES
    + _MOODS
    + _COLORS
    + _ACTIVITIES
    + _PHOTOGRAPHY_SUBJECTS
    + _LIGHTING
    + _WEATHER_TIME
)

# De-duplicate while preserving order
_seen: set[str] = set()
_deduped: list[str] = []
for _t in CLIP_VOCAB_TERMS:
    if _t not in _seen:
        _seen.add(_t)
        _deduped.append(_t)
CLIP_VOCAB_TERMS = _deduped
del _seen, _deduped, _t
