"""
Test cases for evaluating agent prompts.
Each test case has an input request and expected qualities in the output.
"""

# Test cases for the Researcher agent
RESEARCHER_TEST_CASES = [
    {
        "id": "r1",
        "request": "Plan a 3-day trip to Marrakech for 2 people, budget 600 EUR, focused on food and culture.",
        "expected_keywords": ["medina", "souk", "djemaa", "tagine", "riad"],
        "must_mention": ["food", "culture"],
    },
    {
        "id": "r2",
        "request": "I want to visit Lisbon for a week with my partner, we love history and beaches.",
        "expected_keywords": ["alfama", "belem", "tram", "fado"],
        "must_mention": ["history"],
    },
    {
        "id": "r3",
        "request": "5 days in Barcelona, family of 4, budget 2000 EUR, kid-friendly activities.",
        "expected_keywords": ["sagrada", "park guell", "beach", "tapas"],
        "must_mention": ["family", "kid"],
    },
]

# Test cases for the Itinerary agent
ITINERARY_TEST_CASES = [
    {
        "id": "i1",
        "request": "3-day Marrakech itinerary, 600 EUR budget, food-focused.",
        "must_have": ["Day 1", "Day 2", "Day 3"],
        "must_not_have": ["Day 4", "Day 5"],  # Should respect duration
    },
    {
        "id": "i2",
        "request": "Weekend in Paris (2 days), 400 EUR, romantic.",
        "must_have": ["Day 1", "Day 2"],
        "must_not_have": ["Day 3"],
    },
]
