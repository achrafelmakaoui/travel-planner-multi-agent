"""
A/B test framework for prompts.

Compares two versions of an agent prompt on the same test cases,
and scores them on whether outputs contain expected keywords.

Run: python -m evaluation.ab_test
"""
import json
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from config import LLM_MODEL, LLM_TEMPERATURE, GROQ_API_KEY
from evaluation.test_cases import RESEARCHER_TEST_CASES


# Two prompt versions to compare
PROMPT_A = """You are a Travel Researcher. Gather information about the
destination requested by the user. Write a brief summary."""

PROMPT_B = """You are an expert Travel Researcher specializing in cultural
and culinary insights. For each user request:

1. Identify the destination and travel preferences
2. Cover ALL of: top attractions, neighborhoods, local food, cultural tips
3. Use specific names (places, dishes, neighborhoods) — be concrete

Output a structured summary with clear section headers."""


def score_output(output: str, test_case: dict) -> dict:
    """Score an output by counting matched keywords."""
    output_lower = output.lower()

    keyword_hits = sum(
        1 for kw in test_case["expected_keywords"]
        if kw.lower() in output_lower
    )
    keyword_score = keyword_hits / len(test_case["expected_keywords"])

    mention_hits = sum(
        1 for term in test_case["must_mention"]
        if term.lower() in output_lower
    )
    mention_score = mention_hits / len(test_case["must_mention"])

    overall = 0.6 * keyword_score + 0.4 * mention_score
    return {
        "keyword_score": round(keyword_score, 2),
        "mention_score": round(mention_score, 2),
        "overall": round(overall, 2),
        "output_length": len(output),
    }


def run_prompt(prompt: str, request: str) -> str:
    """Send a single (prompt, request) pair to the LLM."""
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=GROQ_API_KEY,
    )
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=request),
    ])
    return response.content


def ab_test():
    """Run both prompts on all test cases and report results."""
    results = {"prompt_A": [], "prompt_B": []}

    for test in RESEARCHER_TEST_CASES:
        print(f"\n--- Test {test['id']} ---")
        print(f"Request: {test['request'][:80]}...")

        out_a = run_prompt(PROMPT_A, test["request"])
        score_a = score_output(out_a, test)
        results["prompt_A"].append({"test": test["id"], **score_a})
        print(f"  Prompt A score: {score_a['overall']}")

        out_b = run_prompt(PROMPT_B, test["request"])
        score_b = score_output(out_b, test)
        results["prompt_B"].append({"test": test["id"], **score_b})
        print(f"  Prompt B score: {score_b['overall']}")

    # Aggregate
    avg_a = sum(r["overall"] for r in results["prompt_A"]) / len(results["prompt_A"])
    avg_b = sum(r["overall"] for r in results["prompt_B"]) / len(results["prompt_B"])

    print(f"\n=== RESULTS ===")
    print(f"Prompt A average: {avg_a:.2f}")
    print(f"Prompt B average: {avg_b:.2f}")
    winner = "B" if avg_b > avg_a else "A"
    print(f"Winner: Prompt {winner}")

    # Save results for the report
    output_path = Path(__file__).parent / "ab_results.json"
    output_path.write_text(json.dumps({
        "prompt_A_avg": round(avg_a, 2),
        "prompt_B_avg": round(avg_b, 2),
        "winner": winner,
        "details": results,
    }, indent=2))
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    ab_test()
