"""Minimal connection test for the Jev API at jevtypesafeai.com.

Set JEV_API_KEY in your local environment before running. Do not put a key
in this file or share it in chat.
"""

import os
import json

import requests


API_URL = "https://jevtypesafeai.com/api/v1/decide"


def main() -> None:
    api_key = os.environ.get("JEV_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set JEV_API_KEY before running this script.")

    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "state": "Customer: I was charged twice and nobody has replied for 3 days.",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Where should this ticket go?",
                    "criteria": {
                        "billing": "payments, refunds, invoices",
                        "bug": "the product is broken",
                        "account": "login or access",
                    },
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent is this?",
                    "criteria": [
                        "routine",
                        "today",
                        "urgent",
                        "critical, about to churn",
                    ],
                },
                "escalate": {
                    "type": "noul",
                    "instructions": "Escalate to a human now?",
                },
            },
        },
        timeout=30,
    )

    if not response.ok:
        raise SystemExit(f"Jev API returned HTTP {response.status_code}: {response.text}")

    result = response.json()
    print("model:", result["model"])
    print("answers:", json.dumps(result["answers"], indent=2, ensure_ascii=False))
    print("usage:", result.get("usage"))


if __name__ == "__main__":
    main()
