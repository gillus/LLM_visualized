"""The curated list of examples to generate.

To add a new example: append a dict here and re-run `python generate/generate.py`.
The generator writes web/examples/<id>.json AND updates web/examples/manifest.json,
so no front-end change is needed. Optional keys:
  note — a classroom aside shown in the Prediction stage (use it when the
         model's top guess is a teachable surprise).

Keep prompts SHORT (<= ~12 tokens). Attention data is O(layers * heads * N^2),
and short prompts make far clearer "meaning maps" for the classroom.

Vet candidates with:  python generate/generate.py --preview "your prompt here"
(check the token split is interesting and the top guesses tell a clean story).
"""

EXAMPLES = [
    {
        "id": "cat-sat",
        "title": "The cat sat on the…",
        "prompt": "The cat sat on the",
    },
    {
        # tokenization demo: "hippopotamus" splits into hipp|opotam|us,
        # and the prediction is still clean (water ~28%)
        "id": "hippo-water",
        "title": "The hungry hippopotamus jumped into the…",
        "prompt": "The hungry hippopotamus jumped into the",
    },
    {
        # attention demo: "she" looks back at "Maria" (strong mid-stack heads)
        "id": "maria-she",
        "title": "When Maria dropped her cup, she…",
        "prompt": "When Maria dropped her cup, she",
    },
    {
        # the clean fact: Paris wins at ~73%
        "id": "france-capital",
        "title": "The capital city of France is called…",
        "prompt": "The capital city of France is called",
    },
    {
        # the deliberate surprise: "the" beats "Paris" — kept BECAUSE it
        # teaches that the model predicts likely text, not facts
        "id": "paris-capital",
        "title": "The capital of France is…",
        "prompt": "The capital of France is",
        "note": "Sorpresa: “Paris” è solo la seconda ipotesi! Tantissime frasi "
                "vere continuano come “The capital of France is the largest "
                "city in the country…”, e così a vincere è “the”. L’IA prevede "
                "quale parola è probabile che venga dopo — non va a cercare i fatti.",
    },
    {
        "id": "sky-color",
        "title": "The sky is…",
        "prompt": "The sky is",
    },
]
