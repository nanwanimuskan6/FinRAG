"""
Evaluation question-answer dataset for FinRAG.

Each question contains:
- question: user query
- answer: expected answer
- pages: PDF pages containing supporting evidence
"""

from typing import TypedDict, NotRequired


class EvaluationItem(TypedDict):
    question: str
    answer: str
    pages: list[int]
    evidence_answers: NotRequired[list[str]]


EVALUATION_DATASET: list[EvaluationItem] = [
    {
        "question": "What was Reliance Industries Limited's total consolidated revenue for FY 2024-25?",
        "answer": "₹10,71,174 crore",
        "pages": [5, 100],
    },
    {
        "question": "What was Reliance Industries Limited's total income for FY 2024-25?",
        "answer": "₹9,98,114 crore",
        "pages": [5, 100],
    },
    {
        "question": "What was Reliance Industries Limited's revenue from operations for FY 2024-25?",
        "answer": "₹9,80,136 crore",
        "pages": [100],
    },
    {
        "question": "What was Reliance Industries Limited's profit for FY 2024-25?",
        "answer": "₹81,309 crore",
        "pages": [5],
    },
    {
        "question": "What was the net profit attributable to owners of the company in FY 2024-25?",
        "answer": "₹69,648 crore",
        "pages": [100],
    },
    {
        "question": "What was Reliance Retail's revenue for FY 2024-25?",
        "answer": "₹3,30,943 crore",
        "pages": [7],
    },
    {
        "question": "What was Reliance Retail's revenue from operations for FY 2024-25?",
        "answer": "₹2,91,043 crore",
        "pages": [7],
    },
    {
        "question": "What was Reliance Retail's EBITDA for FY 2024-25?",
        "answer": "₹25,094 crore",
        "pages": [7],
    },
    {
        "question": "What was Reliance Industries Limited's net worth in FY 2024-25?",
        "answer": "₹7,95,069 crore",
        "pages": [5],
    },
    {
        "question": "What was Reliance Industries Limited's total assets in FY 2024-25?",
        "answer": "₹19,50,121 crore",
        "pages": [5],
    },
    {
        "question": "What was Reliance Industries Limited's market capitalisation in FY 2024-25?",
        "answer": "₹17,25,378 crore",
        "pages": [5],
    },
    {
        "question": "What was the value of sales for Reliance Industries Limited in FY 2024-25?",
        "answer": "₹9,60,355 crore",
        "pages": [100],
    },
    {
        "question": "What was the income from services in FY 2024-25?",
        "answer": "₹1,10,819 crore",
        "pages": [100],
    },
    {
        "question": "What was the consolidated revenue in FY 2023-24?",
        "answer": "₹10,00,122 crore",
        "pages": [5, 100],
    },
    {
        "question": "How much did consolidated revenue increase from FY 2023-24 to FY 2024-25?",
        "answer": "₹71,052 crore",
        "evidence_answers": ["10,71,174", "10,00,122"],
        "pages": [5, 100],
    },
    {
        "question": "What was Reliance Retail's year-on-year revenue growth in FY 2024-25?",
        "answer": "7.9%",
        "pages": [7],
    },
    {
        "question": "What was Reliance Retail's EBITDA growth in FY 2024-25?",
        "answer": "8.6%",
        "pages": [7],
    },
    {
        "question": "What was Reliance Industries Limited's EBDIT in FY 2024-25?",
        "answer": "₹1,83,422 crore",
        "pages": [5],
    },
    {
        "question": "What was depreciation and amortisation in FY 2024-25?",
        "answer": "₹53,136 crore",
        "pages": [5],
    },
    {
        "question": "What was the equity share capital in FY 2024-25?",
        "answer": "₹13,532 crore",
        "pages": [5],
    },
]


def get_evaluation_dataset() -> list[EvaluationItem]:
    """Return the complete evaluation dataset."""
    return EVALUATION_DATASET


def main() -> None:
    """Print the evaluation dataset."""
    dataset = get_evaluation_dataset()

    print("=" * 70)
    print("FINRAG EVALUATION DATASET")
    print("=" * 70)
    print(f"Total questions: {len(dataset)}")
    print()

    for index, item in enumerate(dataset, start=1):
        print(f"{index}. {item['question']}")
        print(f"   Expected: {item['answer']}")
        print(f"   Pages: {item['pages']}")
        print()


if __name__ == "__main__":
    main()