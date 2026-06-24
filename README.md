# AI Quality Gate

Automated testing framework for AI/ML models - built to validate behavior, robustness, and integration of a sentiment analysis system.

![CI/CD](https://github.com/SantiagoSvc/ai-quality-gate/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![pytest](https://img.shields.io/badge/tested%20with-pytest-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)

## What problem does this solve?

AI models fail in ways traditional QA does not catch:
- Unexpected outputs with edge case inputs
- Confidence scores outside valid ranges
- API endpoints returning wrong status codes
- Model behavior breaking after retraining

This project builds a complete QA framework that automatically validates an AI model at every level: unit, integration, and CI/CD pipeline.

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| ML Model | scikit-learn | Sentiment classification |
| API | FastAPI | Expose model as REST API |
| Testing | pytest | Automated test suite |
| CI/CD | GitHub Actions | Run tests on every push |
| HTTP Client | httpx | API testing in pytest |

## Project Structure
ai-quality-gate/

├── src/

│   ├── model.py          # ML model training and prediction

│   └── api.py            # FastAPI endpoints

├── tests/

│   ├── unit/

│   │   └── test_model.py      # 11 unit tests for the ML model

│   └── integration/

│       └── test_api.py        # 10 integration tests for the API

├── .github/workflows/

│   └── tests.yml              # CI/CD pipeline configuration

└── requirements.txt

## Getting Started

### 1. Clone the repository
git clone https://github.com/SantiagoSvc/ai-quality-gate.git

cd ai-quality-gate

### 2. Create and activate virtual environment
python -m venv venv

venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Train the model
python src/model.py

### 5. Run the API
uvicorn src.api:app --reload

API available at: http://127.0.0.1:8000/docs

## Running Tests

### All tests
pytest tests/ -v

### Expected output
21 passed in 3.xx seconds

## What gets tested?

### Unit Tests (11 tests)
- Positive and negative sentiment classification
- Response schema validation
- Confidence score range (0.0 - 1.0)
- Edge cases: empty string, single word, 500-word text
- Robustness: numbers, special characters

### Integration Tests (10 tests)
- Health endpoint returns 200
- Predict endpoint returns correct sentiment
- Empty text rejected with 400 error
- Missing fields rejected with 422 error
- Response fields validation

## Author

Santiago - QA Automation Engineer
Mechatronics background applied to AI/ML quality assurance.
GitHub: https://github.com/SantiagoSvc

## Key Concepts Demonstrated

- Shift-left testing: catching bugs early in the development cycle
- Test pyramid: unit, integration, CI/CD
- AI-specific QA: validating ML model behavior beyond traditional testing
- Pipeline automation: zero manual testing in deployment flow