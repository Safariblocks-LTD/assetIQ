# AssetIQ 

AssetIQ is a novel, real-time asset valuation system for smart connected devices. It combines ensemble forecasting, dynamic self-correction, advanced jump-diffusion (Kou's method), and heavy-tailed volatility adjustment (GARCH with Student's t-distribution) to provide highly accurate predictions.

## Features
- Ensemble of ARIMA, Prophet, and Random Forest with exogenous indicators.
- Self-correction based on recent historical changes.
- Advanced jump-diffusion adjustment for market shocks.
- Dynamic volatility adjustment using a heavy-tailed GARCH model.
- RESTful API service using FastAPI.
- Containerized deployment with Docker and Kubernetes ready.

## Project Structure
AssetIQ_Valuator/ ├── app/ │ ├── init.py │ ├── config.yaml │ ├── main.py │ ├── model.py │ └── utils.py ├── data/ │ └── historical/ │ ├── smartphones.csv │ ├── tablets.csv │ ├── laptops.csv │ └── smartwatches.csv ├── docs/ │ ├── patent_documentation.pdf │ ├── user_manual.pdf │ └── architecture_diagram.png ├── tests/ │ ├── test_api.py │ └── test_model.py ├── Dockerfile ├── requirements.txt └── README.md



## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone <https://github.com/Safariblocks-LTD/assetIQ>
   cd assetIQ

2. **Create a Virtual Environment and Install Dependencies:**

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt

3. **Run Unit Tests:**

    pytest tests/

4. **Run the API Locally:**

    uvicorn app.main:app --reload

Access the API at <http://localhost:8000>

5. **Build and Run Docker Container(if built up):**

    docker build -t assetIQ 
    docker run -p 8000:8000 assetIQ


