# 📈 Universal Sales Forecast Assistant

Forecast any tabular sales CSV in seconds.  
* Streamlit front-end with an LLM agent that auto-detects date & sales columns  
* FastAPI + Prophet back-end for robust forecasting  
* Plotly visualisation + confidence bands  
* Groq-powered `phi.agent` reasoning (LLama-3.3-70B)  
* One-click deploy via Docker / GH Actions

## Quick Start

```bash
git clone https://github.com/NikhithReddyScripts/universal-sales-forecast-assistant
cd universal-sales-forecast-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp example.env .env        # add your GROQ_API_KEY
streamlit run streamlit_app/user_agent.py
