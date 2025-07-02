#!/bin/bash

echo "🔧 Starting mmocr_anonymizer via Streamlit..."
python3 -m streamlit run app.py --server.port 8502 --server.enableCORS false
