#!/bin/bash
# Launch the Fantasy Football Projections Streamlit App

echo "========================================================================"
echo "   Fantasy Football Projections - Streamlit App"
echo "========================================================================"
echo ""
echo "The app will open in your default browser at: http://localhost:8501"
echo ""
echo "Features:"
echo "  • View weekly projections (P10, P50, P90)"
echo "  • Filter by position and scoring system"
echo "  • Compare predictions vs actual results"
echo "  • Download projections as CSV"
echo ""
echo "Press Ctrl+C to stop the app"
echo ""
echo "========================================================================"
echo ""

streamlit run src/app/streamlit_app.py
