"""
Streamlit application for fantasy football projections.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ffproj.config import PROCESSED_DATA_DIR, POSITIONS
from ffproj.io import read_parquet


st.set_page_config(
    page_title="Fantasy Football Projections",
    page_icon="🏈",
    layout="wide"
)


@st.cache_data
def load_predictions(week: int, season: int = 2024):
    """Load predictions for a specific week."""
    try:
        # Try to load from processed predictions
        pred_file = PROCESSED_DATA_DIR / f"preds_week_{season}_{week}.parquet"
        if pred_file.exists():
            return read_parquet(pred_file)

        # Fall back to backtest predictions
        backtest_file = PROCESSED_DATA_DIR.parent / "experiments" / "backtest_predictions.parquet"
        if backtest_file.exists():
            df = read_parquet(backtest_file)
            return df[(df['season'] == season) & (df['week'] == week)]

        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def main():
    st.title("🏈 Fantasy Football Projections")
    st.markdown("*Weekly PPR/Half-PPR projections with uncertainty quantification*")

    # Sidebar controls
    st.sidebar.header("Settings")

    season = st.sidebar.number_input("Season", min_value=2022, max_value=2025, value=2025)
    week = st.sidebar.number_input("Week", min_value=1, max_value=18, value=1)
    scoring = st.sidebar.selectbox("Scoring", ["PPR", "Half-PPR"])
    positions = st.sidebar.multiselect(
        "Positions",
        POSITIONS,
        default=POSITIONS
    )

    min_projection = st.sidebar.slider(
        "Min Projection (filter low scorers)",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=1.0
    )

    # Load data
    preds = load_predictions(week, season)

    if preds is None:
        st.warning(f"No predictions available for Season {season}, Week {week}")
        st.info("""
        **To generate predictions:**
        1. Run the training script: `python -m ffproj.train --backtest`
        2. Or generate weekly predictions and save to `data/processed/preds_week_{season}_{week}.parquet`
        """)
        return

    # Adjust for Half-PPR if needed
    if scoring == "Half-PPR":
        if 'q50_half_ppr' in preds.columns:
            display_col = 'q50_half_ppr'
        else:
            # Approximate conversion (subtract 0.5 * expected receptions)
            st.info("Half-PPR column not found, using PPR values (approximation)")
            display_col = 'q50'
    else:
        display_col = 'q50'

    # Filter by position
    if positions:
        preds = preds[preds['position'].isin(positions)]

    # Filter by minimum projection
    if display_col in preds.columns:
        preds = preds[preds[display_col] >= min_projection]

    # Sort by projection
    preds = preds.sort_values(display_col, ascending=False).reset_index(drop=True)

    # Display metrics in header
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players", len(preds))
    with col2:
        if display_col in preds.columns:
            st.metric("Avg Projection", f"{preds[display_col].mean():.1f}")
    with col3:
        if 'q90' in preds.columns and 'q10' in preds.columns:
            st.metric("Avg Uncertainty", f"{(preds['q90'] - preds['q10']).mean():.1f}")
    with col4:
        if 'y_true' in preds.columns and display_col in preds.columns:
            mae = np.abs(preds['y_true'] - preds[display_col]).mean()
            st.metric("MAE", f"{mae:.2f}")

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Projections", "📈 Analysis", "ℹ️ About"])

    with tab1:
        st.subheader(f"Week {week} Projections ({scoring})")

        # Prepare display dataframe
        display_cols = ['position']

        if 'player_name' in preds.columns:
            display_cols.insert(0, 'player_name')
        elif 'player_id' in preds.columns:
            display_cols.insert(0, 'player_id')

        if 'team' in preds.columns:
            display_cols.append('team')

        if 'opponent' in preds.columns:
            display_cols.append('opponent')

        # Add prediction columns
        if 'q10' in preds.columns:
            display_cols.append('q10')
        if display_col in preds.columns:
            display_cols.append(display_col)
        if 'q90' in preds.columns:
            display_cols.append('q90')

        # Add actual if available
        if 'y_true' in preds.columns:
            display_cols.append('y_true')

        # Select and rename columns
        display_df = preds[display_cols].copy()
        display_df.columns = display_df.columns.str.replace('q50', 'Proj').str.replace('q10', 'P10').str.replace('q90', 'P90').str.replace('y_true', 'Actual')

        # Format numbers
        for col in ['P10', 'Proj', 'P90', 'Actual', display_col]:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(1)

        # Display table
        st.dataframe(
            display_df,
            use_container_width=True,
            height=600
        )

        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Projections (CSV)",
            data=csv,
            file_name=f"projections_week_{week}_{scoring.lower()}.csv",
            mime="text/csv"
        )

    with tab2:
        st.subheader("Projection Analysis")

        if 'y_true' in preds.columns and display_col in preds.columns:
            # Scatter plot: Predicted vs Actual
            fig = px.scatter(
                preds,
                x=display_col,
                y='y_true',
                color='position',
                hover_data=['player_name'] if 'player_name' in preds.columns else None,
                title="Predicted vs Actual Fantasy Points",
                labels={display_col: 'Predicted', 'y_true': 'Actual'}
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, preds['y_true'].max()],
                    y=[0, preds['y_true'].max()],
                    mode='lines',
                    name='Perfect',
                    line=dict(dash='dash', color='gray')
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        # Distribution by position
        if display_col in preds.columns:
            fig = px.box(
                preds,
                x='position',
                y=display_col,
                title="Projection Distribution by Position",
                labels={display_col: 'Projected Points'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Uncertainty analysis
        if 'q10' in preds.columns and 'q90' in preds.columns:
            preds['uncertainty'] = preds['q90'] - preds['q10']

            fig = px.scatter(
                preds,
                x=display_col,
                y='uncertainty',
                color='position',
                title="Projection vs Uncertainty",
                labels={display_col: 'Projected Points', 'uncertainty': 'Uncertainty (P90-P10)'}
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("About This App")
        st.markdown("""
        This application provides weekly fantasy football projections with uncertainty quantification.

        **Features:**
        - Weekly PPR/Half-PPR projections for QB, RB, WR, TE
        - Quantile forecasts (P10, P50, P90) for risk assessment
        - Historical backtest results when available

        **Methodology:**
        - Gradient Boosted Trees (LightGBM) with quantile regression
        - Features: rolling volume, efficiency metrics, opponent strength, game context
        - Trained on historical data (2019-2023)
        - Rolling window backtest for validation

        **Metrics:**
        - **P50 (Projection)**: Median (50th percentile) forecast
        - **P10/P90**: 10th and 90th percentile bounds (80% prediction interval)
        - **MAE**: Mean Absolute Error vs actual points

        **How to Use:**
        1. Select week and scoring system in sidebar
        2. Filter by position and minimum projection
        3. View projections table with uncertainty bands
        4. Download CSV for integration with your fantasy platform

        **Note:** Projections are for informational purposes. Always combine with
        your own research, injury news, and expert analysis.
        """)


if __name__ == '__main__':
    main()
