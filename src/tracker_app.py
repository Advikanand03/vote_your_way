import pandas as pd
import streamlit as st


RESULTS_PATH = "outputs/final_results.csv"
ENRICHED_PATH = "data/processed/final_enriched_dataset.csv"


def load_data():
    results_df = pd.read_csv(RESULTS_PATH)
    enriched_df = pd.read_csv(ENRICHED_PATH)

    merged = results_df.merge(
        enriched_df[
            [
                "promise_id",
                "category",
                "sector",
                "sub_sector",
                "quantifiable",
                "timeline_mentioned",
                "target_year",
                "commitment_type",
            ]
        ],
        on="promise_id",
        how="left",
    )

    merged["confidence"] = pd.to_numeric(merged["confidence"], errors="coerce").fillna(0.0)
    merged["verdict"] = merged["verdict"].fillna("Unknown")
    merged["category"] = merged["category"].fillna("Unknown")
    merged["sector"] = merged["sector"].fillna("Unknown")
    return merged


def main():
    st.set_page_config(page_title="Manifesto Promise Tracker", layout="wide")
    st.title("Vote Your Way")
    st.caption("Simple tracker built from pipeline outputs")

    try:
        df = load_data()
    except Exception as exc:
        st.error(f"Could not load CSV files: {exc}")
        st.info("Expected files: outputs/final_results.csv and data/processed/final_enriched_dataset.csv")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        verdict_filter = st.multiselect(
            "Filter by verdict",
            options=sorted(df["verdict"].dropna().unique().tolist()),
            default=sorted(df["verdict"].dropna().unique().tolist()),
        )
    with col2:
        sector_filter = st.multiselect(
            "Filter by sector",
            options=sorted(df["sector"].dropna().unique().tolist()),
            default=[],
        )
    with col3:
        category_filter = st.multiselect(
            "Filter by category",
            options=sorted(df["category"].dropna().unique().tolist()),
            default=[],
        )

    search = st.text_input("Search promise text")
    min_confidence = st.slider("Minimum confidence", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    filtered = df[df["verdict"].isin(verdict_filter)]
    if sector_filter:
        filtered = filtered[filtered["sector"].isin(sector_filter)]
    if category_filter:
        filtered = filtered[filtered["category"].isin(category_filter)]
    if search.strip():
        filtered = filtered[filtered["promise_text"].str.contains(search.strip(), case=False, na=False)]
    filtered = filtered[filtered["confidence"] >= min_confidence]

    total_promises = len(filtered)
    completed_count = int((filtered["verdict"] == "Completed").sum())
    in_progress_count = int((filtered["verdict"] == "In Progress").sum())
    not_done_count = int((filtered["verdict"] == "Not Done").sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Promises", total_promises)
    m2.metric("Completed", completed_count)
    m3.metric("In Progress", in_progress_count)
    m4.metric("Not Done", not_done_count)

    st.subheader("Status Breakdown")
    status_counts = (
        filtered["verdict"]
        .value_counts()
        .reindex(["Completed", "In Progress", "Not Done"], fill_value=0)
    )
    st.bar_chart(status_counts)

    st.subheader("Promises")
    display_columns = [
        "promise_id",
        "promise_text",
        "verdict",
        "confidence",
        "category",
        "sector",
        "sub_sector",
        "quantifiable",
        "timeline_mentioned",
        "target_year",
        "commitment_type",
    ]
    st.dataframe(filtered[display_columns], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
