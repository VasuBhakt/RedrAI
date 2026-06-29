import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import sys
import subprocess
import os

st.set_page_config(page_title="RedrAI - Candidate Ranker", layout="wide")

st.title("📊 RedrAI - AI Candidate Ranking Engine")
st.markdown("Upload a Job Description (.txt or .docx) to instantly rank the top 100 candidates from our 100k+ pool using our hybrid semantic and rule-based engine.")

st.info("ℹ️ **Hackathon Sandbox Mode:** This hosted environment is pre-loaded with a representative 10,000-candidate sample (`sandbox_data`) to demonstrate end-to-end reproducibility as per the submission specification.")

jd_file = st.file_uploader("Upload Job Description", type=["txt", "docx"])

if jd_file is not None:
    if st.button("Run Ranking Engine"):
        with st.spinner("Processing candidates (usually takes ~2 minutes)..."):
            # Save uploaded file to a temporary location
            suffix = Path(jd_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(jd_file.getvalue())
                tmp_path = tmp.name

            try:
                # Run the rank.py pipeline
                env = os.environ.copy()
                result = subprocess.run(
                    [sys.executable, "-m", "src.rank", "--jd_file", tmp_path],
                    capture_output=True,
                    text=True,
                    env=env
                )
                
                if result.returncode != 0:
                    st.error("An error occurred during execution.")
                    st.code(result.stderr)
                else:
                    st.success("Ranking complete!")
                    
                    # Read the output submission.csv
                    df = pd.read_csv("output/submission.csv")
                    
                    st.subheader("🏆 Top 100 Candidates")
                    st.dataframe(
                        df,
                        column_config={
                            "rank": st.column_config.NumberColumn("Rank"),
                            "candidate_id": st.column_config.TextColumn("Candidate ID"),
                            "score": st.column_config.ProgressColumn(
                                "Score",
                                format="%f",
                                min_value=0,
                                max_value=1,
                            ),
                            "reasoning": st.column_config.TextColumn("Reasoning", width="large")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Provide download button
                    with open("output/submission.csv", "rb") as file:
                        btn = st.download_button(
                            label="Download submission.csv",
                            data=file,
                            file_name="submission.csv",
                            mime="text/csv"
                        )
            finally:
                os.unlink(tmp_path)
