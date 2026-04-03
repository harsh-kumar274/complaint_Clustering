
import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px
import numpy as np

st.set_page_config("Complaint Clusters (server-side)", layout="wide")
st.title("Complaint Clusters Explorer")

DB = os.path.join("output", "complaints.db")
UMAP_SAMPLE_PATH = os.path.join("output", "umap_sample.npy")

if not os.path.exists(DB):
    st.error("Database not found. Run pipeline and import script first.")
    st.stop()

conn = sqlite3.connect(DB, check_same_thread=False)


st.sidebar.header("Filters & Pagination")
label_vals = [str(x[0]) for x in conn.execute("SELECT DISTINCT kmeans_label FROM complaints").fetchall()]
label_choice = st.sidebar.selectbox("Cluster (All)", ["All"] + sorted(label_vals, key=lambda x: int(x) if x.lstrip('-').isdigit() else x))
state_vals = [row[0] for row in conn.execute("SELECT DISTINCT state FROM complaints").fetchall() if row[0] is not None]
state_choice = st.sidebar.selectbox("State (All)", ["All"] + sorted(state_vals))
search_text = st.sidebar.text_input("Search complaint text")
page = st.sidebar.number_input("Page number", min_value=1, value=1, step=1)
page_size = st.sidebar.number_input("Rows per page", min_value=10, max_value=500, value=50, step=10)


where = []
params = []
if label_choice != "All":
    where.append("kmeans_label = ?")
    params.append(int(label_choice))
if state_choice != "All":
    where.append("state = ?")
    params.append(state_choice)
if search_text:
    where.append("complaint LIKE ?")
    params.append(f"%{search_text}%")
where_clause = ("WHERE " + " AND ".join(where)) if where else ""


count_q = f"SELECT COUNT(*) FROM complaints {where_clause}"
total_rows = conn.execute(count_q, params).fetchone()[0]
st.sidebar.markdown(f"Total rows (filtered): **{total_rows:,}**")


dist_q = f"SELECT kmeans_label, COUNT(*) as cnt FROM complaints {where_clause} GROUP BY kmeans_label ORDER BY cnt DESC LIMIT 50"
dist_df = pd.read_sql(dist_q, conn, params=params)
if not dist_df.empty:
    fig = px.bar(dist_df, x="kmeans_label", y="cnt", title="Cluster counts (filtered)")
    st.plotly_chart(fig, use_container_width=True)


if os.path.exists(UMAP_SAMPLE_PATH):
    try:
        arr = np.load(UMAP_SAMPLE_PATH)
        sample_idx = arr[:,0].astype(int)
        coords = arr[:,1:3]
        sample_ids = sample_idx.tolist()
      
        placeholders = ",".join(["?"]*len(sample_ids))
        q = f"SELECT rowid, id, complaint, kmeans_label FROM complaints WHERE rowid IN ({placeholders})"
        rows = conn.execute(q, sample_ids).fetchall()
        if rows:
            df_rows = pd.DataFrame(rows, columns=["rowid","id","complaint","kmeans_label"])
            df_plot = pd.DataFrame({"x": coords[:,0], "y": coords[:,1], "cluster": df_rows["kmeans_label"].astype(str), "complaint": df_rows["complaint"]})
            fig2 = px.scatter(df_plot, x="x", y="y", color="cluster", hover_data=["complaint"], title="UMAP sample")
            st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.write("UMAP sample load error:", e)

st.subheader("Browse complaints (server-side)")


offset = (page - 1) * page_size
q = f"SELECT id, date, state, district, complaint, kmeans_label FROM complaints {where_clause} ORDER BY id LIMIT ? OFFSET ?"
rows = conn.execute(q, params + [page_size, offset]).fetchall()
if rows:
    df_page = pd.DataFrame(rows, columns=["id","date","state","district","complaint","kmeans_label"])
    st.write(df_page)
    csv = df_page.to_csv(index=False).encode("utf-8")
    st.download_button("Download this page as CSV", data=csv, file_name=f"complaints_page{page}.csv", mime="text/csv")
else:
    st.info("No results for this page/filters.")

conn.close()
