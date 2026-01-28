import streamlit as st
import pandas as pd
from .logic import auto_tune_dbscan, run_clustering, process_results, HAS_HDBSCAN

def render_deinterleaving_section(df_input, known_emitters, key_prefix="auto"):
    """
    Renders the complete De-Interleaving UI section.
    """
    st.markdown("---")
    st.header("🔍 De-Interleaving & Analysis")
    st.caption("Apply clustering algorithms to separate the interleaved signal stream below.")

    if df_input is None or df_input.empty:
        st.warning("No PDW data available to analyze.")
        return

    # 1. Feature Selection
    features = render_feature_selection(df_input, key_prefix)
    if not features:
        st.error("Select at least one feature.")
        return

    # 2. Algorithm & Parameters
    algorithm, params, custom_tols = render_algo_selection(known_emitters, df_input, features, key_prefix)

    # 3. Run Button
    if st.button(f"🚀 Run {algorithm}", key=f"{key_prefix}_run_btn", type="primary"):
        with st.spinner(f"Running {algorithm}..."):
            labels = run_clustering(algorithm, df_input, features, params, custom_tols)
            mapped_results, summary = process_results(labels, known_emitters)
            
            # Save results to session state to persist view
            st.session_state[f"{key_prefix}_results"] = mapped_results
            st.session_state[f"{key_prefix}_summary"] = summary
            
            st.toast(f"De-Interleaving Complete! Detected {summary['clusters']} Emitters.", icon="✅")

    # 4. Results Display
    if f"{key_prefix}_results" in st.session_state:
        results = st.session_state[f"{key_prefix}_results"]
        summary = st.session_state[f"{key_prefix}_summary"]
        render_results_view(df_input, results, summary, key_prefix)


def render_feature_selection(df, key_prefix):
    st.subheader("1. Feature Selection")
    
    # Default features
    default_feats = ["freq_MHz", "pri_us"]
    
    c1, c2, c3, c4 = st.columns(4)
    use_freq = c1.checkbox("Frequency", value="freq_MHz" in default_feats, key=f"{key_prefix}_f")
    use_pri  = c2.checkbox("PRI", value="pri_us" in default_feats, key=f"{key_prefix}_p")
    use_pw   = c3.checkbox("PW", value=False, key=f"{key_prefix}_w")
    use_doa  = c4.checkbox("DOA", value=False, key=f"{key_prefix}_d")

    features = []
    if use_freq: features.append("freq_MHz")
    if use_pri:  features.append("pri_us")
    if use_pw and "pw_us" in df.columns: features.append("pw_us")
    if use_doa and "doa_deg" in df.columns: features.append("doa_deg")
    
    return features


def render_algo_selection(known_emitters, df, features, key_prefix):
    st.subheader("2. Algorithm Configuration")
    
    algo_options = ["DBSCAN"]
    if HAS_HDBSCAN: algo_options.append("HDBSCAN")
    algo_options.append("K-Means")

    c_algo, c_params = st.columns([1, 2])
    
    with c_algo:
        algorithm = st.selectbox("Algorithm", algo_options, key=f"{key_prefix}_algo")

    params = {}
    custom_tols = {}

    with c_params:
        if algorithm == "DBSCAN":
            # Auto-Tune option
            if st.checkbox("✨ Auto-Tune Epsilon", value=True, help="Automatically find best Epsilon matching expected emitters.", key=f"{key_prefix}_autotune"):
                 with st.spinner("Auto-tuning..."):
                     tuned = auto_tune_dbscan(df, features, known_emitters)
                     params["eps"] = tuned["eps"]
                     params["min_samples"] = tuned["min_samples"]
                     st.caption(f"Auto-Tuned: eps={tuned['eps']:.2f}")
            else:
                 params["eps"] = st.slider("Epsilon (Radius)", 0.1, 5.0, 0.5, key=f"{key_prefix}_eps")
                 params["min_samples"] = st.slider("Min Samples", 2, 20, 5, key=f"{key_prefix}_ms")

            # Tolerances for Custom Scaling
            with st.expander("🛠️ Advanced Tolerances", expanded=False):
                c1, c2 = st.columns(2)
                custom_tols["freq_MHz"] = c1.number_input("Freq Tol (MHz)", 0.1, 100.0, 10.0, key=f"{key_prefix}_tol_f")
                custom_tols["pri_us"] = c2.number_input("PRI Tol (µs)", 0.1, 500.0, 20.0, key=f"{key_prefix}_tol_p")
                
        elif algorithm == "HDBSCAN":
            params["min_cluster_size"] = st.slider("Min Cluster Size", 2, 50, 5, key=f"{key_prefix}_h_mcs")
            params["min_samples"] = st.slider("Min Samples", 1, 50, 5, key=f"{key_prefix}_h_ms")

        elif algorithm == "K-Means":
             default_k = known_emitters if known_emitters else 3
             params["n_clusters"] = st.slider("Number of Clusters (k)", 2, 20, default_k, key=f"{key_prefix}_k")

    return algorithm, params, custom_tols


def render_results_view(df_input, results, summary, key_prefix):
    
    df_out = df_input.copy()
    df_out["Emitter_ID"] = results
    
    # Save to CSV immediately
    out_dir = st.session_state.get("user_output_dir", "outputs")
    df_out.round(2).to_csv(f"{out_dir}/deinterleaved_pdws.csv", index=False)

    st.markdown("### 📊 De-Interleaving Results")
    
    # Metrics Area
    m1, m2, m3 = st.columns(3)
    m1.metric("Detected Emitters", summary['clusters'], border=True)
    m2.metric("Expected Emitters", summary.get('expected', 'N/A'), border=True)
    m3.metric("Noise Pulses", summary['noise'], border=True)

    # 3-Panel View
    t1, t2, t3 = st.tabs(["🔀 Interleaved Input", "🎯 Detected Emitters", "📡 Emitter Tracks"])
    
    with t1:
        cols = ["toa_us", "freq_MHz", "pri_us", "pw_us", "doa_deg", "amp_dB"]
        st.dataframe(df_out[cols].sort_values("toa_us").round(2), use_container_width=True, height=400)
        
    with t2:
        df_emitters = (
                df_out[df_out["Emitter_ID"] != 0]
                .groupby("Emitter_ID")
                .agg(
                    Pulses=("Emitter_ID", "count"),
                    Mean_Freq_MHz=("freq_MHz", "mean"),
                    Mean_PRI_us=("pri_us", "mean")
                )
                .round(2)
                .reset_index()
            )
        st.dataframe(df_emitters, use_container_width=True)
        
    with t3:
        emitter_ids = sorted(list(set(results)))
        if 0 in emitter_ids: emitter_ids.remove(0)
        
        if not emitter_ids:
            st.info("No emitters found.")
        else:
            sel_id = st.selectbox("Select Emitter to Track", emitter_ids, key=f"{key_prefix}_track_sel")
            track_df = df_out[df_out["Emitter_ID"] == sel_id].sort_values("toa_us")
            st.dataframe(track_df, use_container_width=True)
