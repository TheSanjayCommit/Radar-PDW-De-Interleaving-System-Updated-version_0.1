import streamlit as st
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta


def get_current_time_us():
    now = datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return (now - midnight).total_seconds() * 1_000_000


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)




# =================================================
# MANUAL MODE UI
# =================================================
def manual_mode_ui():

    if "manual_global_time_us" not in st.session_state:
        st.session_state.manual_global_time_us = get_current_time_us()
    if "manual_pdw_buffer" not in st.session_state:
        st.session_state.manual_pdw_buffer = []
    if "manual_running" not in st.session_state:
        st.session_state.manual_running = False

    st.header("Manual Mode – PDW Simulation (Continuous Time)")
    st.info("Emitter parameters are manually configured with realistic tolerances")

    cfg = st.session_state.manual_config

    num_emitters = st.number_input(
        "Number of Emitters", 1, 20, cfg.get("num_emitters", 3)
    )
    pulses_per_emitter = st.number_input(
        "Pulses per Emitter (per 2s)", 1, 1000, cfg.get("pulses", 20)
    )

    cfg["num_emitters"] = num_emitters
    cfg["pulses"] = pulses_per_emitter

    emitters = []

    # =================================================
    # EMITTER CONFIGURATION
    # =================================================
    for i in range(num_emitters):
        st.markdown(f"### Emitter {i+1}")

        # -----------------------------
        # FREQUENCY CONFIG
        # -----------------------------
        # Load from config or default
        saved_freq_type = cfg.get(f"freq_type_{i}", "Fixed")
        freq_opts = ["Fixed", "Agile"]
        try:
            f_idx = freq_opts.index(saved_freq_type)
        except:
            f_idx = 0

        freq_type = st.selectbox(
            "Frequency Type",
            freq_opts,
            index=f_idx,
            key=f"widget_freq_type_{i}"
        )
        cfg[f"freq_type_{i}"] = freq_type  # SAVE

        freqs = []
        if freq_type == "Agile":
            saved_modes = cfg.get(f"num_modes_{i}", 3)
            num_modes = st.number_input(
                "Number of Frequency Modes",
                min_value=2,
                max_value=8,
                value=saved_modes,
                key=f"widget_num_modes_{i}"
            )
            cfg[f"num_modes_{i}"] = num_modes # SAVE

            for m in range(num_modes):
                saved_f = cfg.get(f"freq_{i}_{m}", 9000.0 + m * 50)
                val = st.number_input(
                    f"Mode {m+1} Frequency (MHz)",
                    500.0, 40000.0,
                    float(saved_f),
                    key=f"widget_freq_{i}_{m}"
                )
                cfg[f"freq_{i}_{m}"] = val # SAVE
                freqs.append(val)
        else:
            saved_f = cfg.get(f"freq_{i}", 9000.0 + i * 500.0)
            val = st.number_input(
                "Base Frequency (MHz)",
                500.0, 40000.0,
                float(saved_f),
                key=f"widget_freq_{i}"
            )
            cfg[f"freq_{i}"] = val # SAVE
            freqs.append(val)

        # -----------------------------
        # PRI CONFIG
        # -----------------------------
        saved_pri_type = cfg.get(f"pri_type_{i}", "Fixed")
        pri_opts = ["Fixed", "Jittered", "Staggered"]
        try:
            p_idx = pri_opts.index(saved_pri_type)
        except:
            p_idx = 0

        pri_type = st.selectbox(
            "PRI Type",
            pri_opts,
            index=p_idx,
            key=f"widget_pri_type_{i}"
        )
        cfg[f"pri_type_{i}"] = pri_type # SAVE

        pri_values = []

        if pri_type == "Staggered":
            saved_n_pri = cfg.get(f"num_pri_{i}", 2)
            num_pri = st.number_input(
                "Number of PRI Values",
                min_value=2,
                max_value=5,
                value=saved_n_pri,
                key=f"widget_num_pri_{i}"
            )
            cfg[f"num_pri_{i}"] = num_pri # SAVE

            for p in range(num_pri):
                saved_p = cfg.get(f"pri_{i}_{p}", 2000.0 + p * 100)
                val = st.number_input(
                    f"PRI {p+1} (µs)",
                    2.0, 20000.0,
                    float(saved_p),
                    key=f"widget_pri_{i}_{p}"
                )
                cfg[f"pri_{i}_{p}"] = val # SAVE
                pri_values.append(val)
        else:
            saved_pri = cfg.get(f"pri_{i}", 2000.0 + i * 500.0)
            val = st.number_input(
                "Base PRI (µs)",
                2.0, 20000.0,
                float(saved_pri),
                key=f"widget_pri_{i}"
            )
            cfg[f"pri_{i}"] = val # SAVE
            pri_values.append(val)

        # -----------------------------
        # OTHER PARAMETERS
        # -----------------------------
        saved_pw = cfg.get(f"pw_{i}", 10.0)
        pw = st.number_input(
            "Pulse Width (µs)",
            0.01, 1000.0,
            float(saved_pw),
            key=f"widget_pw_{i}"
        )
        cfg[f"pw_{i}"] = pw # SAVE

        saved_amp = cfg.get(f"amp_{i}", -60.0)
        amp = st.number_input(
            "Amplitude (dB)",
            -200.0, 10.0,
            float(saved_amp),
            key=f"widget_amp_{i}"
        )
        cfg[f"amp_{i}"] = amp # SAVE

        saved_doa = cfg.get(f"doa_{i}", 90.0)
        doa = st.number_input(
            "DOA (deg)",
            0.0, 360.0,
            float(saved_doa),
            key=f"widget_doa_{i}"
        )
        cfg[f"doa_{i}"] = doa # SAVE

        emitters.append({
            "freqs": freqs,
            "pri_type": pri_type,
            "pri_values": pri_values,
            "pw": pw,
            "amp": amp,
            "doa": doa
        })

    # =================================================   
    # CONTROLS
    # =================================================

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("▶ Start / Generate"):
            st.session_state.manual_running = True

    with c2:
        if st.button("⏸ Pause"):
            st.session_state.manual_running = False

    with c3:
        if st.button("🔴 Reset Simulation & Clear Data", type="primary", help="Clears all generated data and resets configuration."):
            st.session_state.manual_running = False
            st.session_state.manual_global_time_us = get_current_time_us()
            st.session_state.manual_pdw_buffer = []
            if "manual_emitter_states" in st.session_state:
                del st.session_state.manual_emitter_states
            # Clear config to force re-load of staggered defaults
            st.session_state.manual_config.clear()
            st.rerun()

    # =================================================
    # GENERATE PDWs
    # =================================================
    if st.session_state.manual_running:
        
        # 1. State Invalidation Logic (Match Auto Mode)
        # Create a signature of the current configuration
        # We include num_emitters and the pulses count, plus a hash of the emitter params
        import json
        try:
            # Simple serialization of cfg dict to string for signature
            # sort_keys=True ensures key order doesn't affect hash
            cfg_sig = json.dumps(cfg, sort_keys=True)
        except:
            cfg_sig = str(cfg)

        last_sig = st.session_state.get("manual_cfg_sig")

        if last_sig != cfg_sig:
            # Config changed! Reset buffer to ensure "Detected = Expected"
            st.session_state.manual_pdw_buffer = []
            if "manual_emitter_states" in st.session_state:
                del st.session_state.manual_emitter_states
            
            st.session_state.manual_cfg_sig = cfg_sig
            st.toast("Configuration changed. Data buffer cleared for consistency.", icon="🧹")
            # Reset global time if you want a true "Fresh Start" from 0? 
            # Or keep time moving? Usually for verification, Time=0 is nicer.
            # Let's reset time too.
            st.session_state.manual_global_time_us = 0 

        # Initialize state tracker if needed
        if "manual_emitter_states" not in st.session_state:
            st.session_state.manual_emitter_states = {}

        rows = []

        window_start = st.session_state.manual_global_time_us
        window_end = window_start + 2e6
        st.session_state.manual_global_time_us = window_end

        for i, e in enumerate(emitters):
            
            # Retrieve or Init State
            if i not in st.session_state.manual_emitter_states:
                # Random start offset
                start_offset = np.random.uniform(0, max(e["pri_values"]))
                st.session_state.manual_emitter_states[i] = {
                    "next_toa": window_start + start_offset,
                    "pulse_idx": 0
                }
            
            state = st.session_state.manual_emitter_states[i]
            
            # Exact Pulse Count Generation
            for _ in range(pulses_per_emitter):
                
                # --- 1. Parameter Selection ---
                
                # FREQUENCY
                if e["freqs"] and len(e["freqs"]) > 1:
                     # Agile: Random choice
                    f_center = np.random.choice(e["freqs"])
                else:
                    f_center = e["freqs"][0]
                
                freq = f_center + np.random.normal(0, 0.5)

                # PRI
                if e["pri_type"] == "Staggered":
                    idx = state["pulse_idx"] % len(e["pri_values"])
                    p_center = e["pri_values"][idx]
                else:
                    p_center = e["pri_values"][0]

                # Jitter
                if e["pri_type"] == "Jittered":
                    # Larger Jitter
                    pri_jitter = np.random.normal(0, 0.05 * p_center)
                else:
                    # Small hardware jitter
                    pri_jitter = np.random.normal(0, 0.01 * p_center)
                
                pri = p_center + pri_jitter
                
                # Other Params
                pw  = e["pw"] + np.random.normal(0, 0.05)
                amp = e["amp"] + np.random.normal(0, 0.5)
                doa = e["doa"] + np.random.normal(0, 0.2)

                # --- 2. Store Pulse ---
                rows.append({
                    "freq_MHz": freq,
                    "pri_us": pri,
                    "pw_us": pw,
                    "doa_deg": doa,
                    "amp_dB": amp,
                    "toa_us": state["next_toa"],
                    "Emitter_ID": i + 1
                })

                # --- 3. Update State ---
                state["next_toa"] += pri
                state["pulse_idx"] += 1

        out_dir = st.session_state.get("user_output_dir", OUTPUT_DIR)
        st.session_state.manual_pdw_buffer.extend(rows)


        df_all = pd.DataFrame(st.session_state.manual_pdw_buffer)

        df_all = df_all.sort_values("toa_us").round(2)
        df_all.to_csv(f"{out_dir}/manual_interleaved.csv", index=False)

        st.session_state.manual_running = False
            
        st.success(f"Generated 2 seconds of PDWs (Total: {len(df_all)})")
        st.dataframe(df_all.tail(20))


