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

np.random.seed(42)



# =================================================
# SESSION STATE
# =================================================
if "global_time_us" not in st.session_state:
    st.session_state.global_time_us = get_current_time_us()

if "pdw_buffer" not in st.session_state:
    st.session_state.pdw_buffer = []

if "auto_running" not in st.session_state:
    st.session_state.auto_running = False


# =================================================
# AUTO MODE UI
# =================================================
# =================================================
# AUTO MODE UI
# =================================================
def auto_mode_ui():

    st.header("Auto Mode – PDW Simulation (Continuous Time)")
    st.info("PDWs are generated in 2-second continuous windows. 'Start / Generate' appends more data.")

    cfg = st.session_state.auto_config

    num_emitters = st.number_input(
        "Number of Emitters", 1, 100, cfg.get("num_emitters", 10), step=1
    )
    pulses_per_emitter = st.number_input(
        "Pulses per Emitter (per 2s window)", 1, 1000, cfg.get("pulses_per_emitter", 20)
    )

    cfg["num_emitters"] = num_emitters
    cfg["pulses_per_emitter"] = pulses_per_emitter

    # -----------------------------
    # EMITTER TYPE DISTRIBUTION
    # -----------------------------
    st.subheader("Emitter Type Distribution (%)")

    fixed_pct = st.number_input("Fixed Emitters (%)", 0, 100, cfg.get("fixed_pct", 100))
    agile_pct = st.number_input("Frequency Agile Emitters (%)", 0, 100, cfg.get("agile_pct", 0))
    stagger_pct = st.number_input("Staggered PRI Emitters (%)", 0, 100, cfg.get("stagger_pct", 0))

    if fixed_pct + agile_pct + stagger_pct != 100:
        st.error("Emitter percentages must sum to 100")
        return

    cfg.update({
        "fixed_pct": fixed_pct,
        "agile_pct": agile_pct,
        "stagger_pct": stagger_pct
    })

    # -----------------------------
    # PARAMETER RANGES
    # -----------------------------
    st.subheader("Parameter Ranges")

    saved_f_min = cfg.get("f_min", 8000.0)
    f_min = st.number_input("Frequency Min (MHz)", 500.0, 40000.0, float(saved_f_min))
    cfg["f_min"] = f_min

    saved_f_max = cfg.get("f_max", 12000.0)
    f_max = st.number_input("Frequency Max (MHz)", 500.0, 40000.0, float(saved_f_max))
    cfg["f_max"] = f_max

    saved_pri_min = cfg.get("pri_min", 2000.0)
    pri_min = st.number_input("PRI Min (µs)", 2.0, 20000.0, float(saved_pri_min))
    cfg["pri_min"] = pri_min

    saved_pri_max = cfg.get("pri_max", 6000.0)
    pri_max = st.number_input("PRI Max (µs)", 2.0, 20000.0, float(saved_pri_max))
    cfg["pri_max"] = pri_max

    saved_pw_min = cfg.get("pw_min", 10.0)
    pw_min = st.number_input("Pulse Width Min (µs)", 0.01, 1000.0, float(saved_pw_min))
    cfg["pw_min"] = pw_min

    saved_pw_max = cfg.get("pw_max", 50.0)
    pw_max = st.number_input("Pulse Width Max (µs)", 0.01, 1000.0, float(saved_pw_max))
    cfg["pw_max"] = pw_max

    saved_amp_min = cfg.get("amp_min", -80.0)
    amp_min = st.number_input("Amplitude Min (dB)", -200.0, 10.0, float(saved_amp_min))
    cfg["amp_min"] = amp_min

    saved_amp_max = cfg.get("amp_max", -30.0)
    amp_max = st.number_input("Amplitude Max (dB)", -200.0, 10.0, float(saved_amp_max))
    cfg["amp_max"] = amp_max

    saved_doa_min = cfg.get("doa_min", 0.0)
    doa_min = st.number_input("DOA Min (deg)", 0.0, 360.0, float(saved_doa_min))
    cfg["doa_min"] = doa_min

    saved_doa_max = cfg.get("doa_max", 360.0)
    doa_max = st.number_input("DOA Max (deg)", 0.0, 360.0, float(saved_doa_max))
    cfg["doa_max"] = doa_max

    # -----------------------------
    # SIMULATION CONTROL
    # -----------------------------

    st.subheader("Simulation Control")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("▶ Start / Generate"):
            st.session_state.auto_running = True
            # DO NOT clear buffer here. We want to append.
            # Emitter generation logic is handled below.

    with c2:
        if st.button("⏸ Pause"):
            st.session_state.auto_running = False

    with c3:
        if st.button("🔴 Reset Simulation & Clear Data", type="primary", help="Clears all generated data, resets emitters, and resets configuration."):
            st.session_state.auto_running = False
            st.session_state.global_time_us = get_current_time_us()
            st.session_state.pdw_buffer = []
            if "active_emitters" in st.session_state:
                del st.session_state.active_emitters # Clear persisted emitters
            st.session_state.auto_config.clear()
            st.rerun()

    # -----------------------------
    # GENERATE PDWs
    # -----------------------------
    if st.session_state.auto_running:

        # 1. Check if we have active emitters. 
        #    Regenerate if N or Distribution percentages have changed.
        current_config_sig = (num_emitters, fixed_pct, agile_pct, stagger_pct)
        
        saved_sig = st.session_state.get("active_emitters_sig")
        
        if "active_emitters" in st.session_state:
            # If config changed or count mismatch, clear state
            if saved_sig != current_config_sig:
                del st.session_state.active_emitters

        if "active_emitters" not in st.session_state:
            st.session_state.active_emitters = generate_emitters_config(
                num_emitters,
                fixed_pct, agile_pct, stagger_pct,
                f_min, f_max, pri_min, pri_max,
                pw_min, pw_max, amp_min, amp_max,
                doa_min, doa_max
            )
            # Save the signature of the current generation
            st.session_state.active_emitters_sig = current_config_sig
            # print(f"Generated {len(st.session_state.active_emitters)} new emitters.")

        # 2. Generate PDWs using the ACTIVE emitters (reusing them)
        df_new = generate_pdws_from_emitters(
            st.session_state.active_emitters,
            pulses_per_emitter
        )

        out_dir = st.session_state.get("user_output_dir", OUTPUT_DIR)
        st.session_state.pdw_buffer.extend(df_new.to_dict("records"))

        df_all = pd.DataFrame(st.session_state.pdw_buffer)
        df_all = df_all.sort_values("toa_us").round(2)

        df_all.to_csv(f"{out_dir}/pdw_interleaved.csv", index=False)

        st.session_state.auto_running = False
        st.success(f"Generated 2 seconds of PDWs. Total Data Points: {len(df_all)}. Detected Emitters will remain consistent.")
        st.dataframe(df_all.tail(20))




# =================================================
# EMITTER GENERATION (PERSISTABLE)
# =================================================
def generate_emitters_config(num_emitters,
                             fixed_pct, agile_pct, stagger_pct,
                             f_min, f_max, pri_min, pri_max,
                             pw_min, pw_max, amp_min, amp_max,
                             doa_min, doa_max):
    """
    Generates a list of emitter configurations.
    Called ONCE at the start of a simulation sequence.
    """
    emitters = []

    n_fixed = int(num_emitters * fixed_pct / 100)
    n_agile = int(num_emitters * agile_pct / 100)
    n_stagger = num_emitters - n_fixed - n_agile

    emitter_types = (
        ["fixed"] * n_fixed +
        ["agile"] * n_agile +
        ["stagger"] * n_stagger
    )
    # Fill remaining if rounding errors
    while len(emitter_types) < num_emitters:
        emitter_types.append("fixed")
        
    np.random.shuffle(emitter_types)

    # Distribute base parameters
    base_freqs = np.linspace(f_min, f_max, num_emitters + 2)[1:-1]
    np.random.shuffle(base_freqs)
    
    base_pris = np.linspace(pri_min, pri_max, num_emitters + 2)[1:-1]
    np.random.shuffle(base_pris)

    for i, etype in enumerate(emitter_types):
        
        base_freq = base_freqs[i] + np.random.uniform(-20, 20) 
        base_pri  = base_pris[i]  + np.random.uniform(-50, 50)
        
        base_pw   = np.random.uniform(pw_min, pw_max)
        base_amp  = np.random.uniform(amp_min, amp_max)
        base_doa  = np.random.uniform(doa_min, doa_max)

        # Config based on Type
        # Agile: Random selection from 3-5 modes
        freq_modes = [base_freq]
        if etype == "agile":
            n_modes = np.random.randint(3, 6)
            # Spread modes around base (Reduced to +/- 50 MHz to ensure separability)
            offsets = np.linspace(-50, 50, n_modes)
            freq_modes = [base_freq + o for o in offsets]
        
        # Staggered: Sequence of 3-5 PRI steps
        pri_modes = [base_pri]
        if etype == "stagger":
            n_steps = np.random.randint(3, 6)
            # Spread sequence around base (Reduced to 5% to ensure separability)
            start_offset = 0.05 * base_pri
            offsets = np.linspace(-start_offset, start_offset, n_steps)
            pri_modes = [base_pri + o for o in offsets]

        # Initial State Tracking
        # Random start within first PRI interval to desynchronize
        next_toa = st.session_state.global_time_us + np.random.uniform(0, max(pri_modes)*2)

        emitters.append({
            "id": i + 1,
            "type": etype,
            "freq_modes": freq_modes,
            "pri_modes": pri_modes,
            "pw": base_pw,
            "amp": base_amp,
            "doa": base_doa,
            # Mutable State
            "next_toa": next_toa,
            "pulse_idx": 0
        })
        
    return emitters


def generate_pdws_from_emitters(emitters, pulses_per_emitter):
    """
    Generates PDWs for a 2-second window using strict time continuity.
    """
    rows = []

    window_start = st.session_state.global_time_us
    window_end = window_start + 2e6 # 2 seconds
    st.session_state.global_time_us = window_end

    for em in emitters:
        
        # Exact Pulse Count Generation
        for _ in range(pulses_per_emitter):
            
            # --- 1. Parameter Selection based on Type ---
            
            # FREQUENCY
            if em["type"] == "agile":
                # Pick random from modes
                f_center = np.random.choice(em["freq_modes"])
            else:
                f_center = em["freq_modes"][0]
            
            # Add small noise (jitter) to Frequency irrespective of type
            freq = f_center + np.random.normal(0, 0.5) 

            # PRI (Determines NEXT TOA)
            if em["type"] == "stagger":
                # Cyclic walk through modes
                idx = em["pulse_idx"] % len(em["pri_modes"])
                p_center = em["pri_modes"][idx]
            else:
                p_center = em["pri_modes"][0]

            # Add jitter to PRI (Simulate real hardware noise)
            # 1-3% jitter is common
            pri_jitter = np.random.normal(0, 0.01 * p_center)
            pri = p_center + pri_jitter
            
            # OTHER PARAMS (Noise added)
            pw  = em["pw"] + np.random.normal(0, 0.05)
            amp = em["amp"] + np.random.normal(0, 0.5)
            doa = em["doa"] + np.random.normal(0, 0.2)

            # --- 2. Store Pulse ---
            rows.append({
                "freq_MHz": freq,
                "pri_us": pri,
                "pw_us": pw,
                "doa_deg": doa,
                "amp_dB": amp,
                "toa_us": em["next_toa"],
                "Emitter_ID": em["id"] # Ground Truth for validation
            })

            # --- 3. Update State ---
            em["next_toa"] += pri
            em["pulse_idx"] += 1

    return pd.DataFrame(rows)
