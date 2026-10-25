import ctypes
import os
import time
from datetime import datetime
import numpy as np

# In case the user doesn't have picosdk installed
try:
    pico_path = r"C:\Program Files\Pico Technology\PicoScope 7 T&M Stable"
    if os.path.exists(pico_path):
        os.environ["PATH"] = pico_path + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(pico_path)

    from picosdk.ps2000 import ps2000 as ps
    from picosdk.functions import assert_pico2000_ok
except ImportError:
    print("Error: The 'picosdk' library was not found.")
    print("Please install it: pip install picosdk")
    print("Also ensure the PicoSDK C drivers (e.g., PicoScope 6/7) are installed.")
    exit(1)

def main():
    # ---------------------------
    # Configuration
    # ---------------------------
    import json
    import sqlite3
    
    config = {
        "N_SAMPLES": 12500,
        "TIMEBASE": 2,
        "TRIGGER_MV": 20.0,
        "CH_RANGE": 7,
        "PRE_TRIGGER_PCT": 40.0,
        "MIN_FREQUENCY_KHZ": 30.0
    }
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                config.update(json.load(f))
        except Exception as e:
            print("Could not load config:", e)

    N_SAMPLES = int(config["N_SAMPLES"])
    TIMEBASE = int(config["TIMEBASE"])
    TRIGGER_MV = float(config["TRIGGER_MV"])
    CH_RANGE = int(config["CH_RANGE"])
    PRE_TRIGGER_PCT = float(config["PRE_TRIGGER_PCT"])
    MIN_FREQUENCY_KHZ = float(config.get("MIN_FREQUENCY_KHZ", 30.0))

    # Range setting:
    range_maxes = {3: 0.1, 4: 0.2, 5: 0.5, 6: 1.0, 7: 2.0, 8: 5.0, 9: 10.0, 10: 20.0}
    MAX_V = range_maxes.get(CH_RANGE, 2.0)
    
    # Max ADC count for 2000 series
    MAX_ADC = 32767
    
    # Prefix for saved files
    TEST_NAME = "measurement"
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # ---------------------------
    # Setup PicoScope
    # ---------------------------
    status = {}
    
    print("Opening PicoScope 2205A (ps2000 driver)...")
    status["openunit"] = ps.ps2000_open_unit()
    assert_pico2000_ok(status["openunit"])
    chandle = status["openunit"]

    try:
        # 1. Setup Channel A
        status["setChA"] = ps.ps2000_set_channel(chandle, 0, 1, 1, CH_RANGE)
        assert_pico2000_ok(status["setChA"])
        
        # Disable Channel B (to ensure full speed & memory availability)
        status["setChB"] = ps.ps2000_set_channel(chandle, 1, 0, 1, CH_RANGE)
        assert_pico2000_ok(status["setChB"])

        # 2. Setup Trigger
        trigger_adc = int((TRIGGER_MV / 1000.0) / MAX_V * MAX_ADC)
        # Delay is -100 to +100 percentage. A 40% pre-trigger means trigger event is 40% into the block,
        # so delay from trigger to start of block is -40%.
        trigger_delay = -int(PRE_TRIGGER_PCT)
        status["trigger"] = ps.ps2000_set_trigger(chandle, 0, trigger_adc, 0, trigger_delay, 0)
        assert_pico2000_ok(status["trigger"])

        # 3. Verify Timebase
        timeInterval_ns = ctypes.c_int32()
        timeUnits = ctypes.c_int16()
        maxSamples = ctypes.c_int32()
        
        status["getTimebase"] = ps.ps2000_get_timebase(
            chandle, TIMEBASE, N_SAMPLES, ctypes.byref(timeInterval_ns), ctypes.byref(timeUnits), 1, ctypes.byref(maxSamples)
        )
        assert_pico2000_ok(status["getTimebase"])
        
        dt_val = timeInterval_ns.value
        # For the PS2000 C API, timeInterval is natively returned in nanoseconds. Ignore timeUnits.
        dt_ns = float(dt_val)
        
        print(f"Timebase {TIMEBASE} configured. Sample interval: {dt_ns} ns ({1000/dt_ns:.2f} MS/s).")

        # ---------------------------
        # Measurement Loop
        # ---------------------------
        print("\nReady! Waiting for trigger (>= 20mV)... Press Ctrl+C to exit.")
        
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metadata.db")
        try:
            # Assuming data_collection.py handles init now, or just let it be.
            # We skip importing app directly.
            pass
        except Exception as e:
            print("Warning: Failed to check database:", e)
            
        stop_flag_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stop_measurement.flag")

        while True:
            if os.path.exists(stop_flag_path):
                print("Stop signal received. Exiting...")
                break

            timeIndisposedms = ctypes.c_int32()
            status["runBlock"] = ps.ps2000_run_block(
                chandle, N_SAMPLES, TIMEBASE, 1, ctypes.byref(timeIndisposedms)
            )
            assert_pico2000_ok(status["runBlock"])

            ready = ctypes.c_int16(0)
            while ready.value == 0:
                if os.path.exists(stop_flag_path):
                    break
                ready.value = ps.ps2000_ready(chandle)
                time.sleep(0.005)
                
            if ready.value == 0:
                continue

            # Retrieve data
            bufferA = (ctypes.c_int16 * N_SAMPLES)()
            bufferB = (ctypes.c_int16 * N_SAMPLES)()
            bufferC = (ctypes.c_int16 * N_SAMPLES)()
            bufferD = (ctypes.c_int16 * N_SAMPLES)()
            overflow = ctypes.c_int16(0)
            
            cmaxSamples = ctypes.c_int32(N_SAMPLES)
            
            status["getValues"] = ps.ps2000_get_values(
                chandle, ctypes.byref(bufferA), ctypes.byref(bufferB), ctypes.byref(bufferC), ctypes.byref(bufferD),
                ctypes.byref(overflow), cmaxSamples
            )
            assert_pico2000_ok(status["getValues"])

            base_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            timestamp = base_timestamp
            counter = 1
            while os.path.exists(os.path.join(out_dir, f"{timestamp}-{TEST_NAME}-scope.csv")):
                timestamp = f"{base_timestamp}_{counter}"
                counter += 1
                
            uhrzeit = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            adc_values = np.ctypeslib.as_array(bufferA)
            V_values = (adc_values.astype(np.float32) / MAX_ADC) * MAX_V
            
            # Form Time axis in ms
            t_ms = np.linspace(0, (N_SAMPLES - 1) * dt_ns * 1e-6, N_SAMPLES)
            # Shift time axis so trigger point (0) is correctly placed based on PRE_TRIGGER_PCT
            trigger_time_offset_ms = (N_SAMPLES * (PRE_TRIGGER_PCT / 100.0)) * dt_ns * 1e-6
            t_ms = t_ms - trigger_time_offset_ms
            
            fft_out = np.fft.rfft(V_values)
            fft_mag = np.abs(fft_out) / N_SAMPLES * 2 
            fft_mag[0] /= 2.0 
            
            V_rms = fft_mag / np.sqrt(2)
            V_rms_clipped = np.clip(V_rms, 1e-12, None)
            dBu = 20 * np.log10(V_rms_clipped / 0.774596669)
            
            dt_s = dt_ns * 1e-9
            freqs_Hz = np.fft.rfftfreq(N_SAMPLES, d=dt_s)
            freqs_MHz = freqs_Hz / 1e6
            
            # The user requested max frequency 1.0 MHz
            valid_idx = freqs_MHz <= 1.0
            freqs_MHz_filtered = freqs_MHz[valid_idx]
            dBu_filtered = dBu[valid_idx]
            
            # Compute dominant frequency (skip DC)
            dom_freq = float(freqs_MHz[np.argmax(fft_mag[1:]) + 1])
            dom_freq_khz = dom_freq * 1000.0
            if dom_freq_khz < MIN_FREQUENCY_KHZ:
                continue

            scope_file = os.path.join(out_dir, f"{timestamp}-{TEST_NAME}-scope.csv")
            spectrum_file = os.path.join(out_dir, f"{timestamp}-{TEST_NAME}-spectrum.csv")
            
            np.savetxt(scope_file, np.column_stack((t_ms, V_values)), 
                       delimiter=',', header='Time (ms),Voltage (V)', comments='', fmt='%.6f')
            np.savetxt(spectrum_file, np.column_stack((freqs_MHz_filtered, dBu_filtered)), 
                       delimiter=',', header='Frequency (MHz),Magnitude (dBu)', comments='', fmt='%.6f')
            
            # Automatic features
            trigger_amplitude = TRIGGER_MV / 1000.0
            peak_amplitude = float(np.max(np.abs(V_values)))
            energy = float(np.sum(V_values**2) * dt_s)
            
            # Positive threshold crossings
            crossings = np.where((V_values[:-1] < trigger_amplitude) & (V_values[1:] >= trigger_amplitude))[0]
            counts = len(crossings)
            duration = float((crossings[-1] - crossings[0]) * dt_s) if counts > 1 else 0.0
            dom_freq = float(freqs_MHz[np.argmax(fft_mag[1:]) + 1]) # skip DC
            try:
                conn = sqlite3.connect(db_path, timeout=20.0)
                conn.row_factory = sqlite3.Row
                c = conn.cursor()

                c.execute('SELECT * FROM session_metadata WHERE id=1')
                row_raw = c.fetchone()
                
                # Default dictionary in case no meta is found
                meta_dict = {}
                if row_raw:
                    meta_dict = dict(row_raw)
                    
                # Fetch live data from arduino if explicitly requested
                if meta_dict.get('auto_sensor') and meta_dict.get('arduino_ip'):
                    ip = meta_dict['arduino_ip']
                    try:
                        import requests
                        res = requests.get(f"http://{ip}/metrics", timeout=10)
                        if res.status_code == 200:
                            data = res.json()
                            # Support both old German and new English keys from Arduino firmware
                            if "temperature" in data:
                                meta_dict["temperature"] = float(data["temperature"])
                            elif "temperatur" in data:
                                meta_dict["temperature"] = float(data["temperatur"])
                            if "humidity" in data:
                                meta_dict["humidity"] = float(data["humidity"])
                            elif "luftfeuchte" in data:
                                meta_dict["humidity"] = float(data["luftfeuchte"])
                            if "soil_moisture" in data:
                                meta_dict["soil_moisture"] = float(data["soil_moisture"])
                            elif "bodenfeuchte" in data:
                                meta_dict["soil_moisture"] = float(data["bodenfeuchte"])
                    except Exception as e:
                        print(f"Warning: Could not fetch automatic sensor data from Arduino ({ip}): {e}")
                
                # Build the insert query
                cols = [
                    "recording_id", "event_timestamp", "trigger_amplitude", "peak_amplitude",
                    "duration", "energy", "counts", "dominant_frequency",
                    "plant", "species", "stem_diameter", "size", "leaf_count", "leaf_size",
                    "sensor_position", "contact_pressure", "sensor_type", "gain", "filter",
                    "temperature", "humidity", "soil_moisture", "light",
                    "stress_type", "stress_type_details", "event_type", "started_at", "experiment_id", "experiment_description", "plant_id"
                ]
                
                vals = [
                    timestamp, uhrzeit, trigger_amplitude, peak_amplitude,
                    duration, energy, counts, dom_freq,
                    meta_dict.get("plant"), meta_dict.get("species"), meta_dict.get("stem_diameter"),
                    meta_dict.get("size"), meta_dict.get("leaf_count"), meta_dict.get("leaf_size"),
                    meta_dict.get("sensor_position"), meta_dict.get("contact_pressure"), meta_dict.get("sensor_type"),
                    meta_dict.get("gain"), meta_dict.get("filter"),
                    meta_dict.get("temperature"), meta_dict.get("humidity"), meta_dict.get("soil_moisture"),
                    meta_dict.get("light"), meta_dict.get("stress_type"), meta_dict.get("stress_type_details"), meta_dict.get("event_type"),
                    meta_dict.get("started_at"), meta_dict.get("experiment_id"), meta_dict.get("experiment_description"), meta_dict.get("plant_id")
                ]
                
                placeholders = ", ".join(["?"] * len(cols))
                insert_sql = f"INSERT INTO recordings ({', '.join(cols)}) VALUES ({placeholders})"
                
                c.execute(insert_sql, tuple(vals))
                conn.commit()
                
                # --- ML Live Prediction ---
                try:
                    from ml_pipeline import predict_live
                    base_features_dict = {'duration': duration, 'counts': counts, 'dominant_frequency': dom_freq}
                    preds = predict_live(timestamp, base_features_dict)
                    
                    if preds:
                        update_cols = []
                        update_vals = []
                        for k, v in preds.items():
                            update_cols.append(f"{k} = ?")
                            update_vals.append(v)
                            
                        if update_cols:
                            update_vals.append(timestamp)
                            update_sql = f"UPDATE recordings SET {', '.join(update_cols)} WHERE recording_id = ?"
                            c.execute(update_sql, tuple(update_vals))
                            conn.commit()
                            print(f"   [ML] Noise Prob: {preds.get('noise_probability', 'N/A')} | Plant: {preds.get('pred_plant', 'N/A')} | Stress: {preds.get('pred_stress', 'N/A')}")
                except Exception as ml_err:
                    print("ML Prediction Error:", ml_err)

                conn.close()
            except Exception as e:
                print("Failed to save automatic metadata to db:", e)

            print(f"[{timestamp}] Triggered! Saved -> {os.path.basename(scope_file)} & {os.path.basename(spectrum_file)}")
            
    except KeyboardInterrupt:
        print("\nStopping continuous measurement...")
        
    finally:
        status["stop"] = ps.ps2000_stop(chandle)
        status["close"] = ps.ps2000_close_unit(chandle)
        print("PicoScope closed gracefully.")

if __name__ == "__main__":
    main()
