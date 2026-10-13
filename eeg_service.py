# eeg_service.py
import serial
import time
import threading
import numpy as np
from collections import deque
from scipy.signal import butter, lfilter, lfilter_zi, iirnotch

# ================== CONFIG ==================
PORT = "COM7"
BAUD_RATE = 230400
SAMPLE_RATE = 500

SYNC1 = 0xC7
SYNC2 = 0x7C
END_BYTE = 0x01
NUM_CHANNELS = 6
HEADER_LEN = 3
PACKET_LEN = NUM_CHANNELS * 2 + HEADER_LEN + 1

ADC_RES = 16383
VREF = 3.3
AMPLIFIER_GAIN = 1000

X_DISPLAY_SECONDS = 5
BUFFER_SIZE = SAMPLE_RATE * X_DISPLAY_SECONDS

# ================== GLOBALS ==================
buffer_a0 = deque(maxlen=BUFFER_SIZE)
time_a0 = deque(maxlen=BUFFER_SIZE)
running = False
ser = None
thread = None
simulation_mode = False

dc_offset = 0.0
t_sim = 0.0

# ================== FILTERS ==================
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype="band")

b_bp, a_bp = butter_bandpass(0.5, 40, SAMPLE_RATE)
zi_bp = lfilter_zi(b_bp, a_bp) * 0

b_notch, a_notch = iirnotch(50 / (SAMPLE_RATE / 2), Q=30)
zi_notch = lfilter_zi(b_notch, a_notch) * 0

# ================== SIMULATED EEG ==================
def simulated_eeg_sample():
    """
    Generate EEG-like signal (alpha + beta + noise)
    """
    global t_sim
    dt = 1.0 / SAMPLE_RATE
    t_sim += dt

    alpha = 50 * np.sin(2 * np.pi * 10 * t_sim)     # 10 Hz
    beta  = 20 * np.sin(2 * np.pi * 20 * t_sim)     # 20 Hz
    noise = np.random.normal(0, 5)

    return alpha + beta + noise
def read_all_packets():
    """Read all available packets from serial and return a list of A0 µV samples"""
    global dc_offset, zi_bp, zi_notch
    samples = []
    buffer = ser.read(ser.in_waiting)
    print(len(buffer))
    i = 0
    while i + PACKET_LEN <= len(buffer):
        if buffer[i]==SYNC1 and buffer[i+1]==SYNC2 and buffer[i+PACKET_LEN-1]==END_BYTE:
            print(buffer[i])
            high = buffer[i+3]
            low  = buffer[i+4]
            adc = (high<<8) | low
            v = (adc/ADC_RES*VREF)*1e6/AMPLIFIER_GAIN
            # DC removal
            DC_ALPHA = 0.995
            dc_offset = DC_ALPHA*dc_offset + (1-DC_ALPHA)*v
            v_dc = v - dc_offset
            # Bandpass filter
            v_bp, zi_bp = lfilter(b_bp, a_bp, [v_dc], zi=zi_bp)
            # Notch filter
            v_notch, zi_notch = lfilter(b_notch, a_notch, v_bp, zi=zi_notch)
            samples.append(v_notch[0])
            i += PACKET_LEN
            print("Get Data")
            print(v_notch[0])
        else:
            i += 1
   
    return samples

# ================== EEG LOOP ==================
def _eeg_loop():
    global  dc_offset, zi_bp, zi_notch
    

    while running:
        try:
            # ---------- REAL EEG ----------
           
            if not simulation_mode:
                print("Port Open")
                new_samples = read_all_packets()
               
                for s in new_samples:
                    buffer_a0.append(s)
                    if time_a0:
                        time_a0.append(time_a0[-1]+1/SAMPLE_RATE)
                    else:
                        time_a0.append(0)
                

  
            else:
                v = simulated_eeg_sample()
                buffer_a0.append(float(v))
                time.sleep(1 / SAMPLE_RATE)

        except Exception as e:
            print("EEG error:", e)
            time.sleep(0.1)

# ================== API FUNCTIONS ==================
def start_eeg():
    global ser, running, thread, simulation_mode

    if running:
        return

    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=0)
        ser.write(b"START\n")
        running = True
      
        time.sleep(2)
        simulation_mode = False
        print("Connected to EEG device on", PORT)
    except Exception as e:
        print("COM not available → Simulation mode enabled")
        ser = None
        simulation_mode = True

    running = True
    thread = threading.Thread(target=_eeg_loop, daemon=True)
    thread.start()

def stop_eeg():
    global running, ser

    running = False
    if ser and ser.is_open:
        ser.close()

    print("EEG stopped")

def get_latest_samples(n=300):
   
    
    if len(buffer_a0) < n:
       
        return list(buffer_a0)
    
    return list(buffer_a0)[-n:]
