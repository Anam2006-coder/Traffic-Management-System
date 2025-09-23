# import streamlit as st
# import sqlite3
# import hashlib
# import numpy as np
# import pandas as pd
# from stable_baselines3 import PPO
# from traffic_env import TrafficEnv

# # ---------------- AUTH SETUP ---------------- #
# def get_db_connection():
#     conn = sqlite3.connect("users.db")
#     conn.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)")
#     return conn

# def hash_password(password):
#     return hashlib.sha256(password.encode()).hexdigest()

# def signup(username, password):
#     conn = get_db_connection()
#     try:
#         conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
#         conn.commit()
#         return True
#     except:
#         return False

# def login(username, password):
#     conn = get_db_connection()
#     cur = conn.cursor()
#     cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hash_password(password)))
#     return cur.fetchone() is not None

# # ---------------- DASHBOARD ---------------- #
# def traffic_dashboard():
#     st.title("🚦 Smart Traffic Management Dashboard")

#     # Load env + model
#     env = TrafficEnv()
#     model = PPO.load("traffic_rl_agent")

#     if "state" not in st.session_state:
#         st.session_state.state = env.reset()
#     if "history" not in st.session_state:
#         st.session_state.history = []

#     col1, col2 = st.columns(2)

#     # Show queues
#     with col1:
#         st.subheader("📊 Current Queue Lengths")
#         lanes = ["North", "East", "South", "West"]
#         for i, lane in enumerate(lanes):
#             st.metric(label=lane, value=int(st.session_state.state[i]))

#     # Show action
#     with col2:
#         st.subheader("🤖 RL Agent Decision")
#         action, _ = model.predict(st.session_state.state)
#         st.write(f"Next action: **{'NS Green (0)' if action == 0 else 'EW Green (1)'}**")

#     # Step button
#     if st.button("▶ Next Step"):
#         action, _ = model.predict(st.session_state.state)
#         new_state, reward, done, _ = env.step(action)
#         st.session_state.history.append(new_state.tolist())
#         st.session_state.state = new_state

#         if done:
#             st.warning("Simulation finished! Resetting environment...")
#             st.session_state.state = env.reset()
#             st.session_state.history = []

#     # Chart
#     st.subheader("📈 Traffic Queue Trends")
#     if st.session_state.history:
#         df = pd.DataFrame(st.session_state.history, columns=["North", "East", "South", "West"])
#         st.line_chart(df)
#     else:
#         st.info("Run the simulation to see queue trends over time.")


# # ---------------- APP ENTRY ---------------- #
# st.set_page_config(page_title="Traffic Dashboard", layout="wide")

# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False
# if "signup_mode" not in st.session_state:
#     st.session_state.signup_mode = False

# if not st.session_state.logged_in:
#     st.title("🔐 Login / Signup")

#     if st.session_state.signup_mode:
#         st.subheader("Create a new account")
#         new_user = st.text_input("Username")
#         new_pass = st.text_input("Password", type="password")
#         if st.button("Sign Up"):
#             if signup(new_user, new_pass):
#                 st.success("✅ Account created! Please login now.")
#                 st.session_state.signup_mode = False
#             else:
#                 st.error("❌ Username already exists. Try another.")

#         if st.button("Back to Login"):
#             st.session_state.signup_mode = False

#     else:
#         st.subheader("Login to your account")
#         username = st.text_input("Username")
#         password = st.text_input("Password", type="password")
#         if st.button("Login"):
#             if login(username, password):
#                 st.session_state.logged_in = True
#                 st.success("✅ Logged in successfully!")
#             else:
#                 st.error("❌ Invalid credentials")

#         if st.button("Sign Up"):
#             st.session_state.signup_mode = True
# else:
#     traffic_dashboard()
#     if st.button("Logout"):
#         st.session_state.logged_in = False




# dashboard.py

# import streamlit as st
# import numpy as np
# from stable_baselines3 import PPO
# from train_traffic_agent import TrafficEnv  # Use same environment

# st.set_page_config(page_title="Smart Traffic Dashboard", layout="wide")
# st.title("🚦 Smart Traffic Management Dashboard")

# # ------------------------
# # Login / Signup
# # ------------------------
# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False

# if not st.session_state.logged_in:
#     login_option = st.radio("Choose an option", ["Login", "Sign Up"])
    
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
    
#     if st.button("Submit"):
#         if username and password:
#             st.session_state.logged_in = True
#             st.session_state.username = username
#             st.success(f"Welcome, {username}!")
#         else:
#             st.error("Please enter both username and password.")
# else:
#     st.write(f"Logged in as: {st.session_state.username} ✅")

#     # ------------------------
#     # Initialize Environment
#     # ------------------------
#     if "env" not in st.session_state:
#         st.session_state.env = TrafficEnv()
#         state, _ = st.session_state.env.reset()
#         st.session_state.state = state

#     env = st.session_state.env
#     state = st.session_state.state

#     # ------------------------
#     # Load RL Agent
#     # ------------------------
#     try:
#         if "model" not in st.session_state:
#             st.session_state.model = PPO.load("traffic_rl_agent")
#         model = st.session_state.model
#         st.success("✅ RL agent loaded successfully")
#     except Exception:
#         st.warning("⚠️ RL agent not found. Run train_traffic_agent.py first.")
#         model = None

#     # ------------------------
#     # Show Current Queue Lengths
#     # ------------------------
#     st.subheader("📊 Current Queue Lengths (vehicles per lane)")
#     lane_names = ["Lane 1", "Lane 2", "Lane 3", "Lane 4"]
#     for i, lane in enumerate(lane_names):
#         st.metric(label=lane, value=int(st.session_state.state[i]))

#     # ------------------------
#     # RL Agent Suggestion
#     # ------------------------
#     if model:
#         action, _ = model.predict(state)
#         st.subheader("Recommended Green Signal Lane by RL Agent")
#         st.write(f"Lane {action + 1}")
#     else:
#         action = 0
#         st.info("RL agent not loaded. Using default lane 1.")

#     # ------------------------
#     # Next Step Simulation
#     # ------------------------
#     if st.button("Next Step"):
#         next_state, reward, done, truncated, info = env.step(action)
#         st.session_state.state = next_state

#         st.subheader("🚦 Updated Vehicle Counts After Action")
#         for i, lane in enumerate(lane_names):
#             st.metric(label=lane, value=int(next_state[i]))
#         st.write("Reward:", reward)

#         if done:
#             st.info("Episode finished. Resetting environment...")
#             state, _ = env.reset()
#             st.session_state.state = state






import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict, deque
import time
import pandas as pd

# ---------------- CONFIG ---------------- #
MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "Traffic1.mp4"

model = YOLO(MODEL_PATH)

LANES = {
    "Lane 1": ((100, 200), (400, 600)),
    "Lane 2": ((500, 200), (800, 600)),
}

EMERGENCY_CLASSES = ["ambulance"]

WATER_ZONES = [
    ((200, 500), (600, 580)),
]

# ---------------- DASHBOARD ---------------- #
st.title("Traffic Management Dashboard")

cap = cv2.VideoCapture(VIDEO_PATH)

vehicle_count_placeholder = st.empty()
queue_length_placeholder = st.empty()
alert_placeholder = st.empty()
video_placeholder = st.empty()
congestion_chart_placeholder = st.empty()

# Keep historical counts for congestion chart (deque for sliding window)
history_length = 50  # last 50 frames
lane_history = {lane: deque(maxlen=history_length) for lane in LANES.keys()}

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    vehicle_counts = defaultdict(int)
    queue_lengths = defaultdict(int)
    emergency_detected = False
    emergency_lane = None
    water_detected = False

    # Waterlogged zones
    for (wx1, wy1), (wx2, wy2) in WATER_ZONES:
        zone = frame[wy1:wy2, wx1:wx2]
        if np.mean(zone[:, :, 0]) > 100:
            water_detected = True
        cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (255, 0, 0), 2)
        cv2.putText(frame, "Waterlogged", (wx1, wy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Process detected vehicles
    for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
        cls_name = model.names[int(cls_id)]
        x1, y1, x2, y2 = map(int, box)

        if cls_name.lower() in EMERGENCY_CLASSES:
            emergency_detected = True
            for lane_name, ((lx1, ly1), (lx2, ly2)) in LANES.items():
                if lx1 <= x1 <= lx2 and ly1 <= y1 <= ly2:
                    emergency_lane = lane_name
                    break

        for lane_name, ((lx1, ly1), (lx2, ly2)) in LANES.items():
            if lx1 <= x1 <= lx2 and ly1 <= y1 <= ly2:
                vehicle_counts[lane_name] += 1
                queue_lengths[lane_name] = max(queue_lengths[lane_name], y2 - ly1)

        color = (0, 0, 255) if cls_name.lower() in EMERGENCY_CLASSES else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, cls_name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Ambulance priority
    if emergency_detected and emergency_lane:
        for lane_name in vehicle_counts.keys():
            if lane_name != emergency_lane:
                vehicle_counts[lane_name] = max(vehicle_counts[lane_name] - 2, 0)

    # Update history for congestion chart
    for lane, count in vehicle_counts.items():
        lane_history[lane].append(count)

    # ---------------- DASHBOARD UPDATES ---------------- #
    vehicle_count_placeholder.table(vehicle_counts)
    queue_length_placeholder.table(queue_lengths)

    if emergency_detected and water_detected:
        alert_placeholder.warning(f"🚨 Emergency in {emergency_lane}! Also waterlogged area ahead!")
    elif emergency_detected:
        alert_placeholder.warning(f"🚨 Emergency vehicle detected in {emergency_lane}! Clear the path.")
    elif water_detected:
        alert_placeholder.warning("⚠️ Waterlogged area detected! Avoid this lane.")
    else:
        alert_placeholder.info("Traffic normal. No emergencies or waterlogging detected.")

    # Show video
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame_rgb)

    # Draw congestion chart
    df = pd.DataFrame({lane: list(lane_history[lane]) for lane in LANES.keys()})
    congestion_chart_placeholder.line_chart(df)

    time.sleep(0.05)

cap.release()
st.success("Video processing completed.")





