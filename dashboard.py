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

import streamlit as st
import numpy as np
from stable_baselines3 import PPO
from train_traffic_agent import TrafficEnv  # Use same environment

st.set_page_config(page_title="Smart Traffic Dashboard", layout="wide")
st.title("🚦 Smart Traffic Management Dashboard")

# ------------------------
# Login / Signup
# ------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_option = st.radio("Choose an option", ["Login", "Sign Up"])
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Submit"):
        if username and password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
        else:
            st.error("Please enter both username and password.")
else:
    st.write(f"Logged in as: {st.session_state.username} ✅")

    # ------------------------
    # Initialize Environment
    # ------------------------
    if "env" not in st.session_state:
        st.session_state.env = TrafficEnv()
        state, _ = st.session_state.env.reset()
        st.session_state.state = state

    env = st.session_state.env
    state = st.session_state.state

    # ------------------------
    # Load RL Agent
    # ------------------------
    try:
        if "model" not in st.session_state:
            st.session_state.model = PPO.load("traffic_rl_agent")
        model = st.session_state.model
        st.success("✅ RL agent loaded successfully")
    except Exception:
        st.warning("⚠️ RL agent not found. Run train_traffic_agent.py first.")
        model = None

    # ------------------------
    # Show Current Queue Lengths
    # ------------------------
    st.subheader("📊 Current Queue Lengths (vehicles per lane)")
    lane_names = ["Lane 1", "Lane 2", "Lane 3", "Lane 4"]
    for i, lane in enumerate(lane_names):
        st.metric(label=lane, value=int(st.session_state.state[i]))

    # ------------------------
    # RL Agent Suggestion
    # ------------------------
    if model:
        action, _ = model.predict(state)
        st.subheader("Recommended Green Signal Lane by RL Agent")
        st.write(f"Lane {action + 1}")
    else:
        action = 0
        st.info("RL agent not loaded. Using default lane 1.")

    # ------------------------
    # Next Step Simulation
    # ------------------------
    if st.button("Next Step"):
        next_state, reward, done, truncated, info = env.step(action)
        st.session_state.state = next_state

        st.subheader("🚦 Updated Vehicle Counts After Action")
        for i, lane in enumerate(lane_names):
            st.metric(label=lane, value=int(next_state[i]))
        st.write("Reward:", reward)

        if done:
            st.info("Episode finished. Resetting environment...")
            state, _ = env.reset()
            st.session_state.state = state
