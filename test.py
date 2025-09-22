import cv2
import streamlit as st
from ultralytics import YOLO
import pyttsx3

# ---------------- Vehicle Detector ---------------- #
class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.engine = pyttsx3.init()
        self.emergency_alerted = False  # avoid repeating alerts

    def detect_vehicles(self, frame):
        results = self.model(frame, verbose=False)[0]
        vehicles = []
        emergency_detected = False

        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            cls = int(cls)
            x1, y1, x2, y2 = map(int, box)

            # Regular vehicles: car, motorcycle, bus, truck
            if cls in [2, 3, 5, 7]:
                vehicles.append((x1, y1, x2, y2))

            # Emergency vehicles (demo: bus/truck treated as emergency)
            if cls in [5, 7]:  
                emergency_detected = True

        return vehicles, emergency_detected

    def draw_boxes(self, frame, vehicles, emergency_detected):
        for (x1, y1, x2, y2) in vehicles:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if emergency_detected:
            cv2.putText(frame, "EMERGENCY VEHICLE DETECTED!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, f"Vehicles: {len(vehicles)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        return frame

    def alert_emergency(self):
        if not self.emergency_alerted:
            self.engine.say("Please clear the traffic and give way to the emergency vehicle")
            self.engine.runAndWait()
            self.emergency_alerted = True

# ---------------- Congestion Metrics ---------------- #
def compute_congestion(vehicle_count, lane_capacity=10):
    return min(vehicle_count / lane_capacity, 1.0)

# ---------------- Signal Optimization ---------------- #
def optimize_signals(congestion_scores, base_time=30):
    green_times = {}
    for lane, congestion in congestion_scores.items():
        green_times[lane] = int(base_time * (1 + congestion))  # up to 2x base
    return green_times

# ---------------- Main System ---------------- #
def main():
    st.title("🚦 Smart Traffic Management System")

    # Initialize detector
    detector = VehicleDetector()

    # Upload or use sample video
    video_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])
    if video_file is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.read())
        video_path = "temp_video.mp4"
    else:
        st.warning("Please upload a traffic video to start.")
        return

    cap = cv2.VideoCapture(video_path)

    lane_vehicle_counts = {"Lane 1": 0, "Lane 2": 0}
    frame_no = 0

    while cap.isOpened() and frame_no < 200:  # process limited frames for demo
        ret, frame = cap.read()
        if not ret:
            break

        # Detect vehicles
        vehicles, emergency_detected = detector.detect_vehicles(frame)
        frame = detector.draw_boxes(frame, vehicles, emergency_detected)

        if emergency_detected:
            detector.alert_emergency()

        # Lane split (simple: left/right half)
        mid = frame.shape[1] // 2
        lane_vehicle_counts["Lane 1"] = sum(1 for (x1, _, x2, _) in vehicles if (x1 + x2) // 2 < mid)
        lane_vehicle_counts["Lane 2"] = len(vehicles) - lane_vehicle_counts["Lane 1"]

        # Congestion
        congestion = {lane: compute_congestion(count) for lane, count in lane_vehicle_counts.items()}

        # Signal Optimization
        signal_times = optimize_signals(congestion)

        # Dashboard display
        st.subheader("Vehicle Counts per Lane")
        st.write(lane_vehicle_counts)

        st.subheader("Congestion Scores")
        st.write(congestion)

        st.subheader("Signal Green Times (seconds)")
        st.write(signal_times)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        frame_no += 1

    cap.release()
    st.success("Simulation Finished ✅")

if __name__ == "__main__":
    main()
