# import cv2
# import numpy as np
# from ultralytics import YOLO
# from rule_based_controller import choose_action  # Make sure this exists
# from tkinter import messagebox, Tk

# MODEL_PATH = "yolov8n.pt"

# class RuleBasedTrafficSignal:
#     def __init__(self, model_path=MODEL_PATH):
#         self.model = YOLO(model_path)
#         self.lane_boundaries = []
#         self.alerted_lanes = set()

#     def detect_vehicles(self, frame):
#         results = self.model.predict(frame, verbose=False)
#         detections = []
#         for r in results:
#             if r.boxes is None:
#                 continue
#             for b in r.boxes.data.tolist():
#                 x1, y1, x2, y2, conf, cls = b
#                 if int(cls) in [2,3,5,7]:  # car, motorcycle, bus, truck
#                     detections.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)))
#         return detections

#     def detect_lanes(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (5,5), 0)
#         edges = cv2.Canny(blur, 50, 150)

#         height, width = frame.shape[:2]
#         mask = np.zeros_like(edges)
#         polygon = np.array([[ (0,height), (width,height), (width,int(height*0.6)), (0,int(height*0.6)) ]], np.int32)
#         cv2.fillPoly(mask, polygon, 255)
#         masked_edges = cv2.bitwise_and(edges, mask)

#         lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=120, minLineLength=100, maxLineGap=50)
#         lane_x = []
#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 if abs(y2 - y1) > 50:
#                     lane_x.append((x1 + x2)//2)
#                     cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 2)

#         lane_x = sorted(list(set(lane_x)))
#         if len(lane_x) >= 2:
#             self.lane_boundaries = [(lane_x[i], lane_x[i+1]) for i in range(len(lane_x)-1)]

#     def assign_vehicles_to_lanes(self, detections):
#         lane_counts = [0]*len(self.lane_boundaries)
#         for x1, y1, x2, y2, conf, cls in detections:
#             cx = (x1 + x2)//2
#             for i, (lx, rx) in enumerate(self.lane_boundaries):
#                 if lx <= cx < rx:
#                     lane_counts[i] += 1
#                     break
#         return lane_counts

#     def draw_vehicle_boxes(self, frame, detections):
#         for x1, y1, x2, y2, conf, cls in detections:
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
#             cv2.putText(frame, f"{conf:.2f}", (x1, y1-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

#     def draw_lane_signal(self, frame, selected_lane):
#         if selected_lane >= len(self.lane_boundaries):
#             return
#         lx, rx = self.lane_boundaries[selected_lane]
#         height = frame.shape[0]
#         cv2.rectangle(frame, (lx, int(height*0.6)), (rx, height), (0,255,0), 4)
#         cv2.putText(frame, f"Green Signal → Lane {selected_lane+1}",
#                     (lx + 10, int(height*0.6) - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

#     def display_counts(self, frame, total_count, lane_counts):
#         cv2.putText(frame, f"Total vehicles: {total_count}",
#                     (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2,
#                     (0,0,255), 5, cv2.LINE_AA)
        
#         for i in range(len(self.lane_boundaries)):
#             count = lane_counts[i]
#             cv2.putText(frame, f"Lane {i+1}: {count}",
#                         (50, 150 + i*60), cv2.FONT_HERSHEY_SIMPLEX,
#                         1.5, (255,255,0), 3, cv2.LINE_AA)

#         # for i, count in enumerate(lane_counts):
#         #     cv2.putText(frame, f"Lane {i+1}: {count}",
#         #                 (50, 150 + i*60), cv2.FONT_HERSHEY_SIMPLEX,
#         #                 1.5, (255,255,0), 3, cv2.LINE_AA)
#         # cv2.putText(frame, f"Lanes detected: {len(self.lane_boundaries)}",
#         #             (50, 150 + len(lane_counts)*60 + 20),
#         #             cv2.FONT_HERSHEY_SIMPLEX, 1.2,
#         #             (0,255,0), 3, cv2.LINE_AA)

#     def check_lane_alerts(self, lane_counts):
#         for i, count in enumerate(lane_counts):
#             if count >= 60 and i not in self.alerted_lanes:
#                 self.alerted_lanes.add(i)
#                 root = Tk()
#                 root.withdraw()
#                 messagebox.showinfo("Traffic Alert",
#                                     f"Lane {i+1} reached 60 vehicles.\nChange signal to GREEN for this lane.")
#                 root.destroy()

#     def process_frame(self, frame):
#         # Fake lane counts for testing rule logic
#         self.detect_lanes(frame)
#         lane_counts = [5, 8, 3, 6]  # You have 4 lanes detected → simulate traffic in each
#         selected_lane = choose_action(lane_counts)
#         detections = []  # Since we are not detecting real vehicles here

#         # detections = self.detect_vehicles(frame)
#         # self.detect_lanes(frame)
#         # lane_counts = self.assign_vehicles_to_lanes(detections) if self.lane_boundaries else []
#         # selected_lane = choose_action(lane_counts) if lane_counts else -1

        

#         self.draw_vehicle_boxes(frame, detections)
#         self.display_counts(frame, len(detections), lane_counts)
#         if selected_lane != -1:
#             self.draw_lane_signal(frame, selected_lane)
#         self.check_lane_alerts(lane_counts)
#         return frame


# if __name__ == "__main__":
#     controller = RuleBasedTrafficSignal()
#     cap = cv2.VideoCapture(r"C:\Users\moham\OneDrive\Documents\Traffic.mp4")

#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cv2.namedWindow("Rule-Based Traffic Signal", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Rule-Based Traffic Signal", width, height)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         output_frame = controller.process_frame(frame)
#         cv2.imshow("Rule-Based Traffic Signal", output_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


import cv2
import numpy as np
from ultralytics import YOLO
from tkinter import messagebox, Tk

# Import your rule-based lane selector
from rule_based_controller import choose_action  

MODEL_PATH = "yolov8n.pt"

class RuleBasedTrafficSignal:
    def __init__(self, model_path=MODEL_PATH):
        self.model = YOLO(model_path)
        self.lane_boundaries = []      # [(x_left, x_right), ...]
        self.alerted_lanes = set()     # Keeps track of alerted lanes

    # ---------------- VEHICLE DETECTION ---------------- #
    def detect_vehicles(self, frame):
        detections = []
        results = self.model.predict(frame, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = b
                if int(cls) in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                    detections.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)))
        return detections

    # ---------------- LANE DETECTION ---------------- #
    def detect_lanes(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        height, width = frame.shape[:2]
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height), (width, height), 
            (width, int(height * 0.6)), (0, int(height * 0.6))
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=120,
                                minLineLength=100, maxLineGap=50)
        lane_x = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) > 50:  # vertical-ish
                    lane_x.append((x1 + x2) // 2)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        lane_x = sorted(set(lane_x))
        if len(lane_x) >= 2:
            self.lane_boundaries = [(lane_x[i], lane_x[i+1]) for i in range(len(lane_x) - 1)]

    # ---------------- LANE ASSIGNMENT ---------------- #
    def assign_vehicles_to_lanes(self, detections):
        lane_counts = [0] * len(self.lane_boundaries)
        for x1, y1, x2, y2, conf, cls in detections:
            cx = (x1 + x2) // 2
            for i, (lx, rx) in enumerate(self.lane_boundaries):
                if lx <= cx < rx:
                    lane_counts[i] += 1
                    break
        return lane_counts

    # ---------------- DRAW HELPERS ---------------- #
    def draw_vehicle_boxes(self, frame, detections):
        for x1, y1, x2, y2, conf, cls in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def draw_lane_signal(self, frame, selected_lane):
        if 0 <= selected_lane < len(self.lane_boundaries):
            lx, rx = self.lane_boundaries[selected_lane]
            height = frame.shape[0]
            cv2.rectangle(frame, (lx, int(height * 0.6)), (rx, height), (0, 255, 0), 4)
            cv2.putText(frame, f"Green → Lane {selected_lane+1}",
                        (lx + 10, int(height * 0.6) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    def display_counts(self, frame, lane_counts):
        total_count = sum(lane_counts)
        cv2.putText(frame, f"Total vehicles: {total_count}",
                    (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        
        for i, count in enumerate(lane_counts):
            cv2.putText(frame, f"Lane {i+1}: {count}",
                        (50, 150 + i * 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255, 255, 0), 3)

    # ---------------- ALERT SYSTEM ---------------- #
    def check_lane_alerts(self, lane_counts, threshold=60):
        for i, count in enumerate(lane_counts):
            if count >= threshold and i not in self.alerted_lanes:
                self.alerted_lanes.add(i)
                root = Tk()
                root.withdraw()
                messagebox.showinfo("Traffic Alert",
                                    f"Lane {i+1} reached {threshold} vehicles.\nSwitch signal to GREEN.")
                root.destroy()

    # ---------------- MAIN PROCESS ---------------- #
    def process_frame(self, frame):
        self.detect_lanes(frame)
        detections = self.detect_vehicles(frame)
        lane_counts = self.assign_vehicles_to_lanes(detections) if self.lane_boundaries else []
        selected_lane = choose_action(lane_counts) if lane_counts else -1

        self.draw_vehicle_boxes(frame, detections)
        self.display_counts(frame, lane_counts)
        if selected_lane != -1:
            self.draw_lane_signal(frame, selected_lane)
        self.check_lane_alerts(lane_counts)

        return frame


if __name__ == "__main__":
    controller = RuleBasedTrafficSignal()
    cap = cv2.VideoCapture(r"C:\Users\moham\OneDrive\Documents\Traffic1.mp4")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow("Rule-Based Traffic Signal", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Rule-Based Traffic Signal", width, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = controller.process_frame(frame)
        cv2.imshow("Rule-Based Traffic Signal", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
