import cv2
import numpy as np
from ultralytics import YOLO

# Path to YOLOv8 model (smallest one for speed)
MODEL_PATH = "yolov8n.pt"

class VehicleLaneCounter:
    def _init_(self, model_path=MODEL_PATH):
        self.model = YOLO(model_path)

    def detect_vehicles(self, frame):
        results = self.model.predict(frame, verbose=False)
        detections = []

        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = b
                # Optional: filter for vehicle classes (car=2, motorcycle=3, bus=5, truck=7, etc.)
                if int(cls) in [2, 3, 5, 7]:
                    detections.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)))
        return detections

    def detect_lanes(self, frame):
        """Detect lane lines using Canny + Hough and overlay them."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        height, width = frame.shape[:2]
        mask = np.zeros_like(edges)

        # ROI: bottom 40% of the image
        polygon = np.array([[
            (0, height),
            (width, height),
            (width, int(height * 0.6)),
            (0, int(height * 0.6))
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Hough Transform for line detection
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=50
        )

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def draw_vehicle_boxes(self, frame, detections):
        for x1, y1, x2, y2, conf, cls in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def display_vehicle_count(self, frame, count):
        # Large red text displaying the count
        cv2.putText(
            frame,
            f"Detected vehicles: {count}",
            (50, 100),                      # Position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,                            # Font scale
            (0, 0, 255),                    # Red color
            5,                              # Thickness
            cv2.LINE_AA
        )

    def process_frame(self, frame):
        detections = self.detect_vehicles(frame)
        self.detect_lanes(frame)
        self.draw_vehicle_boxes(frame, detections)
        self.display_vehicle_count(frame, len(detections))
        return frame


if __name__ == "_main_":
    detector = VehicleLaneCounter()

    # ⚠ Replace this with the path to your video file
    cap = cv2.VideoCapture(r"C:\Users\moham\OneDrive\Documents\Traffic.mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = detector.process_frame(frame)

        # Display the frame
        cv2.namedWindow("Vehicle & Lane Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Vehicle & Lane Detection", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

class DynamicTrafficController:
    def __init__(self, lane_names=None, cycle_time=60, emergency_green_time=30):
        """
        lane_names: Optional list like ["North", "East", "South", "West"]
        cycle_time: Total cycle time for all green lights (in seconds)
        emergency_green_time: Fixed green time for emergency lane
        """
        self.lane_names = lane_names or []
        self.num_lanes = len(self.lane_names)
        self.lane_traffic = [0] * self.num_lanes
        self.cycle_time = cycle_time
        self.emergency_active = False
        self.emergency_lane = None
        self.emergency_green_time = emergency_green_time

    def update_lane_names(self, names):
        """Dynamically set lane names and initialize counts."""
        self.lane_names = names
        self.num_lanes = len(names)
        self.lane_traffic = [0] * self.num_lanes

    def update_traffic_counts(self, counts):
        """Update vehicle counts for each lane."""
        if len(counts) != self.num_lanes:
            raise ValueError("Mismatch between number of lanes and traffic counts.")
        self.lane_traffic = counts

    def set_emergency(self, lane_index):
        """Trigger emergency mode for a specific lane."""
        if lane_index < 0 or lane_index >= self.num_lanes:
            raise IndexError("Invalid emergency lane index.")
        self.emergency_active = True
        self.emergency_lane = lane_index

    def reset_emergency(self):
        """Disable emergency mode."""
        self.emergency_active = False
        self.emergency_lane = None

    def get_signal_plan(self):
        """Returns green time (in seconds) for each lane."""

        if self.emergency_active:
            # Emergency override: one lane gets fixed green time, others get 0
            return [
                self.emergency_green_time if i == self.emergency_lane else 0
                for i in range(self.num_lanes)
            ]

        total_vehicles = sum(self.lane_traffic)

        if total_vehicles == 0:
            # No vehicles anywhere — all red or fixed minimum time
            return [0] * self.num_lanes

        # Allocate green time proportionally
        signal_plan = [
            (count / total_vehicles) * self.cycle_time
            for count in self.lane_traffic
        ]

        # Ensure total time adds up to exactly cycle_time
        remainder = self.cycle_time - sum(signal_plan)
        signal_plan[-1] += remainder  # adjust last lane

        return [round(t) for t in signal_plan]  # Round to whole seconds for simplicity

    def print_plan(self):
        """Debug printout of the current signal plan."""
        plan = self.get_signal_plan()
        for i, time in enumerate(plan):
            lane = self.lane_names[i] if self.lane_names else f"Lane {i+1}"
            label = " (EMERGENCY)" if self.emergency_active and i == self.emergency_lane else ""
            print(f"{lane}: {time}s{label}")




# import cv2
# import numpy as np
# from ultralytics import YOLO

# # Path to YOLOv8 model
# MODEL_PATH = "yolov8n.pt"


# # ---------------- Vehicle + Lane Detection ---------------- #
# class VehicleLaneCounter:
#     def __init__(self, model_path=MODEL_PATH):
#         self.model = YOLO(model_path)

#     def detect_vehicles(self, frame):
#         results = self.model(frame, verbose=False)[0]  # first result
#         detections = []

#         if results.boxes is not None:
#             for b in results.boxes.data.tolist():
#                 x1, y1, x2, y2, conf, cls = b
#                 if int(cls) in [2, 3, 5, 7]:  # Car, motorcycle, bus, truck
#                     detections.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)))

#         return detections

#     def detect_lanes(self, frame):
#         """Detect lane lines using Canny + Hough and overlay them."""
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (5, 5), 0)
#         edges = cv2.Canny(blur, 50, 150)

#         height, width = frame.shape[:2]
#         mask = np.zeros_like(edges)

#         polygon = np.array([[
#             (0, height),
#             (width, height),
#             (width, int(height * 0.6)),
#             (0, int(height * 0.6))
#         ]], np.int32)
#         cv2.fillPoly(mask, polygon, 255)
#         masked_edges = cv2.bitwise_and(edges, mask)

#         lines = cv2.HoughLinesP(
#             masked_edges,
#             rho=1,
#             theta=np.pi / 180,
#             threshold=100,
#             minLineLength=100,
#             maxLineGap=50
#         )

#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     def draw_vehicle_boxes(self, frame, detections):
#         for x1, y1, x2, y2, conf, cls in detections:
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

#     def display_vehicle_count(self, frame, count):
#         cv2.putText(
#             frame,
#             f"Total vehicles: {count}",
#             (50, 100),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             2.0,
#             (0, 0, 255),
#             4,
#             cv2.LINE_AA
#         )

#     def process_frame(self, frame):
#         detections = self.detect_vehicles(frame)
#         self.detect_lanes(frame)
#         self.draw_vehicle_boxes(frame, detections)
#         self.display_vehicle_count(frame, len(detections))
#         return frame, detections


# # ---------------- Dynamic Traffic Controller ---------------- #
# class DynamicTrafficController:
#     def __init__(self, lane_names=None, cycle_time=60, emergency_green_time=30):
#         self.lane_names = lane_names or []
#         self.num_lanes = len(self.lane_names)
#         self.lane_traffic = [0] * self.num_lanes
#         self.cycle_time = cycle_time
#         self.emergency_active = False
#         self.emergency_lane = None
#         self.emergency_green_time = emergency_green_time

#     def update_lane_names(self, names):
#         self.lane_names = names
#         self.num_lanes = len(names)
#         self.lane_traffic = [0] * self.num_lanes

#     def update_traffic_counts(self, counts):
#         if len(counts) != self.num_lanes:
#             raise ValueError("Mismatch between number of lanes and traffic counts.")
#         self.lane_traffic = counts

#     def set_emergency(self, lane_index):
#         if lane_index < 0 or lane_index >= self.num_lanes:
#             raise IndexError("Invalid emergency lane index.")
#         self.emergency_active = True
#         self.emergency_lane = lane_index

#     def reset_emergency(self):
#         self.emergency_active = False
#         self.emergency_lane = None

#     def get_signal_plan(self):
#         if self.emergency_active:
#             return [
#                 self.emergency_green_time if i == self.emergency_lane else 0
#                 for i in range(self.num_lanes)
#             ]

#         total_vehicles = sum(self.lane_traffic)

#         if total_vehicles == 0:
#             return [0] * self.num_lanes

#         signal_plan = [
#             (count / total_vehicles) * self.cycle_time
#             for count in self.lane_traffic
#         ]

#         remainder = self.cycle_time - sum(signal_plan)
#         signal_plan[-1] += remainder

#         return [round(t) for t in signal_plan]

#     def print_plan(self):
#         plan = self.get_signal_plan()
#         for i, time in enumerate(plan):
#             lane = self.lane_names[i] if self.lane_names else f"Lane {i+1}"
#             label = " (EMERGENCY)" if self.emergency_active and i == self.emergency_lane else ""
#             print(f"{lane}: {time}s{label}")


# # ---------------- Main Execution ---------------- #
# if __name__ == "__main__":
#     detector = VehicleLaneCounter()
#     controller = DynamicTrafficController(lane_names=["Left", "Center", "Right"], cycle_time=60)

#     cap = cv2.VideoCapture(r"C:\Users\moham\OneDrive\Documents\Traffic.mp4")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         output_frame, detections = detector.process_frame(frame)

#         # --- Divide detections into lanes (3 lanes by x-coordinates) ---
#         height, width = frame.shape[:2]
#         lane_counts = [0, 0, 0]  # Left, Center, Right

#         for x1, y1, x2, y2, conf, cls in detections:
#             cx = (x1 + x2) // 2  # center x
#             if cx < width // 3:
#                 lane_counts[0] += 1
#             elif cx < 2 * width // 3:
#                 lane_counts[1] += 1
#             else:
#                 lane_counts[2] += 1

#         print("Lane counts:", lane_counts)

#         # Update controller
#         controller.update_traffic_counts(lane_counts)
#         signal_plan = controller.get_signal_plan()
#         controller.print_plan()

#         # --- Draw lane overlays ---
#         colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255)]  # Different colors
#         lane_width = width // 3
#         for i in range(3):
#             x_start = i * lane_width
#             x_end = (i + 1) * lane_width
#             cv2.rectangle(output_frame, (x_start, 0), (x_end, height),
#                           colors[i], 2)
#             cv2.putText(output_frame, f"{controller.lane_names[i]}: {lane_counts[i]}",
#                         (x_start + 20, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors[i], 3)

#         # --- Draw traffic lights (top of each lane) ---
#         for i in range(3):
#             x_center = (i * lane_width + (i + 1) * lane_width) // 2
#             y_pos = 100  # top margin

#             if signal_plan[i] > 0:  # Green light
#                 cv2.circle(output_frame, (x_center, y_pos), 30, (0, 255, 0), -1)
#                 cv2.putText(output_frame, f"{signal_plan[i]}s",
#                             (x_center - 20, y_pos + 60),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
#             else:  # Red light
#                 cv2.circle(output_frame, (x_center, y_pos), 30, (0, 0, 255), -1)

#         cv2.namedWindow("Vehicle & Lane Detection", cv2.WINDOW_NORMAL)
#         cv2.imshow("Vehicle & Lane Detection", output_frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
