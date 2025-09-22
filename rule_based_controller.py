# # rule_based_controller.py

# def choose_action(lane_counts):
#     """
#     Simple rule: pick the lane with the most vehicles.
#     """
#     if not lane_counts:
#         return 0
#     max_count = max(lane_counts)
#     return lane_counts.index(max_count)


import time

last_switch_time = time.time()
green_duration = 10  # seconds
current_green_lane = 0  # start with lane 0

def choose_action(lane_counts):
    global last_switch_time, current_green_lane
    now = time.time()

    if now - last_switch_time > green_duration:
        # Find next lane with most traffic
        if not lane_counts:
            return 0
        max_count = max(lane_counts)
        best_lanes = [i for i, count in enumerate(lane_counts) if count == max_count]
        current_green_lane = best_lanes[0]  # pick the first
        last_switch_time = now

    return current_green_lane
