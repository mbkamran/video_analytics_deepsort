import cv2
import numpy as np
import time
import torch
import csv
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.experimental import attempt_load
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


class YoloDetector():
    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.classes = ['person']  # Specify the class names manually for your custom model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.start_time = time.time()  # Record start time
        self.interval = 60  # Time interval in seconds
        self.person_count = 0
        self.time_intervals = []
        self.written_track_ids = set()  # Initialize an empty set to keep track of written track_ids



        # Open CSV file in append mode
        self.csv_file = open('./persons_data.csv', mode='a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # Write header if file is empty
        if self.csv_file.tell() == 0:
            self.csv_writer.writerow(['ID', 'Time'])

    def __del__(self):
        # Close CSV file when YoloDetector object is deleted
        self.csv_file.close()

    def load_model(self, model_name):
        model = torch.hub.load(
            "",
            "custom",
            path="./best.pt",
            source="local",
        )
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        downscale_factor = 2
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width, height))

        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame, height, width, confidence=0.5):
        labels, cord = results
        detections = []

        n = len(labels)
        x_shape, y_shape = width, height

        for i in range(n):
            row = cord[i]

            if row[4] >= confidence:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)

                color = (255, 0, 0)  # Blue color
                thickness = 2  # Increase thickness for better visibility

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 2)

                detections.append(([x1, y1, x2 - x1, y2 - y1], row[4].item(), self.class_to_label(labels[i])))
                #print(f"Detected {self.class_to_label(labels[i])} at ({x1}, {y1}) to ({x2}, {y2})")
                self.person_count += 1

        return frame, detections

    def update_counters(self):
        current_time = time.time()
        if current_time - self.start_time >= self.interval:
            self.time_intervals.append((self.start_time, self.person_count))
            self.start_time = current_time
            self.person_count = 0

    def display_results(self):
        for interval_start, count in self.time_intervals:
            interval_end = interval_start + self.interval
            #print(
            #    f"From {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(interval_start))} to {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(interval_end))}: {count} persons")


    # def write_to_csv(self, track_id):
    #     current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    #     self.csv_writer.writerow([track_id, current_time])

    def write_to_csv(self, track_id):
        if track_id not in self.written_track_ids:  # Check if the track_id is not already written
            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            
            self.csv_writer.writerow([track_id, current_time])
            self.written_track_ids.add(track_id)  # Add the track_id to the set of written track_ids
            return self.written_track_ids
    # def out_time(self, x, y):
    #     set_of_existing_id = list(x)
    #     set_of_new_id = y
    #     difference = [item for item in set_of_existing_id if item not in set_of_new_id]
    #     print(difference) 
    #     with open('H:/SOLUTYICS/yolodeep/yolov5/persons_data.csv', 'r') as file:
    #         reader = csv.reader(file)
    #         rows = list(reader)

    #     for row in rows:
    #         if row[0] in difference:
    #             # Assuming you have a function get_time_out() to get the time-out value
    #             time_out = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())  # You need to implement this function
    #             row.append(time_out)
    #     print("##########################################",rows)
    #     with open('H:/SOLUTYICS/yolodeep/yolov5/output_csv', 'a', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerows(rows)\
        

    def out_time(self, x, y):
        set_of_existing_id = set(x)  # Convert to set for faster lookup
        set_of_new_id = set(y)
        difference = set_of_existing_id - set_of_new_id  # Use set difference for faster computation
        print("IDs to update:", difference) 


        updated_rows = []  # Store modified rows
        with open('./persons_data.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            print("*********************************888888888888888888",list(reader))
            file.seek(0)  # Reset the file pointer to the beginning of the file

            for row in reader:
                print("#########################################",row)

                if row[0] in difference:
                    time_out = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())  
                    row.append(time_out)
                updated_rows.append(row)

        print("Updated rows:", updated_rows)

        with open('./output.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(updated_rows)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = YoloDetector(model_name=None)
object_tracker = DeepSort(max_age=5,
                          n_init=2,
                          nms_max_overlap=1.0,
                          max_cosine_distance=0.3,
                          nn_budget=None,
                          override_track_class=None,
                          embedder="mobilenet",
                          half=True,
                          bgr=True,
                          embedder_gpu=True,
                          embedder_model_name=None,
                          embedder_wts=None,
                          polygon=False,
                          today=None)


# Prompt the user to enter the video file path
video_path = "D:/deepsort/No2.mp4"

# Create a VideoCapture object using the video file path
# cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

global x, y
x = set()
# detector = YourDetector()  # Initialize your detector object
# object_tracker = YourObjectTracker()  # Initialize your object tracker object

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Failed to read frame")
        break

    start = time.perf_counter()
    results = detector.score_frame(img)
    img, detections = detector.plot_boxes(results, img, height=img.shape[0], width=img.shape[1], confidence=0.5)
    detector.update_counters()  # Update counters based on time intervals

    tracks = object_tracker.update_tracks(detections, frame=img)
    y = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        y.append(track_id)
        ltrb = track.to_ltrb()
        bbox = ltrb

        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)

        x = detector.write_to_csv(track_id)  # Write ID and time to CSV

    print("set of total ids : ", x)
    print("set of current ids : ", y)
    if x is not None:
        detector.out_time(x, y)
    detector.display_results()  # Display the number of people detected within each time interval

    end = time.perf_counter()
    totalTime = end - start
    fps = 1 / totalTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 255, 0), 2)
    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release and destroy all windows before termination
cap.release()
cv2.destroyAllWindows()