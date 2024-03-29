import cv2
import numpy as np
import time
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.experimental import attempt_load
import pathlib
import csv

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def update_csv_with_start_times(csv_file, id_list, start_time):
    # Read existing data from CSV file
    data = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # try:
    #     print(type(data[0]))
    # except Exception as e:
    #     print(f"Exception: {e}")
    
    # print(f"Before: {data}")
            
    print(f"id_list: {id_list}")
    print(f"data: {data}")


    # Update Out_Time for specified IDs
    for row in data:
        if row['IDs'] in id_list:
            row['Out_Time'] = str(start_time)

    # print(f"After: {data}")
    # Write updated data back to CSV file
    with open(csv_file, 'w', newline='') as file:
        fieldnames = ['IDs', 'In_Time', 'Out_Time']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

class YoloDetector():

    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.classes = ['person']  # Specify the class names manually for your custom model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ##print("Using Device: ", self.device)
        self.start_time = time.time()  # Record start time
        self.interval = 60  # Time interval in seconds
        self.person_count = 0
        self.time_intervals = []
        self.persons = {}  # Dictionary to track person IDs and their time-in
        self.active_ids = []  # List to store active person IDs
        self.csv_file = "./persons_data.csv"  # CSV file to store ID and creation time
        # self.init_csv_file()

    def init_csv_file(self):
        with open(self.csv_file, 'w', newline='') as csvfile:
            fieldnames = ['IDs', 'In_Time', 'Out_Time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

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

    def plot_boxes(self, results, frame, height, width, confidence=0.3):
        labels, cord = results
        detections = []

        n = len(labels)
        x_shape, y_shape = width, height

        for i in range(n):
            row = cord[i]

            if row[4] >= confidence:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)

                color = (255, 0, 0)  # Blue color
                thickness = 2  # Increase thickness for better visibility

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                detections.append(([x1, y1, x2-x1, y2-y1], row[4].item(), self.class_to_label(labels[i])))
                ##print(f"Detected {self.class_to_label(labels[i])} at ({x1}, {y1}) to ({x2}, {y2})")

                # Add or update person's time-in and active IDs
                self.update_persons(row, time.time())
                self.update_active_ids(row)
                ##print(f"Active Ids: {self.active_ids}")
                ##print(row) 

        return frame, detections

    def update_persons(self, row, current_time):
        try:
            person_id = int(row[5])  # Convert person ID to int
            if person_id not in self.persons:
                # New person detected, record time-in
                self.persons[person_id] = {'time_in': current_time}
                # Write the new person's data to CSV
                ##print("------------------------------------------------------------------------------------------------------------------")
                # self.write_to_csv(person_id, current_time)
            else:
                # Person already in dictionary, update time-out and calculate time difference
                ##print("**********************************************************************************************************************")
                self.persons[person_id]['time_out'] = current_time
                time_in = self.persons[person_id]['time_in']
                time_out = self.persons[person_id]['time_out']
                time_difference = time_out - time_in
                # Add time difference to time_intervals for average
                self.time_intervals.append((person_id, time_in, time_out, time_difference))
        except Exception as e:
            pass
            # Handle index error
            ##print(f"Error in update_persons: {e}")
            # ##print("Index error occurred. Person ID not found.")

    def update_active_ids(self, row):
        try:
            person_id = int(row[5])  # Convert person ID to int
            if person_id not in self.active_ids:
                # Add the person's ID to the list of active IDs
                self.active_ids.append(person_id)
        except IndexError:
            # Handle index error
            ##print("Index error occurred. Person ID not found.")
            pass

    # def write_to_csv(self, person_id, creation_time):
    #     # Write the person's ID and creation time to the CSV file
    #     with open(self.csv_file, 'a', newline='') as csvfile:
    #         fieldnames = ['ID', 'In_Time', 'Out_Time']
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #         writer.writerow({'ID': person_id, 'In_Time': creation_time})
    #         ##print(creation_time)

    def display_results(self):
        for interval_start, count in self.time_intervals:
            interval_end = interval_start + self.interval
            ##print(f"From {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(interval_start))} to {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(interval_end))}: {count} persons")

    def update_counters(self):
        current_time = time.time()
        if current_time - self.start_time >= self.interval:
            self.time_intervals.append((self.start_time, self.person_count))
            self.start_time = current_time
            self.person_count = 0

        def display_results(self):
            for interval_start, count in self.time_intervals:
                interval_end = interval_start + self.interval
                ##print(f"From {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(interval_start))} to {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(interval_end))}: {count} persons")

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

detector.active_ids = []
temp_ids = []

while cap.isOpened():
    success, img = cap.read()
    if not success:
        ##print("Failed to read frame")
        break

    start = time.perf_counter()
    results = detector.score_frame(img)
    img, detections = detector.plot_boxes(results, img, height=img.shape[0], width=img.shape[1], confidence=0.5)
    detector.update_counters()  # Update counters based on time intervals

    tracks = object_tracker.update_tracks(detections, frame=img)
    
    temp_ids = [track.track_id for track in tracks if track.is_confirmed()]   

    ##print(f"detector.active_ids= {detector.active_ids}")
    ##print(f"temp_ids= {temp_ids}")

    if temp_ids != detector.active_ids:
        # Find ids not present in temp_ids anymore
        missing_ids = [id for id in detector.active_ids if id not in temp_ids]
        
        # Remove missing ids from detector.active_ids
        detector.active_ids = [id for id in detector.active_ids if id in temp_ids]
        
        # ##print missing ids
        csv_file = 'persons_data.csv'
        start_time = time.perf_counter()
        update_csv_with_start_times(csv_file, missing_ids, start_time)
        print(f"-------------------------------------Missing ids: {missing_ids}------------------------------------")

    detector.active_ids = temp_ids
    startTime = time.perf_counter()

    csv_file_path = "persons_data.csv"

    # Read existing data from the CSV file into a list of dictionaries
    existing_data = []
    try:
        with open(csv_file_path, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            existing_data = list(reader)
    except FileNotFoundError:
        # File doesn't exist, no need to read existing data
        pass

    # Extract existing IDs to check for duplicates
    existing_ids = set(int(row['IDs']) for row in existing_data)
    ##print(f"HELLOOO {existing_ids}")

    # Prepare the data to be written, appending to existing data
    data_to_write = []
    for id in detector.active_ids:
        if int(id) not in existing_ids:
            ##print("In the looopoyeeee")
            ##print((id))
            ##print(existing_ids)
            ##print("WHAAATTT")
            data_to_write.append({'IDs': id, 'In_Time': startTime, 'Out_Time': None})
            # Append the new data to the existing CSV file
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['IDs', 'In_Time', 'Out_Time'])

                # Write header only if the file was empty
                if file.tell() == 0:
                    writer.writeheader()

                for data in data_to_write:
                    writer.writerow(data)
        else:
            ##print(f"ID {id} already exists and cannot be re-entered.")
            pass

    

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id

        ltrb = track.to_ltrb()
        bbox = ltrb

        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        ##print(f"Tracked object ID: {track_id} with bounding box: {bbox}")

        # ##print the ID and creation time
        ##print(f"ID {track_id} is created at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")

        detector.display_results()  # Display the number of people detected within each time interval

        end = time.perf_counter()
        totalTime = end - start
        fps = 1 / totalTime

        cv2.putText(img, f'Frame Per Second: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 255, 0), 2)

        # Display Peak Hours, Person Count, and Time of a Person (id) in frame
        if detector.time_intervals:
            peak_hours = max(set([time.localtime(start)[3] for start, _ in detector.time_intervals]), key=[time.localtime(start)[3] for start, _ in detector.time_intervals].count)
        else:
            peak_hours = "No data Yet"

        person_count = len(tracks)

        # Collect person information for tracks with valid IDs
        valid_tracks_info = []
        for track in tracks:
            if track.track_id in detector.persons:
                time_in = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(detector.persons[track.track_id]['time_in']))
                valid_tracks_info.append(f"ID: {track.track_id}, Time: {time_in}")

        # Join the collected information into a string
        person_info = ", ".join(valid_tracks_info)

        cv2.putText(img, f'Peak Hours starts from: {peak_hours}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, f'Person Count at instance: {person_count}', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, f'Person Info: {person_info}', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

# Release and destroy all windows before termination
cap.release()
cv2.destroyAllWindows()