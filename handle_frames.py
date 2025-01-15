import argparse
import os
import time
from queue import Queue
from threading import Thread

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataPath", required=True, help="path to dataset folder")
parser.add_argument("-o", "--outputPath", required=True, help="path to output folder")

args = parser.parse_args()

class VideoStream:
    def __init__(self, path, max_queue_size = 128):
        self.stream = cv2.VideoCapture(path)
        self.queue = Queue(maxsize = max_queue_size)
        self.stopped = False

    def update(self):
        while True:
            if self.stopped:
                return
            if not self.queue.full():
                (grabbed, frame) = self.stream.read()

                if not grabbed:
                    self.stop()
                    return
                self.queue.put(frame)

    def start(self):
        thread = Thread(target = self.update, args = ())
        thread.daemon = True
        thread.start()
        return self

    def stop(self):
        self.stopped = True

    def read(self):
        return self.queue.get()

    def is_more(self):
        return self.queue.qsize() > 0

data_path = args.dataPath
output_path = args.outputPath

video_list = os.listdir(data_path)
print('video list: ', video_list)

for category in os.listdir(data_path):
    category_path = os.path.join(data_path, category)
    if os.path.isdir(category_path):
        output_category_path = os.path.join(output_path, category)
        os.makedirs(output_category_path, exist_ok=True)

        for video_file in os.listdir(category_path):
            video_path = os.path.join(category_path, video_file)
            if not video_path.endswith(('.mp4', '.avi', '.mov')):
                continue

            print(f"Processing {video_file} in category {category}")
            video_stream = VideoStream(video_path).start()
            time.sleep(1) # delay for buffer

            # Create a folder for the video inside the category output folder
            video_output_path = os.path.join(output_category_path, os.path.splitext(video_file)[0])
            os.makedirs(video_output_path, exist_ok=True)

            frame_count = 0

            while video_stream.is_more():
                frame = video_stream.read()
                frame_output_path = os.path.join(video_output_path, f"{frame_count}.jpg")
                cv2.imwrite(frame_output_path, frame)
                frame_count += 1

            video_stream.stop()
