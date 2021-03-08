"""Demo of running EfficientDet detection."""
from argparse import ArgumentParser

import cv2

from detector import Detector

parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--output", type=str, default=None,
                    help="Video file to be written.")
args = parser.parse_args()

# This is the label names for the default checkpoint file with COCO dataset.
label_map = {
    # 0: 'background',
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush',
}

if __name__ == '__main__':
    # Summon a model runner.
    detector = Detector('saved_model')

    # Set the threshold for valid detections.
    threshold = 0.4

    # Construct video source.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error opening input video: {}'.format(args.video))
    else:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use video writer to write processed video file.
    if args.output:
        video_writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc('a', 'v', 'c', '1'),
            cap.get(cv2.CAP_PROP_FPS),
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Use meter to calculate FPS.
    meter = cv2.TickMeter()

    # Capture frame-by-frame
    while cap.isOpened():
        # Start the timmer.
        meter.start()

        # Read a frame.
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare the input image.
        frame_raw = detector.preprocess(frame)

        # Run the model
        predictions = detector.predict(frame_raw, threshold)

        # Stop the timmer.
        meter.stop()

        # Draw the bounding boxes and the class names.
        boxes, scores, classes = predictions
        for box, score, class_ in zip(boxes, scores, classes):
            y0, x0, y1, x1 = [int(b) for b in box]
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, "{}:{:.2f}".format(label_map[class_], score),
                        (x0, y0-7), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0),
                        1, cv2.LINE_AA)

        # Draw FPS on the screen's top left corner.
        cv2.putText(frame, "FPS: {:.0f}".format(meter.getFPS()), (7, 14),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # show the frame online, mainly used for real-time speed test.
        cv2.imshow('Frame', frame)

        # And write the processed video file.
        if args.output:
            video_writer.write(frame)

        # Press ESC to quit.
        if cv2.waitKey(1) == 27:
            break
