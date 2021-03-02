"""Demo of running EfficientDet detection."""

from argparse import ArgumentParser
import cv2
from detector import Detector

parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
args = parser.parse_args()

if __name__ == '__main__':
    # Summon a model runner.
    detector = Detector('saved_model')

    # Set the threshold for valid detections.
    threshold = 0.4

    # Construct video source.
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print('Error opening input video: {}'.format(args.video))

    # Capture frame-by-frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare the input image.
        frame = cv2.resize(frame, (512, 512))
        frame_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run the model
        detections = detector.predict([frame_raw], 0.4)

        # Draw the bounding boxes. This is the MINIMAL code to show the
        # detection result. I believe you are able to draw the class names and
        # scores on the original frames.
        # for boxes, scores, classes in detections:
        #     for box, score, class_ in zip(boxes, scores, classes):
        #         cv2.rectangle(frame, (int(box[1]), int(box[0])),
        #                       (int(box[3]), int(box[2])), (0, 255, 0), 2)

        # show the frame online, mainly used for real-time speed test.
        cv2.imshow('Frame', frame)

        # Press ESC to quit.
        if cv2.waitKey(1) == 27:
            break
