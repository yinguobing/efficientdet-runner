"""Tool to run a model."""
import tensorflow as tf


class Runner(object):
    """Mini module to run the EfficientDet model."""

    def __init__(self, saved_model):
        """Build an EfficientDet model runner.

        Args:
            saved_model: the string path to the SavedModel.
        """
        # Load the SavedModel object.
        imported = tf.saved_model.load('saved_model')
        self.predict = imported.signatures["serving_default"]


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
    args = parser.parse_args()

    # Set GPU memory.
    devices = tf.config.list_physical_devices('GPU')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Summon a model runner.
    runner = Runner('saved_model')

    # Set the threshold for valid detections.
    threshold = 0.4

    # Construct video source.
    import cv2
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
        frame_raw = tf.constant(frame_raw, dtype=tf.uint8)
        frame_raw = tf.expand_dims(frame_raw, axis=0)

        # Run the model
        detections = runner.predict(frame_raw)

        # Get the detection results.
        boxes = detections['output_0'].numpy()[0]
        scores = detections['output_1'].numpy()[0]
        classes = detections['output_2'].numpy()[0]

        # Filter out the results by scores.
        mask = scores > threshold
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

        # Draw the bounding boxes. This is the MINIMAL code to show the
        # detection result. I believe you are able to draw the class names and
        # scores on the original frames.
        for box, score, class_ in zip(boxes, scores, classes):
            cv2.rectangle(frame, (int(box[1]), int(box[0])),
                          (int(box[3]), int(box[2])), (0, 255, 0), 2)

        # show the frame online, mainly used for real-time speed test.
        cv2.imshow('Frame', frame)

        # Press ESC to quit.
        if cv2.waitKey(1) == 27:
            break
