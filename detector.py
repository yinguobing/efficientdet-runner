"""Tool to run a model."""
import cv2
import numpy as np
import tensorflow as tf


class Detector(object):
    """Mini module to run the EfficientDet model."""

    def __init__(self, saved_model):
        """Build an EfficientDet model runner.

        Args:
            saved_model: the string path to the SavedModel.
        """
        self.scale_width = 0
        self.scale_height = 0
        self.input_size = 512

        # Load the SavedModel object.
        imported = tf.saved_model.load('saved_model')
        self.__predict_fn = imported.signatures["serving_default"]

    def preprocess(self, image):
        """Preprocess the input image."""

        # Scale the image first.
        height, width, _ = image.shape
        self.ratio = self.input_size / max(height, width)
        image = cv2.resize(
            image, (int(self.ratio * width), int(self.ratio * height)))

        # Convert to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Then pad the image to input size.
        self.padding_h = self.input_size - int(self.ratio * width)
        self.padding_v = self.input_size - int(self.ratio * height)
        image = cv2.copyMakeBorder(
            image, 0, self.padding_v, 0, self.padding_h, cv2.BORDER_CONSTANT, (0, 0, 0))

        return image

    def __filter(self, detections, threshold):
        """Filter the detection results by score threshold."""
        # Get the detection results.
        boxes = detections['output_0'].numpy()[0]
        scores = detections['output_1'].numpy()[0]
        classes = detections['output_2'].numpy()[0]

        # Filter out the results by score threshold.
        mask = scores > threshold
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

        return boxes, scores, classes

    def predict(self, image, threshold):
        """Run inference with image inputs.

        Args:
            image: a numpy array as an input image.

        Returns:
            predictions: result.
        """
        frame_tensor = tf.constant(image, dtype=tf.uint8)
        frame_tensor = tf.expand_dims(frame_tensor, axis=0)
        detections = self.__predict_fn(frame_tensor)
        boxes, scores, classes = self.__filter(detections, threshold)

        # Scale the box back to the original image size.
        boxes /= self.ratio

        # Crop out the padding area.
        boxes[:, 2] = np.minimum(
            boxes[:, 2], (self.input_size - self.padding_v)/self.ratio)
        boxes[:, 1] = np.minimum(
            boxes[:, 1], (self.input_size - self.padding_h)/self.ratio)

        return boxes, scores, classes
