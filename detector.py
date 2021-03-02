"""Tool to run a model."""
import tensorflow as tf


class Detector(object):
    """Mini module to run the EfficientDet model."""

    def __init__(self, saved_model):
        """Build an EfficientDet model runner.

        Args:
            saved_model: the string path to the SavedModel.
        """
        # Load the SavedModel object.
        imported = tf.saved_model.load('saved_model')
        self.__predict_fn = imported.signatures["serving_default"]

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

    def predict(self, images, threshold):
        """Run inference with image inputs.

        Args:
            images: a list of numpy array as input images.

        Returns:
            predictions: result batch.
        """
        frame_tensor = tf.constant(images, dtype=tf.uint8)
        detections = self.__predict_fn(frame_tensor)
        results = self.__filter(detections, threshold)

        return results
