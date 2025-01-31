import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None
        self.full_model = None

    def get_base_model(self) -> None:
        """
        Load the base model dynamically based on the configuration.
        """
        try:
            logging.info(f"Loading the base model: {self.config.params_base_model}")

            # Map model names to their corresponding TensorFlow functions
            model_mapping = {
                "VGG16": tf.keras.applications.VGG16,
                "MobileNet": tf.keras.applications.MobileNet,
                "ResNet50": tf.keras.applications.ResNet50,
                "InceptionV3": tf.keras.applications.InceptionV3,
            }

            # Get the model function from the mapping
            model_fn = model_mapping.get(self.config.params_base_model)
            if not model_fn:
                raise ValueError(f"Unsupported model: {self.config.params_base_model}")

            # Load the model
            self.model = model_fn(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top
            )
            logging.info(f"{self.config.params_base_model} loaded successfully.")
            self.save_model(path=self.config.base_model_path, model=self.model)
        except Exception as e:
            logging.error(f"Error loading the base model: {e}")
            raise e

    @staticmethod
    def _prepare_full_model(
        model: tf.keras.Model,
        classes: int,
        freeze_all: bool,
        freeze_till: int,
        learning_rate: float
    ) -> tf.keras.Model:
        """
        Prepare the full model by adding custom layers on top of the base model.
        """
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif freeze_till is not None and freeze_till > 0:
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self) -> None:
        """
        Update the base model by adding custom layers and compiling it.
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model) -> None:
        """
        Save the model to the specified path.
        """
        model.save(path)
        logging.info(f"Model saved at: {path}")