import json
import os

import joblib
import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Iranian License Plate Character Recognition",
    page_icon="IR",
    layout="wide",
    initial_sidebar_state="expanded"
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "knn_model.pkl"
)

MODEL_INFO_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_info.json"
)

ASSETS_DIR = os.path.join(
    BASE_DIR,
    "assets"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_model_info():
    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def preprocess_image(image, model_info):

    processed_image = image.convert("L")

    image_width = model_info["image_width"]
    image_height = model_info["image_height"]

    processed_image = processed_image.resize(
        (image_width, image_height)
    )

    image_array = np.array(processed_image)

    flattened_image = image_array.flatten()

    normalized_image = (
        flattened_image.astype("float32") / 255.0
    )

    model_input = normalized_image.reshape(1, -1)

    return processed_image, model_input


def show_asset(filename, title):

    asset_path = os.path.join(
        ASSETS_DIR,
        filename
    )

    if os.path.exists(asset_path):
        st.subheader(title)
        st.image(
            asset_path,
            use_container_width=True
        )
    else:
        st.warning(
            f"Asset not found: {filename}"
        )


try:
    model = load_model()
    model_info = load_model_info()

except Exception as error:
    st.error(
        f"Error loading project files: {error}"
    )
    st.stop()


st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Recognition",
        "Model Performance",
        "Dataset & Analysis"
    ]
)


if page == "Recognition":

    st.title(
        "Iranian License Plate Character Recognition"
    )

    st.write(
        "Upload an image containing a single character "
        "to classify it using the trained KNN model."
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Character Image",
        type=["png", "jpg", "jpeg", "bmp"]
    )

    if uploaded_file is not None:

        try:

            image = Image.open(uploaded_file)

            processed_image, model_input = (
                preprocess_image(
                    image,
                    model_info
                )
            )

            prediction = model.predict(
                model_input
            )[0]

            probabilities = model.predict_proba(
                model_input
            )[0]

            top_indices = np.argsort(
                probabilities
            )[-3:][::-1]

            top_classes = model.classes_[
                top_indices
            ]

            top_probabilities = probabilities[
                top_indices
            ]

            st.subheader("Image Processing")

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    image,
                    caption="Original Image",
                    use_container_width=True
                )

            with col2:

                st.image(
                    processed_image,
                    caption=(
                        f"Processed Image "
                        f"({model_info['image_width']} × "
                        f"{model_info['image_height']})"
                    ),
                    use_container_width=True
                )

            st.divider()

            col1, col2, col3 = st.columns(
                [1, 2, 1]
            )

            with col2:

                st.subheader("Prediction Result")

                st.metric(
                    "Predicted Character",
                    prediction
                )

            st.divider()

            st.subheader("Top Predictions")

            for label, probability in zip(
                top_classes,
                top_probabilities
            ):

                percentage = probability * 100

                col1, col2 = st.columns(
                    [1, 5]
                )

                with col1:
                    st.write(
                        f"**{label}**"
                    )

                with col2:

                    st.progress(
                        float(probability)
                    )

                    st.caption(
                        f"{percentage:.2f}%"
                    )

        except Exception as error:

            st.error(
                f"Error processing image: {error}"
            )


elif page == "Model Performance":

    st.title("Model Performance")

    st.write(
        "Performance information for the trained "
        "classification model."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Test Accuracy",
            f'{model_info["test_accuracy"]}%'
        )

    with col2:

        st.metric(
            "Algorithm",
            model_info["algorithm"]
        )

    with col3:

        st.metric(
            "K Value",
            model_info["k_value"]
        )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Samples",
            f'{model_info["total_samples"]:,}'
        )

    with col2:

        st.metric(
            "Training Samples",
            f'{model_info["training_samples"]:,}'
        )

    with col3:

        st.metric(
            "Testing Samples",
            f'{model_info["testing_samples"]:,}'
        )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Number of Classes",
            model_info["number_of_classes"]
        )

    with col2:

        st.metric(
            "Image Size",
            f'{model_info["image_width"]} × '
            f'{model_info["image_height"]}'
        )

    with col3:

        st.metric(
            "Features per Image",
            model_info["features_per_image"]
        )

    st.divider()

    st.subheader("Preprocessing Pipeline")

    preprocessing_steps = model_info.get(
        "preprocessing",
        []
    )

    for index, step in enumerate(
        preprocessing_steps,
        start=1
    ):

        st.write(
            f"{index}. {step}"
        )

    st.divider()

    st.subheader("Final Model")

    st.write(
        f"The final model uses "
        f"{model_info['algorithm']} "
        f"with K = {model_info['k_value']}."
    )

    st.write(
        f"It achieved a test accuracy of "
        f"{model_info['test_accuracy']}%."
    )


elif page == "Dataset & Analysis":

    st.title("Dataset & Model Analysis")

    st.write(
        "Dataset statistics and visual analysis "
        "of the model."
    )

    st.divider()

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Samples",
            f'{model_info["total_samples"]:,}'
        )

    with col2:

        st.metric(
            "Number of Classes",
            model_info["number_of_classes"]
        )

    with col3:

        st.metric(
            "Features per Image",
            model_info["features_per_image"]
        )

    st.divider()

    show_asset(
        "class_distribution.png",
        "Class Distribution"
    )

    st.divider()

    show_asset(
        "k_comparison.png",
        "K Value Comparison"
    )

    st.divider()

    show_asset(
        "confusion_matrix.png",
        "Confusion Matrix"
    )

    st.divider()

    st.subheader("K Comparison Results")

    k_comparison = model_info.get(
        "k_comparison",
        {}
    )

    if k_comparison:

        k_columns = st.columns(
            len(k_comparison)
        )

        for column, (k, accuracy) in zip(
            k_columns,
            k_comparison.items()
        ):

            with column:

                st.metric(
                    f"K = {k}",
                    f"{accuracy}%"
                )

    else:

        st.info(
            "K comparison results are not available."
        )

    st.divider()

    st.subheader("Top Misclassifications")

    errors = [
        ("3", "2", 43),
        ("2", "3", 26),
        ("Sad", "Sin", 24),
        ("Sin", "Sad", 14),
        ("5", "D", 7),
        ("6", "4", 6),
        ("7", "2", 4),
        ("2", "7", 3),
        ("9", "1", 3),
        ("2", "1", 2)
    ]

    for true_label, predicted_label, count in errors:

        col1, col2, col3 = st.columns(
            [2, 2, 1]
        )

        with col1:
            st.write(
                f"True: {true_label}"
            )

        with col2:
            st.write(
                f"Predicted: {predicted_label}"
            )

        with col3:
            st.write(
                f"{count} images"
            )
