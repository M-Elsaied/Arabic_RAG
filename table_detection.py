
import os
from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection
import matplotlib.pyplot as plt

# Set the directory containing the PDF files and the directory for outputting the processed images
pdf_directory = 'docs'
output_directory = 'images'
os.makedirs(output_directory, exist_ok=True)

# Load the pre-trained Table Detection model and its corresponding feature extractor
feature_extractor = DetrFeatureExtractor()
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

def plot_and_save_results(pil_img, scores, labels, boxes, file_path):
    """
    Plots the results of the table detection and saves the images if tables are detected.
    Only saves images where the detected tables have a score above the threshold.

    Parameters:
    - pil_img (PIL.Image): The image on which to plot the detections.
    - scores (list): The confidence scores for each detected object.
    - labels (list): The labels for each detected object.
    - boxes (list): The bounding boxes for each detected object.
    - file_path (str): The path where the image will be saved.
    """
    # Check if any tables were detected with confidence score above the threshold
    if any(score > 0.7 for score in scores):
        plt.figure(figsize=(16, 10))
        plt.imshow(pil_img)
        ax = plt.gca()
        colors = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                  [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]] * 100
        for score, label, (xmin, ymin, xmax, ymax), color in zip(scores, labels, boxes, colors):
            if score > 0.5:  # Filter results by confidence score
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=3))
                label_id = label.item()  # Convert tensor to integer
                label_description = model.config.id2label[label_id]
                text = f'{label_description}: {score:.2f}'
                ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        plt.savefig(file_path)
        plt.close()
    else:
        print(f"No tables detected in {file_path}, image not saved.")

# Process each PDF in the specified directory
for pdf_file in os.listdir(pdf_directory):
    if pdf_file.endswith('.pdf'):
        # Convert PDF pages to images
        images = convert_from_path(os.path.join(pdf_directory, pdf_file))
        for i, image in enumerate(images):
            # Encode the image for the model
            encoding = feature_extractor(images[i], return_tensors="pt")
            # Perform object detection
            with torch.no_grad():
                outputs = model(**encoding)
            # Rescale bounding boxes to the size of the image
            width, height = image.size
            results = feature_extractor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[(height, width)])[0]
            # Construct the output file path
            output_file_path = os.path.join(output_directory, f"{os.path.splitext(pdf_file)[0]}_page_{i+1}.png")
            # Plot and save the results
            plot_and_save_results(image, results['scores'], results['labels'], results['boxes'], output_file_path)
