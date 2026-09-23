import os
import json
import base64
import argparse
from PIL import Image


def image_to_base64(image_path):
    """Convert image to base64 string for imageData field."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_label_json(template_path, image_path, output_path):
    """Generate a label JSON for a specific image using the template."""
    # Load template
    with open(template_path, "r") as f:
        label_data = json.load(f)

    # Open image to get dimensions
    with Image.open(image_path) as img:
        width, height = img.size

    # Update file-specific metadata
    label_data["imagePath"] = os.path.basename(image_path)
    label_data["imageData"] = image_to_base64(image_path)
    label_data["imageHeight"] = height
    label_data["imageWidth"] = width

    # Write output
    with open(output_path, "w") as f:
        json.dump(label_data, f, indent=2)

    print(f"Created: {output_path}")


def process_directory(
    template_path, image_dir, output_dir, extensions=(".png", ".jpg", ".jpeg")
):
    """Process all images in a directory."""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(extensions):
            continue

        image_path = os.path.join(image_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".json"
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            print(f"Skipping existing: {output_path}")
            continue

        generate_label_json(template_path, image_path, output_path)


if __name__ == "__main__":

    template = "ocean_land.json"
    image_dir = "c:/users/michaelself/tamu/research/sat-rad-ml/images/20220610"
    output_dir = image_dir
    process_directory(template, image_dir, output_dir)
