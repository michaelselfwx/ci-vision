import json
import os
from pathlib import Path
from PIL import Image, ImageDraw
import argparse
import numpy as np


def load_label_json(json_path):
    """Load label data from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def visualize_labels(image_path, json_path, output_path=None):
    """
    Load image and JSON labels, draw annotations, and save/display result.

    Args:
        image_path (str): Path to PNG image
        json_path (str): Path to JSON label file
        output_path (str): Optional path to save annotated image
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.float32)
    height, width = img_array.shape[:2]

    # Load labels
    label_data = load_label_json(json_path)

    # Define label hierarchy (higher value = higher priority, overwrites lower priority)
    label_hierarchy = {
        "OCEAN": 0,
        "LAND": 1,
        "OBSCR": 2,
        "SHALLOW": 3,
        "HCRF": 4,
        "SBF": 5,
        "OUTFLOW": 6,
        "DEEP": 7,
    }

    # Assign colors based on label (RGB only, no alpha)
    color_map = {
        "OCEAN": (0, 100, 200),  # Blue
        "LAND": (34, 139, 34),  # Forest green
        "OBSCR": (64, 64, 64),  # Dark grey
        "SHALLOW": (128, 128, 128),  # Light Gray
        "HCRF": (144, 238, 144),  # Light Green
        "DEEP": (255, 0, 0),  # Red
        "SBF": (128, 0, 128),  # Purple
        "OUTFLOW": (255, 255, 0),  # Yellow
    }

    # Create priority mask (stores hierarchy value for each pixel)
    priority_mask = np.full((height, width), -1, dtype=np.int32)
    label_mask = np.full((height, width), "", dtype=object)

    # Process each shape and update masks based on hierarchy
    shapes = label_data.get("shapes", [])

    for shape in shapes:
        shape_type = shape.get("shape_type", "").lower()
        points = shape.get("points", [])
        label = shape.get("label", "")

        hierarchy_value = label_hierarchy.get(label, 999)

        if shape_type == "polygon" and len(points) >= 3:
            # Convert points to format PIL expects
            polygon_points = [tuple(p) for p in points]

            # Create temporary image for this polygon
            temp_img = Image.new("L", (width, height), 0)
            temp_draw = ImageDraw.Draw(temp_img)
            temp_draw.polygon(polygon_points, fill=255)
            temp_array = np.array(temp_img)

            # Update masks only where polygon exists and has higher priority
            mask = temp_array > 0
            update_mask = mask & (priority_mask < hierarchy_value)
            pixels_updated = np.sum(update_mask)
            if pixels_updated > 0:
                priority_mask[update_mask] = hierarchy_value
                label_mask[update_mask] = label
                # print(f"  {label} (priority {hierarchy_value}): updated {pixels_updated} pixels")

        elif shape_type == "rectangle" and len(points) >= 2:
            # Rectangle: points are [[x1, y1], [x2, y2]]
            x1, y1 = points[0]
            x2, y2 = points[1]

            # Create temporary image for this rectangle
            temp_img = Image.new("L", (width, height), 0)
            temp_draw = ImageDraw.Draw(temp_img)
            temp_draw.rectangle([(x1, y1), (x2, y2)], fill=255)
            temp_array = np.array(temp_img)

            # Update masks only where rectangle exists and has higher priority
            mask = temp_array > 0
            update_mask = mask & (priority_mask < hierarchy_value)
            pixels_updated = np.sum(update_mask)
            if pixels_updated > 0:
                priority_mask[update_mask] = hierarchy_value
                label_mask[update_mask] = label
                # print(f"  {label} (priority {hierarchy_value}): updated {pixels_updated} pixels")

    # Apply labels to image with transparency (60% blend)
    for y in range(height):
        for x in range(width):
            if priority_mask[y, x] >= 0:
                label = label_mask[y, x]
                if label in color_map:
                    color = np.array(color_map[label])
                    # 60% transparency: blend image with label color
                    img_array[y, x] = 0.6 * img_array[y, x] + 0.4 * color

    img = Image.fromarray(np.uint8(img_array))

    # Add timestamp (first token of filename) in top-left corner
    draw = ImageDraw.Draw(img, "RGBA")
    filedate_token = os.path.splitext(os.path.basename(image_path))[0].split("_")[0]
    filetime_token = os.path.splitext(os.path.basename(image_path))[0].split("_")[1]
    timestamp_text = f"{filedate_token} @ {filetime_token}z"
    bbox = (
        draw.textbbox((0, 0), timestamp_text)
        if hasattr(draw, "textbbox")
        else (0, 0, *draw.textsize(timestamp_text))
    )
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = 4
    box_coords = [(5, 5), (5 + text_w + 2 * pad, 5 + text_h + 2 * pad)]
    draw.rectangle(
        box_coords, fill=(255, 255, 255, 120), outline=(0, 0, 0, 180), width=1
    )
    draw.text((5 + pad, 5 + pad), timestamp_text, fill=(0, 0, 0, 220))

    # Add legend to the image (bottom right corner)
    draw = ImageDraw.Draw(img, "RGBA")

    # Get labels that are actually present in this image
    present_labels = sorted(
        set(label_mask.flatten()) - {""}, key=lambda l: label_hierarchy.get(l, 999)
    )

    if present_labels:
        # Legend dimensions (smaller)
        box_size = 12
        padding = 4
        line_height = box_size + 3
        legend_width = 75
        legend_height = len(present_labels) * line_height + 2 * padding

        # Position legend in bottom-right corner
        legend_x = width - legend_width - 5
        legend_y = height - legend_height - 5

        # Draw more transparent background
        draw.rectangle(
            [(legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height)],
            fill=(255, 255, 255, 120),
            outline=(0, 0, 0, 180),
            width=1,
        )

        # Draw each label
        for i, label in enumerate(present_labels):
            y_pos = legend_y + padding + i * line_height

            # Draw color box (smaller)
            color = color_map.get(label, (255, 0, 0))
            draw.rectangle(
                [
                    (legend_x + padding, y_pos),
                    (legend_x + padding + box_size, y_pos + box_size),
                ],
                fill=color + (255,),
                outline=(0, 0, 0, 180),
                width=1,
            )

            # Draw label text (smaller font size via positioning)
            draw.text(
                (legend_x + padding + box_size + 5, y_pos + 2),
                label,
                fill=(0, 0, 0, 220),
            )

    # Save or display
    if output_path:
        img.save(output_path)
        print(f"Saved annotated image to {output_path}")

    return img


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (255, 0, 0)  # Default to red


def process_directory(image_dir, output_dir=None, extensions=(".png", ".jpg", ".jpeg")):
    """Process all images in a directory with their associated JSON labels."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(os.listdir(image_dir)):
        if not filename.lower().endswith(extensions):
            continue

        image_path = os.path.join(image_dir, filename)
        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(image_dir, json_filename)

        if not os.path.exists(json_path):
            print(f"Skipping {filename} (no associated JSON)")
            continue

        output_path = None
        if output_dir:
            output_filename = os.path.splitext(filename)[0] + "_labeled.png"
            output_path = os.path.join(output_dir, output_filename)

        print(f"Processing {filename}...")
        visualize_labels(image_path, json_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize image labels from JSON annotation files"
    )
    parser.add_argument(
        "--image-dir", required=True, help="Directory containing images and JSON files"
    )
    parser.add_argument(
        "--output-dir", help="Directory to save labeled images (optional)"
    )

    args = parser.parse_args()
    process_directory(args.image_dir, args.output_dir)
