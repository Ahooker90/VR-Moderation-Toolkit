import os
import time
import torch
import numpy as np
from PIL import ImageGrab, Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
import cv2
import psutil
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")  # Lightweight YOLO model

# Load CLIP model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# BASE PROMPT FOR NONHUMAN VS HUMAN
base_prompt = "A VR scene featuring characters or avatars, including non-humanoids like animals or aliens, or with no characters present."

# NONHUMAN VS HUMAN PROMPTS
nonhuman_vs_human_inputs = [
    [f"{base_prompt} an anthropomorphic game character", f"{base_prompt} a human game character"],
    [f"{base_prompt} a character with animal ears", f"{base_prompt} a character with human ears"],
    [f"{base_prompt} a character with fur", f"{base_prompt} skin and possible tattoos"],
]

# NONHUMAN PROMPTS
explicit_prompt = "showing explicit nudity, including naked breasts, genitals, tail, or buttocks, or suggestive scenes without characters."
safe_prompt = "showing no explicit nudity, with characters fully or partially clothed, or a scene without nudity and no characters present."
nonhuman_inputs = [
    f"{base_prompt} {explicit_prompt}",
    f"{base_prompt} {safe_prompt}",
]

# HUMAN PROMPTS
base_prompt = "A VR scene featuring characters or avatars, or with no characters present."
explicit_prompt = "showing explicit nudity, including naked breasts, genitals, or buttocks, or suggestive scenes without characters."
safe_prompt = "showing no explicit nudity, with characters fully or partially clothed, or a scene without nudity and no characters present."
human_inputs = [
    f"{base_prompt} {explicit_prompt}",
    f"{base_prompt} {safe_prompt}",
]


# Monitoring parameters
sampling_rate = 1.0  # Frames per second
screen_region = None  # None for full screen or specify (x1, y1, x2, y2)
resource_threshold = 80  # Max CPU/GPU usage percentage


def is_system_overloaded():
    """Check if the system resources are under heavy load."""
    cpu_usage = psutil.cpu_percent(interval=0.1)
    gpu_usage = 0
    try:
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated(device) / torch.cuda.max_memory_allocated(device) * 100
    except:
        pass
    return cpu_usage > resource_threshold or gpu_usage > resource_threshold

def take_consensus(probs):
    """
    Calculate the average of each column in probs.
    If column 0's (nonhuman) average is higher than column 1's (human), return "nonhuman".
    Else if column 1's (human) average is higher than column 0's (nonhuman), return "human".
    """
    # Calculate the column averages
    num_rows = len(probs)
    column_averages = [sum(row[i] for row in probs) / num_rows for i in range(2)]
    
    if column_averages[0] > column_averages[1]:
        return "nonhuman"
    else:
        return "human"

def classify_region(image):
    """Classify the content of a region."""
    # Step 1: Human vs Nonhuman Classification
    probs = []
    for prompts in nonhuman_vs_human_inputs:
        inputs = clip_processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs.append(logits_per_image.softmax(dim=1).detach().cpu().numpy()[0])  # Detach tensor before converting to numpy

    classification = take_consensus(probs)

    # Step 2: NSFW vs Safe Classification
    nsfw_prompts = nonhuman_inputs if classification == "nonhuman" else human_inputs
    inputs = clip_processor(text=nsfw_prompts, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    nsfw_prob = logits_per_image.softmax(dim=1).detach().cpu().numpy()[0, 0]  # Detach tensor before converting to numpy

    nsfw_classification = "UNSAFE" if nsfw_prob > 0.5 else "SAFE"
    return classification, nsfw_classification

def monitor_screen():
    """Monitor the screen in real time and classify content."""
    ## OpenCV implementation was not tested during paper evaulations this is shown as a proof of concept and needs further testing to validate
    ## TODO: This currently on grabs images with 'person' classification to reduce noise. This should instead explore the OpenCV classification patterns for nonhuman avatars.
    try:
        while True:
            # Check system resources
            if is_system_overloaded():
                logging.warning("System overloaded! Slowing down processing...")
                time.sleep(2)
                continue

            # Capture the screen
            screenshot = ImageGrab.grab(bbox=screen_region)  # Capture the screen
            frame = np.array(screenshot)  # Convert to numpy array
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency

            # Detect objects using YOLO
            results = yolo_model(frame)
            detections = results[0].boxes

            for detection in detections:
                # Process only "person" class (class ID 0)
                class_id = int(detection.cls[0])
                if class_id != 0:  # Skip non-person detections
                    continue

                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                cropped = frame[y1:y2, x1:x2]
                cropped_pil = Image.fromarray(cropped)  # Convert cropped frame to PIL image (RGB)

                # Classify cropped region
                entity_classification, nsfw_classification = classify_region(cropped_pil)

                # Draw bounding box and label
                color = (0, 255, 0) if nsfw_classification == "SAFE" else (255, 0, 0)
                #label = f"{entity_classification}--{nsfw_classification}"
                label = f"{nsfw_classification}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Convert back to BGR for OpenCV display
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the results
            cv2.imshow("Screen Monitor", display_frame)

            # Break loop on 'q' key press
            if cv2.waitKey(int(1000 / sampling_rate)) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logging.info("Monitoring interrupted. Exiting safely...")
    except Exception as e:
        logging.error(f"Error encountered: {e}")
    finally:
        cv2.destroyAllWindows()



if __name__ == "__main__":
    logging.info("Starting screen monitor with safety checks...")
    monitor_screen()
