import matplotlib.pyplot as plt
import torch
import numpy as np
from contextlib import contextmanager
import cv2
import time
import math
import yaml

@contextmanager
def timer(code_part = "No part name"):
    start = time.perf_counter()
    yield  # Everything in the `with` block executes here
    end = time.perf_counter()
    take_time = (end - start)*1000
    print(f"***This {code_part} took about {take_time:.2f} millisecond***")
    
class Webcam:
    def __init__(self, cam_prop_yaml, cam_index=0 ):
        with open(cam_prop_yaml, "r") as file:
            config = yaml.safe_load(file)
            
        self.cam_prop = {
            "res" : config["CAM_PROP"]["res"],
            "fps" : config["CAM_PROP"]["fps"]
            }
        self.scale_factor = config["SCALE_FACTOR"]
        
        self.cam_index = cam_index
        self.cap = None

    def __enter__(self):
        with timer("get device"):
            self.cap = cv2.VideoCapture(self.cam_index)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, math.ceil(self.cam_prop["res"][0] * self.scale_factor))
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, math.ceil(self.cam_prop["res"][1] * self.scale_factor)) 
                self.cap.set(cv2.CAP_PROP_FPS, self.cam_prop["fps"]) 
            else: 
                raise RuntimeError("Failed to open webcam") 
        return self.cap  # Assign `cap` to `as cap` 

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Webcam released, all windows closed.")

        
def put_multiline_text(image, text, position, font, font_scale, text_color, thickness, bg_color, alpha=0.5, line_type=cv2.LINE_AA, line_space=30, padding=10):
    """
    Draws multiline text on an image with a semi-transparent rectangle background.
    
    :param image: The image to draw on.
    :param text: A list of strings, each representing a line.
    :param position: (x, y) starting position.
    :param font: OpenCV font type.
    :param font_scale: Scale of the font.
    :param text_color: Color of the text (B, G, R).
    :param thickness: Thickness of the text.
    :param bg_color: Background rectangle color (B, G, R).
    :param alpha: Transparency level of the background (0 = fully transparent, 1 = fully opaque).
    :param line_type: Line type for the text.
    :param line_space: Space between lines.
    :param padding: Padding around the text inside the rectangle.
    """
    overlay = image.copy()
    x, y = position
    y_offset = y
    max_text_width = 0
    text_heights = []

    # Calculate text sizes and max width
    for line in text:
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_text_width = max(max_text_width, text_width)
        text_heights.append(text_height)

    # Compute rectangle coordinates
    rect_x1 = x - padding
    rect_y1 = y - padding
    rect_x2 = x + max_text_width + padding
    rect_y2 = y + sum(text_heights) + (len(text) - 1) * line_space + padding

    # Draw a filled rectangle on the overlay
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Draw text on top
    for i, line in enumerate(text):
        cv2.putText(image, line, (x, y_offset + text_heights[i]), font, font_scale, text_color, thickness, line_type)
        y_offset += text_heights[i] + line_space  # Move the Y position down for the next line


def show_sample(dataloader, show_size = 10):
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    data_size = labels.size(0)
    if show_size <= data_size:
        images = images[:show_size]
        labels = labels[:show_size]
    else:
        show_size = data_size
    
    side1 = side2 = int(np.round(show_size**0.5))
    if  show_size**0.5 > np.round(show_size**0.5):
        side2 += 1
    fig, ax = plt.subplots(side1, side2, figsize = (10,10))
    
    for i in range(show_size):
        image = images[i].numpy().squeeze()
        pred = labels[i].item()
        collect = "o" if pred == labels[i]else "x"
        ax[i//side2,i%side2].imshow(image, cmap='gray')
        ax[i//side2,i%side2].set_title(f"{pred} ({collect})")
        ax[i//side2,i%side2].axis('off')
        ax[i//side2,i%side2].set_aspect("equal")   
    plt.tight_layout()
    plt.show()

            
    
def visual_sample(dataloader, model, device):
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    show_size = len(images)
    if show_size >= 16:
        show_size = 16
        images = images[:16]
        labels = labels[:16]
    
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    side1 = side2 = int(np.round(show_size**0.5))
    if  show_size**0.5 > np.round(show_size**0.5):
        side2 += 1
    fig, ax = plt.subplots(side1, side2, figsize = (10,10))
    
    for i in range(show_size):
        image = images[i].detach().cpu().numpy().squeeze()
        pred = predicted[i].item()
        collect = "o" if pred == labels[i]else "x"
        ax[i//side2,i%side2].imshow(image, cmap='gray')
        ax[i//side2,i%side2].set_title(f"{pred} ({collect})")
        ax[i//side2,i%side2].axis('off')
        ax[i//side2,i%side2].set_aspect("equal")
        
    plt.tight_layout()
    plt.show()
    

def display_images(images):
    #Check that image is list type or not.
    if not isinstance(images, list):
        #If not list type
        plt.figure(figsize=(12, 12))
        plt.imshow(images)  # Convert BGR to RGB
        plt.axis('off')
        return

    n = len(images)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    print("n = ",n)
    print("cols = ",cols)
    print("rows = ",rows)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    print("axes before: ",len(axes))
    axes = axes.flatten()  # Flatten the axes array for easier indexing
    print("axes after: ",len(axes))

    for i in range(n):
        if images[i].ndim == 2:
            axes[i].imshow(images[i], cmap="gray")  # Convert BGR to RGB
        elif images[i].ndim > 2:
            axes[i].imshow(images[i])
        axes[i].axis('on') #hide axis


    # Hide any unused subplots
    for j in range(i + 1, rows * cols):
        axes[j].axis('on')

    plt.tight_layout()
    plt.show()
            
    