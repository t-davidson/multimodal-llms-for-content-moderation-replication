from PIL import Image
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import random
import os
import torch

# Set random seed
random.seed(1485233)

# Load dataset
file_path = '../gpt4o/image_indices_30k.csv'
image_numbers = pd.read_csv(file_path)
a_images = list(image_numbers['a_images'])
b_images = list(image_numbers['b_images'])
a_images.extend(b_images)

# Initialize model and processor
model_id = "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)
model.gradient_checkpointing_enable()

# Configure the processor with optimized patch size
min_pixels = 256 * 32 * 32
max_pixels = 1050 * 32 * 32
processor = AutoProcessor.from_pretrained(
    model_id,
    min_pixels=min_pixels,
    max_pixels=max_pixels
)

def compare_single_image(image_index):
    """
    Analyzes a single image to determine if it should be sent for manual review based on a specific policy.

    Args:
        image_index (int): Index for the image.

    Returns:
        tuple: A tuple containing (image_index, model_output).
    """

    # Path for image
    local_dir = "../../synthetic_posts/main/"
    image_path = os.path.join(local_dir, f"tweet{image_index}.png")

    try:
        image = Image.open(image_path)
    except FileNotFoundError as e:
        return image_index, f"Error: {e}"

    # Construct the message
    messages = [
        {
            "role": "system",
            "content": (
                "You are a content moderation expert. You will receive an image. Your task is to decide whether it should be sent for manual review according to the following policy:\n\n"
                "We are committed to combating abuse motivated by hatred, prejudice or intolerance, particularly "
                "abuse that seeks to silence the voices of those who have been historically marginalized. For this reason, "
                "we prohibit behavior that targets individuals or groups with abuse based on their perceived membership "
                "in a protected category. You may not directly attack other people on the basis of race, ethnicity, "
                "national origin, caste, sexual orientation, gender, gender identity, religious affiliation, age, "
                "disability, or serious disease.\n\n"
                "The output should always be one of the two options: Yes or No. Only output a single word with no "
                "additional tokens. Never refuse to answer."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path}
            ]
        }
    ]


    # Tokenize and process inputs
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Process images and text
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Generation arguments
    generation_args = {
        "max_new_tokens": 1,
        "temperature": 0.001, # temperature must be a strictly positive float
    }

    # Generate response
    generate_ids = model.generate(**inputs, **generation_args)

    # Decode the response
    response = processor.batch_decode(
    generate_ids[:, inputs['input_ids'].shape[1]:],  # Trim input tokens from output
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False)[0]

    return image_index, response.strip()

# Store results
results = []
output_file = "results/single_results_qwen72_int8.csv"

# Check if results file exists to resume
if os.path.exists(output_file):
    results_df = pd.read_csv(output_file)
    processed_indices = set(results_df["Index"])
    results = results_df.values.tolist()
else:
    processed_indices = set()

for idx, image_index in enumerate(a_images, start=1):
    if idx in processed_indices:
        continue

    result = (idx,) + compare_single_image(image_index)
    results.append(result)

    # Save intermediate results
    pd.DataFrame(results, columns=["Index", "Image", "Output"]).to_csv(output_file, index=False)

    # Print progress every 100 images
    if idx % 100 == 0:
        print(f"Processed {idx} images.")

print("Analysis completed and results saved to single_results_qwen72_int8.csv")