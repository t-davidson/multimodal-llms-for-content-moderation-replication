from PIL import Image
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import random
import os
import torch

# Set random seed
random.seed(1485233)

# Load dataset
file_path = 'image_indices_30k.csv'
dataset = pd.read_csv(file_path)
image_pairs = dataset[['a_images', 'b_images']].values.tolist()

# Initialize model and processor
model_id = "Qwen/Qwen2-VL-7B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
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

def compare_images(index_a, index_b):
    """
    Compares two images to determine which should be prioritized for manual review based on a specific policy.

    Args:
        index_a (int): Index for Image A.
        index_b (int): Index for Image B.

    Returns:
        tuple: A tuple containing (index_a, index_b, model_output).
    """

    # Paths for images
    local_dir = "images"
    image_a_path = os.path.join(local_dir, f"tweet{index_a}.png")
    image_b_path = os.path.join(local_dir, f"tweet{index_b}.png")

    try:
        image_a = Image.open(image_a_path)
        image_b = Image.open(image_b_path)
    except FileNotFoundError as e:
        return index_a, index_b, f"Error: {e}"

    # Construct the message
    messages = [
        {
            "role": "system",
            "content": (
                "You are a content moderation expert. You will receive pairs of images labeled as 'Image A' and 'Image B'. "
                "Your task is to decide which image should be prioritized for manual review according to the following policy:\n\n"
                "We are committed to combating abuse motivated by hatred, prejudice or intolerance, particularly "
                "abuse that seeks to silence the voices of those who have been historically marginalized. For this reason, "
                "we prohibit behavior that targets individuals or groups with abuse based on their perceived membership "
                "in a protected category. You may not directly attack other people on the basis of race, ethnicity, "
                "national origin, caste, sexual orientation, gender, gender identity, religious affiliation, age, "
                "disability, or serious disease.\n\n"
                "The output should always be one of the two options: A or B. Only output a single character with no "
                "additional tokens. Never refuse to answer."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_a_path, "label": "Image A"},
                {"type": "image", "image": image_b_path, "label": "Image B"}
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
        images=[image_a, image_b],
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

    return index_a, index_b, response.strip()

# Store results
results = []
output_file = "results/baseline_results_qwen.csv"

# Check if results file exists to resume
if os.path.exists(output_file):
    results_df = pd.read_csv(output_file)
    processed_indices = set(results_df["Index"])
    results = results_df.values.tolist()
else:
    processed_indices = set()

for idx, pair in enumerate(image_pairs, start=1):
    if idx in processed_indices:
        continue

    index_a, index_b = pair
    result = (idx,) + compare_images(index_a, index_b)
    results.append(result)

    # Save intermediate results
    pd.DataFrame(results, columns=["Index", "Image A", "Image B", "Output"]).to_csv(output_file, index=False)

    # Print progress every 100 pairs
    if idx % 100 == 0:
        print(f"Processed {idx} image pairs.")

print("Comparison completed and results saved to baseline_results_qwen.csv")
