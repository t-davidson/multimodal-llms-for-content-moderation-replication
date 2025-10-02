from PIL import Image
import pandas as pd
from transformers import AutoProcessor, AutoModelForImageTextToText
import random
import os
import torch

# Set random seed
random.seed(1485233)

# Load dataset
file_path = 'image_indices_30k.csv'
image_numbers = pd.read_csv(file_path)
a_images = list(image_numbers['a_images'])
b_images = list(image_numbers['b_images'])
a_images.extend(b_images)

# Initialize model and processor
model_id = "OpenGVLab/InternVL3-2B-hf"

# Load model using the same pattern as Gemma with memory optimizations
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# Configure the processor (similar to Gemma)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

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

    # Construct the message using the exact same structure as Gemma
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text", 
                    "text": (
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
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image}
            ]
        }
    ]

    try:
        # Process inputs using the same pattern as Gemma
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        # Generation arguments (same as Gemma)
        generation_args = {
            "max_new_tokens": 1,
            "do_sample": False
        }

        # Generate response
        with torch.inference_mode():
            generate_ids = model.generate(**inputs, **generation_args)

        # Decode the response (same pattern as Gemma)
        input_len = inputs["input_ids"].shape[-1]
        response = processor.decode(
            generate_ids[0][input_len:],
            skip_special_tokens=True
        )

        return image_index, response.strip()
        
    except Exception as e:
        return image_index, f"Generation Error: {e}"

# Store results
results = []
output_file = "results/single_results_internvl3_2b.csv"

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

print("Analysis completed and results saved to single_results_internvl3_2b.csv")