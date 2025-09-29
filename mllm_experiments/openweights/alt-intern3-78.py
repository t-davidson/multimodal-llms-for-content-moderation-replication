from PIL import Image
import pandas as pd
from transformers import AutoProcessor, AutoModelForImageTextToText
import random
import os
import torch

# Set random seed
random.seed(1485233)

def map_remote_path_to_local(remote_path):
    """
    Map remote paths to local directory paths for Amarel
    output_alt/ -> alt/
    output_2024/ -> images/
    """
    if remote_path.startswith('output_alt/'):
        return remote_path.replace('output_alt/', 'alt/')
    elif remote_path.startswith('output_2024/'):
        return remote_path.replace('output_2024/', 'images/')
    else:
        # Default fallback
        return f"alt/{remote_path}"

# Load dataset
file_path = 'image_indices_alt_99750.csv'
dataset = pd.read_csv(file_path)
image_data = dataset[['a_images', 'b_images', 'a_paths', 'b_paths']].values.tolist()

# Initialize model and processor
model_id = "OpenGVLab/InternVL3-78B-hf"

# Load model using the same pattern as Gemma with memory optimizations
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    load_in_8bit=True,  # Use 8-bit quantization
    low_cpu_mem_usage=True  # Reduce CPU memory usage
)

# Configure the processor (similar to Gemma)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def compare_images(index_a, index_b, path_a, path_b):
    """
    Compares two images to determine which should be prioritized for manual review based on a specific policy.

    Args:
        index_a (int): Index for Image A.
        index_b (int): Index for Image B.
        path_a (str): Remote path for Image A.
        path_b (str): Remote path for Image B.

    Returns:
        tuple: A tuple containing (index_a, index_b, model_output).
    """

    # Map remote paths to local paths
    image_a_path = map_remote_path_to_local(path_a)
    image_b_path = map_remote_path_to_local(path_b)

    try:
        image_a = Image.open(image_a_path)
        image_b = Image.open(image_b_path)
    except FileNotFoundError as e:
        return index_a, index_b, f"Error: {e}"

    # Construct the message using the exact same structure as Gemma
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text", 
                    "text": (
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
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_a},
                {"type": "text", "text": "Image A"},
                {"type": "image", "image": image_b},
                {"type": "text", "text": "Image B"}
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

        return index_a, index_b, response.strip()
        
    except Exception as e:
        return index_a, index_b, f"Generation Error: {e}"

# Store results
results = []
output_file = "results/alt_results_internvl3_78b.csv"

# Check if results file exists to resume
if os.path.exists(output_file):
    results_df = pd.read_csv(output_file)
    processed_indices = set(results_df["Index"])
    results = results_df.values.tolist()
else:
    processed_indices = set()

for idx, data in enumerate(image_data, start=1):
    if idx in processed_indices:
        continue

    index_a, index_b, path_a, path_b = data
    result = (idx,) + compare_images(index_a, index_b, path_a, path_b)
    results.append(result)

    # Save intermediate results
    pd.DataFrame(results, columns=["Index", "Image A", "Image B", "Output"]).to_csv(output_file, index=False)

    # Print progress every 100 pairs
    if idx % 100 == 0:
        print(f"Processed {idx} image pairs.")

print("Comparison completed and results saved to alt_results_internvl3_78b.csv")