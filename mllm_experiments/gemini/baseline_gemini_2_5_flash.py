from google import genai
from google.genai import types
import pandas as pd
import random
import os
import yaml
import asyncio
import aiofiles
import time
from datetime import datetime

# Set random seed for reproducibility
random.seed(1485233)

class RateLimiter:
    """Thread-safe rate limiter to respect API limits"""
    def __init__(self, max_requests_per_minute=950):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to respect rate limits"""
        async with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            if len(self.requests) >= self.max_requests:
                # Wait until the oldest request is more than 1 minute old
                sleep_time = 60 - (now - self.requests[0]) + 0.1
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    # Clean up again after waiting
                    now = time.time()
                    self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            self.requests.append(now)

def load_api_key(yaml_file='key.yaml'):
    """Load API key from YAML file"""
    with open(yaml_file, 'r') as file:
        yaml_data = yaml.safe_load(file)
        return yaml_data['key']

async def load_image_bytes(image_path):
    """Async load image bytes"""
    try:
        async with aiofiles.open(image_path, 'rb') as f:
            return await f.read()
    except FileNotFoundError:
        return None

async def compare_images_async(client, rate_limiter, index_a, index_b, local_dir="../output/"):
    """
    Compare two images using Gemini 2.0 Flash - async version
    Returns: (index_a, index_b, output)
    """
    try:
        # Wait for rate limit approval
        await rate_limiter.acquire()
        
        # Construct image paths
        image_a_path = os.path.join(local_dir, f"tweet{index_a}.png")
        image_b_path = os.path.join(local_dir, f"tweet{index_b}.png")

        # Load images concurrently
        image_a_bytes, image_b_bytes = await asyncio.gather(
            load_image_bytes(image_a_path),
            load_image_bytes(image_b_path)
        )
        
        if image_a_bytes is None:
            return index_a, index_b, f"Error: Image A not found ({image_a_path})"
        if image_b_bytes is None:
            return index_a, index_b, f"Error: Image B not found ({image_b_path})"

        # Create image parts
        image_a_part = types.Part.from_bytes(
            data=image_a_bytes,
            mime_type='image/png'
        )
        
        image_b_part = types.Part.from_bytes(
            data=image_b_bytes,
            mime_type='image/png'
        )

        # System instruction (exact match to original Qwen script)
        system_instruction = (
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

        # Generate content using Gemini 2.5 Flash (note additional argument for thinking)
        response = await client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                system_instruction,
                image_a_part,
                image_b_part
            ],
            config=types.GenerateContentConfig( 
                thinking_config=types.ThinkingConfig(thinking_budget=0), # Ensures thinking is not active, see https://ai.google.dev/gemini-api/docs/thinking#set-budget
                max_output_tokens=1,
                temperature=0
            )
        )
        
        return index_a, index_b, response.text.strip()
        
    except Exception as e:
        return index_a, index_b, f"Error: {str(e)}"

async def main():
    """Main async function"""
    # Load dataset
    file_path = 'image_indices_30k.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return
    
    try:
        dataset = pd.read_csv(file_path)
        # Process first 10 rows for testing (remove [:10] for full run)
        image_pairs = dataset[['a_images', 'b_images']].values.tolist()
        print(f"Loaded {len(image_pairs)} image pairs from {file_path}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Load API key
    try:
        api_key = load_api_key()
        print("API key loaded successfully")
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading API key: {e}")
        return
    
    # Create client and rate limiter
    client = genai.Client(api_key=api_key)
    rate_limiter = RateLimiter(max_requests_per_minute=950)
    
    # Setup output
    output_file = "results/baseline_results_gemini_2_5_flash_async.csv"
    os.makedirs("results", exist_ok=True)
    
    # Check for existing results to resume
    results = []
    processed_indices = set()
    
    if os.path.exists(output_file):
        try:
            results_df = pd.read_csv(output_file)
            processed_indices = set(results_df["Index"])
            results = results_df.values.tolist()
            print(f"Resuming from {len(processed_indices)} already processed pairs")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
            print("Starting fresh...")
    
    # Filter remaining pairs
    remaining_pairs = []
    for idx, pair in enumerate(image_pairs, start=1):
        if idx not in processed_indices:
            remaining_pairs.append((idx, pair))
    
    if not remaining_pairs:
        print("All pairs already processed!")
        return
    
    print(f"Processing {len(remaining_pairs)} remaining pairs using Gemini 2.5 Flash")
    print(f"Rate limit: 950 requests/minute")
    print(f"Expected completion time: ~{len(remaining_pairs)/950:.1f} minutes")
    print("=" * 60)
    
    start_time = time.time()
    
    # Process with controlled concurrency
    semaphore = asyncio.Semaphore(10)  # Conservative limit for stability
    
    async def process_single_pair(idx, pair):
        """Process a single pair with semaphore control"""
        async with semaphore:
            index_a, index_b = pair
            return idx, await compare_images_async(client, rate_limiter, index_a, index_b)
    
    # Create all tasks
    tasks = [process_single_pair(idx, pair) for idx, pair in remaining_pairs]
    
    # Process in chunks for progress tracking
    chunk_size = 25  # Smaller chunks for more frequent progress updates
    total_chunks = (len(tasks) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(0, len(tasks), chunk_size):
        chunk_tasks = tasks[chunk_idx:chunk_idx + chunk_size]
        chunk_start_time = time.time()
        
        try:
            # Process chunk with exception handling
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            # Process results
            chunk_success_count = 0
            for result in chunk_results:
                if isinstance(result, Exception):
                    print(f"Task exception: {result}")
                    continue
                
                try:
                    idx, (index_a, index_b, output) = result
                    results.append([idx, index_a, index_b, output])
                    chunk_success_count += 1
                except Exception as e:
                    print(f"Result processing error: {e}")
            
            # Save intermediate results
            try:
                df = pd.DataFrame(results, columns=["Index", "Image A", "Image B", "Output"])
                df.to_csv(output_file, index=False)
            except Exception as e:
                print(f"Warning: Could not save intermediate results: {e}")
            
            # Progress reporting
            chunk_time = time.time() - chunk_start_time
            total_processed = len(results) - len(processed_indices)
            total_time = time.time() - start_time
            current_chunk = chunk_idx // chunk_size + 1
            
            print(f"Chunk {current_chunk}/{total_chunks}: {chunk_success_count}/{len(chunk_tasks)} successful in {chunk_time:.1f}s")
            print(f"Total progress: {total_processed}/{len(remaining_pairs)} pairs ({total_processed/len(remaining_pairs)*100:.1f}%)")
            
            if total_time > 0 and total_processed > 0:
                rate = total_processed / total_time * 60
                print(f"Average rate: {rate:.1f} pairs/minute")
                
                remaining = len(remaining_pairs) - total_processed
                if remaining > 0:
                    eta_minutes = remaining / rate
                    print(f"ETA: {eta_minutes:.1f} minutes")
            
            print("-" * 40)
            
        except Exception as e:
            print(f"Chunk processing error: {e}")
            # Continue with next chunk
    
    # Final summary
    total_time = time.time() - start_time
    final_processed = len(results) - len(processed_indices)
    
    print(f"\nProcessing completed!")
    print(f"Time taken: {total_time/60:.1f} minutes")
    print(f"Pairs processed: {final_processed}/{len(remaining_pairs)}")
    
    if total_time > 0 and final_processed > 0:
        print(f"Average rate: {final_processed/(total_time/60):.1f} pairs/minute")
    
    print(f"Results saved to: {output_file}")
    
    # Final save
    try:
        df = pd.DataFrame(results, columns=["Index", "Image A", "Image B", "Output"])
        df.to_csv(output_file, index=False)
        print(f"Final results saved: {len(df)} total rows")
    except Exception as e:
        print(f"Error saving final results: {e}")

def run_async():
    """Entry point to run the async main function"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    run_async()
