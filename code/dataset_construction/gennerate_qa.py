import os
import csv
import time
import base64
import re
import pandas as pd
from PIL import Image
from openai import AzureOpenAI
import tiktoken

# ---------- Utility functions ----------
def compress_image(input_path, output_path, max_size=(800, 600), quality=70):
    """Resize and compress an image to reduce payload size for LLM requests."""
    try:
        img = Image.open(input_path)
        img.thumbnail(max_size)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(output_path, "JPEG", quality=quality)
    except Exception as e:
        print(f"[X] Error compressing {input_path}: {e}")

def convert_image_to_data_url(image_path):
    """Convert an image file to a base64-encoded data URL."""
    with open(image_path, "rb") as image_file:
        data = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


# ---------- GPT client wrapper ----------
class GPTAzureRAG():
    def __init__(self, deployment_name):
        print("GPTAzureRAG is in use.\n")
        self.set_API_key()
        self.deployment_name = deployment_name
        self.temperature = 0.7
        print(f"Deployment name: {self.deployment_name}")

    def set_API_key(self):
        self.client = AzureOpenAI(
            api_key="YOUR_API-KEY", 
            api_version="YOUR_API_VERSION",
            azure_endpoint="YOUR_AZURE_ENDPOINT"
        )

    def calc_tokens(self, msg):
        encoding = tiktoken.encoding_for_model("gpt-4o")
        return len(encoding.encode(msg))

    def query(self, msg, image_urls=None, try_num=0, icl_num=0):
        time.sleep(1)
        try:
            return self.__do_query(msg, image_urls)
        except Exception as e:
            print(f"[X] Query failed: {e}")
            time.sleep(10)
            return self.query(msg, image_urls, try_num + 1)

    def __do_query(self, msg, image_urls=None):
        messages = [
            {"role": "system", "content": "You are a helpful assistant for VR scene analysis."},
            {"role": "user", "content": [{"type": "text", "text": msg}]}
        ]
        if image_urls:
            for url in image_urls:
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": url}
                })

        completion = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            temperature=self.temperature
        )
        return completion.choices[0].message.content

# ---------- Main pipeline ----------
gpt_client = GPTAzureRAG(deployment_name="gpt-4o")
base_dir = r'./dataset/architecture'
all_records = []

folder_names = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f.isdigit()]
folder_names.sort(key=lambda x: int(x))

for folder in folder_names:
    print(f"=== Processing folder: {folder} ===")

    gt_data_csv = os.path.join(base_dir, folder, "groundtruth", "data.csv")
    sc_data_csv = os.path.join(base_dir, folder, "screenshot", "data.csv")
    selected_csv = os.path.join(base_dir, folder, "screenshot", "after", "selected.csv")

    if not os.path.exists(selected_csv):
        print(f"No selected.csv in folder {folder}, skipping...")
        continue

    # Load selected screenshot filenames
    selected_df = pd.read_csv(selected_csv)
    selected_filenames = set(selected_df["Filename"].tolist())

    gt_df = pd.read_csv(gt_data_csv)
    sc_df = pd.read_csv(sc_data_csv)

    sc_df = sc_df[sc_df["ScreenshotFilename"].isin(selected_filenames)]

    for _, row in sc_df.iterrows():
        sc_filename = row["ScreenshotFilename"]
        index_value = row["Index"]

        # Align screenshot index to the corresponding GT entry
        gt_row = gt_df[gt_df["Index"] == index_value]
        if gt_row.empty:
            print(f"[!] No matching screenshot for {sc_filename}")
            continue

        gt_filename = gt_row.iloc[0]["ScreenshotFilename"]

        print(f"Processing GT: {gt_filename} + SC: {sc_filename}")

        # Original file paths
        gt_image_path = os.path.join(base_dir, folder, "groundtruth", "after", gt_filename)
        sc_image_path = os.path.join(base_dir, folder, "screenshot", "after", sc_filename)

        # Create a local folder for compressed copies
        compressed_folder = os.path.join(base_dir, folder, "compressed")
        os.makedirs(compressed_folder, exist_ok=True)
        
        gt_stem = os.path.splitext(os.path.basename(gt_filename))[0]
        sc_stem = os.path.splitext(os.path.basename(sc_filename))[0]

        gt_compressed = os.path.join(compressed_folder, f"{gt_stem}_compressed.jpg")
        sc_compressed = os.path.join(compressed_folder, f"{sc_stem}_compressed.jpg")

        compress_image(gt_image_path, gt_compressed)
        compress_image(sc_image_path, sc_compressed)

        # Convert compressed images to data URLs for the vision model
        gt_data_url = convert_image_to_data_url(gt_compressed)
        sc_data_url = convert_image_to_data_url(sc_compressed)

        # Prompt for generating QA pairs
        prompt = f"""You are given two images of the same VR scene taken at different times. 
        The first image: previous screenshot, 
        the second image: current screenshot.

Your task:
Generate 10 simple question-answer pairs comparing the images, focusing on object disappearance or persistence.

Rules:
- Use spatial references (e.g., "the chair near the window"). The reference should be the environment such as the left and right walls, Windows and arches, rather than specific objects.
- If an object was visible in the first image but is missing in the second image, answer: "It has disappeared."
- If an object was present in both, answer: "It has always been here."
- If an object was missing in the first image, answer: "It was never there."
- Each time, ask 4 disappearing objects, 3 always present objects and 3 objects that have never been in the scene. (If there are less than four disappearing objects in the scene, you can repeatedly ask questions about the same disappearing object using different sentences.) But as far as possible to ask different objects.
- Only questions about whether an object existed in the past are allowed. Only question like "Did a specific object exist in a past scene?" is allowed. No other content can be asked. (When asking questions, never mention the two pictures and do not use words like "the differences or changes between the two pictures".). Use the past tense.
- When asking questions, the tone should seem as if you can only see the second picture (that is, the current picture), so when referring to relative positions, do not use objects that existed in the previous picture but not in the current one as references. When answering, you can see the information of the two pictures.
- Respect facts, not an illusion.

Output Format:
1. <question>
   Answer: <answer>

2. <question>
   Answer: <answer>
"""

        try:
            reply = gpt_client.query(prompt, image_urls=[gt_data_url, sc_data_url])
            print(f"\n[DEBUG] GPT reply:\n{reply}\n{'-'*60}")

            lines = reply.splitlines()
            qa_buffer = []

            for line in lines:
                line = line.strip()
                q_match = re.match(r"^\d+\.\s*(.+\?)\s*$", line)
                a_match = re.match(r"^Answer:\s*(.+)", line, re.IGNORECASE)

                if q_match:
                    qa_buffer.append({"question": q_match.group(1).strip()})
                elif a_match and qa_buffer:
                    qa_buffer[-1]["answer"] = a_match.group(1).strip()

            for qa in qa_buffer:
                if "question" in qa and "answer" in qa:
                    all_records.append({
                        "Folder": folder,
                        "Image": gt_filename,
                        "SC_Image": sc_filename,
                        "Question": qa["question"],
                        "Answer": qa["answer"]
                    })

        except Exception as e:
            print(f"[X] Error generating QA for {gt_filename}: {e}")

# Save generated QA pairs to CSV
csv_path = os.path.join(base_dir, "generated_QA.csv")

df = pd.DataFrame(all_records)
df.to_csv(csv_path, index=False, encoding="utf-8")
print(f"✅ All QAs saved to: {csv_path}")
