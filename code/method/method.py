import os
import base64
import pandas as pd
import numpy as np
from PIL import Image
from openai import AzureOpenAI
import time

# =========================
# Global configuration
# =========================

MAX_QA = 10000
TOP_K = 3  # Number of retrieved images used for reasoning

# Lightweight candidate pool parameters
MIN_ANG, CAP_ANG = 12, 30      
MIN_POS, CAP_POS = 36, 80      
ANG_PER_K = 3                 
POS_PER_ANG = 2.5            


# =========================
# Image utilities
# =========================

def compress_image(input_path, output_path, max_size=(800, 600), quality=70):
    """Compress and resize an image to reduce payload size for LLM queries."""
    img = Image.open(input_path).convert("RGB")
    img.thumbnail(max_size)
    img.save(output_path, "JPEG", quality=quality)


def convert_image_to_data_url(image_path):
    """Convert an image file to a base64-encoded data URL."""
    with open(image_path, "rb") as image_file:
        data = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


# =========================
# Parsing utilities
# =========================

def parse_position(pos_str):
    """Parse a 3D position string into a numpy array."""
    try:
        x, y, z = map(float, pos_str.strip("()").split(","))
        return np.array([x, y, z])
    except:
        return np.zeros(3)


def parse_rotation(rot_str):
    """Parse a quaternion rotation string into a numpy array."""
    try:
        x, y, z, w = map(float, rot_str.strip("()").split(","))
        return np.array([x, y, z, w])
    except:
        return np.array([0, 0, 0, 1])


def rotation_diff_deg(q1, q2):
    """Compute angular difference (in degrees) between two quaternions."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    angle_rad = 2 * np.arccos(abs(dot))
    return np.degrees(angle_rad)


# =========================
# Prompt construction
# =========================

def get_pairwise_rag_prompt(question, current_image_url, retrieved_image_url, step_num):
    """
    Construct a pairwise comparison prompt between a retrieved historical image
    and the current image for object presence reasoning.
    """
    instruction = f"""You are comparing two VR scene images to help answer a question about object presence.

The two images are:
- The first picture: Retrieved scene image (from an earlier time)
- The second picture: Current scene image

---
This is an example for your reference.

Example Question:  
"Was the trash bin at this location before?"

Answer: 
"In the retrieved picture, you can clearly see a cylindrical trash bin beside the column near the entrance. In the current picture, that bin is no longer there. So it has disappeared."

---

Your task is to carefully observe the two pictures and answer the following question.

Question: "{question}"

Choose one of the following answers:
- If the item in the question appears in the retrieved picture but not in the current one, answer: "In the retrieved picture...So it has disappeared."
- If the item in the question appears both in the retrieved picture and in the current picture, answer: "In the retrieved picture...So it has always been here."
- If the item in the question neither appears in the retrieved picture nor in the current picture, answer: "In the retrieved picture...So it was never there."
- If the two pictures seem to have no connection at all, answer: "Not enough information."

Carefully examine the pictures and don't miss any details!
"""
    content_list = [{"type": "text", "text": instruction}]
    content_list.append({"type": "image_url", "image_url": {"url": retrieved_image_url}})
    content_list.append({"type": "image_url", "image_url": {"url": current_image_url}})
    
    return [
        {"role": "system", "content": "You are a helpful assistant for reasoning about object changes in VR scenes."},
        {"role": "user", "content": content_list}
    ]


def get_final_reasoning_prompt(question, observations):
    """
    Aggregate multiple pairwise observations into a final reasoning prompt
    to resolve inconsistencies caused by viewpoint or temporal differences.
    """
    joined_obs = "\n".join([f"{i+1}. {obs}" for i, obs in enumerate(observations)])
    summary_instruction = f"""
To compare the changes of items at a certain location in the scene compared to the previous time, we retrieved three pictures from the previous time by time, compared them with the current pictures respectively, and gave answers to the current questions respectively. We offer you these three sub-answers. Due to inconsistent shooting angles, some sub-answers might be incorrect (for instance, assume there are no objects at the location just because no objects were observed). You can correct the errors based on the other two sub-answers and arrive at the final answer.

---

This is two example for your reference.

Example Question 1:  
"Was the dinosaur fossil at this location before?"

Example sub-answers based on three retrieved images:
1. The fossil is present in the retrieved image, but it is not present in the current image. So it has disappeared.

2. The fossil is present in the retrieved image, but it is not present in the current image. So it has disappeared. 

3. The fossil is not present in both images. So it was never there.

Final answer:  
"It has disappeared. Because the fossil appear in the first and second retrieved pictures, but not in the third retrieved picture and the current picture, and these pictures are in chronological order, it indicates that the fossil may has disappeared over time."

Example Question 2:  
"Was there an bench next to the trash bin before?"

Example sub-answers based on three retrieved images:
1. The bench does not appear in either the retrieved image or the current image. So it was never there.

2. The bench next to the trash bin does not appear in either the retrieved image or the current image. So it was never there.

3. The bench is present in the retrieved image, but it is not present in the current image. So it has disappeared. 

Final answer:  
"It has disappeared. Although there is no bench in the first and second retrieved images, there is one in the third retrieved image. The inconsistent sub-answers might be due to the incorrect shooting angles of the first and second retrieved images, which failed to capture the bench. Only the third retrieved image has the correct Angle and captured the bench. It indicates that there was a bench here before. However, the current image does not have a bench, indicating that the bench has disappeared."

---

This is the question you need to answer:

Question: "{question}"

These are the sub-answers:
{joined_obs}

Now, based on these sub-answers, give me the final answer:

Choose one of the following answers:
- "It has disappeared. Because..."
- "It has always been here. Because..."
- "It was never there. Because..."
- "Not enough information."

Note that these sub-answers are based on pictures from different times and perspectives. Therefore, when the three sub-answers are inconsistent, you need to deduce the true answer.
"""
    
    return [
        {"role": "system", "content": "You are a helpful assistant for reasoning about object changes in VR scenes."},
        {"role": "user", "content": summary_instruction}
    ]


# =========================
# Azure GPT
# =========================

class GPTAzureRAG():
    """Wrapper for Azure OpenAI chat completion API."""
    def __init__(self, deployment_name):
        self.set_API_key()
        self.deployment_name = deployment_name
        self.temperature = 0.7

    def set_API_key(self):
        self.client = AzureOpenAI(
            api_key="YOUR_API_KEY",
            api_version="YOUR_API_VERSION",
            azure_endpoint="YOUR_AZURE_ENDPOINT"
        )

    def query(self, messages, retries=3):
        """Query the LLM with retry logic."""
        for attempt in range(retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=self.temperature
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                print(f"[X] GPT query failed (Attempt {attempt+1}): {e}")
                time.sleep(2)
        return "Not enough information."


# =========================
# pipeline
# =========================

def generate_rag_results_for_folder(folder_path, gpt, qa_df):
    """Run RAG-based object change reasoning for a single trajectory folder."""
    folder_name = os.path.basename(folder_path)
    screenshot_path = os.path.join(folder_path, "screenshot")
    compressed_path = os.path.join(screenshot_path, "compressed")
    os.makedirs(compressed_path, exist_ok=True)

    scene_csv = os.path.join(screenshot_path, "data.csv")
    rag_out_path = os.path.join(folder_path, "results.csv")

    if os.path.exists(rag_out_path):
        return
    if not os.path.exists(scene_csv):
        return

    scene_df = pd.read_csv(scene_csv)
    qa_df = qa_df[qa_df["Folder"].astype(str) == folder_name]
    if MAX_QA is not None:
        qa_df = qa_df.head(MAX_QA)

    all_indices, all_img_paths, all_positions, all_rotations = [], [], [], []

    # Preload all scene metadata
    for _, row in scene_df.iterrows():
        img_file = row["ScreenshotFilename"]
        idx = int(row["Index"])
        pos = parse_position(row["Position"])
        rot = parse_rotation(row["Rotation"])
        raw_path = os.path.join(screenshot_path, img_file)
        compressed_img_name = os.path.splitext(img_file)[0] + "_compressed.jpg"
        compressed_path_full = os.path.join(compressed_path, compressed_img_name)

        if not os.path.exists(compressed_path_full):
            compress_image(raw_path, compressed_path_full)

        all_indices.append(idx)
        all_img_paths.append(compressed_path_full)
        all_positions.append(pos)
        all_rotations.append(rot)

    rag_records = []

    for i, (_, row) in enumerate(qa_df.iterrows()):
        img_name = row["SC_Image"]
        question = row["Question"]
        answer = row["Answer"]

        match = scene_df[scene_df["ScreenshotFilename"] == img_name]
        if match.empty:
            continue

        query_idx = int(match["Index"].values[0])
        query_pos = parse_position(match["Position"].values[0])
        query_rot = parse_rotation(match["Rotation"].values[0])

        current_img_file = scene_df[scene_df["Index"] == query_idx]["ScreenshotFilename"].values[0]
        current_compressed = os.path.splitext(current_img_file)[0] + "_compressed.jpg"
        current_img_path = os.path.join(compressed_path, current_compressed)

        print(f"\n Processing Q{i}: {question}")
        print(f" Folder: {folder_name} | Image: {current_img_file} | Index: {query_idx}")

        # Select only historical frames
        subset_mask = [idx < query_idx for idx in all_indices]
        subset_indices = np.array(all_indices)[subset_mask]
        subset_paths = np.array(all_img_paths)[subset_mask]
        subset_positions = np.array(all_positions)[subset_mask]
        subset_rotations = np.array(all_rotations)[subset_mask]

        # Positional distance filtering
        dists = np.linalg.norm(subset_positions - query_pos, axis=1)

        # Adaptive candidate budgeting
        K_ang = min(len(dists), CAP_ANG, max(MIN_ANG, ANG_PER_K * TOP_K))
        K_ang = max(int(np.ceil(K_ang)), TOP_K)

        K_pos = min(len(dists), CAP_POS, max(MIN_POS, int(np.ceil(POS_PER_ANG * K_ang))))
        K_pos = max(K_pos, K_ang)

        # Positional filtering
        idx_pos = np.argpartition(dists, K_pos - 1)[:K_pos]

        # Rotation filtering
        selected_rotations = subset_rotations[idx_pos]
        angles = np.array([rotation_diff_deg(rot, query_rot) for rot in selected_rotations])

        K_ang = min(len(idx_pos), K_ang)
        idx_ang_rel = np.argpartition(angles, K_ang - 1)[:K_ang]

        # Temporal ordering and final TOP_K selection
        final_idx = idx_pos[idx_ang_rel]
        final_indices = np.array(subset_indices)[final_idx]
        order_rel = np.argsort(final_indices)[:TOP_K]
        chosen_rel = final_idx[order_rel]

        retrieved_indices = subset_indices[chosen_rel]
        top_k_image_urls = [convert_image_to_data_url(subset_paths[i]) for i in order_rel]

        current_image_url = convert_image_to_data_url(current_img_path)

        observations = []
        for j, url in enumerate(top_k_image_urls):
            pairwise_prompt = get_pairwise_rag_prompt(question, current_image_url, url, j+1)
            obs = gpt.query(pairwise_prompt)
            print(obs)
            observations.append(obs)

        summary_prompt = get_final_reasoning_prompt(question, observations)
        generated = gpt.query(summary_prompt)

        print(f" GPT Answer: {generated}")

        rag_records.append({
            "Folder": folder_name,
            "Index": query_idx,
            "Image": current_img_file,
            "Question": question,
            "Answer": answer,
            "GeneratedAnswer": generated,
            "Sub_Answers": observations,
            "TopK": TOP_K,
            "RetrievedIndices": ",".join(map(str, retrieved_indices))
        })

    out_df = pd.DataFrame(rag_records)
    out_df.to_csv(rag_out_path, index=False)


# =========================
# Run
# =========================

def run_all_rag(base_dir):
    """Run RAG for all trajectory folders in a domain."""
    gpt = GPTAzureRAG(deployment_name="gpt-4o")
    qa_path = os.path.join(base_dir, "generated_QA.csv")
    if not os.path.exists(qa_path):
        print("[X] generated_QA.csv not found.")
        return
    qa_df = pd.read_csv(qa_path)

    for folder in sorted(os.listdir(base_dir)):
        if not folder.isdigit():
            continue
        folder_path = os.path.join(base_dir, folder)
        generate_rag_results_for_folder(folder_path, gpt, qa_df)


def run_all_domains(dataset_root="./dataset", domains=None):
    """Run RAG pipeline across multiple domains."""
    if domains is None:
        domains = ["architecture", "fastfood", "market", "museum", "village"]
    for name in domains:
        base_dir = os.path.join(dataset_root, name)
        if not os.path.isdir(base_dir):
            print(f"[!] Skip missing: {base_dir}")
            continue
        print(f"\n===== Processing domain: {name} =====")
        run_all_rag(base_dir)


if __name__ == "__main__":
    run_all_domains(
        dataset_root="./dataset",
        domains=["architecture", "fastfood", "market", "museum", "village"]
    )
