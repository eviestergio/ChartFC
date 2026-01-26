from openai import OpenAI

client = OpenAI(api_key='9Y7GbYUw3f65G60UYPSZT3BlbkFJrrxHa78SVepPHKXffWs2')
from PIL import Image
import base64
import requests
from io import BytesIO

# Set up OpenAI API key

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Function to interact with OpenAI GPT-4 API
def gpt4_with_image(image_path, prompt):
    encoded_image = encode_image(image_path)

    response = client.chat.completions.create(model="gpt-4-vision",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ],
    functions=[
        {
            "name": "interpret_image",
            "parameters": {
                "image": {
                    "content": encoded_image,
                },
            },
        }
    ])

    return response.choices[0].message.content

# Example usage
if __name__ == "__main__":
    image_path = "seed_datasets_12/5_final_dataset_12/train/png/chartQA_multi_col_20204-train.png"  # Change to the path of your image
    prompt = "Given the chart in the image input, please explain the reasoning and then answer: What is the appropriate label ('supports', 'refutes', or 'not enough information') for the following claim: '517 people committed suicide in 2009.'?"

    result = gpt4_with_image(image_path, prompt)
    print("Output Text: ", result)
