# # With pipeline, just specify the task and the model id from the Hub.
# from transformers import pipeline
# pipe = pipeline("text-generation", model="distilbert/distilgpt2", device=0)

# import torch
# print(torch.cuda.is_available())

# # If you want more control, you will need to define the tokenizer and model.
# from transformers import AutoTokenizer, AutoModelForCausalLM
# tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
# model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

from PIL import Image
import requests
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch

torch.hub.download_url_to_file('https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_1229.png', 'chart_example_1.png')

image_path = "seed_datasets_12/5_final_dataset_12/train/png/plotQA_37698-train.png"
input_text = "Given the chart in the image input, what is the appropriate label ('supports', 'refutes', or 'not enough information') for the following claim: 'There are 3 bars on the 2nd tick from the top.'? Why?"

# Load Model
model = PaliGemmaForConditionalGeneration.from_pretrained("ahmed-masry/chartgemma", torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Process Inputs
image = Image.open(image_path).convert('RGB')
inputs = processor(text=input_text, images=image, return_tensors="pt")
prompt_length = inputs['input_ids'].shape[1]
inputs = {k: v.to(device) for k, v in inputs.items()}


# Generate
generate_ids = model.generate(**inputs, num_beams=4, max_new_tokens=512)
output_text = processor.batch_decode(generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output_text)