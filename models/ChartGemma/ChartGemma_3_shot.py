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

# Load Model
model = PaliGemmaForConditionalGeneration.from_pretrained("ahmed-masry/chartgemma", torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

torch.hub.download_url_to_file('https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_1229.png', 'chart_example_1.png')

image_path = "seed_datasets_12/5_final_dataset_12/train/png/chartQA_multi_col_20204-train.png"
chart = image_path 
claim = '517 people committed suicide in 2009.'

chart1 = "seed_datasets/FigureQA/test/png/figureQA_2-3-test.png"
chart2 = "seed_datasets/PlotQA/test/png/plotQA_39-test.png"
chart3 = "seed_datasets/ChartQA/test/png/chartQA_166.png"

# Correct placeholders for charts and claims
input_text = f"""

Given the chart '{chart}', what is the correct label ('supports', 'refutes', or 'not enough information') for the claim '{claim}'? Why? 
Output the result as a JSON object with the following keys: "label" and "explanation". The format should strictly follow this structure:
{{
"label": "your label for the claim",
"explanation": "your explanation for the label selected"
}}

Here are some input-output examples:
<
1. Input:
{{
"chart": '{chart1}',
“claim”: “Crimson is the maximum.”
}}
Output:
{{
"label": "refutes",
"explanation": "The claim directly contradicts the chart, which clearly shows that Gold has the maximum value at 59.05. By asserting that Crimson is the maximum, it directly opposes the factual information provided, refuting the chart."
}}
2. Input:
{{
"chart": '{chart2}',
“claim": "The income share held by the highest 10% of the population was greater than the average income share held by the highest 10% of the population in 1 year."
}}
Output:
{{
"label": "supports",
"explanation": "The chart shows data over several years where the income shares of the highest 10% are listed. Comparing these figures reveals that in one specific year, the income share of the top 10% exceeded the average income share held by the top 10% across all years."
}}
3. Input:
{{
"chart": '{chart3}',
“claim”: “President Donald Trump's approach to international relations contributes to their perception as Dangerous.”
}}
Output:
{{
"label": "not enough information",
"explanation": "This claim speculates on information not provided in the chart, as there is no data regarding international relations, making the connection to the perception of being 'Dangerous' unverifiable."
}}
>

"""

# Process Inputs
image = Image.open(chart).convert('RGB')
image1 = Image.open(chart1).convert('RGB')
image2 = Image.open(chart2).convert('RGB')
image3 = Image.open(chart3).convert('RGB')

images = [image, image1, image2, image3]

inputs = processor(text=input_text, images=images, return_tensors="pt")
prompt_length = inputs['input_ids'].shape[1]
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate output
generate_ids = model.generate(**inputs, num_beams=4, max_new_tokens=1000)
output_text = processor.batch_decode(generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(output_text)

# input_text = """

# Given the chart {chart}, what is the correct label ('supports', 'refutes', or 'not enough information') for the claim {claim}? Why? 
# Output the result as a JSON object with the following keys: "label" and "explanation". The format should strictly follow this structure:
# {{
# "label": "your label for the claim",
# "explanation": "your explanation for the label selected"
# }}

# Here are some input-output examples:
# <
# 1. Input:
# {{
# "chart": {chart1}
# “claim”: “Crimson is the maximum.”
# }}
# Output:
# {{
# "label": "refutes",
# "explanation": "The claim directly contradicts the chart, which clearly shows that Gold has the maximum value at 59.05. By asserting that Crimson is the maximum, it directly opposes the factual information provided, refuting the chart."
# }}
# 2. Input:
# {{
# "chart": {chart2}
# “claim": "The income share held by the highest 10% of the population was greater than the average income share held by the highest 10% of the population in 1 year."
# }}
# Output:
# {{
# "label": "supports",
# "explanation": "The chart shows data over several years where the income shares of the highest 10% are listed. Comparing these figures reveals that in one specific year, the income share of the top 10% exceeded the average income share held by the top 10% across all years.."
# }}
# 3. Input:
# {{
# "chart": {chart3}
# “claim”: “President Donald Trump's approach to international relations contributes to their perception as Dangerous.”
# }}
# Output:
# {{
# “label”: “not enough information”,
# “explanation”: “This claim speculates on information not provided in the chart, as there is no data regarding international relations, making the connection to the perception of being 'Dangerous' unverifiable.”
# }}
# >

# """



# # Process Inputs
# image = Image.open(chart).convert('RGB')
# image1 = Image.open(chart1).convert('RGB')
# image2 = Image.open(chart2).convert('RGB')
# image3 = Image.open(chart3).convert('RGB')

# images = [image, image1, image2, image3]

# inputs = processor(text=input_text, images=images, return_tensors="pt")
# prompt_length = inputs['input_ids'].shape[1]
# inputs = {k: v.to(device) for k, v in inputs.items()}


# # Generate
# generate_ids = model.generate(**inputs, num_beams=4, max_new_tokens=1000)
# output_text = processor.batch_decode(generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# print(output_text)
