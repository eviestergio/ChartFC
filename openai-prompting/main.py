from model import QueryModel
import json
import os


def process_file(input_file):
    model = QueryModel(query_type='chat')
    
    with open(input_file, 'r') as file:
            entries = json.load(file)

    results = []
    
    for entry in entries:

        # 2. Few-shot
        prompt = f"""
            You will be provided with a data entry in JSON format deliminated by 3 single quotes. 

            Each data entry contains the following keys:
            'image', 'question', 'answer'

            Using the 4 input-output examples deliminated by angle brackets, your task is to convert 
            each 'question' and 'answer' pair to a claim that supports the information.

            Examples: < 
            1. Input: {
                "image": "chartQA_multi_col_803.png",
                "question": "How many stores did Saint Laurent operate in Western Europe in 2020?",
                "answer": "47"
            }
            Output: "Saint Laurent operated 47 stores in Western Europe in 2020." 
            2. Input: {
                "image": "plotQA_11.png",
                "question": "What is the title of the graph ?",
                "answer": "Net disbursements of loans from International Monetary Fund"
            }
            Output: "The title of the graph is Net disbursements of loans from International Monetary Fund."
            3. Input: {
                "image": "figureQA_400.png",
                "question": "Is Turquoise the roughest?",
                "answer": "No"
            }
            Output: "Turquoise is not the roughest."
            4. Input: {
                "image": "figureQA_514.png",
                "question": "Is Periwinkle greater than Green Yellow?",
                "answer": "Yes"
            }
            Output: "Periwinkle is greater than Green Yellow."
            >

            Output only the text for the claim derived from converting the question and answer pair in double quotes.

            Data entry: '''{entry}'''
            """
        
        response = model(model_name='gpt-4', query=prompt)
        
        result_entry = {
            "image": entry.get("image", ""),
            "claim": response,
            "label": "Supports"
        }

        results.append(result_entry)

    output_file = os.path.basename(input_file).replace('preprocessed', 'converted')
    with open(output_file, 'w') as output_file:
        json.dump(results, output_file, indent=4)

    print(f"Conversion completed for {os.path.basename(input_file)}. Results saved to {output_file.name}.")

def main():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    input_directory = os.path.join(current_folder, "../data_preprocessing") # directory containing JSON files to convert

    for filename in os.listdir(input_directory):
        if filename.endswith('.json') and filename.startswith('preprocessed_'):
            input_file = os.path.join(input_directory, filename)
            process_file(input_file)

if __name__ == '__main__':
    main()