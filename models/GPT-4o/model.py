## Currently not in use 

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv('API_KEY'))

class QueryModel:
    """
    The class that handles model queries and responses.
    """

    def __init__(self, query_type=None, params_dict=None):
        """
        The constructor.
        Args:
            query_type (string): The type of response expected ('completion' or 'chat').
            params_dict (dict): The parameters of the model in dict format.
        """
        self.type = query_type if query_type is not None else 'chat'
        self.params = params_dict if params_dict is not None else {
            'temperature': 0,
            'max_tokens': 50,
            'top_p': 1,
            'frequency_penalty': 0,
            'presence_penalty': 0
        }

    def __call__(self, model_name, query, *args, **kwargs):
        """
        A function that returns the response of an OpenAI model to a query.
        Args:
            model_name (string): The name of the model - note: GPT-3.5 and up only work on 'chat' query_type, whereas text-davinci-003 and below only work in completion mode.
            query (string): The query string.
        Return:
            response_text (string): The response from the model.
        """
        if self.type == 'completion':
            response = client.completions.create(model=model_name, prompt=query, **self.params)
            response_text = response.choices[0].text.strip()
            print("Answer is: " + response_text)
            return response_text
        else:
            response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": query}])
            response_text = response.choices[0].message.content.strip()
            print("Answer is: " + response_text)
            return response_text
