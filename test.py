import openai
from dotenv import load_dotenv
import os

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # Make sure this is correctly named in your .env file

# Create an instance of the OpenAI client
client = openai.OpenAI(api_key=api_key)

def generate_response(prompt):
    try:
        # Using the client to create a chat completion with the specified model
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Make sure this model is available in your API plan
            messages=[{"role": "user", "content": prompt}]
        )
        # Correctly accessing the text content of the response
        return response.choices[0].message.content
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    prompt = "Is this working?"
    result = generate_response(prompt)
    print("Response from OpenAI GPT:", result)
