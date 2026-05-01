import os
from dotenv import load_dotenv
from openai import OpenAI

# Load the API key from your .env file
load_dotenv()

# Set up the OpenRouter client
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

print("Connecting to OpenRouter...")

try:
    response = client.chat.completions.create(
      model="openai/gpt-oss-120b:free", # Using the best free model
      messages=[
        {"role": "system", "content": "You are a helpful support triage agent. Reply in strict JSON format."},
        {"role": "user", "content": "how many r's is in strawberry ? Format as {'answer': value}"}
      ]
    )
    
    print("\nSuccess! Here is the AI's response:")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"\nError: {e}")