import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
try:
    client = AzureOpenAI(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
    )
    print("Azure OpenAI client initialized successfully")
except Exception as e:
    print(f"Failed to initialize Azure OpenAI client: {str(e)}")
    client = None

# Test the client with a simple request
if client:
    try:
        response = client.chat.completions.create(
            model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello to the world."}
            ],
            temperature=0.5,
            max_tokens=50
        )
        print("Successfully received Azure OpenAI response")
        print("Response:", response.choices[0].message.content)
    except Exception as e:
        print(f"Error making request to Azure OpenAI: {str(e)}")
