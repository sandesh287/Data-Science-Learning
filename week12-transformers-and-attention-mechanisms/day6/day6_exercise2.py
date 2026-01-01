# Use the GPT-3 API for text generation and analyze the quality of the generated text.

# Note: Make sure to install the OpenAI Python client library using:
# pip install openai

# Since we need payment to use the GPT-3 API, I have included a placeholder for the API key. Please replace 'your_api_key_here' with your actual API key to run the code.



# libraries
from openai import OpenAI


# Set OpenAI API key
client = OpenAI(
  api_key='your_api_key_here'
)


try:
  # Generate text using GPT-3.5 Turbo
  response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[
      {'role':'system', 'content':'You are a helpful assistant.'},
      {'role':'user', 'content':'Write a short story about a robot learning to cook.'}
    ],
    max_tokens=150,
    temperature=0.7
  )
  
  print('Generated Text: \n', response['choices'][0]['message']['content'].strip())

except Exception as e:
  print(f'An error occured: {e}')
