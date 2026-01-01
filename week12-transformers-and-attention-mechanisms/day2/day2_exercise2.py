# Set up an environment for working with Transformers using PyTorch

# Set up environment for Transformer in PyTorch
# Must install torch, torchvision, transformers



# libraries
from transformers import BertTokenizer, BertModel


# Load a pre-defined BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# Tokenize a sample input
text = 'Transformers are powerful models for NLP tasks'
inputs = tokenizer(text, return_tensors='pt')


# Pass the input through the model
outputs = model(**inputs)
print(f'\nHidden States Shape: {outputs.last_hidden_state.shape}')