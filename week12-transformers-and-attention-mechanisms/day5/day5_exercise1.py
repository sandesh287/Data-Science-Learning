# Hands-on with Pre-Trained Transformers BERT and GPT
# Use a Hugging Face's Transformers library to fine-tune a pre-trained BERT and GPT model for a text classification task

# pip install transformers datasets

# Since the dataset is too big, it requires more resources, which cannot be handled by our personal computer. It is going to take around 8-10 hours to train the model. I have tried training the model, but my computer only handles for 1% training.



# libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset


# load dataset
dataset = load_dataset('imdb')


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


# tokenize the dataset
def tokenize_function(examples):
  return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


# Prepare data for training
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets.set_format('torch')

train_dataset = tokenized_datasets['train']
test_dataset = tokenized_datasets['test']


# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)


# Define training arguments
training_args = TrainingArguments(
  output_dir='./results',
  evaluation_strategy='epoch',
  learning_rate=2e-5,
  per_device_train_batch_size=8,
  per_device_eval_batch_size=8,
  num_train_epochs=3,
  weight_decay=0.01,
  logging_dir='.logs',
  logging_steps=10,
  save_steps=500
)


# Train the model
trainer = Trainer(
  model=model, 
  args=training_args, 
  train_dataset=train_dataset, 
  eval_dataset=test_dataset,
  tokenizer=tokenizer
)

trainer.train()


# Evaluate model
results = trainer.evaluate()

print(f'Evaluation Results: {results}')





# Experiment with GPT

from transformers import AutoModelForCausalLM

gpt_model = AutoModelForCausalLM.from_pretrained('gpt2')

input_text = 'Once upon a time'
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = gpt_model.generate(input_ids, max_length=50, num_return_sequences=1)

print(f'Generated Text: {tokenizer.decode(output[0], skip_special_token=True)}')