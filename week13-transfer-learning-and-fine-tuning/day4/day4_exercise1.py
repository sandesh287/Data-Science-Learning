# Transfer Learning in NLP
# Fine-tune a pre-trained BERT for sentiment analysis using Hugging Face's Transformers library.
# Preprocess the text data, tokenize it, and evaluate the model



# libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset


# Load imdb dataset
dataset = load_dataset('imdb')


# load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


# Tokenize the dataset
def tokenize_function(examples):
  return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


# Prepare data for training
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets.set_format('torch')


# Get train and test dataset
train_dataset = tokenized_datasets['train']
test_dataset = tokenized_datasets['test']


# Load pre-trained BERT model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)


# Define training arguments
training_args = TrainingArguments(
  output_dir='./results',
  evaluation_strategy='epoch',
  learning_rate=2e-5,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=16,
  num_train_epochs=1,
  weight_decay=0.01,
  save_total_limit=2
)


# Fine-tune the model
trainer = Trainer(
  model = model,
  args=training_args,
  train_dataset=train_dataset,
  eval_dataset=test_dataset,
  tokenizer=tokenizer
)


# Train the model
trainer.train()


# Evaluation
results = trainer.evaluate()
print(f'Evaluation Results: {results}')