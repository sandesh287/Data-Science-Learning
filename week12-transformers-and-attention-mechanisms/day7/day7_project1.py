# Transformer Project: Text Summarization or Translation
# Fine-tune a pre-trained Transformer model (eg. T5 or BART) for text summarization or translation and evaluate its performace

# Dataset: CNN daily mail for data summarization, WMT14 for translation

# Note: Training may take several hours depending on the hardware. Consider using a smaller subset of the dataset for quicker experimentation.



# libraries
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer


# Load dataset for summarization
dataset = load_dataset('cnn_dailymail', '3.0.0')

print(dataset['train'][0])


# Datset for translation
# dataset = load_dataset('wmt14', 'en-fr')


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('t5-small')

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')


# Tokenize for summarization
def tokenize_function(examples):
  inputs = ['summarize: ' + doc for doc in examples['article']]
  model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
  
  # Tokenize targets
  with tokenizer.as_target_tokenizer():
    labels = tokenizer(examples['highlights'], max_length=150, truncation=True, padding='max_length')
  
  model_inputs['labels'] = labels['input_ids']
  return model_inputs


tokenized_datasets = dataset.map(tokenize_function, batched=True)


train_data = tokenized_datasets['train'].shuffle(seed=42).select(range(2000))
val_data = tokenized_datasets['validation'].shuffle(seed=42).select(range(500))


# Training arguments
training_args = TrainingArguments(
  output_dir='./results',
  evaluation_strategy='epoch',
  save_strategy='epoch',
  learning_rate=2e-5,
  per_device_train_batch_size=8,
  per_device_eval_batch_size=8,
  num_train_epochs=3,
  weight_decay=0.01,
  save_total_limit=2,
  load_best_model_at_end=True
)


# Trainer
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=train_data,
  eval_dataset=val_data,
  tokenizer=tokenizer
)


# Train the model
trainer.train()


# Define a sample text for summarization
sample_text = 'The Transformer model has revolutionized NLP by enabling parallel processing of sequences.'
inputs = tokenizer('summarize: ' + sample_text, return_tensors='pt', max_length=512, truncation=True)
outputs = model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)

print('Generated Summary: ', tokenizer.decoder(outputs[0], skip_special_token=True))


# Load Metric and evaluate
# metric = load_metric('rouge')
# predictions = outputs['generated_text']
# references = dataset['validation']['highlights']

# results = metric.compute(predictions=predictions, references=references)
# print(results)