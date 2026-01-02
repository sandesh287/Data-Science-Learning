# Domain Adaptation and Transfer Learning Chanllenges
# Fine-tune a pre-trained model on a domain-specific dataset (eg. BERT for medical text classification) and experiment with domain-specific embeddings

# The dataset we are using is very large dataset and it cannot be done on a simple computer. You need a very powerful machine or somewhere on the cloud.



# libraries
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split


# Load PubMed 20k RCT dataset
dataset = load_dataset("nanyy1025/pubmed_rct_20k")

print(dataset['train'].column_names)
# print(dataset['train'][0])
print(dataset)


# Split 90% train and 10% validation
tokenized_datasets = dataset['train'].train_test_split(test_size=0.1, seed=42)

print(tokenized_datasets)


# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


# Tokenize data
def preprocess_data(examples):
  # Join the list of sentences into a single string
  texts = [" ".join(x) if isinstance(x, list) else x for x in examples["text"]]
  return tokenizer(texts, truncation=True, padding="max_length", max_length=128)


# Apply tokenizer to flattened datasets
tokenized_datasets = tokenized_datasets.map(preprocess_data, batched=True)


# Encode labels
# Create label mapping
label_list = list(set(x for row in dataset['train']['target'] for x in row))
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}


# Convert string labels to integer IDs
def encode_labels(examples):
    examples['labels'] = [label2id[x[0]] for x in examples['target']]  # take first label in list
    return examples


tokenized_datasets = tokenized_datasets.map(encode_labels, batched=True)

# Keep only the columns we need for Trainer
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# Load pretrained BERT or Domain Specific model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

# For Domain Specific Model (for BioBERT)
# model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1", num_labels=5)


# Fine-tune model

# Training Arguments
training_args = TrainingArguments(
  output_dir='./results',
  evaluation_strategy='epoch',
  learning_rate=2e-5,
  per_device_train_batch_size=4,
  per_device_eval_batch_size=4,
  num_train_epochs=1,
  weight_decay=0.01,
  report_to='none'
)


# Create trainer
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(2000)),
  eval_dataset=tokenized_datasets["test"].shuffle(seed=42).select(range(500)),
  tokenizer=tokenizer
)


# Train model
trainer.train()


# Evaluate model
results = trainer.evaluate()

print(f'Evaluation Results: {results}')



# Note: Addressing data mismatch with Augmentation: can use paraphrasing or synonym substitution to augment the datasets

# Data mismatch with Data Augmentation
import random

def augment_text(text):
  synonyms = {'cancer': ['tumor', 'malignancy'], 'study': ['research', 'experiment']}
  words = text.split()
  new_words = [random.choice(synonyms[word]) if word in synonyms else word for word in words]
  return ' '.join(new_words)

# Apply augmentation
augmented_data = [augment_text(sample['text_combined']) for sample in dataset['train'][:5]]

print(augmented_data)