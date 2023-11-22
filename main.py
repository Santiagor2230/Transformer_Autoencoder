import torch
import torch.nn as nn
from encoder_model import Encoder
from decoder_model import Decoder
from transformer_model import Transformer
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
from training import train

model_checkpoint = 'Helsinki-NLP/opus-mt-en-es'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

encoder = Encoder(vocab_size = tokenizer.vocab_size + 1,
                  max_len = 512,
                  d_k=16,
                  d_model=64,
                  n_heads = 4,
                  n_layers=2,
                  dropout_prob=0.1)
decoder = Decoder(vocab_size=tokenizer.vocab_size + 1,
                  max_len=512,
                  d_k=16,
                  d_model=64,
                  n_heads=4,
                  n_layers=2,
                  dropout_prob=0.1)
transformer = Transformer(encoder, decoder)

raw_dataset = load_dataset("csv", data_files="\content\spa.csv")


split = raw_dataset["train"].train_test_split(test_size=0.3, seed=42)

#sequence length
max_input_length = 128
max_target_length = 128

def preprocess_function(batch):
  model_inputs = tokenizer(
      batch["en"], max_length=max_input_length, truncation=True
  )

  # Set up the tokenizer for targets
  labels = tokenizer(
      text_target=batch["es"], max_length = max_target_length, truncation=True
  )

  model_inputs["labels"] = labels["input_ids"]
  return model_inputs


#tokenizing the dataset that is split
tokenized_datasets = split.map(
    preprocess_function,
    batched=True,
    remove_columns=split["train"].column_names,
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

#cuda for CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
encoder.to(device)
decoder.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(transformer.parameters())


train_loader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=32,
    collate_fn=data_collator
)

valid_loader = DataLoader(
    tokenized_datasets["test"],
    batch_size = 32,
    collate_fn = data_collator
)

train_losses, test_losses = train(
    transformer, criterion, optimizer, train_loader, valid_loader, epochs=20
)

def translate(input_sentence):
  # get encoder output first
  enc_input = tokenizer(input_sentence, return_tensors='pt').to(device)
  enc_output = encoder(enc_input['input_ids'], enc_input['attention_mask'])

  # setup initial decoder input
  dec_input_ids = torch.tensor([[65_001]], device=device)
  dec_attn_mask = torch.ones_like(dec_input_ids, device=device)

  # now do the decoder loop
  for _ in range(32):
    dec_output = decoder(
        enc_output,
        dec_input_ids,
        enc_input['attention_mask'],
        dec_attn_mask,
    )

    # choose the best value (or sample)
    prediction_id = torch.argmax(dec_output[:, -1, :], axis=-1)

    # append to decoder input
    dec_input_ids = torch.hstack((dec_input_ids, prediction_id.view(1, 1)))

    # recreate mask
    dec_attn_mask = torch.ones_like(dec_input_ids)

    # exit when reach </s>
    if prediction_id == 0:
      break

  translation = tokenizer.decode(dec_input_ids[0, 1:])
  print(translation)
  
sentence = input("English words:/n")
translate(sentence)