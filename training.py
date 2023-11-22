import numpy as np
import torch
from datetime import datetime
from transformers import AutoTokenizer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_checkpoint = 'Helsinki-NLP/opus-mt-en-es'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# A function to encapsulate th training loop
def train(model, criterion, optimizer, train_loader, valid_loader, epochs):
  train_losses = np.zeros(epochs)
  test_losses = np.zeros(epochs)

  for it in range(epochs):
    model.train()
    t0 = datetime.now()
    train_loss = []
    for batch in train_loader:
      #move data to GPU (enc_input, enc_mask, translation)
      batch = {k: v.to(device) for k,v in batch.items()}

      #zero the parameter gradients
      optimizer.zero_grad()

      #encoder inputs and masking
      enc_input = batch["input_ids"]
      enc_mask = batch["attention_mask"]

      #decoder target
      targets = batch["labels"]

      #shift targets forwards to det decoder_input
      #this is the opposite of before since we are getting the input not target
      dec_input = targets.clone().detach()
      dec_input = torch.roll(dec_input, shifts=1, dims=1)
      dec_input[:, 0] = 65_001

      #also convert all -100 to pad token id
      dec_input = dec_input.masked_fill(
          dec_input == -100, tokenizer.pad_token_id
      )

      #make decoder input mask
      dec_mask = torch.ones_like(dec_input)
      dec_mask = dec_mask.masked_fill(dec_input == tokenizer.pad_token_id, 0)

      #forward pass
      outputs = model(enc_input, dec_input, enc_mask, dec_mask) #transformer model
      loss = criterion(outputs.transpose(2,1), targets)

      #backward and optimize
      loss.backward()
      optimizer.step()
      train_loss.append(loss.item())

    # Get  train loss and test loss
    train_loss = np.mean(train_loss)

    model.eval()
    test_loss = []
    for batch in valid_loader:
      batch = {k: v.to(device) for k,v in batch.items()}

      enc_input = batch["input_ids"]
      enc_mask = batch["attention_mask"]
      targets = batch["labels"]

      #shift targets forwards to get decoder_input
      dec_input = targets.clone().detach()
      dec_input = torch.roll(dec_input, shifts=1, dims=1)
      dec_input[:, 0] = 65_001

      #also convert all -100 to pad token id
      dec_input = dec_input.masked_fill(
          dec_input == -100, tokenizer.pad_token_id
      )

      dec_mask = torch.ones_like(dec_input)
      dec_mask = dec_mask.masked_fill(dec_input == tokenizer.pad_token_id, 0)

      outputs = model(enc_input, dec_input, enc_mask, dec_mask)
      loss = criterion(outputs.transpose(2,1), targets)
      test_loss.append(loss.item())
    test_loss = np.mean(test_loss)

    #save losses
    train_losses[it] = train_loss
    test_losses[it]= test_loss

    dt = datetime.now() - t0
    print(f"Epoch {it+1}/{epochs}, Train Loss: {train_loss: .4f}, \
    Test Loss: {test_loss:.4f}, Duration {dt}")
  return train_losses, test_losses