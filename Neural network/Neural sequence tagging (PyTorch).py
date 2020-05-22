#!/usr/bin/env python
# coding: utf-8

# # BERT for sequence labelling tasks (PyTorch example)
# <sup>This notebook is a part of Natural Language Processing class at the University of Ljubljana, Faculty for computer and information science. Please contact [slavko.zitnik@fri.uni-lj.si](mailto:slavko.zitnik@fri.uni-lj.si) for any comments.</sub>
# 
# This notebook uses the same data and model as the previous notebook, except it was coded in TensorFlow 2.0.
# 
# *Prerequisite libraries (wrt. previous notebooks)*
# 
# ```
# torch torchvision
# ```
# 
# We will use a [Kaggle dataset](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus) which is based on Groningen Meaning Bank dataset for named entity recognition.
# 
# The model example was inspired and parts of code are taken from [Tobias Sterbak's blog post](https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/).

# In[ ]:


import pandas as pd
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import transformers
from transformers import BertForTokenClassification, AdamW

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")


# In[ ]:


df_data = pd.read_csv("nn_data.csv", encoding="latin1").fillna(method="ffill")
df_data.shape


# In[ ]:


print(df_data.head)


# In[ ]:


tag_list = df_data.Tag.unique()
tag_list = np.append(tag_list, "PAD")
print(f"Tags: {', '.join(map(str, tag_list))}")


# In[ ]:


x_train, x_test = train_test_split(df_data, test_size=0.20, shuffle=False, random_state = 42)
x_val, x_test = train_test_split(x_test, test_size=0.50, shuffle=False, random_state = 42)


# In[ ]:


x_train.shape, x_val.shape, x_test.shape


# In[ ]:


agg_func = lambda s: [ [w,t] for w,t in zip(s["Word"].values.tolist(),s["Tag"].values.tolist())]


# In[ ]:


x_train_grouped = x_train.groupby("Sentence #").apply(agg_func)
x_val_grouped = x_val.groupby("Sentence #").apply(agg_func)
x_test_grouped = x_test.groupby("Sentence #").apply(agg_func)


# In[ ]:


x_train_sentences = [[s[0] for s in sent] for sent in x_train_grouped.values]
x_val_sentences = [[s[0] for s in sent] for sent in x_val_grouped.values]
x_test_sentences = [[s[0] for s in sent] for sent in x_test_grouped.values]


# In[ ]:


x_train_tags = [[t[1] for t in tag] for tag in x_train_grouped.values]
x_val_tags = [[t[1] for t in tag] for tag in x_val_grouped.values]
x_test_tags = [[t[1] for t in tag] for tag in x_test_grouped.values]


# In[ ]:


x_train_sentences[0]


# In[ ]:


x_train_tags[0]


# In[ ]:


label2code = {label: i for i, label in enumerate(tag_list)}
code2label = {v: k for k, v in label2code.items()}
label2code


# In[ ]:


MAX_LENGTH = 128
BATCH_SIZE = 32

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
#n_gpu = torch.cuda.device_count()

#if torch.cuda.is_available():
 #   print(f"GPU device: {torch.cuda.get_device_name(0)}")

#device = "cpu"

# In[ ]:


tokenizer = BertTokenizer.from_pretrained('slo-hr-en-bert-pytorch', from_pt=True, do_lower_case=False)


# In[ ]:


def convert_to_input(sentences,tags):
    input_id_list = []
    attention_mask_list = []
    label_id_list = []
    
    for x,y in tqdm(zip(sentences,tags),total=len(tags)):
        tokens = []
        label_ids = []
        
        for word, label in zip(x, y):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label2code[label]] * len(word_tokens))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_id_list.append(input_ids)
        label_id_list.append(label_ids)

    input_id_list = pad_sequences(input_id_list,
                          maxlen=MAX_LENGTH, dtype="long", value=0.0,
                          truncating="post", padding="post")
    label_id_list = pad_sequences(label_id_list,
                     maxlen=MAX_LENGTH, value=label2code["PAD"], padding="post",
                     dtype="long", truncating="post")
    attention_mask_list = [[float(i != 0.0) for i in ii] for ii in input_id_list]

    return input_id_list, attention_mask_list, label_id_list


# In[ ]:


input_ids_train, attention_masks_train, label_ids_train = convert_to_input(x_train_sentences, x_train_tags)
input_ids_val, attention_masks_val, label_ids_val = convert_to_input(x_val_sentences, x_val_tags)
input_ids_test, attention_masks_test, label_ids_test = convert_to_input(x_test_sentences, x_test_tags)


# In[ ]:


np.shape(input_ids_train), np.shape(attention_masks_train), np.shape(label_ids_train)


# In[ ]:


np.shape(input_ids_val), np.shape(attention_masks_val), np.shape(label_ids_val)


# In[ ]:


np.shape(input_ids_test), np.shape(attention_masks_test), np.shape(label_ids_test)


# In[ ]:


train_inputs = torch.tensor(input_ids_train)
train_tags = torch.tensor(label_ids_train)
train_masks = torch.tensor(attention_masks_train)

val_inputs = torch.tensor(input_ids_val)
val_tags = torch.tensor(label_ids_val)
val_masks = torch.tensor(attention_masks_val)

test_inputs = torch.tensor(input_ids_test)
test_tags = torch.tensor(label_ids_test)
test_masks = torch.tensor(attention_masks_test)


# In[ ]:


train_data = TensorDataset(train_inputs, train_masks, train_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(test_inputs, test_masks, test_tags)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)


# In[ ]:


model = BertForTokenClassification.from_pretrained(
    "slo-hr-en-bert-pytorch",
    num_labels=len(label2code),
    output_attentions = False,
    output_hidden_states = False
)


# In[ ]:


#if torch.cuda.is_available():
#   model.cuda()


# In the part below we must pass all the parameters that can be finetuned to the optimizer. If we set *FULL_FINETUNING* to False, we will finetune just the model head. Otherwise the whole model weights will be updated. 
# 
# Gamma and beta are parameters by the *BERTLayerNorm* and should not be regularized. We can include also all parameters to the regularization and will achieve similar results.

# In[ ]:


FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)


# In[ ]:


model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"The model has {params} trainable parameters")


# In[ ]:


from transformers import get_linear_schedule_with_warmup

epochs = 3
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


# In[ ]:


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# In[ ]:


## Store the average loss after each epoch so we can plot them.
"""loss_values, validation_loss_values = [], []

for _ in trange(epochs, desc="Epoch"):
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.

    # Put the model into training mode.
    model.train()
    # Reset the total loss for this epoch.
    total_loss = 0

    # Training loop
    for step, batch in tqdm(enumerate(train_dataloader)):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        # forward pass
        # This will return the loss (rather than the model output)
        # because we have provided the `labels`.
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        # get the loss
        loss = outputs[0]
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # track train loss
        total_loss += loss.item()
        # Clip the norm of the gradient
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)


    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        eval_accuracy += flat_accuracy(logits, label_ids)
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [code2label[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if code2label[l_i] != "PAD"]
    valid_tags = [code2label[l_i] for l in true_labels
                                  for l_i in l if code2label[l_i] != "PAD"]
    print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
    print()


# In[ ]:


# Save model
torch.save(model, 'ner_bert_pt2.pt')"""


# In[ ]:


# Loading a model (see docs for different options)
model = torch.load('ner_bert_pt2.pt', map_location=torch.device('cpu'))


# In[ ]:


# Uncommend inline and show to show within the jupyter only.
"""import matplotlib.pyplot as plt
#%matplotlib inline

import seaborn as sns

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# Plot the learning curve.
plt.plot(loss_values, 'b-o', label="training loss")
plt.plot(validation_loss_values, 'r-o', label="validation loss")

# Label the plot.
plt.title("Learning curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

#plt.show()
plt.savefig("training2.png")


# The figure above should show similart to the following:
# 
# ![](training-pt.png)

# In[ ]:
"""

# TEST
predictions , true_labels = [], []
for batch in tqdm(test_dataloader):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)

    logits = outputs[1].detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    true_labels.extend(label_ids)

results_predicted = [code2label[p_i] for p, l in zip(predictions, true_labels)
                             for p_i, l_i in zip(p, l) if code2label[l_i] != "PAD"]
results_true = [code2label[l_i] for l in true_labels
                              for l_i in l if code2label[l_i] != "PAD"]


# In[ ]:


print(f"F1 score: {f1_score(results_true, results_predicted)}")
print(f"Accuracy score: {accuracy_score(results_true, results_predicted)}")
print(classification_report(results_true, results_predicted))


# The expected output of the above should be similar as follows:
# 
# ```
# F1 score: 0.8276047261009667
# Accuracy score: 0.9615467524499643
# 
# 
#            precision    recall  f1-score   support
# 
#       org       0.76      0.70      0.73      3762
#       gpe       0.94      0.95      0.95      1841
#       per       0.79      0.80      0.79      2749
#       geo       0.84      0.90      0.87      5956
#       tim       0.87      0.83      0.85      2252
#       nat       0.31      0.38      0.34        21
#       art       0.15      0.09      0.11        80
#       eve       0.33      0.30      0.32        33
# 
# micro avg       0.82      0.83      0.83     16694
# macro avg       0.82      0.83      0.83     16694
# ```
# 
# After that we can observe also the results of the following example, fed to the algorithm:

# In[ ]:


test_sentence = """
Dr. Marko Robnik-Šikonja is lecturing a course on NLP at the University of Ljubljana in Slovenia. 
Dr. Žitnik is having labs every Tuesday and Friday. His lectures are recorded at 
the Televizija Slovenija national tv station. 
"""


# In[ ]:


tokenized_sentence = tokenizer.encode(test_sentence)

#if torch.cuda.is_available():
 #   input_ids = torch.tensor([tokenized_sentence]).gpu()
#else:
input_ids = torch.tensor([tokenized_sentence])


# In[ ]:


with torch.no_grad():
    output = model(input_ids)
label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)


# In[ ]:


# Join BPE split tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
new_tokens, new_labels = [], []
for token, label_idx in zip(tokens, label_indices[0]):
    if token.startswith("##"):
        new_tokens[-1] = new_tokens[-1] + token[2:]
    else:
        new_labels.append(code2label[label_idx])
        new_tokens.append(token)


# In[ ]:


for token, label in zip(new_tokens, new_labels):
    print("{}\t{}".format(label, token))


# The expected output should recognize entities:
# 
# * (PER) Dr. Marko Robnik-Šikonja
# * (ORG) NLP
# * (ORG) University
# * (GEO) Ljubljana
# * (GEO) Slovenia
# * (PER) Dr. Žitnik
# * (TIM) Tuesday and Friday
# * (ORG) Televizija Slovenija
