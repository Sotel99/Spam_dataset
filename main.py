import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from transformers import TFTrainer, TFTrainingArguments
import os

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

#load dataset
df = pd.read_csv(
    "https://raw.githubusercontent.com/Sotel99/Spam_dataset/main/spam.csv", encoding="latin-1"
)
df.head(n=10)

#preprocessing data
df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True, axis=1)
df.rename({"v1": "label", "v2": "msg"}, axis=1, inplace=True)
df["label"].replace({"ham": 0, "spam": 1}, inplace=True)

#feature creation
df["content"]=df["content"].apply(str)
df["nwords"] = df["content"].apply(lambda s: len(re.findall(r"\w+", s)))
df["message_len"] = df["content"].apply(len)
df["nupperchars"] = df["content"].apply(
    lambda s: sum(1 for c in s if c.isupper())
)
df["nupperwords"] = df["content"].apply(
    lambda s: len(re.findall(r"\b[A-Z][A-Z]+\b", s))
)
df["is_free_or_win"] = df["content"].apply(
    lambda s: int("free" in s.lower() or "win" in s.lower())
)
df["is_url"] = df["content"].apply(
    lambda s: 1
    if re.search(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        s,
    )
    else 0
)


#spam, no spam ratio
n_sms = pd.value_counts(df["is_spam"], sort=True)
n_sms.plot(kind="pie", labels=["ham", "spam"], autopct="%1.0f%%")

plt.title("SMS Distribution")
plt.ylabel("")
plt.show()


#length of spam and no spam messages
_, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot(
    df.loc[df.is_spam == 0, "message_len"],
    shade=True,
    label="Ham",
    clip=(-50, 250),
)
sns.kdeplot(df.loc[df.is_spam == 1, "message_len"], shade=True, label="Spam")
ax.set(
    xlabel="Length",
    ylabel="Density",
    title="Length of SMS.",
)
ax.legend(loc="upper right")
plt.show()

#no. of words in spam and no spam messages
_, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot(
    df.loc[df.is_spam == 0, "nwords"],
    shade=True,
    label="Ham",
    clip=(-10, 50),
)
sns.kdeplot(df.loc[df.is_spam == 1, "nwords"], shade=True, label="Spam")
ax.set(
    xlabel="Words",
    ylabel="Density",
    title="Number of Words in SMS.",
)
ax.legend(loc="upper right")


#further preprocessing of data
#lemmatization
_, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot(
    df.loc[df.is_spam == 0, "nwords"],
    shade=True,
    label="Ham",
    clip=(-10, 50),
)
sns.kdeplot(df.loc[df.is_spam == 1, "nwords"], shade=True, label="Spam")
ax.set(
    xlabel="Words",
    ylabel="Density",
    title="Number of Words in SMS.",
)
ax.legend(loc="upper right")


#creating train and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["is_spam"], stratify=df["is_spam"],test_size=0.2
)

#DistilBert
#Tokenization
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

#generating encodings
train_encodings = tokenizer(
    X_train.tolist(),
    max_length=96,
    padding="max_length",
    truncation=True,
)
test_encodings = tokenizer(
    X_test.tolist(),
    max_length=96,
    padding="max_length",
    truncation=True,
)
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True)

#transformation of labels and encodings
train_dataset = tf.data.Dataset.from_tensor_slices(
    (dict(train_encodings), y_train)
)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (dict(test_encodings), y_test)
)

#Fine-tuning and training
training_args = TFTrainingArguments(
    output_dir=r"C:\Users\darek\Desktop\NlP\results\distilbert",
    num_train_epochs=8,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=r"C:\Users\darek\Desktop\NlP\logs\distilbert",
    logging_steps=10,
)

#model training
from transformers import TFDistilBertForSequenceClassification

with training_args.strategy.scope():
    model = TFDistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased"
    )

trainer = TFTrainer(
    model=model, args=training_args, train_dataset=train_dataset
)
trainer.train()

#saving model
trainer.save_model(r"C:\Users\darek\Desktop\NlP\models\distilbert")
tokenizer.save_pretrained(training_args.output_dir)

#predictions
preds, label_ids, metrics = trainer.predict(test_dataset)

#normalization
preds = np.argmax(preds, axis=1)

#confusion matrix
plt.figure(figsize=(10, 4))

heatmap = sns.heatmap(
    data=pd.DataFrame(confusion_matrix(y_test, preds)),
    annot=True,
    fmt="d",
    cmap=sns.color_palette("Blues", 50),
)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), fontsize=14)
heatmap.yaxis.set_ticklabels(
    heatmap.yaxis.get_ticklabels(), rotation=0, fontsize=14
)

plt.title("Confusion Matrix")
plt.ylabel("Ground Truth")
plt.xlabel("Prediction")


#KNN
#fine-tuning and training
from sklearn.neighbors import KNeighborsClassifier

knn = GridSearchCV(
    Pipeline(
        [
            ("bow", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", KNeighborsClassifier()),
        ]
    ),
    {
        "clf__n_neighbors": (8, 15, 20, 25, 40, 55),
    }
)
knn.fit(X=X_train, y=y_train)
preds = knn.predict(X_test)

#confusion matrix
plt.figure(figsize=(10, 4))

heatmap = sns.heatmap(
    data=pd.DataFrame(confusion_matrix(y_test, preds)),
    annot=True,
    fmt="d",
    cmap=sns.color_palette("Blues", 50),
)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), fontsize=14)
heatmap.yaxis.set_ticklabels(
    heatmap.yaxis.get_ticklabels(), rotation=0, fontsize=14
)

plt.title("Confusion Matrix")
plt.ylabel("Ground Truth")
plt.xlabel("Prediction")


#Multinomial Naive Bayes
#Fine-tuning and training
from sklearn.naive_bayes import MultinomialNB

mnbayes = GridSearchCV(
    Pipeline(
        [
            ("bow", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", MultinomialNB()),
        ]
    ),
    {
        "tfidf__use_idf": (True, False),
        "clf__alpha": (0.1, 1e-2, 1e-3),
        "clf__fit_prior": (True, False),
    },
)
mnbayes.fit(X=X_train, y=y_train)

#predictions
preds = mnbayes.predict(X_test)

#confusion matrix
plt.figure(figsize=(10, 4))

heatmap = sns.heatmap(
    data=pd.DataFrame(confusion_matrix(y_test, preds)),
    annot=True,
    fmt="d",
    cmap=sns.color_palette("Blues", 50),
)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), fontsize=14)
heatmap.yaxis.set_ticklabels(
    heatmap.yaxis.get_ticklabels(), rotation=0, fontsize=14
)

plt.title("Confusion Matrix")
plt.ylabel("Ground Truth")
plt.xlabel("Prediction")

#SVM
#fine-tuning and training
from sklearn.svm import SVC

svc = GridSearchCV(
    Pipeline(
        [
            ("bow", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", SVC(gamma="auto", C=1000)),
        ]
    ),
    dict(tfidf=[None, TfidfTransformer()], clf__C=[500, 1000, 1500]),
)
svc.fit(X=X_train, y=y_train)

#predictions
preds = svc.predict(X_test)

#confusion matrix
plt.figure(figsize=(10, 4))

heatmap = sns.heatmap(
    data=pd.DataFrame(confusion_matrix(y_test, preds)),
    annot=True,
    fmt="d",
    cmap=sns.color_palette("Blues", 50),
)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), fontsize=14)
heatmap.yaxis.set_ticklabels(
    heatmap.yaxis.get_ticklabels(), rotation=0, fontsize=14
)

plt.title("Confusion Matrix")
plt.ylabel("Ground Truth")
plt.xlabel("Prediction")
