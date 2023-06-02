import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
import random
import os

# Start with a blank Russian model
nlp = spacy.blank("ru")

# Add a TextCategorizer to the pipeline
textcat = nlp.add_pipe("textcat")

# Add labels to the TextCategorizer
textcat.add_label("YES")
textcat.add_label("NO")
textcat.add_label("UNKNOWN")

# Prepare training data in the format (TEXT, {'cats': {'YES': True/False, 'NO': True/False, 'UNKNOWN': True/False}})
train_data = [
    ("да", {'cats': {'YES': True, 'NO': False, 'UNKNOWN': False}}),
    ("даа", {'cats': {'YES': True, 'NO': False, 'UNKNOWN': False}}),
    ("д", {'cats': {'YES': True, 'NO': False, 'UNKNOWN': False}}),
    ("дъ", {'cats': {'YES': True, 'NO': False, 'UNKNOWN': False}}),
    ("дв", {'cats': {'YES': True, 'NO': False, 'UNKNOWN': False}}),
    ("конечно", {'cats': {'YES': True, 'NO': False, 'UNKNOWN': False}}),
    ("наверное да", {'cats': {'YES': True, 'NO': False, 'UNKNOWN': False}}),
    ("ага", {'cats': {'YES': True, 'NO': False, 'UNKNOWN': False}}),
    ("согласен", {'cats': {'YES': True, 'NO': False, 'UNKNOWN': False}}),
    ("пожалуй", {'cats': {'YES': True, 'NO': False, 'UNKNOWN': False}}),
    ("нет", {'cats': {'YES': False, 'NO': True, 'UNKNOWN': False}}),
    ("никогда", {'cats': {'YES': False, 'NO': True, 'UNKNOWN': False}}),
    ("наверное нет", {'cats': {'YES': False, 'NO': True, 'UNKNOWN': False}}),
    ("я не думаю", {'cats': {'YES': False, 'NO': True, 'UNKNOWN': False}}),
    ("нет конечно", {'cats': {'YES': False, 'NO': True, 'UNKNOWN': False}}),
    ("ноуп", {'cats': {'YES': False, 'NO': True, 'UNKNOWN': False}}),
    ("найн", {'cats': {'YES': False, 'NO': True, 'UNKNOWN': False}}),
    ("бббббб", {'cats': {'YES': False, 'NO': False, 'UNKNOWN': True}}),
    ("билебирда", {'cats': {'YES': False, 'NO': False, 'UNKNOWN': True}}),
    ("любая длинная фраза", {'cats': {'YES': False, 'NO': False, 'UNKNOWN': True}}),
    ("джигурда", {'cats': {'YES': False, 'NO': False, 'UNKNOWN': True}}),
    ("а", {'cats': {'YES': False, 'NO': False, 'UNKNOWN': True}}),
    ("Я не знаю что ты хочешь", {'cats': {'YES': False, 'NO': False, 'UNKNOWN': True}}),
    ("поговори со мной", {'cats': {'YES': False, 'NO': False, 'UNKNOWN': True}}),
    ("уй", {'cats': {'YES': False, 'NO': False, 'UNKNOWN': True}}),
    ("я подумаю", {'cats': {'YES': False, 'NO': False, 'UNKNOWN': True}}),

]

# Convert the training data to the SpaCy format
train_data = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in train_data]

# Train the model
random.seed(1)
spacy.util.fix_random_seed(1)
optimizer = nlp.initialize()

losses = {}
for epoch in range(20):
    random.shuffle(train_data)
    batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        nlp.update(batch, sgd=optimizer, losses=losses)
    print(f"Losses at iteration {epoch} - {losses['textcat']:.3f}")


current_dir = os.getcwd()
nlp.to_disk(os.path.join(current_dir, 'model'))

