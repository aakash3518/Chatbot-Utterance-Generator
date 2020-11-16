from flask import Flask, request
from flask_restful import Resource,Api
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from rouge import Rouge
from datetime import datetime

app=Flask(__name__)
api=Api(app)

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)
set_seed(42)



class Generator(Resource):
    def get(self,text):


        sentence='Paraphrase: %s </s>'%text
        encoding = tokenizer.encode_plus(sentence, pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
        beam_outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            do_sample=True,
            max_length=256,
            top_k=50,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=10
        )
        final_outputs = []
        for beam_output in beam_outputs:
            sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if sent.lower() != text.lower() and sent not in final_outputs:
                final_outputs.append(sent)
        messages=[text]
        messages.extend(final_outputs)
        embeddings = embed(messages)
        correlation = np.inner(embeddings, embeddings)
        correlation = correlation[0][1:]
        messages = messages[1:]
        selected = []
        for i in range(len(messages)):
            if correlation[i] >= 0.80:
                selected.append(messages[i])
        rouge = Rouge()
        rouge_l = []
        for sent in selected:
            scores = rouge.get_scores(sentence, sent)
            rouge_l.append(scores[0]['rouge-l']['p'])
        final_selected = []
        for i in range(len(selected)):
            if rouge_l[i] <= 0.6:
                final_selected.append(selected[i])
        return {'Filtered utterances': final_selected}

api.add_resource(Generator, '/generate/<string:text>')

if __name__=='__main__':
    global model
    global embed
    global tokenizer
    model = T5ForConditionalGeneration.from_pretrained(
        r'.\t5_paraphrase')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print("device ", device)
    model = model.to(device)

    embed = hub.load(r'.\universal-sentence-encoder_4', options=tf.saved_model.LoadOptions(
        experimental_io_device='/CPU:0'))
    app.run(debug=False)