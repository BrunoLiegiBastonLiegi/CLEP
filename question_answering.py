import json, sys, argparse
from pathlib import Path
from tqdm import tqdm
from os.path import basename


parser = argparse.ArgumentParser(description='Question Answering.')
parser.add_argument('--load_model', default=None, help='Path to pretrained model.')
parser.add_argument('--text_encoder', default='distilbert-base-uncased')
parser.add_argument('--clip', action='store_true')
parser.add_argument('--skip_training', action='store_true')
parser.add_argument('--epochs', default=3, type=int)

args = parser.parse_args()


# ------------ Prepare the Data ---------------------------------------------------------------------

def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    ids = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                qa_id = qa['id']
                question = qa['question']
                if len(qa['answers']) == 0:
                    ids.append(qa_id)
                    contexts.append(context)
                    questions.append(question)
                    answers.append({'text':'', 'answer_start':0})
                else:
                    for answer in qa['answers']:
                        ids.append(qa_id)
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)

    return contexts, questions, answers, ids

train_contexts, train_questions, train_answers, train_ids = read_squad('data/squad/train-v2.0.json')
val_contexts, val_questions, val_answers, val_ids = read_squad('data/squad/dev-v2.0.json')

def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if len(gold_text) == 0:
            answer['answer_end'] = 0
        elif context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

add_end_idx(train_answers, train_contexts)
add_end_idx(val_answers, val_contexts)
    

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained(args.text_encoder)

#from transformers import GPT2Tokenizer # <-------- GPT2 is not supported for question answering
                                       #           Change it to GPTJ or BERT

#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#tokenizer.padding_side, tokenizer.pad_token = 'left', tokenizer.bos_token


train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        if answers[i]['text'] == '':
            start_positions.append(-1)
            end_positions.append(0)
        else:
            start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
        # if None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(val_encodings, val_answers)
import torch

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, ids):
        self.encodings = encodings
        self.ids = ids

    def __getitem__(self, idx):
        d = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        d['id'] = self.ids[idx]
        return d

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = SquadDataset(train_encodings, train_ids)
val_dataset = SquadDataset(val_encodings, val_ids)

# ---------------- Training Routine ---------------------------------------------------------------------------

def get_answers(input_ids, outputs):
    start_tokens = torch.nn.functional.softmax(outputs.start_logits, dim=-1).argmax(-1)
    end_tokens = torch.nn.functional.softmax(outputs.end_logits, dim=-1).argmax(-1)
    answers = []
    for ids, s, e in zip(input_ids, start_tokens, end_tokens):
        answers.append(ids[s:e+1])
    return tokenizer.batch_decode(
        answers,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

from transformers import AdamW

def train(model, train_loader, val_loader, epochs=1):
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
    print_step = 100
    
    for epoch in range(epochs):
        running_train_loss, running_val_loss = torch.zeros(1), torch.zeros(1)
        print(f'---------- Epoch {epoch} ----------')
        model.train()
        for i,batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            bs = input_ids.shape[0]
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs[0]
            running_train_loss += loss.item()
            loss.backward()
            optim.step()
            if i % print_step == print_step - 1:
                print(f'> Train Loss: {running_train_loss.item()/print_step:.4f}')
                running_train_loss = torch.zeros(1)
        print('\n')

        model.eval()
        for batch in tqdm(val_loader):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
                loss = outputs[0]
                running_val_loss += loss.item()
        print(f'> Val Loss: {running_val_loss.item()/len(val_loader):.4f}')


# ------------ Prepare the Model ---------------------------------------------------------------------

from transformers import DistilBertForQuestionAnswering, AutoModelForQuestionAnswering
from model import GPT2CaptionEncoder, BertCaptionEncoder, RGCN, CompGCNWrapper, CLIP_KB
from collections import OrderedDict
from torch.utils.data import DataLoader

model = AutoModelForQuestionAnswering.from_pretrained(args.text_encoder)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

if args.load_model is not None:
    if args.clip:
        params_dict = {} 
        for k,v in torch.load(args.load_model).items():
            if 't_encoder' in k:
                params_dict.update({k.replace('t_encoder.model.', 'distilbert.'): v})
            elif 't_mlp' in k:
                params_dict.update({k.replace('t_mlp.nn.0.', 'linear.'): v})
            else:
                params_dict.update({k: v})

        head = torch.nn.Sequential(OrderedDict([
            ('linear', torch.nn.Linear(768, 200)),
            ('relu', torch.nn.ReLU()),
            ('dout', torch.nn.Dropout(p=0.1, inplace=False)),
            ('qa_outputs', torch.nn.Linear(200, 2)),
        ]))
        model.qa_outputs = head
        model.load_state_dict(params_dict, strict=False)
    else:
        model.load_state_dict(torch.load(args.load_model))

model.to(device)
model.train()

if (args.load_model is None or args.clip) and not args.skip_training:
    for p in model.parameters():
        p.requires_grad = True
    epochs = args.epochs
    train(model, train_loader, val_loader, epochs=epochs)
    filename = 'data/squad/qa_squad_{}'.format(epochs)
    if args.clip:
        filename += '_clip'
        filename += '_{}'.format(basename(args.load_model))
    torch.save(model.state_dict(), filename + '.pt')

answers = {}
gt = {}
model.eval()
for batch in tqdm(val_loader):
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        gt.update(dict(zip(
            batch['id'],
            tokenizer.batch_decode(
                [ i[s:e+1] for s,e,i in zip(start_positions, end_positions, input_ids) ],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
        )))
        ans = get_answers(input_ids, outputs)
        answers.update(dict(zip(batch['id'], ans)))
        
for k,v in gt.items():
    gt[k] = v
for k,v in answers.items():
    answers[k] = v

ans_file = 'data/squad/answers'
if args.clip:
    ans_file += '_{}'.format(basename(args.load_model))
with open(ans_file + '.json', 'w') as f:
    json.dump(answers, f)

with open('data/squad/answers_true.json', 'w') as f:
    json.dump(gt, f)
    

