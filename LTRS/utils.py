import os
import random

import torch
import subprocess
from transformers import BertTokenizer, BertConfig, BertModel
import pandas as pd
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_tokenizer(name):
	tokenizer = BertTokenizer.from_pretrained(name)
	return tokenizer

def load_word_senses(path):
	word_senseIDs={}
	senseID_sense={}
	with open('./data/word_data.txt',encoding='utf-8') as f:
		rst=f.readlines()

	for line in rst:
		line=line.split()
		gloss=line[5]
		id=line[0]
		word=line[1]
		pos=line[3].replace('@待定','')
		if word not in word_senseIDs:
			word_senseIDs[word]=[]
		if id not in word_senseIDs[word]:
			word_senseIDs[word].append(id)
		senseID_sense[id]=gloss
	
	with open('./data/morph_data.txt',encoding='utf-8') as f:
		rst=f.readlines()
	for line in rst:
		line=line.split()
		gloss=line[6]
		id=line[0]
		word=line[1]
		pos=line[4].replace('@待定','')
		if '素' in pos:continue
		if word not in word_senseIDs:
			word_senseIDs[word]=[]
		if id not in word_senseIDs[word]:
			word_senseIDs[word].append(id)
		senseID_sense[id]=gloss
	

    
	max_num=0
	for word in word_senseIDs:
		max_num = max(max_num,len(word_senseIDs[word]))
		assert len(word_senseIDs[word])>0

	print('max sense number per word',max_num)
	return {'word_senseIDs':word_senseIDs, 'senseID_sense':senseID_sense}

def evaluate_output(eval_preds,pos='ALL',pos_dict=None):
    if pos=='ALL':
        y_true=[l[0] for l in eval_preds]
        y_pred=[l[1] for l in eval_preds]
    elif pos=='other':
        y_true=[]
        y_pred=[]
        for l in eval_preds:
            if pos_dict.get_id_pos(l[0]) not in ['名词','动词','形容词','副词']:#noun,verb,adj.,adv.
                y_true.append(l[0])
                y_pred.append(l[1])
    else:
        y_true=[]
        y_pred=[]
        for l in eval_preds:
            if pos_dict.get_id_pos(l[0])==pos:
                y_true.append(l[0])
                y_pred.append(l[1])
    print(pos,len(y_true))
    
    micro_f1= f1_score(y_true, y_pred,average='micro')
    macro_f1= f1_score(y_true, y_pred,average='macro')
    return micro_f1,macro_f1

def load_data(data_path,is_train=False):
	data_table = pd.read_csv(data_path)

	data_list=[]
	for idx, data in data_table.iterrows():
        
			word= data['word']
			#if word in ['睛']:continue
			context=data['context']
			gloss = data['gloss']
			gloss_id = data['id']
			data_list.append({'word':word,'context':context,'gloss':gloss,'id':gloss_id})
	print(data_path,'dataset size',len(data_list))
	if is_train:
		print(data_path,'dataset size',len(data_list))
		return data_list
	else:
		print(data_path,'dataset size',len(data_list))
		return data_list 



def token_encode(tokenizer, d, max_len, sep_id=102, cls_id=101, pad_id=0,mask_id=103,word_sense_utils=None):
	# [CLS] token_text [SEP] hard_prompt [MASK] [SEP]
	word = d['word']
	context = d['context']
	assert '～' in context
	context = context.split('～')
	left_context = context[0]

	right_context = ''.join(context[1:]).replace('～', word)

	left_context_tokens = [cls_id] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(left_context))
	word_tokens =[mask_id]#tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
	word_index_list = [i for i in range(len(left_context_tokens), len(left_context_tokens) + len(word_tokens))]
	assert len(left_context_tokens) + len(word_tokens) < max_len
	right_context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(right_context))[
						   :max_len - 1 - len(left_context_tokens) - len(word_tokens)] + [sep_id]

	context_tokens = left_context_tokens + word_tokens + right_context_tokens

	token_type_ids = [0] * max_len
	attention_mask = [1] * len(context_tokens)
	assert len(context_tokens) <= max_len
	attention_mask += [pad_id] * (max_len - len(context_tokens))
	context_tokens += [pad_id] * (max_len - len(context_tokens))

	assert len(attention_mask) == len(context_tokens) == max_len == len(token_type_ids)
	return {'input_ids':context_tokens,
			'attention_mask':attention_mask,
			'token_type_ids':token_type_ids,
			'word_index_list':word_index_list}

def compare_set(seta,setb):
	max_s=0.0
	for a in seta:
		for b in setb:
			max_s=max(max_s,get_rank_score(a,b))
	return max_s

def get_rank_score(cid_a, cid_b):

	if cid_a[0] != cid_b[0]:
		return 0.0
	elif cid_a[0:2] != cid_b[0:2]:
		return 2.0
	elif cid_a[0:4] != cid_b[0:4]:
		return 4.0
	elif cid_a[0:5] != cid_b[0:5]:
		return 6.0
	elif cid_a[0:7] != cid_b[0:7]:
		return 8.0
	else:
		return 10.0


def get_tokenize(bge_model,word_sense_utils, batch_data, tokenizer, context_max_len,gloss_max_len):  # ,word_senseIDs, senseID_sense,senseID_cate
	context_input_ids = []
	context_attention_mask = []
	context_token_type_ids = []
	context_word_index_lists = []

	candinate_gloss_id_list = []
	target_gloss_id_list = []
	for d in batch_data:
		tmp_dict=token_encode(tokenizer,d,context_max_len,)
		context_attention_mask.append(tmp_dict['attention_mask'])
		context_input_ids.append(tmp_dict['input_ids'])
		context_token_type_ids.append(tmp_dict['token_type_ids'])
		context_word_index_lists.append(tmp_dict['word_index_list'])
		for id_ in word_sense_utils['word_senseIDs'][d['word']]:
			if id_ not in candinate_gloss_id_list:
				candinate_gloss_id_list.append(id_)
		target_gloss_id_list.append(d['id'])
	context_input_ids_tensor = torch.tensor(context_input_ids)
	context_attention_mask_tesnor = torch.tensor(context_attention_mask)
	context_token_type_ids_tensor = torch.tensor(context_token_type_ids)


	candinate_gloss_list = [word_sense_utils['senseID_sense'][cgid] for cgid in candinate_gloss_id_list]
	target_gloss_list = [word_sense_utils['senseID_sense'][cgid] for cgid in target_gloss_id_list]

	embeddings_1 = bge_model.encode(target_gloss_list, normalize_embeddings=True)
	embeddings_2 = bge_model.encode(candinate_gloss_list, normalize_embeddings=True)

                                                                    
	target_score = embeddings_1 @ embeddings_2.T
	target_score_tensor=torch.tensor(target_score)
	candinate_gloss_inputs =tokenizer(candinate_gloss_list,max_length=gloss_max_len,padding='max_length',truncation=True,return_tensors='pt')
	return {
		'context_input_ids':context_input_ids_tensor,
		'context_attention_mask':context_attention_mask_tesnor,
		'context_token_type_ids':context_token_type_ids_tensor,
		'context_word_indexes':context_word_index_lists,
		'candinate_gloss_inputs':candinate_gloss_inputs,
		'target_score':target_score_tensor
	}







def get_batch_data(data_list, word_sense_utils,gloss_batch_size,shuffle=True):
	word_senseIDs=word_sense_utils['word_senseIDs']
	if shuffle:
		random.shuffle(data_list)
	batch_datas_list = []
	gloss_num = 0
	tmp_data_list = []
	for d in data_list:
		
		if gloss_num + len(word_senseIDs[d['word']]) <= gloss_batch_size:
			tmp_data_list.append(d)
			gloss_num += len(word_senseIDs[d['word']])
		else:
			if len(tmp_data_list)!=0:
				batch_datas_list.append(tmp_data_list)
			gloss_num = len(word_senseIDs[d['word']])

			tmp_data_list = []
			tmp_data_list.append(d)
            
	if len(tmp_data_list)>0:
		batch_datas_list.append(tmp_data_list)
	sum_=sum([len(l) for  l in batch_datas_list])

	assert sum_==len(data_list)


	print('batch num', len(batch_datas_list))
	print(batch_datas_list[0][0])

	return batch_datas_list

class POS_dict:
    def __init__(self):
        with open('./data/word_data.txt',encoding='utf-8') as f:
            rst=f.readlines()
        self.id_pos_dict={}
        for line in rst:
            line=line.split()
            wid=line[0]
            pos=line[3].replace('@待定','')

            self.id_pos_dict[wid]=pos
        with open('./data/morph_data.txt',encoding='utf-8') as f:
            rst=f.readlines()

        for line in rst:
            line=line.split()
            wid=line[0]
            pos=line[4].replace('@待定','')

            self.id_pos_dict[wid]=pos
    def get_id_pos(self,wid):
        if wid in self.id_pos_dict:
            return self.id_pos_dict[wid]
        else:
            print('error')
            return ''
        
    


def tokenize_glosses_for_evaluate(gloss_list, tokenizer, max_len):
	text_encoding = tokenizer(gloss_list,max_length=max_len,padding='max_length', return_tensors='pt',truncation = True,return_attention_mask = True)
	return text_encoding



def load_and_preprocess_glosses(tokenizer,word_sense_utils, max_len=32):
	word_gtokens = {}
	word_senseIDs = word_sense_utils['word_senseIDs']
	senseID_sense = word_sense_utils['senseID_sense']
	for word in tqdm(word_senseIDs):
		gloss_list = [senseID_sense[s] for s in word_senseIDs[word]]
		gloss_token_inputs=tokenize_glosses_for_evaluate(gloss_list, tokenizer, max_len)
		word_gtokens[word]=gloss_token_inputs
	return word_gtokens

def get_context_tokenize_for_evaluate(batch_data, tokenizer, word_sense_utils,context_max_len):  # ,word_senseIDs, senseID_sense,senseID_cate
	context_input_ids = []
	context_attention_mask = []
	context_token_type_ids = []
	context_word_index_lists = []

	candinate_gloss_id_list = []
	target_gloss_id_list = []
	for d in batch_data:
		tmp_dict=token_encode(tokenizer,d,context_max_len,word_sense_utils=word_sense_utils)
		context_attention_mask.append(tmp_dict['attention_mask'])
		context_input_ids.append(tmp_dict['input_ids'])
		context_token_type_ids.append(tmp_dict['token_type_ids'])
		context_word_index_lists.append(tmp_dict['word_index_list'])

		candinate_gloss_id_list += [id_ for id_ in word_sense_utils['word_senseIDs'][d['word']]]
		target_gloss_id_list.append(d['id'])

	context_input_ids_tensor = torch.tensor(context_input_ids)
	context_attention_mask_tesnor = torch.tensor(context_attention_mask)
	context_token_type_ids_tensor = torch.tensor(context_token_type_ids)

	return {
		'context_input_ids':context_input_ids_tensor,
		'context_attention_mask':context_attention_mask_tesnor,
		'context_token_type_ids':context_token_type_ids_tensor,
		'context_word_indexes':context_word_index_lists,
	}





if __name__ == "__main__":
	tokenizer = BertTokenizer.from_pretrained('./model/chinese-bert-wwm-ext')
	tp=token_encode(tokenizer, {'word':'含蓄','context':'她是一个～的人，不怎么爱说话。'}, 20)
	print(tp)
	print(tokenizer('她是一个含蓄的人，不怎么爱说话,习惯就好。',max_len=20))
	word_sense_utils=load_word_senses('./data')
	train_data_list=load_data('./data/train_data.tsv')
	trian_batch_datas_list=get_batch_data(train_data_list,word_sense_utils,64)




