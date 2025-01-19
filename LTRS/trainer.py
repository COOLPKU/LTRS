from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import os
import sys
import argparse
from tqdm import tqdm
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from utils import *
from models import BiEncoderModel
import random
from sentence_transformers import SentenceTransformer
parser = argparse.ArgumentParser(description='LTRS for WSD')

# training arguments
parser.add_argument('--rand_seed', type=int, default=12345, help='Random seed for reproducibility.')
parser.add_argument('--grad_norm', type=float, default=1.0, help='Maximum norm for gradient clipping.')
parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate.')
parser.add_argument('--warmup', type=int, default=10000, help='Number of warmup steps for learning rate.')
parser.add_argument('--context_max_len', type=int, default=64, help='Maximum length of the context.')
parser.add_argument('--gloss_max_len', type=int, default=32, help='Maximum length of the gloss.')

parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
parser.add_argument('--gloss_bsz', type=int, default=64, help='Batch size for gloss.')
parser.add_argument('--gradient_accumulation_step', type=int, default=40, help='Number of steps for gradient accumulation.')
parser.add_argument('--encoder-name', type=str, default='model', help='Name of the encoder.')
parser.add_argument('--bge_model_path', type=str, default='./bge_model', help='Path to the BGE model.')

parser.add_argument('--train_data_path', type=str, default='./data/MF_train_data.tsv', help='Path to the training data.')
parser.add_argument('--test_data_path', type=str, default='./data/MF_dev_data.tsv', help='Path to the test data.')
parser.add_argument('--dev_data_path', type=str, default='./data/MF_dev_data.tsv', help='Path to the development data.')
parser.add_argument('--word_sense_path', type=str, default='./data_sense', help='Path to the word sense data.')

parser.add_argument('--ckpt', type=str, default='./output', help='Path to save the model checkpoint.')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the training on, e.g., cuda or cpu.')

parser.add_argument('--tau1', type=float, default=20.0, help='Value of hyperparameter tau1.')
parser.add_argument('--tau2', type=float, default=20.0, help='Value of hyperparameter tau2.')
parser.add_argument('--tau3', type=float, default=20.0, help='Value of hyperparameter tau3.')
parser.add_argument('--steps', type=int, default=1000, help='number of training steps per eval.')

parser.add_argument('--loss_fn_type', type=str, default='listnet', help='Type of loss function,e.g., listnet or list_mle.')
parser.add_argument('--mode', type=str, default='train', help='Mode to run, e.g., train or evaluate.')


def train_epoch(
        model,
        bge_model,
        loss_fn,
        optimizer,
        tokenizer,
        word_sense_utils,
        args,
        train_data_list,
        dev_data_list,best_f1,epoch
):
    model = model.train()  # train mode
    losses = []

    trian_batch_datas_list= get_batch_data(train_data_list, word_sense_utils, args.gloss_bsz,shuffle=True)

    i=0
    for bd in tqdm(trian_batch_datas_list):
        token_data=get_tokenize(bge_model,word_sense_utils, bd, tokenizer, args.context_max_len, args.gloss_max_len)

        context_embeddings=model.context_forward(token_data['context_input_ids'].to(args.device),token_data['context_attention_mask'].to(args.device),token_data['context_token_type_ids'].to(args.device),token_data['context_word_indexes'])
        gloss_embeddings = model.gloss_forward(token_data['candinate_gloss_inputs'].to(args.device))

        target_score=token_data['target_score'].to(args.device)
        pred_logits=context_embeddings @ gloss_embeddings.T
        loss = loss_fn(pred_logits, target_score)
        
        
        losses.append(loss.item())
        loss /= args.gradient_accumulation_step
        loss.backward() 
    
        if ((i+1)%args.gradient_accumulation_step)==0 or i==(len(trian_batch_datas_list)-1):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step() 
            optimizer.zero_grad()
        i+=1
        if i%args.steps==0 or i==len(trian_batch_datas_list)-1:
            # eval model on dev set 
            print('After {} epochs, {} steps,train loss = {}'.format(epoch,i,np.mean(losses)))
            eval_preds = _eval(model,tokenizer,word_sense_utils,args,dev_data_list)
            # run predictions through scorer
            dev_f1,_ = evaluate_output(eval_preds)
            print('After {} epochs, {} steps, Dev f1 = {}'.format(epoch,i, dev_f1))
            sys.stdout.flush()
            
            if dev_f1 >= best_f1 and dev_f1<80.3:
                print('updating best model at epoch {}  steps{}...'.format(epoch,i))
                sys.stdout.flush()
                best_f1 = dev_f1
            # save to file if best probe so far on dev set
                model_name = os.path.join(args.ckpt, 'best_model.ckpt')
                with open(model_name, 'wb') as f:
                    torch.save(model.state_dict(), f)
                sys.stdout.flush()
            # generate predictions file
                pred_filepath = os.path.join(args.ckpt, 'dev_pred.txt')
                with open(pred_filepath, 'w') as f:
                    for inst, prediction in eval_preds:
                        f.write('TRUE:{}\tPREDICTION:{}\n'.format(inst, prediction))
                
            with open('training_log.txt', "a") as file:
                file.write(f"{str(epoch)}\t{str(i)}\t{str(dev_f1)}\n")
    
            

            print('Best dev f1 = {}'.format(best_f1))
        
        
    return best_f1



def _eval(model,tokenizer,word_sense_utils,args,dev_data_list):
    model.eval()
    eval_preds = []
    dev_batch_datas_list = get_batch_data(dev_data_list, word_sense_utils, args.gloss_bsz, shuffle=False)


    for bd in tqdm(dev_batch_datas_list):
        with torch.no_grad():
            context_inputs=get_context_tokenize_for_evaluate(bd, tokenizer, word_sense_utils,args.context_max_len)
            context_outputs=model.context_forward(context_inputs['context_input_ids'].to(args.device),context_inputs['context_attention_mask'].to(args.device),context_inputs['context_token_type_ids'].to(args.device),context_inputs['context_word_indexes'])
            for one_context_output, d in zip(context_outputs.split(1, dim=0), bd):
                # run example's glosses through gloss encoder
                word=d['word']
                label=d['id']
                gloss_inputs=word_sense_utils['word_gtokens'][word].to(args.device)

                gloss_output = model.gloss_forward(gloss_inputs)
                output = one_context_output @ gloss_output.T


                pred_idx = output.topk(1, dim=-1)[1].squeeze().item()
                pred_label = word_sense_utils['word_senseIDs'][word][pred_idx]
                if label not in word_sense_utils['word_senseIDs'][word]:
                    print('error',label)
                eval_preds.append((label, pred_label))

    return eval_preds

def eval_model(args):

    model = BiEncoderModel(args.encoder_name)
    model_path = os.path.join(args.ckpt, 'best_model.ckpt')
    model.load_state_dict(torch.load(model_path))
    model = model.to(args.device)

    word_sense_utils = load_word_senses(args.word_sense_path)
    tokenizer = load_tokenizer(args.encoder_name)  #
    word_gtokens = load_and_preprocess_glosses(tokenizer, word_sense_utils, max_len=args.gloss_max_len)

    word_sense_utils['word_gtokens'] = word_gtokens


    test_data_list = load_data(args.test_data_path)

    test_preds = _eval(model,tokenizer,word_sense_utils,args,test_data_list)


    pred_filepath = os.path.join(args.ckpt, 'test_pred.txt')
    with open(pred_filepath, 'w') as f:
        for inst, prediction in test_preds:
            f.write('TRUE:{}\tPREDICTION:{}\n'.format(inst, prediction))
    pos_dict=POS_dict()
    test_f1 = evaluate_output(test_preds)
    print('Test f1 = {}'.format(str(test_f1)))
    
    test_f1 = evaluate_output(test_preds,'名词',pos_dict)
    print('Test f1 = {}'.format(str(test_f1)))
    
    test_f1 = evaluate_output(test_preds,'动词',pos_dict)
    print('Test f1 = {}'.format(str(test_f1)))
    
    test_f1 = evaluate_output(test_preds,'形容词',pos_dict)
    print('Test f1 = {}'.format(str(test_f1)))
    
    test_f1 = evaluate_output(test_preds,'副词',pos_dict)
    print('Test f1 = {}'.format(str(test_f1)))
    
    test_f1 = evaluate_output(test_preds,'other',pos_dict)
    print('Test f1 = {}'.format(str(test_f1)))
    
    sys.stdout.flush()
    return


def train_model(args):
    print('Training bi-encoder model...')

    # create passed in ckpt dir if doesn't exist
    if not os.path.exists(args.ckpt):
        os.mkdir(args.ckpt)

    ################################################################
    ################################################################
    # LOAD PRETRAINED TOKENIZER & DATA

    print('Loading tokenzier & data, preprocessing data...')
    sys.stdout.flush()

    tokenizer = load_tokenizer(args.encoder_name)  

    epochs = args.epochs  # 20

    model = BiEncoderModel(args.encoder_name)

    model = model.to(args.device)

    if args.loss_fn_type =='listmle':

        loss_fn = list_mle
    
    else:

        loss_fn = listnet

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    word_sense_utils = load_word_senses(args.word_sense_path)

    word_gtokens = load_and_preprocess_glosses(tokenizer, word_sense_utils, max_len=args.gloss_max_len)

    word_sense_utils['word_gtokens'] = word_gtokens


    train_data_list = load_data(args.train_data_path,is_train=True)

    dev_data_list = load_data(args.dev_data_path)
    
    bge_model = SentenceTransformer(args.bge_model_path)


    best_f1 = 0.,
    print('Training probe...')
    sys.stdout.flush()

    for epoch in range(1, epochs + 1):

        best_f1=train_epoch(model,bge_model,loss_fn,optimizer,tokenizer,word_sense_utils,args,train_data_list,dev_data_list,best_f1,epoch)

    return 



def listnet(predict, target):
    # predict : batch x n_items
    # target : batch x n_items
   
    top1_target = F.softmax(target*args.tau1, dim=1)
    top1_predict = F.softmax(predict*args.tau2, dim=1)
    return torch.mean(-torch.sum(top1_target * torch.log(top1_predict),dim=1))


def list_mle(y_pred, y_true, k=10):
    # y_pred : batch x n_items
    # y_true : batch x n_items 
    y_pred = y_pred*args.tau3
 
    y_true_sorted, indices = y_true.sort(descending=True, dim=-1)

    pred_sorted_by_true = y_pred.gather(dim=1, index=indices)
 
    cumsums = pred_sorted_by_true.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
    
    pred_sorted_by_true = pred_sorted_by_true[:,:k]
    cumsums = cumsums[:,:k]
    
    y_true_weights = F.softmax(y_true_sorted[:,:k]*args.tau3, dim=1)

 
    listmle_loss = (torch.log(cumsums + 1e-10) - pred_sorted_by_true)

    
    listmle_loss_weighted = listmle_loss * y_true_weights

    
    
    return listmle_loss_weighted.sum(dim=1).mean()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Need available GPU(s) to run this model...")
        quit()

    # parse args
    args = parser.parse_args()
    print(args)

    # set random seeds
    torch.manual_seed(args.rand_seed)
    os.environ['PYTHONHASHSEED'] = str(args.rand_seed)
    torch.cuda.manual_seed(args.rand_seed)
    torch.cuda.manual_seed_all(args.rand_seed)
    np.random.seed(args.rand_seed)
    random.seed(args.rand_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.loss_fn_type =='train':
        train_model(args)
    eval_model(args)




# EOF
