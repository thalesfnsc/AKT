from load_data import PID_DATA
from load_data import DATA
from run import test
import torch
from akt import AKT
import pickle
import pandas as pd
from run import test
import math 
import numpy as np
from scipy.stats import pearsonr
import time
transpose_data_model = {'akt'}
device = torch.device("cpu")


######

with open('/home/thales/KT-Models/AKT/pickle/students_index_skills.pickle','rb') as file:
   students_index_skills =  pickle.load(file)['student_index']


with open('/home/thales/KT-Models/AKT/pickle/problems_per_skills.pickle','rb') as file:
    problems_per_skills = pickle.load(file)
    

with open('/home/thales/KT-Models/AKT/pickle/student_interations.pickle','rb') as file:
    student_interations = pickle.load(file)

with open('/home/thales/KT-Models/AKT/data/errex/errex_dropped.csv','rb') as file:
    df = pd.read_csv(file)

######
seqlen = 350

batch_size = 598
n_question = 4

dat = PID_DATA(n_question=4,seqlen=seqlen,separate_char=',')
train_q_data, train_qa_data, train_pid = dat.load_data('/home/thales/KT-Models/AKT/data/errex_pid/errex_pid_train1.csv')


with open('/home/thales/KT-Models/AKT/data/errex/errex_dropped.csv','rb') as file:
    df_2 = pd.read_csv(file)



######

students = df_2['student_id'].unique()


#Reproducing test for predictions

checkpoint = torch.load('/home/thales/KT-Models/AKT/checkpoints/errex_pid_new.pt',map_location=torch.device('cpu'))
model = AKT(n_question=n_question,n_pid=238,n_blocks=1,d_model=256,dropout=0.05,kq_same=1,model_type='akt',l2=1e-5)
model.load_state_dict(checkpoint['model_state_dict'])   
model.eval()

students_list = list(students_index_skills.keys())

student_mean_of_correctness = {}
pid_flag = False
model_type = 'akt'

N = int(math.ceil(float(len(train_q_data))/float(1)))

pid_flag = True
pid_data = train_pid.T
q_data = train_q_data.T #(200,598)
qa_data = train_qa_data.T #(200,598)

seq_num = train_q_data.shape[1]
pred_list = []
target_list = []


count = 0
true_el = 0
element_count = 0
preds = []

start = time.time()

for idx in range(N):

        q_one_seq = q_data[:, idx*batch_size:(idx+1)*batch_size]
        if pid_flag:
            pid_one_seq = pid_data[:, idx *
                                   batch_size:(idx+1) * batch_size]
        input_q = q_one_seq[:, :]  # Shape (seqlen, batch_size)
        qa_one_seq = qa_data[:, idx *
                             batch_size:(idx+1) * batch_size]
        input_qa = qa_one_seq[:, :]  # Shape (seqlen, batch_size)

        # print 'seq_num', seq_num
        if model_type in transpose_data_model:
            # Shape (seqlen, batch_size)
            input_q = np.transpose(q_one_seq[:, :])
            # Shape (seqlen, batch_size)
            input_qa = np.transpose(qa_one_seq[:, :])
            target = np.transpose(qa_one_seq[:, :])
            if pid_flag:
                input_pid = np.transpose(pid_one_seq[:, :])
        else:
            input_q = (q_one_seq[:, :])  # Shape (seqlen, batch_size)
            input_qa = (qa_one_seq[:, :])  # Shape (seqlen, batch_size)
            target = (qa_one_seq[:, :])
            if pid_flag:
                input_pid = (pid_one_seq[:, :])
        target = (target - 1) /n_question
        target_1 = np.floor(target)
        #target = np.random.randint(0,2, size = (target.shape[0],target.shape[1]))

        input_q = torch.from_numpy(input_q).long().to(device)
        input_qa = torch.from_numpy(input_qa).long().to(device)
        target = torch.from_numpy(target_1).float().to(device)
        
        if pid_flag:
            input_pid = torch.from_numpy(input_pid).long().to(device)
    
        print(input_q.shape)
        print(input_q[0])
        print(input_pid[0])
        print(input_qa[0])

        
        with torch.no_grad():
            if pid_flag:
                loss, pred, ct = model(input_q, input_qa, target, input_pid)
            else:
                loss, pred, ct = model(input_q, input_qa, target)
        pred = pred.cpu().numpy()  # (seqlen * batch_size, 1)
        true_el += ct.cpu().numpy()
        print(pred.shape)
        break

        #target = target.cpu().numpy()
        if (idx + 1) * batch_size > seq_num:
            real_batch_size = seq_num - idx * batch_size
            count += real_batch_size
        else:
            count += batch_size

        # correct: 1.0; wrong 0.0; padding -1.0
        target = target_1.reshape((-1,))
        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]
        preds.append(pred_nopadding)
        

print(N)
print(q_data.shape)
end = time.time()

print("elapsed time:", end - start)

'''
with open('preds.pickle','wb') as file:
    pickle.dump({'preds':preds},file)


with open('/home/thales/KT-Models/AKT/preds.pickle','rb') as file:
    preds = pickle.load(file)['preds']
    file.close()

for i in range(len(students)):

    index_ordDecimals = np.where(train_q_data[i]==1)
    index_placeNumber = np.where(train_q_data[i]==2)
    index_completeSeque = np.where(train_q_data[i] ==3)
    index_decimaAddition = np.where(train_q_data[i]==4)


    mean_of_correctness = {}
    mean_of_correctness['OrderingDecimals'] = np.mean(preds[i][index_ordDecimals])
    mean_of_correctness['PlacementOnNumberLine'] = np.mean(preds[i][index_placeNumber])
    mean_of_correctness['CompleteTheSequence'] = np.mean(preds[i][index_completeSeque])
    mean_of_correctness['DecimalAddition'] = np.mean(preds[i][index_decimaAddition])

    student_mean_of_correctness[students[i]] = mean_of_correctness










with open('/home/thales/KT-Models/Errex data/ErrEx posttest data.xlsx','rb') as file:
    df_2 = pd.read_excel(file)



df_2 = df_2.drop([0,1], axis=0)
df_2 = df_2.drop(df_2.columns[1:5],axis=1)


df_post_dropped = df_2.drop_duplicates('Anon Student Id',keep='last')
df_post_dropped = df_post_dropped.drop(df_post_dropped.index[598])

decimal_addition_post = df_post_dropped['Unnamed: 171'].values
ordering_decimals_post=  df_post_dropped['Unnamed: 172'].values 
complete_sequence_post = df_post_dropped['Unnamed: 173'].values 
placement_number_post = df_post_dropped['Unnamed: 174'].values 



ord_decimals = [i['OrderingDecimals'] for i in  list(student_mean_of_correctness.values())]
place_number = [i['PlacementOnNumberLine'] for i in  list(student_mean_of_correctness.values())]
complet_sequence = [i['CompleteTheSequence'] for i in  list(student_mean_of_correctness.values())]
decimal_addition =  [i['DecimalAddition'] for i in  list(student_mean_of_correctness.values())]


pearson_correlations = {}


pearson_correlations['OrderingDecimals'] = pearsonr(ord_decimals,ordering_decimals_post)[0]
pearson_correlations['PlacementOnNumberLine'] = pearsonr(place_number,placement_number_post)[0]
pearson_correlations['CompleteTheSequence'] = pearsonr(complet_sequence,complete_sequence_post)[0]
pearson_correlations['DecimalAddition'] = pearsonr(decimal_addition,decimal_addition_post)[0]

print(pearson_correlations)

'''


'''
question = student_interations[students[0]][0]
answers = student_interations[students[0]][1]
students_index_skills = {}
cont = 0
for student in students:
    problem = student_interations[student][0]
    problem_list = problem.tolist()
    answer = student_interations[student][1]
    with torch.no_grad():
        problem = torch.LongTensor(problem)
        problem = torch.unsqueeze(problem,dim=0)
        answer = torch.LongTensor(answer)
        answer = torch.unsqueeze(answer,dim=0)
        #pred = 

    skill_index = {}
    for skill in problems_per_skills.keys():
        index = []
        for i in problem_list:
            if i in problems_per_skills[skill]:
                index.append(problem_list.index(i))
            skill_index[skill] = index

    students_index_skills[student] = skill_index
    


with open('/home/thales/KT-Models/AKT/pickle/students_index_skills.pickle','wb') as file:
     pickle.dump({'student_index':students_index_skills},file)
'''
