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
n_question = 238
batch_size = 24
seqlen = 200

dat = DATA(n_question=238,seqlen=200,separate_char=',')
train_q_data, train_qa_data, train_pid = dat.load_data('/home/thales/KT-Models/AKT/data/errex/errex_train1.csv')

######
def knowledge_estimate(df,student_interations,problems_per_skills):

    students = df['student_id'].unique()
    student_mean_of_correctness = {}

######

students = df['student_id'].unique()


#Reproducing test for predictions

checkpoint = torch.load('/home/thales/KT-Models/AKT/checkpoints/_b24_nb1_gn-1_lr1e-05_s224_sl200_do0.05_dm256_ts1_kq1_l21e-05_150.pt',map_location=torch.device('cpu'))
model = AKT(n_question=n_question,n_pid=-1,n_blocks=1,d_model=256,dropout=0.05,kq_same=1,model_type='akt',l2=1e-5)
model.load_state_dict(checkpoint['model_state_dict'])   
model.eval()

students_list = list(students_index_skills.keys())

student_mean_of_correctness = {}
pid_flag = False
model_type = 'akt'

N = int(math.ceil(float(len(train_q_data))/float(1)))

train_q_data = train_q_data.T #(200,598)
train_qa_data = train_qa_data.T #(200,598)

seq_num = train_q_data.shape[1]
pred_list = []
target_list = []

count = 0
true_el = 0
element_count = 0
for idx in range(N):

    q_one_seq = train_q_data[:, idx*1:(idx+1)*1]
    if pid_flag:
        pid_one_seq = train_pid[:, idx *
                                1:(idx+1) * 1]
    input_q = q_one_seq[:, :]  # Shape (seqlen, batch_size)
    qa_one_seq = train_qa_data[:, idx*1:(idx+1) * 1]
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
    target = (target - 1) / 238
    target_1 = np.floor(target)

    input_q = torch.from_numpy(input_q).long().to(device)
    input_qa = torch.from_numpy(input_qa).long().to(device)
    target = torch.from_numpy(target_1).float().to(device)


    with torch.no_grad():
        loss,pred,ct = model(input_q,input_qa,target)

    pred = pred.cpu().numpy()  # (seqlen * batch_size, 1)
    true_el += ct.cpu().numpy()
    #target = target.cpu().numpy()
    if (idx + 1) * 1 > seq_num:
        real_batch_size = seq_num - idx * 1
        count += real_batch_size
    else:
        count += 1

    # correct: 1.0; wrong 0.0; padding -1.0
    target = target_1.reshape((-1,))
    nopadding_index = np.flatnonzero(target >= -0.9)
    nopadding_index = nopadding_index.tolist()
    pred_nopadding = pred[nopadding_index]
    target_nopadding = target[nopadding_index]
    element_count += pred_nopadding.shape[0]


    skill_index = students_index_skills[students_list[idx]]

    mean_of_correctness = {}
    for skill in skill_index.keys():
        mean_of_correctness[skill] = np.mean([pred_nopadding[i] for i in skill_index[skill] ])

    student_mean_of_correctness[students_list[idx]] = mean_of_correctness




#print(student_mean_of_correctness)
    #now use the students skill index and calculated the knowledge estimate for each iteration




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


#try to adapt the test method , modifying only the part that concatenate the prediction
#try to generate prediction for an individual student, passing only one student per time
#and use the student_index_skills to acess the prediction in the student sequence 
#for each skill separated

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
