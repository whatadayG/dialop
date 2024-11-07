import gspread
import re
from google.oauth2.service_account import Credentials
import pdb
import json

SERVICE_ACCOUNT_FILE = '/Users/georgiazhou/research_machine/dialop/dialop-8759580d9f40.json'
SCOPE = ["https://spreadsheets.google.com/feeds", 
         "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPE)
client = gspread.authorize(credentials)
spreadsheet = client.open('Dialop-GPT4-Planning')

columns = ["Experiment 1", "Experiment 2", "Experiment 3"]

worksheet1 = spreadsheet.sheet1
worksheet2 = worksheet = spreadsheet.get_worksheet(1)

#worksheet1.update('B1:D1', [columns])
#worksheet2.update('B1:D1', [columns])

experiment_1 = "3"
experiment_2 = "4-2"
experiment_3 = "4-3"

def extract_messages(experiment, col):
    collected_lines = []
    for i in range(113):
        filename = f'/Users/georgiazhou/research_machine/dialop/dialop/results/old_result/planning-{experiment}/{i}_0.out'
        with open(filename, 'r') as file:
            lines = file.readlines()
        normalized_reward = 0
        user_prompt = 0
        record = False
        result = False
        #collected = f'Dialogue {i}\n'
        collected = ''
        for line in lines:
            if record and user_prompt == 2:
                if result and line.startswith(r">"):           
                    json_string = line.strip()
                    json_string = json_string.replace('<', '').replace('>', '')
                    dic = json.loads(json_string) #collected += re.sub(r'\n\s*\n+', '\n', (line + '\n'))
                    if dic.get('user') == '[accept]':
                        normalized_reward = round(dic.get('info')["reward_normalized"], 2)
                        collected += "Result: " + str(normalized_reward)
                    else:
                        collected += "Error" 
                if re.search(r'Result', line):
                    result = True
                if line.startswith(r"You:[message]") or line.startswith(r"Agent") or line.startswith(r"You:[reject]") or line.startswith(r"You:[accept]"):
        
                        
                    collected += re.sub(r'\n\s*\n+', '\n', (line + '\n'))
            if re.search(r'Final user Prompt', line):
                record = True
            if re.search(r'CITY 2.', line):
                user_prompt = user_prompt + 1
        collected_lines.append([i, collected, normalized_reward])
      
    sorted_lines = sorted(collected_lines, key=lambda x: x[2], reverse=True)
    num_sorted = []
    dialo_sorted = []
    score_sorted = []
    for dialo in sorted_lines:
        num_sorted.append([dialo[0]])
        dialo_sorted.append([dialo[1]])
        score_sorted.append([dialo[2]])
    
    
    worksheet1.update(range_name=f'{chr(65 + col)}{2}:{chr(65 + col)}{113+2}', values=num_sorted)
    worksheet1.update(range_name=f'{chr(65 + 1 + col)}{2}:{chr(65 + 1 + col)}{113+2}', values=dialo_sorted)
    worksheet1.update(range_name=f'{chr(65 + 2 + col)}{2}:{chr(65 + 2 +col)}{113+2}', values=score_sorted)

  

#extract_messages(experiment_1, 0)
#extract_messages(experiment_2, 3)
#extract_messages(experiment_3, 6)


def extract_full_dialogue(experiment, col):
    collected_lines = []
    for i in range(113):
        filename = f'/Users/georgiazhou/research_machine/dialop/dialop/results/old_result/planning-{experiment}/{i}_0.out'
        with open(filename, 'r') as file:
            lines = file.readlines()
        agent_prompt = 0
        user_prompt = False
        record = False
        collected = ''
        #checklist = False
        result = False
        pre_result = False
        record_preferences = True
        preferences = ''
        normalized_reward = 0

        for line in lines:
            
            if re.search(r"Final user Prompt", line):
                pre_result = True
                record = False
            
            if pre_result:
                if user_prompt:
                    
                    if line.startswith(r"Travel Preferences"):
                        record_preferences = True
                    if line.startswith(r"You:"):
                        record_preferences = False
                    if record_preferences:
                        preferences += re.sub(r'\n\s*\n+|\n', r';', line)
                    
                    
                    if re.search(r'Result', line):
                        result = True
                if re.search(r'CITY 2.', line):
                    user_prompt = True
                if result and line.startswith(r">"):           
                    json_string = line.strip()
                    json_string = json_string.replace('<', '').replace('>', '')
                    dic = json.loads(json_string) #collected += re.sub(r'\n\s*\n+', '\n', (line + '\n'))
                    if dic.get('user') == '[accept]':
                        normalized_reward = round(dic.get('info')["reward_normalized"], 2)
                
            if record and agent_prompt == 2:
                
                #if line.startswith(r"Proposal Score"):
                #   collected += re.sub(r'\n\s*\n+', '\n', (line + '\n'))
                #   checklist = True 
                
                collected += re.sub(r'\n\s*\n+', '\n', (line + '\n'))
                
            
            if re.search(r'Final agent Prompt', line):
                record = True
            if re.search(r'USER 2.', line):
                agent_prompt += 1
        preferences = preferences + collected
        collected_lines.append([i, preferences, normalized_reward])
    sorted_lines = sorted(collected_lines, key=lambda x: x[2], reverse=True)
    num_sorted = []
    dialo_sorted = []
    score_sorted = []
    for dialo in sorted_lines:
        num_sorted.append([dialo[0]])
        dialo_sorted.append([dialo[1]])
        score_sorted.append([dialo[2]])
    
    worksheet2.update(range_name=f'{chr(65 + col)}{2}:{chr(65 + col)}{113+2}', values=num_sorted)
    worksheet2.update(range_name=f'{chr(65 + 1 + col)}{2}:{chr(65 + 1 + col)}{113+2}', values=dialo_sorted)
    worksheet2.update(range_name=f'{chr(65 + 2 + col)}{2}:{chr(65 + 2 +col)}{113+2}', values=score_sorted)


#extract_full_dialogue(experiment_1, 0)
#extract_full_dialogue(experiment_2, 3)
#extract_full_dialogue(experiment_3, 6)