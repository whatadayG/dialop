#evaluate exclusivity from generated conversations? 
import gspread
import re
from google.oauth2.service_account import Credentials
import pdb
import json
import openai
from openai import OpenAI
import ast
import pathlib


try:
    with open(pathlib.Path(__file__).parent / ".api_key") as f:
        
        x = json.load(f)
        
        client = OpenAI(api_key=x["api_key"], organization = x["organization"])
        #pdb.set_trace()


        # TODO: The 'openai.organization' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(organization=x["organization"])'
        
        
    print("Loaded .api_key")
except Exception as e:
    #openai.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print(e)

if not client.api_key:
    print("Warning: no OpenAI API key loaded.")


#SERVICE_ACCOUNT_FILE = '/Users/georgiazhou/research_machine/dialop/dialop-8759580d9f40.json'
#SCOPE = ["https://spreadsheets.google.com/feeds", 
#         "https://www.googleapis.com/auth/drive"]
#credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPE)
#paper = gspread.authorize(credentials)
#spreadsheet = paper.open('Dialop-GPT4-Planning')

columns = ["Experiment 1", "Experiment 2", "Experiment 3"]

#worksheet1 = spreadsheet.sheet1
#worksheet2 = spreadsheet.get_worksheet(1)
#search_sheet = spreadsheet.get_worksheet(2)
#other_sheet_Q = spreadsheet.get_worksheet(3)
#cleaned_data_sheet = spreadsheet.get_worksheet(4)


def extract_tools_from_worksheet(col):
    collected_lines = []
    ranges = [f'H{i}' for i in range(2, 115)] 
    ranges_num = [f'G{i}' for i in range(2, 115)] 
        #val = worksheet2.acell(f'H{i}').value
    records = worksheet2.batch_get(ranges)
    records_num = worksheet2.batch_get(ranges_num)
    #pdb.set_trace()
    for i in range(len(records)):
        text = records[i][0][0]
        lines = text.split('\n')
        num = records_num[i][0][0]
        collected = ''
        record = False
        for line in lines:
            if line.startswith(r"You:[tool]"):
                record = True
            else:
                if line.startswith(r"You") or line.startswith(r"User"):
                    record = False
            if record:
                collected += line
        collected_lines.append([num, collected])
          
    num = []
    dialop = []
    for dialo in collected_lines:
        num.append([dialo[0]])
        dialop.append([dialo[1]])
    
    
    search_sheet.update(range_name=f'{chr(65 + col)}{2}:{chr(65 + col)}{113+2}', values=num)
    search_sheet.update(range_name=f'{chr(65 + 1 + col)}{2}:{chr(65 + 1 + col)}{113+2}', values=dialop)
                
    
#extract_tools_from_worksheet(1)


def get_data(i = 0, batch = False, file_num = False, fil_n = None):
    def get_data_single(num):
        filename = f'/Users/georgiazhou/research_machine/dialop/dialop/results/old_result/planning-4-3/{num}_0.out'
        with open(filename, 'r') as file:
            lines = file.readlines()
        query = False
        data = ''
        database = False
        
        for line in lines:
            if re.search(r'Query Executor Prompt', line):
                query = True
            if query:
                #pdb.set_trace()
                if line.startswith('Query'):
                    database = False
                    query = False
                if database:
                    data += line
                if line.startswith(">Database"):
                    database = True
        return data
    if batch:
        results = []
        #ranges = [f'C{i}' for i in range(2, 115)] 
        ranges_num = [f'B{i}' for i in range(2, 115)] 
        #val = worksheet2.acell(f'H{i}').value
        #records = search_sheet.batch_get(ranges)
        if file_num:
            records_num = fil_n
        else:
            records_num = search_sheet.batch_get(ranges_num)
        for i in range(len(records_num)):
            num = records_num[i][0][0]
            data = get_data_single(num)
            results.append(data)
        return results, records_num
    else:
        results = []
        num = search_sheet.acell(f'B{i}').value
        results.append(get_data_single(num))
        return results, None

#check if toolcall return was right 
#pdb.set_trace()

def get_tool_list(col = 2, file_num = False, fil_n = None):
    results = []
    if file_num:
        for num_str in fil_n:
            try:
                cell_B = search_sheet.find(num_str, in_column=2)
                value_C = search_sheet.cell(cell_B.row, 3).value
                results.append(value_C)
            except:
                pdb.set_trace()
    else:
        ranges_num = [f'{chr(65 + col)}{i}' for i in range(2, 115)] 
        records_num = search_sheet.batch_get(ranges_num)
        for i in range(len(records_num)):
            try:
                text = records_num[i][0][0]
                results.append(text)
            except:
                pdb.set_trace()
    return results

#check if exclusivity works 

#prompt = search_sheet.acell('C2').value
#prompt = prompt + "The above is a list of search queries done by you. Please tell me if any number of items are exclusive to each other and why."


def Agent_tell(string, temperature = 0.8): 
    prompt = {'role': 'system', 'content': string,}      
    model_kwargs = dict(
            
            model='gpt-4o',
            temperature=temperature,
            top_p=.95,
            frequency_penalty=0,
            messages = [prompt]
        )

    #kwargs.update(**self.model_kwargs)
        # 'message' key added for new openai API; role is the role of the mssage
    #pdb.set_trace()
    response = client.chat.completions.create(**model_kwargs)
    return response.choices[0].message.content

#def accuracy_score(results):
def clean(n):
    # Clean the string by replacing commas with dots and removing trailing dots
    cleaned = n.replace('.', '')
    cleaned = cleaned.replace('`', '')# Replace period with dots
    cleaned = cleaned.strip() 
    try:
        cleaned = int(cleaned)
    except:
        pdb.set_trace()
    
    return cleaned
    
def check_spreadsheet(max_num):
    all_score = []
    def check_api_result(tool, agent_tell):
        nonlocal all_score
        num_q = len(re.findall(r'You', tool,))
        lines = agent_tell.split('\n')
        for line in lines:
            if re.search(r'\[result\]', line):
                #pdb.set_trace()
                result = line.split('::')[1].strip()
                results = result.split(',')
                for i in range(len(results)):
                    try:
                        results[i] = int(results[i])
                    except:
                        
                        results[i] = clean(results[i])
                # currently there can be '3.' which is why int() won't work. see computergpt history for how to fix
                return_q = results[0]
                return_incorrect = results[1]
                return_correct = results[2]
                score = round(return_correct / return_q, 2)
                if return_q == num_q and (return_incorrect + return_correct) == num_q:
                        #average_score = ((num_score * average_score) + score) / (num_score + 1)
                    all_score.append(score)
                        
                    return [True, line, score]
                else:
                    return [False, line, num_q, score]
        
    results = []
    list_data, records_num = get_data(batch = True)
    list_tool = get_tool_list()
    save_prompt_list = []
    save_number_list = []
    for i in range(2, max_num):
        try:
            data = list_data[i-2]
            tool = list_tool[i-2]
        except:
            pdb.set_trace()
        
        #the total number of queries, the number of incorrect queries, and the number of correct queries
        prompt = data + tool + "The above is a dataset and a list of search queries done by you. Please double check the dataset and and return the total number of queries, the number of incorrect queries, and the number of correct queries in the form of [result]::total number of queries, number of incorrect queries, number of correct queries. You can tell the begining of a query by 'You:[tool] Search'. Examples of return: [result]:: 5, 0, 5; [result]:: 3, 1, 2. You should think step by step."
        agent_tell = Agent_tell(prompt)
        if type(agent_tell) == list:
            agent_tell = str(agent_tell)
        file_path = "/Users/georgiazhou/research_machine/dialop/dialop/results/search/" + f"tooluse_evaluate{i}.txt"
        with open(file_path, "w") as text_file:
            text_file.write(agent_tell)
        api_result = check_api_result(tool, agent_tell)
        if api_result[0]:
            print(api_result)
            results.append([str(api_result)])
            if api_result[2] == 1.0:
                save_prompt = data + tool + "Based on the provided data, tool calls, and return of the tool calls, are any of the events mutually exclusive? If so, please explain why and list them as [exclusive]:: [even-name1, event-name2], [even-name3, event-name4. You should think step by step."
                save_prompt_list.append([save_prompt])
                save_number_list.append([records_num[i][0][0]])
# write seperating line between each return; look at the tooluse_eval file. may need to channge some in check_api since 
# also right now it would re-write. so either figure how not to do that or just put new thing in a new file
        else:
            results.append([str(api_result)])
            # or append [None]
            pdb.set_trace()
        
        
            
    
        
    print(sum(all_score) / len(all_score))
    file_path_1 = "/Users/georgiazhou/research_machine/dialop/dialop/results/search/" + f"save_prompt_list.txt"
    file_path_2 = "/Users/georgiazhou/research_machine/dialop/dialop/results/search/" + f"save_number_list.txt"
    file_path_3 = "/Users/georgiazhou/research_machine/dialop/dialop/results/search/" + f"results.txt"
    with open(file_path_1, "w") as text_file:
        text_file.write(str(save_prompt_list))
    with open(file_path_2, "w") as text_file:
        text_file.write(str(save_number_list))
    with open(file_path_3, "w") as text_file:
        text_file.write(str(results))
    
    #other_sheet_Q
    #.update(range_name=f'C{2}:C{len(save_prompt_list)+1}', values=save_prompt_list)
    #other_sheet_Q
    #.update(range_name=f'B{2}:B{len(save_number_list)+1}', values=save_number_list)
    search_sheet.update(range_name=f'L{2}:L{max_num}', values=results)


#check_spreadsheet(115)

#    list_data, records_num = get_data(batch = True)
#    list_tool = get_tool_list()
def find_good_results():
    good_results = []
    exclusive_results = []
    list_data, records_num = get_data(batch = True)
    eval_D = get_tool_list(3)
    eval_H = get_tool_list(7)
    eval_L = get_tool_list(11)
    eval_P = get_tool_list(15)
    #pdb.set_trace()
    for i in range(len(eval_D)):
        try:
            if ast.literal_eval(eval_D[i])[2] == ast.literal_eval(eval_H[i])[2] == ast.literal_eval(eval_L[i])[2] == ast.literal_eval(eval_P[i])[2]:
                num = records_num[i][0][0]
                good_results.append(num)
                if float(ast.literal_eval(eval_D[i])[2]) == 1.0:
                    #pdb.set_trace()
                    exclusive_results.append(num)
        except:
            pass
    print(good_results)
    print(exclusive_results)
    return good_results, exclusive_results

#good_results, exclusive_results = find_good_results()


#vertical_list = [[num] for num in numbers]
#cleaned_data_sheet.update(range_name=f'B{2}:B{len(vertical_list)+1}', values=vertical_list)

#Don't need API
def make_exclusive_data(numbers):
    list_data, records_num = get_data(batch = True, file_num = True, fil_n = numbers)
    list_tool = get_tool_list(file_num = True, fil_n = numbers)
    prompt_list = []
    record_list = []
    #pdb.set_trace()
    for i in range(len(numbers)):       
        data = list_data[i]
        tool = list_tool[i]
        save_prompt = data + tool
        record = tool + '\n' + "You are a helpful assistant analysing the tool search and return of a dataset. You should identify features that can't be satisfied at the same time by the same event. If the return of a search is no result, it means that there's no place that satisfies the search requirements. I know that the boolean values of the same feature are mutually exclusive (parking:True and parking:False are mutually exclusive) so you don't need to mention cases similar to this. Please explain and list features that can't be satisfied at the same time by the same event as [exclusive]:: [[even-name1, event-name2, ... event-name-n], [event-name-2-1, event-name-2-2, ... event-name-2-n]]. You should think step by step."
        record_list.append([record])
        prompt_list.append([save_prompt])
    #pdb.set_trace()
    #cleaned_data_sheet.update(range_name=f'C{2}:C{len(prompt_list)+2}', values=prompt_list)
    cleaned_data_sheet.update(range_name=f'D{2}:D{len(record_list)+2}', values=record_list)

numbers = ['19', '32', '41', '66', '92', '72', '84', '93', '54', '27', '80', '13', '79', '96', '47', '70', '73']


def check_exclusive():
    api_results = []
    ranges = [f'D{i}' for i in range(2, 19)]
    prompt_list = cleaned_data_sheet.batch_get(ranges)
    #record_list = cleaned_data_sheet.batch_get(range_name=f'D{2}:D{len(record_list)+2}')
    for i in range(len(prompt_list)):
        prompt = prompt_list[i][0][0]
        agent_tell = Agent_tell(prompt)
        api_results.append([agent_tell])
    cleaned_data_sheet.update(range_name=f'F{2}:F{len(api_results)+2}', values=api_results)

#make_exclusive_data(numbers)
#check_exclusive()


    
# now for all the ones that score 1, use them to check if exclusivity works
# and check A>B, C>D, if they know to ask something that decides whether AD or BC is better
# maybe put what's below in a function?
#all_others = []
#
#exclusive_numbers = numbers
#all_valid_numbers = ['6', '10', '111', '19', '18', '32', '41', '66', '81', '92', '74', '101', '1', '29', '34', '20', '24', '104', '72', '84', '85', '93', '90', '54', '23', '27', '80', '25', '3', '4', '13', '33', '79', '106', '58', '96', '47', '59', '70', '112', '37', '5', '40', '73', '109']
#other_num_str = [num for num in all_valid_numbers if num not in exclusive_numbers]
#other_numbers = [[num] for num in all_valid_numbers if num not in exclusive_numbers]
#other_sheet_Q.update(range_name=f'B{2}:B{len(other_numbers)+1}', values=other_numbers)
#list_data, records_num = get_data(batch = True, file_num = True, fil_n = other_numbers)
#list_tool = get_tool_list(file_num = True, fil_n = other_num_str)
#for i in range(len(other_numbers)):
#    data = list_data[i]
#    tool = list_tool[i]
#    save_prompt = data + tool
#    all_others.append([save_prompt])
#pdb.set_trace()
#other_sheet_Q.update(range_name=f'C{2}:C{len(all_others)+1}', values=all_others)
    

