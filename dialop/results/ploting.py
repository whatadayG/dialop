
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import pdb


failed = []
def extract_result_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()  # Read all lines into a list
            if lines:
                return lines[-1].strip()  # Get the last line and remove extra whitespace
            else:
                return "The file is empty."
    except FileNotFoundError:
        return 0


# Example usage
#script_dir = os.path.dirname(os.path.abspath(__file__))

def extract_result(file_path):
    i = 0
    all_result = []
    while i < 113:
        file_name = f'{i}_0.out'    
        file_path_c = file_path + file_name
        extracted = extract_result_from_file(file_path_c)
        if extracted != 0:
            extracted = extracted.replace('<', '').replace('>', '')
            all_result.append(extracted)
        i += 1
    return all_result




#set the below variable to global when needing to print them and allow print
def check_all_result(all_result):
    '''global x
    global y
    global agent_tool
    global user_message
    global agent_message
    global agent_think
    global user_think
    global user_reject
    global failed_other
    global reject 
    global tool 
    global message 
    global think 
    global info 
    global agent_message_info 
    global agent_tool_info 
    global user_reject_info
    '''
    global x
    global y 

    agent_tool = 0
    user_message = 0
    agent_message = 0
    agent_think = 0
    user_think = 0
    user_reject = 0
    failed_other = 0
    reject = "[reject]"
    tool = "[tool]"
    message = "[message]"
    think = "[think]"
    info = []
    agent_message_info = []
    agent_tool_info= []

    averaged_reward = []

    user_reject_info = []
    user_message_info = []
    machine_error = 0

    reward_batch = 0
    def average_reward_word(dic):
        nonlocal reward_batch
        nonlocal averaged_reward
        num = dic["num_words"]
        reward = dic['info']["reward_normalized"]
        averaged = round(reward / (num*0.001), 2) 
        reward_batch += averaged
        averaged_reward.append(averaged)

    for i, dic in enumerate(all_result):
        try:
            json_string = dic
            json_string = json_string.strip() 
            dic = json.loads(json_string)
        except:
            pdb.set_trace()
        #pdb.set_trace()
        
        if dic.get('user') == '[accept]':
            x.append(dic["num_words"]) 
            y.append(dic['info']["reward_normalized"])
            average_reward_word(dic)
        else:
            if '\nError' in dic.get('user'):
                machine_error += 1
            elif tool in dic.get('agent'):
                agent_tool += 1
                agent_tool_info.append(i)
            elif message in dic.get('agent'):
                agent_message += 1
                agent_message_info.append(i)
            elif think in dic.get('agent'):
                agent_think += 1
            elif think in dic.get('user'):
                user_think += 1
            elif message in dic.get('user'):
                user_message += 1
                user_message_info.append(i)
            else:
                info.append(dic.get('user'))
                if '[reject]' == dic.get('user'):
                    user_reject +=1
                    user_reject_info.append(i)
                else:
                    failed_other += 1
    def print_info():
        print(agent_tool, "agent_tool")
        print(agent_message, "agent_message")
        print(agent_think, "agent_think")
        print(user_message, "user_message")
        print(user_reject, "user_reject")
        print(machine_error, "machine_error")
        print(failed_other, "failed_other")
        #print(info, "info")
        print(agent_message_info, "agent_message_info")
        print(agent_tool_info, "agent_tool_info")
        print(user_reject_info, "user_reject_info")
        print(user_message_info, "user_message_info")
    
    print_info()
    #pdb.set_trace()
    return reward_batch, averaged_reward, 
    

def print_info():
    print(agent_tool, "agent_tool")
    print(agent_message, "agent_message")
    print(agent_think, "agent_think")
    print(user_message, "user_message")
    print(user_reject, "user_reject")
    print(failed_other, "failed_other")
    #print(info, "info")
    print(agent_message_info, "agent_message_info")
    print(agent_tool_info, "agent_tool_info")
    print(user_reject_info, "user_reject_info")
    print(user_message_info, "user_message_info")
    
    

def make_scatter (num, path):
    path = path
    all_result = extract_result(path)
    reward_batch, averaged_reward = check_all_result(all_result)
    all_reward_averaged = reward_batch/len(averaged_reward)
    print(round(all_reward_averaged, 2), "all_reward_averaged")
    print(len(x), "valid")
    plt.scatter(x, y, color='blue', marker='o')

    # Add labels and title
    plt.title('Scatter Plot of GPT4')
    plt.xlabel('Number of Words')
    plt.ylabel('Reward')

    # Show the plot
    plt.savefig(f'Scatter_Plot_gpt4_selfplay{num}.png')

    plt.close()


#path1 = "/Users/georgiazhou/research_machine/dialop/dialop/results/planning/Self-play-gpt4/"
#all_result1 = extract_result(path1)
#path2 = "/Users/georgiazhou/research_machine/dialop/dialop/old_result/planning-3/"
#all_result2 = extract_result(path2)
path3 = "/Users/georgiazhou/research_machine/dialop/dialop/old_result/planning-4-2/"
all_result3 = extract_result(path3)
#pdb.set_trace() 



x=[]
y=[]

agent_tool = 0
user_message = 0
agent_message = 0
agent_think = 0
user_think = 0
user_reject = 0
failed_other = 0
reject = "[reject]"
tool = "[tool]"
message = "[message]"
think = "[think]"
info = []
agent_message_info = []
agent_tool_info= []

averaged_reward = []

user_reject_info = []
user_message_info = []


#path = "/Users/georgiazhou/research_machine/dialop/dialop/old_result/planning-2/"
#make_scatter(-1, path)




reward_batch3, averaged_reward3 = check_all_result(all_result3)
all_reward_averaged3 = reward_batch3/len(averaged_reward3)
print(all_reward_averaged3, "all_reward_averaged3")
print(reward_batch3, "reward_batch3")
print(len(averaged_reward3), "len(averaged_reward3)")


#pdb.set_trace()
#reward_batch1, averaged_reward1 = check_all_result(all_result1)
#all_reward_averaged1 = reward_batch1/len(averaged_reward1)
#reward_batch2, averaged_reward2 = check_all_result(all_result2)
#all_reward_averaged2 = reward_batch2/len(averaged_reward2)

#print(averaged_reward, "averaged_reward_conversation")
#print(round(all_reward_averaged1, 2), "all_reward_averaged1")
#print(round(all_reward_averaged2, 2), "all_reward_averaged2")
#print(round(all_reward_averaged3, 2), "all_reward_averaged3")
#print_info(2)
#print(len(x), "valid")



#count3 = []
#for i in range(len(averaged_reward3)):
#    count3.append(float(i))
#count1 = []
#for i in range(len(averaged_reward1)):
#    count1.append(float(i))
#
#count2 = []
#for i in range(len(averaged_reward2)):
#    count2.append(float(i))
#    
#
#plt.hist(count1, bins=30, alpha = 0.5, weights=averaged_reward1, label="experiment3", color="red")
#plt.hist(count2, bins=30, alpha = 0.5, weights=averaged_reward2, label="experiment1", color="cyan")
##plt.hist(count3, bins=30, alpha = 0.5, weights=averaged_reward3, label="experiment2", color="blue")
#
#plt.title('reward averaged per word')
#plt.xlabel('the ith conversation')
#plt.ylabel('Reward')
#plt.legend()
#
# Show the plot
#plt.savefig('reward_averaged_per_word_3.1_experiment.png')
#
#plt.close()




'''
##### trying to display histogram together from difference experiment batches
import matplotlib.pyplot as plt
from PIL import Image

# Open the histogram images
img1 = Image.open('reward_averaged_per_word1.png')
img2 = Image.open('reward_averaged_per_word2.png')

# Create a figure
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Display each histogram image in its respective subplot
axs[0].imshow(img1)
axs[0].axis('off')  # Hide the axis
axs[0].set_title('Histogram 1')

axs[1].imshow(img2)
axs[1].axis('off')  # Hide the axis
axs[1].set_title('Histogram 2')

# Display the combined figure
plt.savefig('Combined_histogram.png')
'''

