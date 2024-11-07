import sys
import os
import re
import pdb
import random
import json
# Get the parent dir
# Add the parent directory to the system path
sys.path.append('/Users/georgiazhou/research_machine/dialop/dialop/')
# Now you can import the module
from skills import Agent_tell

def generate_all_features():
    prompt = "You are a helpful assistant. List all features of a traveler relavent to day trip planing, such as 'has kids', 'casual', etc. Say the number of features you listed in the end. "
    agent_tell_response = Agent_tell(prompt)
    open('/Users/georgiazhou/research_machine/dialop/dialop/RL/features.txt', 'w').write(agent_tell_response)
    print(agent_tell_response)

def generate_assumption_as_dic():
    assumptions = {}
    assumptions['known'] = []
    assumptions['unknown'] = []
##generate_all_features()

def generate_personas():
    prompt = "You are a helpful assistant. List all communication styles of clients important to day trip planing, such as 'doesn't offer a lot of information so needs questions prompting, 'like to know a lot of information beforehand', etc. Say the number of personas you listed in the end. "
    agent_tell_response = Agent_tell(prompt)
    open('/Users/georgiazhou/research_machine/dialop/dialop/RL/communication_styles.txt', 'w').write(agent_tell_response)
    print(agent_tell_response)

##generate_personas()
communication_styles_file = '/Users/georgiazhou/research_machine/dialop/dialop/RL/communication_styles.txt'

def extract_text_between_symbols(line, symbol="**"):
    matches = re.findall(r'\*\*(.*?)\*\*', line)
    return matches

def extract_text_and_value(line):
    matches = re.findall(r'\*\*(.*?)\*\*', line)
    if len(matches) >= 1:
        key = matches[0]
        # Find everything after the second **
        after_second_star = re.split(r'\*\*.*?\*\*', line, maxsplit=1)
        if len(after_second_star) > 1:
            value = after_second_star[1].strip(': ').strip()
            return {key: value}
    return {}


def generate_persona_tokens():
    with open(communication_styles_file, 'r') as file:
        lines = file.readlines()
    all_tokens = []
    for line in lines:
        tokens = extract_text_between_symbols(line)
        if tokens:
            all_tokens.append(tokens)
    return all_tokens

def generate_persona_explanation():
    with open(communication_styles_file, 'r') as file:
        lines = file.readlines()
    all_tokens = []
    for line in lines:
        tokens = extract_text_and_value(line)
        if tokens:
            all_tokens.append(tokens)
    return all_tokens

def tuning_personas():
    ##all_personal_dict = {}
    
    all_explanation_dict = generate_persona_explanation()
    
    merged_dict = {}
    for d in all_explanation_dict:
        merged_dict.update(d)
    
   ##all_tokens = generate_persona_tokens()
   ##for i in range(20):
   ##    per_personal_tokens = random.sample(all_tokens, 3)
   ##    per_personal_dict = dict([(per_personal_tokens[0][0], round(random.uniform(0, 1), 2)), (per_personal_tokens[1][0], round(random.uniform(0, 1), 2)), (per_personal_tokens[2][0], round(random.uniform(0, 1), 2))])
   ##    all_personal_dict[i] = per_personal_dict
   ##
    with open('/Users/georgiazhou/research_machine/dialop/dialop/RL/explanation_per_persona.txt', 'w') as f:
        json.dump(merged_dict, f, indent=4)


tuning_personas()
        


