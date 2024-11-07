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
import random
from skills import Agent_tell


#try:
#    with open(pathlib.Path(__file__).parent / ".api_key") as f:
#        
#        x = json.load(f)
#        
#        client = OpenAI(api_key=x["api_key"], organization = x["organization"])
#        #pdb.set_trace()
#
#
#        # TODO: The 'openai.organization' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(organization=x["organization"])'
#        
#        
#    print("Loaded .api_key")
#except Exception as e:
#    #openai.api_key = os.getenv("OPENAI_API_KEY")
#    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#    print(e)
#
#if not client.api_key:
#    print("Warning: no OpenAI API key loaded.")


SERVICE_ACCOUNT_FILE = '/Users/georgiazhou/research_machine/dialop/dialop-8759580d9f40.json'
SCOPE = ["https://spreadsheets.google.com/feeds", 
         "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPE)
paper = gspread.authorize(credentials)
spreadsheet = paper.open('Dialop-GPT4-Planning')

columns = ["Experiment 1", "Experiment 2", "Experiment 3"]

worksheet1 = spreadsheet.sheet1
worksheet2 = spreadsheet.get_worksheet(1)
search_sheet = spreadsheet.get_worksheet(2)
other_sheet_Q = spreadsheet.get_worksheet(3)
cleaned_data_sheet = spreadsheet.get_worksheet(4)
ordering_sheet = spreadsheet.get_worksheet(5)
abstract_sheet = spreadsheet.get_worksheet(6)



def extract_preferences_from_worksheet(col):
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
            if line.startswith(r"Travel Preferences"):
                record = True
            else:
                if line.startswith(r"You") or line.startswith(r"User"):
                    record = False
            if record:
                formated = line.split(';')
                for i in range(len(formated)):
                    formated[i] = formated[i] + '\n'
                formated = ''.join(formated)
                collected += formated
        collected_lines.append([num, collected])
          
    num = []
    dialop = []
    for dia in collected_lines:
        num.append([dia[0]])
        dialop.append([dia[1]])
    


    
    ordering_sheet.update(range_name=f'{chr(65 + col)}{2}:{chr(65 + col)}{113+2}', values=num)
    ordering_sheet.update(range_name=f'{chr(65 + 1 + col)}{2}:{chr(65 + 1 + col)}{113+2}', values=dialop)
                
##extract_preferences_from_worksheet(1)



def label_ground_truth_ordering_pref(custom = False, num_list = None, sheet = ordering_sheet):
    if not custom:
        ranges = [f'C{i}' for i in range(2, 115)] 
        preferences = ordering_sheet.batch_get(ranges)
    all_random_orders = []
        #pdb.set_trace()
    try:
        for i in range(0, 113):
            if not custom:
                pref = preferences[i][0][0].split('\n')
                pref = list(filter(bool, pref))
                num = range(len(pref))
                num_list = list(num)
            random.shuffle(num_list)
            all_random_orders.append([str(num_list)])
            print(num_list, i)
    except:
        pdb.set_trace()
   #pdb.set_trace()
    sheet.update(range_name=f'{chr(65 + 1)}{2}:{chr(65 + 1)}{113+2}', values=all_random_orders)

#label_ground_truth_ordering_pref(custom = True, num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], sheet = abstract_sheet)


#def producing_ordering(custom = False, num_list = None, sheet = abstract_sheet):

def mapping(seed_list):
    num_letter_mapping = {}
    letter_num_mapping = {}
    for i in range(len(seed_list)):
        num_letter_mapping[seed_list[i]] = f'{chr(65 + i)}'
        letter_num_mapping[f'{chr(65 + i)}'] = seed_list[i]
    return num_letter_mapping, letter_num_mapping

def insert_ordered_comparison(letter_list, letter_num_mapping):
    ordered_list = []
    for i in range(len(letter_list)):
        if i == 0:
            continue
        else:
            x1 = letter_list[i-1]
            x2 = letter_list[i]
            value_1 = letter_num_mapping[x1]
            value_2 = letter_num_mapping[x2]
            if value_1 < value_2:
                relation = "<"
            elif value_1 > value_2:
                relation = ">"
            else:
                relation = "="
            ordered_list.append(f"{x1} {relation} {x2}")
        
    return ordered_list


def insert_random_comparison(input_list):
    if len(input_list) < 2:
        return input_list
    
    result = []
    comparison_symbols = ['>', '<', '=']
    
    for i, item in enumerate(input_list):
        result.append(item)
        if i < len(input_list) - 1: 
            result.append(random.choice(comparison_symbols))
    
    return result


    
    

def produce_three_variations_letters(input_list, variation_num = 3):
    number_letter_mapping, letter_number_mapping = mapping(input_list)
    variations = []
    for i in range(variation_num):
        random.shuffle(input_list)
        new_letter_list = [number_letter_mapping[x] for x in input_list]
        variation = insert_ordered_comparison(new_letter_list, letter_number_mapping)
        variations.append(variation)
    return variations, letter_number_mapping, number_letter_mapping
    

import networkx as nx
import itertools
from itertools import combinations

def approximate_sort(pairs):
    # Create a directed graph
    graph = nx.DiGraph()

    # Add directed edges based on known pairs
    for greater, lesser in pairs:
        graph.add_edge(greater, lesser)

    # Check if the graph has cycles
    if not nx.is_directed_acyclic_graph(graph):
        print("Cycle detected; inconsistent information")
        return None
    
    def check_ambiguities(G):
        nodes = list(G.nodes())
        ambiguous_pairs = []

        for node1, node2 in combinations(nodes, 2):
            path1 = nx.has_path(G, node1, node2)
            path2 = nx.has_path(G, node2, node1)

            if not path1 and not path2:
                ambiguous_pairs.append((node1, node2))

        return ambiguous_pairs
    
    # Perform topological sorting

    ambiguous = check_ambiguities(graph)
    print(ambiguous)
    
    approximate_order = nx.topological_sort(graph)   
    approximate_order = ' > '.join(approximate_order)
    return approximate_order, ambiguous

def convert_to_greater_than_pairs(variations):
    pairs = []
    for v in variations:  
        for comparison in v:
            left, relation, right = comparison.split()
            if relation == '>':
                pairs.append((left, right))
            elif relation == '<':
                pairs.append((right, left))
            # We ignore '=' relations as they don't imply a 'greater than' relationship
    return pairs

def set_up_spreadsheet(fixed_variations = False):
    ranges = [f'B{i}' for i in range(2, 115)] 
    seeds = abstract_sheet.batch_get(ranges)
    variation_list = []
    letter_num_mapping_list = []
    approximate_order_list = []
    number_letter_mapping_list = []
    flexible_nodes_list = []
    if fixed_variations:
        ranges = [f'C{i}' for i in range(2, 115)] 
        variations_list = abstract_sheet.batch_get(ranges)
        variations_list = [ast.literal_eval(v[0][0]) for v in variations_list]
    for i in range(len(seeds)):
        seed = ast.literal_eval(seeds[i][0][0])
        if not fixed_variations:
            variations, letter_num_mapping, number_letter_mapping = produce_three_variations_letters(seed)
            variation_list.append([str(variations)])
            letter_num_mapping_list.append([str(letter_num_mapping)])
            number_letter_mapping_list.append([str(number_letter_mapping)])
        variations = variations_list[i]
        greater_than_pairs = convert_to_greater_than_pairs(variations)
        approximate_order, flexible_nodes = approximate_sort(greater_than_pairs)
        approximate_order_list.append([str(approximate_order)])
        flexible_nodes_list.append([str(flexible_nodes)])
   
    if not fixed_variations:
        abstract_sheet.update(range_name=f'{chr(65 + 2)}{2}:{chr(65 + 2)}{113+2}', values=variation_list)
        abstract_sheet.update(range_name=f'{chr(65 + 3)}{2}:{chr(65 + 3)}{113+2}', values=letter_num_mapping_list)
        abstract_sheet.update(range_name=f'{chr(65 + 4)}{2}:{chr(65 + 4)}{113+2}', values=number_letter_mapping_list)
    abstract_sheet.update(range_name=f'{chr(65 + 5)}{2}:{chr(65 + 5)}{113+2}', values=approximate_order_list)
    abstract_sheet.update(range_name=f'{chr(65 + 6)}{2}:{chr(65 + 6)}{113+2}', values=flexible_nodes_list)
    


#set_up_spreadsheet(fixed_variations = True)

def api_call_for_ordering():
    ranges = [f'C{i}' for i in range(2, 115)] 
    variations = abstract_sheet.batch_get(ranges)
    results = []
    for i in range(len(variations)):
        v = variations[i][0][0]
        prompt = f"The following is a list of lists of comparisons between different items. Please determine the overall order of the items from the comparisons. Output format from the largest item to the smallest item: item > item > ... > item. \n {v}"
        api_result = Agent_tell(prompt)
        file_path = "/Users/georgiazhou/research_machine/dialop/dialop/results/ordering/" + f"{i}ordering.txt"
        with open(file_path, "w") as text_file:
            text_file.write(api_result)
        lines = api_result.split('\n')
        line = lines[len(lines) - 1]
        results.append([line])
    result_path = "/Users/georgiazhou/research_machine/dialop/dialop/results/ordering/allordering.txt"
    with open(result_path, "w") as text_file:
            text_file.write(str(results))
    #abstract_sheet.update(range_name=f'{chr(65 + 7)}{2}:{chr(65 + 7)}{113+2}', values=results)

def extract_ordering_from_api_result(i):
    file_path = "/Users/georgiazhou/research_machine/dialop/dialop/results/ordering/" + f"{i}ordering.txt"
    with open(file_path, "r") as text_file:
        lines = text_file.readlines()
        line = lines[len(lines) - 1]
        return line
    

# verify the order of ground truth is actually perfect
def verify_graph_ordering():
    ranges = [f'F{i}' for i in range(2, 115)] 
    ground_truth_list = abstract_sheet.batch_get(ranges)
    ranges = [f'D{i}' for i in range(2, 115)] 
    letter_number_mapping_list = abstract_sheet.batch_get(ranges)
    #all not right below
    def verify_ordering(ordering_string, letter_number_mapping):
        # Convert the string to a list of letters
        order = ordering_string.replace(' ', '').split('<')
        
        # Convert letter_number_mapping to a dict if it's not already
        if isinstance(letter_number_mapping, list):
            letter_number_mapping = {item[0][0]: int(item[0][1]) for item in letter_number_mapping}
        
        # Check if the ordering is correct
        for i in range(len(order) - 1):
            current = letter_number_mapping[order[i]]
            next = letter_number_mapping[order[i+1]]
            if current >= next:
                return False
        return True

    # Verify each ground truth ordering
    incorrect_orderings = []
    
    total_orderings = len(ground_truth_list)
    for i in range(total_orderings):
        ground_truth = ground_truth_list[i][0][0]
        letter_number_mapping = letter_number_mapping_list[i][0][0]
        if not verify_ordering(ground_truth, letter_number_mapping):
            incorrect_orderings.append(i)
            print(f"Incorrect ordering at index {i + 2}: {ground_truth}")

    print(f"Incorrect orderings: {len(incorrect_orderings)}/{total_orderings}")
    print(incorrect_orderings)

import Levenshtein
#verify_graph_ordering()
def calculate_similarity_score(api_str, true_str, complex = False):
    """
    Calculate a similarity score between two strings.
    The score is based on the Levenshtein distance, normalized to a 0-1 scale.
    
    Args:
    str1 (str): The first string
    str2 (str): The second string
    
    Returns:
    float: A similarity score between 0 and 1, where 1 is a perfect match
    """
    
    edited_api_str = None
    
    # Split the string at '. Ambiguous: '
    if complex:
        api_str = api_str.replace('/n', '').replace(' ', '').replace('.', '')
        parts = api_str.split('Ambiguous:')

    
    # Find first occurrence of any of the keywords
        keywords = ['vibe', 'location', 'price', 'rating']
        first_idx = len(api_str)

        for keyword in keywords:
            # Check both lowercase and uppercase versions
            idx_lower = api_str.lower().find(keyword.lower())
            if idx_lower != -1 and idx_lower < first_idx:
                first_idx = idx_lower

        # Trim everything before first keyword
        api_str = api_str[first_idx:]
    
    # If Ambiguous list is empty (i.e. '[]'), just use the ordering part
        try:
            if parts[1].strip() == '[]':
                api_str = parts[0]
                distance = Levenshtein.distance(api_str, true_str)
            else:
                distance = -1
        except:
            pdb.set_trace()
    else: 
        distance = Levenshtein.distance(api_str, true_str)
        
    # Calculate the Levenshtein distance between the cleaned strings
    
    
    # Normalize the score
    #max_length = max(len(str1), len(str2))
    #similarity = 1 - (distance / max_length)
    
    return distance, edited_api_str

def calculate_similarity_score_for_all():
    ranges = [f'F{i}' for i in range(2, 115)] 
    ground_truth_list = abstract_sheet.batch_get(ranges)
    ranges = [f'H{i}' for i in range(2, 115)] 
    api_result_list = abstract_sheet.batch_get(ranges)
    scores = []
    betters = []
    for i in range(len(ground_truth_list)):
        ground_truth = ground_truth_list[i][0][0]
        api_result = api_result_list[i][0][0]
        score = calculate_similarity_score(ground_truth, api_result)
        #print(score)
        scores.append([score])
        if score < 5:
            betters.append(i)
    #abstract_sheet.update(range_name=f'{chr(65 + 8)}{2}:{chr(65 + 8)}{113+2}', values=scores)
    print(betters)

##calculate_similarity_score_for_all()
less_sheet = spreadsheet.get_worksheet(7)
def api_input_prompt(variations: str, true_order: str):

    A = 'location'
    B = 'price'
    C = 'rating'
    D = 'vibe'
    # Create a mapping between numbers and words
    true_order = true_order.replace('A', f'{A}').replace('B', f'{B}').replace('C', f'{C}').replace('D', f'{D}')
    translated_variations = []
    for v in variations:
        for str in v:
            str = str.replace('A', f'{A}').replace('B', f'{B}').replace('C', f'{C}').replace('D', f'{D}') 
            translated_variations.append(str)
    prompt = f'You are a traveler explaining your preferences. State it in the form of " I care about something more than something". Here is how much you care about each of the four preferences: {translated_variations}. Simply translate the preferences into 6 comparisons in the form of " I care about something more than something". You should write one comparison per line. Remember, only write 6 lines. '
    return prompt


def easy_word_sentence_ordering_set_up(only_api_prompt = False):

    preferences = [1, 2, 3, 4]
    variation_list = []
    letter_num_mapping_list = []
    number_letter_mapping_list = []
    approximate_order_list = []
    flexible_nodes_list = []
    all_api_prompts = []
    if only_api_prompt:
        ranges = [f'F{i}' for i in range(2, 52)] 
        variation_list = less_sheet.batch_get(ranges)
        ranges = [f'E{i}' for i in range(2, 52)] 
        approximate_order_list = less_sheet.batch_get(ranges)
        
    easy_api_generated_prompts_path = '/Users/georgiazhou/research_machine/dialop/dialop/results/ordering/easy_api_generated_prompts/'
    for i in range(50):
        
        if not only_api_prompt:
            random.shuffle(preferences)
            variations, letter_number, number_letter = produce_three_variations_letters(preferences, variation_num = 2)
            variation_list.append([str(variations)])
            letter_num_mapping_list.append([str(letter_number)])
            number_letter_mapping_list.append([str(number_letter)])
            greater_than_pairs = convert_to_greater_than_pairs(variations)
            approximate_order, flexible_nodes = approximate_sort(greater_than_pairs)
            flexible_nodes_list.append([str(flexible_nodes)])
            print(flexible_nodes)
            
        else:
            variations = variation_list[i][0][0]
            approximate_order = approximate_order_list[i][0][0]
        api_prompt = api_input_prompt(str(variations), str(approximate_order)) 
        api_prompt_done = Agent_tell(api_prompt)
        with open(f'{easy_api_generated_prompts_path}{i}api_prompt_done.txt', 'w') as f:
            f.write(api_prompt_done)
        all_api_prompts.append([api_prompt_done])
        approximate_order_list.append([str(approximate_order)])
        
    ##less_sheet.update(range_name=f'{chr(65 + 1)}{2}:{chr(65 + 1)}{50+2}', values=variation_list)
    ##less_sheet.update(range_name=f'{chr(65 + 2)}{2}:{chr(65 + 2)}{50+2}', values=letter_num_mapping_list)
    ##less_sheet.update(range_name=f'{chr(65 + 3)}{2}:{chr(65 + 3)}{50+2}', values=number_letter_mapping_list)
    ##less_sheet.update(range_name=f'{chr(65 + 4)}{2}:{chr(65 + 4)}{50+2}', values=approximate_order_list)
    ##less_sheet.update(range_name=f'{chr(65 + 5)}{2}:{chr(65 + 5)}{50+2}', values=all_api_prompts)   
    ##less_sheet.update(range_name=f'{chr(65 + 6)}{2}:{chr(65 + 6)}{50+2}', values=flexible_nodes_list)   
   

##easy_word_sentence_ordering_set_up(only_api_prompt = True)

def easy_word_sentence_api_response():
    ranges = [f'F{i}' for i in range(2, 52)]
    api_prompts = less_sheet.batch_get(ranges)
    api_responses = []
    prompt = "You are an helpful assistant. Based on the setences given here, please list this traveler's preferences from most prefered to least preferred. When the relationship between two items is ambiguous, write their relationship as item1 == item2. Example output when can't tell if price is more important than rating: location > vibe > price == rating "  
    api_responses = []
    api_response_file_path = '/Users/georgiazhou/research_machine/dialop/dialop/results/ordering/easy_api_response/'
    
    for i in range(len(api_prompts)):
        prompt_full = prompt + api_prompts[i][0][0]
        api_response = Agent_tell(prompt_full)
        with open(f'{api_response_file_path}{i}api_response.txt', 'w') as f:
            f.write(api_response)
        api_responses.append([api_response])
        
    less_sheet.update(range_name=f'{chr(65 + 7)}{2}:{chr(65 + 7)}{50+2}', values=api_responses)

def easy_word_eval():
    ranges = [f'H{i}' for i in range(2, 52)]
    api_responses = less_sheet.batch_get(ranges)
    ranges = [f'E{i}' for i in range(2, 52)]
    true_orders = less_sheet.batch_get(ranges)
    scores = []
    A = 'location'
    B = 'price'
    C = 'rating'
    D = 'vibe'
    true_order_in_words = []
    full_scores = []
    #list_edited_api_response = []
    # Create a mapping between numbers and words
    for i in range(len(api_responses)):
        true_order = true_orders[i][0][0].replace('A', f'{A}').replace('B', f'{B}').replace('C', f'{C}').replace('D', f'{D}')
        true_order_in_words.append([true_order])
        api_response = api_responses[i][0][0]
        score, _ = calculate_similarity_score(api_response, true_order)

        if score == 0:
            full_scores.append(i)
        scores.append([score])
        #list_edited_api_response.append([edited_api_response])
    print(f"Total score: {len(full_scores)}/{len(api_responses)}")
    print(full_scores)
    less_sheet.update(range_name=f'{chr(65 + 8)}{2}:{chr(65 + 8)}{50+3}', values=scores)
    less_sheet.update(range_name=f'{chr(65 + 9)}{2}:{chr(65 + 9)}{50+3}', values=true_order_in_words)
    #less_sheet.update(range_name=f'{chr(65 + 10)}{2}:{chr(65 + 10)}{49+2}', values=list_edited_api_response)
##easy_word_sentence_api_response()
##easy_word_eval()

def api_test_ambiguity():
    ranges = [f'F{i}' for i in range(2, 52)]
    api_prompts = less_sheet.batch_get(ranges)
    prompt = "You are a helpful assitant. From what a traveler says about their preferences, can you tell whether the description contain enough information for you to rank the preferences from most prefered to least preferred?  If you can, please output 'yes'. If you cannot, please output 'no' and explain why."
    list_ambiguous = []
    for i in range(len(api_prompts)):
        full_prompt = prompt + api_prompts[i][0][0]
        api_response = Agent_tell(full_prompt)
        list_ambiguous.append([api_response])
        print(api_response)
    less_sheet.update(range_name=f'{chr(65 + 11)}{2}:{chr(65 + 11)}{50+2}', values=list_ambiguous)

##api_test_ambiguity()

def convert_ambiguity_to_score():
    ranges = [f'L{i}' for i in range(2, 52)]
    ambiguities = less_sheet.batch_get(ranges)
    ranges = [f'G{i}' for i in range(2, 52)]
    true_ambiguities = less_sheet.batch_get(ranges)
    scores = []
    scores_right = []
    
    for i in range(len(ambiguities)):
        true_ambiguity = ast.literal_eval(true_ambiguities[i][0][0])
        if true_ambiguity == []:
            true_state = True
        else:
            print(true_ambiguity)
            true_state = False
        if ambiguities[i][0][0].lower().startswith('yes'):
            api_state = True
        elif ambiguities[i][0][0].lower().startswith('no'):
            api_state = False
        else:
            pdb.set_trace()
            
        if api_state == true_state:
            scores.append([1])
            scores_right.append(i)
        else:
            scores.append([-1])
    print(len(scores_right)/len(scores), scores_right)
    less_sheet.update(range_name=f'{chr(65 + 12)}{2}:{chr(65 + 12)}{50+2}', values=scores)



convert_ambiguity_to_score()







# all the below functions are for evaluate the api_prompts 
def convert_preference_to_inequality(preference):
    # Remove leading/trailing whitespace and split the sentence
    parts = preference.strip().lower().replace('.', '').split()
    
    def extract_attribute(parts):
        keywords = ['location', 'price', 'rating', 'vibe']

        for i, part in enumerate(parts):
            for keyword in keywords:
                if keyword == part:
                    return parts[i:]
        else:
            pdb.set_trace()
    
    # Extract the two attributes being compared
    parts = extract_attribute(parts)

    if parts[1] == 'more':
        sign = '>'
    elif parts[1] == 'less':
        sign = '<'
    else:
        sign = '='
    
    # Construct the inequality
    return f"{parts[0]}{sign}{parts[3]}"

def convert_inequalities_to_order(inequalities):
    # Create a directed graph
    graph = nx.DiGraph()

    # Add edges to the graph based on inequalities
    for inequality in inequalities:
        if '<' in inequality:
            lesser, greater = inequality.split('<')
            graph.add_edge(lesser, greater)
        elif '>' in inequality:
            greater, lesser = inequality.split('>')
            graph.add_edge(lesser, greater)
        # Ignore '=' relations as they don't contribute to the ordering

    # Check if the graph has cycles
    if not nx.is_directed_acyclic_graph(graph):
        print("Cycle detected; inconsistent information", inequality)

    # Perform topological sorting
    order = list(nx.topological_sort(graph))
    
    order_f = '<'.join(order)
    order_finish = re.sub(r'(\w)([<>])(\w)', r'\1 \2 \3', order_f)

    # Join the items with '<' to create the final order string
    return order_finish


def parse_word_prompt(prompt):
    prompt_parsed = []
    for line in prompt.split('\n'):
        prompt_parsed.append(convert_preference_to_inequality(line))

    return prompt_parsed
def double_check_api_prompts():
    ranges = [f'F{i}' for i in range(2, 49)]
    api_prompts = less_sheet.batch_get(ranges)
    all_orders = []
    for i in range(len(api_prompts)):
        prompt = api_prompts[i][0][0]
        prompt_parsed = parse_word_prompt(prompt)
        try:
            order = convert_inequalities_to_order(prompt_parsed)
        except:
            pdb.set_trace()
        all_orders.append([order])
    less_sheet.update(range_name=f'{chr(65 + 10)}{2}:{chr(65 + 10)}{49+2}', values=all_orders)

#double_check_api_prompts()
# Highlights row 5


# Example usage:
# score = calculate_similarity_score("ABCDE", "ABCDF")
# print(f"Similarity score: {score}")

##lines = []
##for i in range(113):
##    line = extract_ordering_from_api_result(i)
##    lines.append([line])
##abstract_sheet.update(range_name=f'{chr(65 + 7)}{2}:{chr(65 + 7)}{113+2}', values=lines)
    
#api_call_for_ordering()

# Known relationships A > C and B > D

#num_list = [1, 2, 3, 4, 5]
#variations, letter_num_mapping = produce_three_variations_letters(num_list)
#print(variations)
#print(letter_num_mapping)
#true_order = discover_true_ordering(variations)
#if true_order:
#    print("The discovered true ordering of letters is:", ' < '.join(true_order))
#else:
#    print("No valid ordering found.")
##num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#letter_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
#variations, letter_num_mapping = produce_three_variations_letters(num_list)
#print(variations)
#print(letter_num_mapping)
##result = insert_random_comparison(letter_list)
##print(result)
# so first there's a letter number mapping from a seed list. then translate three variations


# Example usage:
# input_list = ['A', 'B', 'C', 'D']
# result = insert_random_comparison(input_list)
# print(result)  # Might output: ['A', '<', 'B', '>', 'C', '=', 'D']
#[['E < H', 'H > B', 'B < D', 'D < J', 'J > G', 'G > F', 'F > A', 'A < C', 'C < I'], ['B < G', 'G > F', 'F > D', 'D < H', 'H > E', 'E < J', 'J > A', 'A < C', 'C < I'], ['I > D', 'D < E', 'E < F', 'F < J', 'J > A', 'A < B', 'B < C', 'C < G', 'G < H']]
#{'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10}


#the input is a list of strings. insert ramdom choice symbol out of ">", "<", and "=" between strings and form a new list.

# Example usage:
# random_num = generate_random_number(10)  # Generates a random number between 1 and 10