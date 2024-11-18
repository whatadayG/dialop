
import sys
print(sys.executable)
sys.path.append('..')
import pickle
from pathlib import Path
from rich import print
from rich.console import Console
import pdb
import json
import random
import copy
import time
import re
from ruamel.yaml import YAML 
console = Console()

# Import your modules
from dialop.evaluate import CheckpointManager
from dialop.responses_class import ParallelConversations, ResponseManager, Conversation
from dialop.envs.planning import PlanningEnv
from skills import Agent_tell
from dialop.games.planning import PlanningGame
print("Imports successful!")


def style_judger(parallel_convs):
    with open ("/Users/georgiazhou/research_machine/dialop/dialop/RL/explanation_per_persona.txt", "r") as f:
        explanation_per_persona = json.load(f)
    user_base_prompt = parallel_convs.c_players['user'].user_prompt_obss
    first_you = user_base_prompt.find("You:")
    second_you = user_base_prompt.find("You:", first_you + 1)
    user_base_prompt = user_base_prompt[second_you:]
    prompt = "You are a communication expert. Please first judge whether the following messages on conform to one of the styles listed. You should only confirm if you are very confident. If you don't think the message conforms to any of the styles, output: {}. If you find one or two that match, output the style(s) that you think it matches as {style : explanation, ...}. Here is conversation history: " + "\n" + user_base_prompt + "Here are the styles you should judge against: " + "\n" + str(explanation_per_persona) + "Remember you should only output the style(s) that you are very confident about. If you are not very confident about any of the options, output: {}.  "
    response = Agent_tell(prompt)
    
    return response

def clean_user_conversation(base_user_prompt_obss,  parallel_convs):
    user_conversations_no_pref = []
    
    for i in range(3):
        user_conve_record = base_user_prompt_obss + parallel_convs.players_dict[i]['user'].user_prompt_obss
        first_you = user_conve_record.find("You:")
        second_you = user_conve_record.find("You:", first_you + 1)

    # Get everything from second "You:" onwards
        user_conversations_no_pref.append(user_conve_record[second_you:])
    return user_conversations_no_pref



def n_different_responses(players, strategy_response_list, t):
    
    
    resp1 = players['agent'].respond(t, 35, vary=True, strategy = strategy_response_list[0])
    resp2 = players['agent'].respond(t, 35, vary=True, strategy = strategy_response_list[1])
    resp3 = players['agent'].respond(t, 35, vary=True, strategy = strategy_response_list[2])
    return [resp1, resp2, resp3]


def initialize_parallel_conversations(agent_3_responses, strategy_response_list, players, env, t):
    # Create parallel conversation manager
    parallel_convs = ParallelConversations(
        num_streams=3,
        players=players,
        env_ctor=lambda: PlanningEnv(),
        env=env,
        max_length=35
    )
    
    # Generate initial responses
    
    # Initialize parallel streams
    parallel_convs.initialize_streams(agent_3_responses, strategy_response_list, t)
    
    print("\nInitialized parallel conversations:")
    for i, conv in enumerate(parallel_convs.conversations):
        state = conv.get_current_state()
        print(f"\nStream {i+1}:")
        print(f"Initial response: {agent_3_responses[i][:100]}...")
        print(f"Features extracted: {state.features}")
        
    return parallel_convs

# Initialize parallel conversations


def get_conversation_history(game):
    # Full action log (includes messages, proposals, and responses)
    full_history = game.action_log  # List of dicts with type, message, player, time
    
    # Messages only
    messages_only = game.message_history  # List of message strings
    
    # Format by player
    all_conversation = ''
    for action in game.action_log:
        if action["type"] == "message":
            speaker = "agent" if action["player"] == 1 else "user"
            conversation_str = f"{speaker}: {action['message']}\n"
            all_conversation += conversation_str
    
    return all_conversation

def strategy_sampling(env):
    with open("/Users/georgiazhou/research_machine/dialop/dialop/envs/data/planning_strategy_sampling.txt", "r") as f:
        strategy_prompt = f.read()
    
    chat_history = get_conversation_history(env.game)

    whole_prompt = strategy_prompt + "\n" + chat_history
    
    
    strategy_response = Agent_tell(whole_prompt)
    strategy_response = re.sub(r'\n\s*\n', '\n', strategy_response)
    strategy_response_list = strategy_response.split('\n')

    
    return strategy_response_list


def step_parallel_conversations(parallel_convs):
    """Step conversations until each hits a user turn"""
    
    # Track which conversations are waiting for user input
    waiting_for_user = [False] * len(parallel_convs.conversations)
    
    
    while not all(waiting_for_user):
        # Get responses for conversations where it's agent's turn
    
        active_convs = {0:'', 1:'', 2:''}  # Track which conversations get stepped
        
        for i, conv in enumerate(parallel_convs.conversations):
            if waiting_for_user[i]:
                active_convs[i] = None  # Placeholder to maintain indexing
                continue
                
            state = conv.get_current_state()
            if state.obss["turn_player"] == "agent":
                response = conv.players["agent"].respond(vary=True)
                active_convs[i] = response
            
            else:
                # Hit user turn - mark this conversation as waiting
                waiting_for_user[i] = True
                active_convs[i] = None
                print(f"----Conversation {i} is waiting for user input---")
        
        # Step only the active conversations forward
        if active_convs:

            print("\nStepping conversations:", active_convs)
            for i, response in active_convs.items():
                if response is not None:
                    print(f"\nConversation {i} response:\n{response[:100]}...")
                    conv = parallel_convs.conversations[i]
                    players = parallel_convs.players_dict[i]
                    print(response)
                    
                    conv.step(str(response), conv.get_current_state().turn + 1)
                    [player.observe(conv.state.obss[pname]) for pname, player in players.items()]
                
        
    
    print("\nAll conversations are now waiting for user input")
    # Print current state of each conversation
    for i, conv in enumerate(parallel_convs.conversations):
        state = conv.get_current_state()
        print(f"\nConversation {i} (Turn {state.turn}):")
        print(f"Last observation: {state.obss.get('user', '')[:100]}...")
    
    return parallel_convs

# Use it like this:




def sample_preferences(known_preferences):
    potential_prefs_list = json.loads(open("/Users/georgiazhou/research_machine/dialop/dialop/RL/feature_dict.txt").read())
    # it's a dictionary, so we need to get the keys; known should also be a dictionary
    # make sure known preferences are keys or dicts/
    available_prefs = list(set(potential_prefs_list.keys()) - set(known_preferences))
    pref_list = random.sample(available_prefs, (10-len(known_preferences)))
    pref_dict = {}
    for pref in pref_list:
        if potential_prefs_list[pref] == 'binary':
            pref_dict[pref] = random.choice(['like', 'dislike'])
        else:
            pref_dict[pref] = 'like'
        
    return pref_dict
def get_column_letter(n):
    """Convert number to Excel column letters (A->1, B->2, AA->27, etc)"""
    if n == 0:
        return 'A'
    while n > 0:
        n, remainder = divmod(n, 26)
        result = chr(65 + remainder) 
    print(result)
    return result

global_preferences_only = []

def n_1_user_personas(formatted_sampled_preferences, extracted_features, record = None, style_response = None):
    """
    Create 10 user personas for each conversation; no longer use sample_preferences
    """

    global global_preferences_only
    if re.search('Communication Style', formatted_sampled_preferences):
        formatted_sampled_preferences = formatted_sampled_preferences[:formatted_sampled_preferences.find('Communication')]
    # currently there's 1 user per agent reply. 
    #for j,record in enumerate(user_conversations_no_pref):
    #prompts = []
    #preferences = []
    #for i in range(10):
    # If formatted_sampled_preferences is a list of strings, join them with newlines
    if isinstance(extracted_features, dict):
        formatted_features = []
        for key, value in extracted_features.items():
            if value[1] == 'Unknown':
                importance = random.choice(['High', 'Medium', 'Low'])
                formatted_features.append(f"{key}, {value[0]} : importance {importance}")
            else:
                formatted_features.append(f"{key}, {value[0]} : importance {value[1]}")
        extracted_features = '\n'.join(formatted_features)
    
    if style_response:
        preference = 'Here are your preferences: ' + str(formatted_sampled_preferences) + str(extracted_features)  + "\n" + "Here are your communication styles: " + str(style_response)
    else:
        preference = 'Here are your preferences: ' +  str(formatted_sampled_preferences) + str(extracted_features) 
        
                
    if record:
        features_prompt = preference + '\n' + record
    else:
        features_prompt = preference
    
    global_preferences_only.append(str(formatted_sampled_preferences) + str(extracted_features))
    #prompts.append(features_prompt)
    #preferences.append(preference)

    #worksheet.update(f'{get_column_letter(10*j)}{5}:{get_column_letter(10*j + 9)}{5}', [preferences])

        
    return features_prompt


def create_10_new_user_per_conv(half_prompts, parallel_convs):
    players = {}
    for i, half_per_conv in enumerate(half_prompts):
        players[i] = parallel_convs.create_pair_new_players(parallel_convs.players_dict[i], multiple_user_personas_incomplete_prompt=half_per_conv)
    return players
    


#print(agent_user_10_pairs_dict[0]['user_0'].prompt, agent_user_10_pairs_dict[0]['user_1'].prompt)


# In[65]:


def format_user_game_state(base_env, incomplete_action_log = False):
    """create custom prefereces and score function (see game class)"""
    if incomplete_action_log:
        action_log = []
    else:
        action_log = base_env.game.action_log
    game_state = {
                "action_log": action_log,
                "events": base_env.game.events,
                "preferences": base_env.game.prefs,  # Must be in the format [(desc, weight, cls_name, cls_args), ...]
                "persona_styles": base_env.game.persona_styles
                }
    return game_state



def initialize_10_user_conversations(parallel_convs, user_conversations_no_pref, style_response, extracted_features, reset_sonlist = False):
    if reset_sonlist:
        for conversation in parallel_convs.conversations:
            conversation.son_convs = []
    # we want 10 user personas per conversation; they should be the 10 sam ones but copies 
    n_10_versions = {}
    for i in range(10):
        # any conversation is fine since action log is not counted
        env_copy = copy.deepcopy(parallel_convs.conversations[0].env)
        game_state = format_user_game_state(env_copy, incomplete_action_log = True)
        #parsed_features = parse_extracted_features(extracted_features)
        obss = env_copy.reset(game_state, known_user_preferences_num=(10-len(extracted_features.keys())), extracted_features = extracted_features)
        bundle = {'env': env_copy, 'obss': obss}
        n_10_versions[i] = bundle
    
    # user and agent are using different games; user needs to be new, agent is not 
    for conv_idx, conversation in enumerate(parallel_convs.conversations):
        base_players = parallel_convs.players_dict[conv_idx]
        base_chat_history = user_conversations_no_pref[conv_idx]
        for a_bundle in range(len(n_10_versions)):
            env_copy = copy.deepcopy(n_10_versions[a_bundle]['env'])
            env_copy.game.action_log = copy.deepcopy(conversation.env.game.action_log)
            obss = copy.deepcopy(n_10_versions[a_bundle]['obss'])
            #print(obss['user'])
            #extracted_features = parse_extracted_features(extracted_features)
            n_1_prompts = n_1_user_personas(obss['user'], extracted_features, record = base_chat_history, style_response = style_response)
            players = parallel_convs.create_pair_new_players(base_players, user_personas_incomplete_prompt=n_1_prompts)
            new_conv = Conversation(players=players, env=env_copy, max_length=parallel_convs.max_length, parent_conv=conversation)
            conversation.son_convs.append(new_conv)
            #new_conv = Conversation(players=pairs_3_10_2_pairs_dict[conv_idx][f'pair_{i}'], env=env_copy, max_length=parallel_convs.max_length, parent_conv=conversation)
            #conversation.son_convs.append(new_conv)
    global global_preferences_only
    return parallel_convs, global_preferences_only




def step_all_user_responses( parallel_convs):
    """
    For each of the 3 conversation paths:
        - Get all 10 users to respond to their agent
        - Step the environment with each response
    """
    
    all_states = []
    all_envs = []
    
    # For each conversation path (0,1,2)
    for conversation in parallel_convs.conversations:
        if conversation.propose_made:
            continue
        envs = []
        conv_states = []
   
        current_state = conversation.get_current_state()
        # this get the obss of 3 main branches 
        
        
        
        
        # Get responses from all 10 users for this conversation
        for son_conv in conversation.son_convs:
            user = son_conv.players['user']
            
            
            ##state = conversation.reverse_state()
            ##print(state.obss)
            ##break
            
            
            
            # Have user observe current state
            #user.observe(current_state.obss['user'])
            
            # Get user's response
            while True:
                user_response = user.respond(t = current_state.turn, max_len = parallel_convs.max_length, vary = True)
                new_state = son_conv.step(user_response, current_state.turn + 1)
                if new_state.obss["turn_player"] == 'agent':
                    break
                user.observe(new_state.obss['user'])
            
            # Step environment with user response
                
            print(son_conv.env.game.turn_player)
            conv_states.append(new_state)
            
            # Have agent observe the new state
        
        all_states.append(conv_states)
    
    return all_states

# Use the function




def run_until_agent_proposals(parallel_convs):
    """
    For each conversation path:
        - Allow agent to tool call or think multiple times
        - Continue until agent makes a proposal
        - Reject regular messages - only allow tool calls, thinks, or proposals
    """
    all_proposals = []
    
    # For each conversation path (0,1,2)
    for conv_idx, conversation in enumerate(parallel_convs.conversations):
        if conversation.propose_made:
            continue
        path_proposals = []
        
        # For each son conversation
        for son_conv in conversation.son_convs:
            agent = son_conv.players['agent']
            agent.observe(son_conv.state.obss['agent'])
            
            # Keep getting responses until we get a proposal
            must_propose = False
            while True:
                response = agent.respond(t=son_conv.state.turn + 1, vary=True, propose=must_propose)
                if '[tool]' in response or '[think]' in response:
                    # Valid tool/think response
                    state = son_conv.step(response, son_conv.state.turn + 1)
                    agent.observe(state.obss['agent'])
                    continue
                if '[message]' in response:
                    must_propose = True
                    continue
                state = son_conv.step(response, son_conv.state.turn + 1, propose = must_propose)
                # Check response type
                if (must_propose or ('[propose]' in state.obss['agent'])) and 'Error' not in state.obss['agent']:
                    # Got a proposal - save it and move on
                    #son_conv.step(response, son_conv.state.turn + 1, propose=True)
                    path_proposals.append(response)
                    break
                agent.observe(state.obss['agent'])

                
            print(f"Final proposal: {response}")
            
        all_proposals.append(path_proposals)
    
    return all_proposals   
        

def extract_total_score(obss_string):
    # Find the line with "TOTAL SCORE:"
    for line in obss_string.split('\n'):
        if "TOTAL SCORE:" in line:
            # Extract the number after the equals sign
            score = int(line.split('=')[-1])
            return score
    return None


def get_tool_calls(conversation):
    tool_calls = []
    for state in conversation.history:
        if state.obss['agent'].startswith('[tool]'):
            tool_calls.append(state.obss['agent'])
    tool_calls = '\n'.join(tool_calls)
    return tool_calls


def api_score_extraction(response):
    try:
        score = int(response.split('TOTAL SCORE:')[-1].strip())
    except:
        # Extract just the number after TOTAL SCORE:
        score = []
        letters = list(response.split('TOTAL SCORE:')[-1].strip())
        for letter in letters:
            if letter.isdigit():
               score.append(letter)
        score = int(''.join(score))
    return score

def extract_proposal(conversation):
    for state in conversation.history:
        if '[propose]' in state.obss['agent'] and 'Error' not in state.obss['agent']:
            proposal = state.obss['agent']
    return proposal




def api_rate_agent_proposals(agent_proposals, parallel_convs):
    # here can use the global_preferences_only
    all_prompts = []
    USER_EVAL_PROMPT = """You are a user evaluating an AI travel agent's itinerary proposals. You have specific preferences, and you should evaluate how well the agent's proposals match these preferences.

Your preferences are:
{preferences}

For each proposal, you will see:
1. The proposed itinerary
2. The agent's search history showing attributes of each location

The proposal is:
{proposal}

The search history is:
{tool_calls}

Please evaluate each proposal by:
1. Reading the proposal and search history carefully
2. Checking if the locations match your stated preferences
3. output a score between 0 and 100 in the format of 'TOTAL SCORE: 100'"""
    global global_preferences_only
    all_tool_calls = []
    count = 0
    all_scores = []
    for i, conversation in enumerate(parallel_convs.conversations):
        tool_calls = get_tool_calls(conversation)
        all_tool_calls.append(tool_calls)
        if conversation.propose_made:
            proposal = extract_proposal(conversation)
        else:
            proposal_list = agent_proposals[i]
        prompts = []
        scores = []
        i = 0
        while i < 10:
            if not conversation.propose_made:
                proposal = proposal_list[i]
            preferences = global_preferences_only[count%10]
            tailored_prompt = USER_EVAL_PROMPT.format(preferences = preferences, proposal = proposal, tool_calls = tool_calls)
            response = Agent_tell(tailored_prompt)
            try:
                score = api_score_extraction(response)
            except:
                score = -1
            #print(score)
            if 0 <= score <= 100:
                count += 1
                scores.append(score)
                prompts.append(tailored_prompt)
                i += 1
        all_scores.append(scores)
        all_prompts.append(prompts)
    return all_scores, all_prompts
            
 
def choose_best_path(scores):
    best_score = 0
    best_path = None
    for i, score_list in enumerate(scores):
        path_average = sum(score_list) / len(score_list) if score_list else 0
        if path_average > best_score:
            best_score = path_average
            best_path = i
    return best_path




# In[93]:


def save_best_path_as_checkpoint(parallel_convs, best_path, checkpoint_manager):
    """Save the best conversation path as a checkpoint"""
    best_conv = parallel_convs.conversations[best_path]
    
    
    # Create state dict from best conversation
    state = {
        'players': best_conv.players,
        'history': best_conv.history,
        'env_state': best_conv.env.game.get_game_info(override_events=True),
        'turn': best_conv.history[-1].turn if best_conv.history else 0
    }
    
    # Save as checkpoint 
    checkpoint_name = f"best_path_{int(time.time())}"
    checkpoint_manager.save_state(state, checkpoint_name)
    return checkpoint_name





# In[ ]:


def wait_for_initial_checkpoint(checkpoint_mgr):
    """Wait for initial checkpoint from evaluate.py"""
    print("Waiting for initial checkpoint from evaluate.py...")
    while True:
        # Look for initial_state checkpoint
        if (checkpoint_mgr.checkpoint_dir / "initial_state.pkl").exists():
            print("Found initial checkpoint!")
            # Load the initial state
            state = checkpoint_mgr.load_full_state("initial_state")
            return state
        time.sleep(3)  # Check every second




