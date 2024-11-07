from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from players import LLMPlayer
import copy
import json
import random


@dataclass
class ConversationState:
    """Represents the state of a single conversation"""
    obss: Dict
    features: List[str]
    turn: int
    responses: List[str] = None
    player_states: Dict = None

class Conversation:
    def __init__(self, players, env, max_length: int, parent_conv: Optional['Conversation'] = None):
        self.players = players
        self.env = env
        self.max_length = max_length
        self.state = None
        self.history = []
        self.parent_conv = parent_conv
        
    def step(self, response: str, t: int) -> ConversationState:
        """Take a step in the conversation"""
        obss, resample, features, ready = self.env.step(response, t >= (self.max_length-5))
        
        #print(obss)
        
        self.state = ConversationState(
            obss=obss,
            features=features,
            turn=t
        )
        self.history.append(self.state)
        return self.state
    
    def reverse_state(self) -> ConversationState:
        """Revert to previous state and all related observations

        Returns:
            ConversationState: Previous state, or None if no history
        """
        if len(self.history) > 1:
            # Remove current state
            current_state = self.history.pop()
            # Set previous state as current
            previous_state = self.history[-1]
            self.state = ConversationState(
            obss=previous_state.obss.copy(),  # Make a copy of previous observations
            features=previous_state.features,
            turn=previous_state.turn,
            responses=previous_state.responses,
                player_states=previous_state.player_states
            )

            # Revert environment state
            self.env.num_msgs -= 1
            if hasattr(self.env, 'game'):
                self.env.game.action_log.pop()
                self.env.game.message_history.pop()

            # Revert player observations
            for player_name, player in self.players.items():
                if self.state.obss.get(player_name):
                    if player.role == "user":
                        # Remove the last observation from user_prompt_obss
                        last_obs_start = player.user_prompt_obss.rfind(player.prefix)
                        last_obs = player.prompt.rfind(player.prefix)
                        if last_obs_start != -1:
                            player.user_prompt_obss = player.user_prompt_obss[:last_obs_start]
                            player.prompt = player.prompt[:last_obs]

                    if hasattr(player, 'temp_prompt') and player.temp_prompt:
                        # Revert temporal prompts if they exist
                        if player.count_agent_path > 0:
                            player.count_agent_path -= 1
                            if player.count_agent_path in player.temp_prompt:
                                # Remove last observation from temporal prompt
                                last_special_prompt = player.temp_prompt[player.count_agent_path]
                                if '\n--special prompt--\n' in last_special_prompt:
                                    base, obs = last_special_prompt.split('\n--special prompt--\n')
                                    player.temp_prompt[player.count_agent_path] = base

                    # Revert main prompt to previous state
                    # Find and remove the last observation
                    last_obs = current_state.obss[player_name]
                    if last_obs in player.prompt:
                        player.prompt = player.prompt.replace(last_obs, '', 1)

                    # Observe the previous state
                    player.observe(previous_state.obss[player_name], ignore_obs=False)

            return self.state
        return None

    
    def get_current_state(self) -> ConversationState:
        return self.state

class ParallelConversations:
    """Manages multiple parallel conversation streams"""
    def __init__(self, *, num_streams: int, players, env_ctor, env, max_length: int):
        self.num_streams = num_streams
        self.conversations = []
        self.c_players = players
        self.env_ctor = env_ctor
        self.env = env
        self.max_length = max_length
    
    
    def create_pair_new_players(self, current_players, multiple_user_personas_incomplete: Optional[List[str]] = None):
        agent = current_players['agent']
        user = current_players['user']
        if not multiple_user_personas_incomplete:
            players = {"agent": LLMPlayer(agent.prompt, agent.role, agent.console,
                                     optional=agent.optional,
                                     model_kwargs={"temperature": 0.8}),
                       "user":  LLMPlayer(user.prompt, user.role, user.console,
                                    optional=user.optional,
                                    model_kwargs={"temperature": 0.8})}
        else: 
            with open('/Users/georgiazhou/research_machine/dialop/dialop/prompts/planning_user.txt', 'r') as f:
                base_prompt = f.read()
            players = {}
            for i, halfprompt in enumerate(multiple_user_personas_incomplete):
                prompt = base_prompt + halfprompt
                players[f'pair_{i}'] = {"agent": LLMPlayer(agent.prompt, agent.role, agent.console,
                                     optional=agent.optional,
                                     model_kwargs={"temperature": 0.8}),
                                     "user" : LLMPlayer(prompt, user.role, user.console,
                                    optional=user.optional,
                                    model_kwargs={"temperature": 0.8})}
        return players
    def initialize_streams(self, initial_responses: List[str], t: int): # 3 different agent responses 
        """Initialize parallel conversations with different responses"""
        self.conversations = []
        #self.code_names = {}
        self.envs = {}
        self.players_dict = {}
        for i, resp in enumerate(initial_responses):
            players = self.create_pair_new_players(self.c_players)
            
            self.players_dict[i] = players
            env_copy = copy.deepcopy(self.env)
            self.envs[i] = env_copy
            conv = Conversation(players, env_copy, self.max_length)
            conv.step(resp, t)
            [player.observe(conv.state.obss[pname]) for pname, player in players.items()]
            print(conv.state.obss)
            self.conversations.append(conv)
    
    def step_all(self, responses: List[str], t: int) -> List[ConversationState]:
        """Step all conversations forward"""
        states = []
        for conv, resp in zip(self.conversations, responses):
            state = conv.step(resp, t)
            states.append(state)
        return states
    
    def get_all_states(self) -> List[ConversationState]:
        """Get current state of all conversations"""
        return [conv.get_current_state() for conv in self.conversations]

class ResponseManager:
    """Manages response generation and evaluation"""
    def __init__(self, players):
        self.players = players
        
    def generate_responses(self, t: int, max_length: int, n: int = 3) -> List[str]:
        """Generate n different responses"""
        responses = []
        for _ in range(n):
            resp = self.players['agent'].respond(t, max_length, vary=True)
            responses.append(resp)
        return responses
    
    def evaluate_responses(self, states: List[ConversationState]) -> Dict:
        """Evaluate different response streams"""
        evaluations = {}
        for i, state in enumerate(states):
            evaluations[i] = {
                'features': state.features,
                'observation': state.obss
            }
        return evaluations