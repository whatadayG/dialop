from copy import deepcopy
import json
import os
import pathlib
import time
from typing import Optional, Literal
from itertools import cycle
import pdb
import random
import pickle
from pathlib import Path

from openai import APIConnectionError, RateLimitError
import numpy as np
import tyro
from ruamel.yaml import YAML
from rich import print
from rich.console import Console
console = Console()

from envs import (
    PlanningEnv,
    OptimizationEnv,
    MediationEnv,
    WordLimit,
    ForceProposal,
    AsymmetricForceProposal
)
from players import (
    LLMPlayer,
    HumanPlayer,
    DryRunPlayer,
    OutOfContextError
)
from utils import Logger, retry, count_words

FPATH = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
RESDIR = pathlib.Path("/Users/georgiazhou/research_machine/dialop/dialop/results")
#RESDIR = pathlib.Path("/Users/user/Projects/collaborative-dialogue/results/")
DATADIR = pathlib.Path("/Users/georgiazhou/research_machine/dialop/dialop/data")

GAME_CLSS = {
    "matching": OptimizationEnv,
    "planning": PlanningEnv,
    "mediation": MediationEnv,
}

class ResampleError(Exception):
    pass

class CheckpointManager:
    def __init__(self, checkpoint_dir='/Users/georgiazhou/research_machine/dialop/dialop/checkpoints/debug_states/'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_state(self, state, name):
        """Save state while handling unpickleable objects"""
        clean_state = {}
        
        # Only save the essential player data
        if 'players' in state:
            clean_state['players'] = {}
        for player_name, player in state['players'].items():
            if not hasattr(player, 'prompt'):
                pdb.set_trace()
            clean_state['players'][player_name] = {
                'prompt': player.prompt,
                'role': player.role,
                'user_prompt_obss': getattr(player, 'user_prompt_obss', ''),
                'temp_prompt': getattr(player, 'temp_prompt', {})
            }
        
    
        # Copy other state attributes
        for key, value in state.items():
            if key != 'players':
                clean_state[key] = value
            
        with open(self.checkpoint_dir / f"{name}.pkl", 'wb') as f:
            pickle.dump(clean_state, f)
        
        print(f"DEBUG: Successfully saved checkpoint to {self.checkpoint_dir.absolute()}")
    
    def load_state(self, name):
        """Load state and verify player objects"""
        path = self.checkpoint_dir / f"{name}.pkl"
        with open(path, 'rb') as f:
            state = pickle.load(f)
            
        # Verify player objects are accessible
        if 'players' in state:
            players = state['players']
            # Verify key player attributes are accessible
            for player_name, player in players.items():
                assert hasattr(player, 'prompt'), f"Player {player_name} missing prompt"
                assert hasattr(player, 'role'), f"Player {player_name} missing role"
                if player.role == 'user':
                    assert hasattr(player, 'user_prompt_obss'), f"User player missing prompt observations"
        
        return state
    def load_full_state(self, name):
        """Load state and recreate player objects with full functionality"""
        path = self.checkpoint_dir / f"{name}.pkl"
        with open(path, 'rb') as f:
            state = pickle.load(f)
    
        # Recreate player objects with full functionality
        if 'players' in state:
            restored_players = {}
            for player_name, player_data in state['players'].items():
                # Create new player instance based on role
                if player_data['role'] in ['user', 'agent', 'user0', 'user1']:
                    player = LLMPlayer(
                    prompt=player_data['prompt'],
                    role=player_data['role'],
                    console=Console(),  # Create new console instance
                )
                    # Restore additional attributes
                    player.prompt = player_data['prompt']
                    player.user_prompt_obss = player_data.get('user_prompt_obss', '')
                    player.temp_prompt = player_data.get('temp_prompt', {})
                else:
                    # Handle other player types (HumanPlayer, DryRunPlayer) if needed
                    player = DryRunPlayer(
                        prompt=player_data['prompt'],
                        role=player_data['role'],
                        console=Console()
                    )
                restored_players[player_name] = player
        
            state['players'] = restored_players
    
        return state



def selfplay(
    game_cls,
    games,
    samples_per_game,
    resume,
    end
):
    #pdb.set_trace()
    for game_idx, game in enumerate(games[resume:end]):
        #pdb.set_trace()
#        data = game["games"][0]
        original_log = game["action_log"]
        data = deepcopy(game)
        # Clear action log so env doesn't initialize with a message history
        data["action_log"] = []
        if game_cls == OptimizationEnv:
            score = data["proposal_reward"]
            score_norm = data["result"]["norm"]
        else:
#            score = data["action_log"][-3]["scores"]["total"]
            score = data["result"]["score"]
            score_norm  = data["result"]["norm"]
        metadata = {
            "hh_turns": len(original_log),
            "hh_words": count_words(original_log),
            "hh_score": score,
            "hh_score_norm": score_norm,
        }
        for sidx in range(samples_per_game):
            name = f"{game_idx + resume}_{sidx}"
            yield data, name, metadata

def prompted_selfplay(
    game_cls,
    games,
    samples_per_game,
    resume,
    end,
):
    for game_idx, game in enumerate(games[resume:end]):
        if game_cls == OptimizationEnv:
            data = deepcopy(game)
            original_log = game["action_log"]
        elif game_cls == PlanningEnv:
            data = deepcopy(game)
            original_log = game["action_log"]
        elif game_cls == MediationEnv:
            data = deepcopy(game)
            original_log = game["action_log"]

        if game_cls == PlanningEnv:
            try:
                score = data["action_log"][-3]["scores"]["total"]
            except:
                for turn in range(0, len(data["action_log"])):
                    if data["action_log"][turn]["type"] == "proposal":
                        score = data["action_log"][turn]["scores"]["total"]
        elif game_cls == OptimizationEnv:
            score = data["proposal_reward"]
        elif game_cls == MediationEnv:
            score = data["result"]["score"]

        total_word_count = count_words(original_log)
        prefix_word_counts = []
        for turn in range(0, len(data["action_log"])):
            num_words = count_words(original_log[:turn])
            prefix_word_counts.append(num_words)
        # Get turns closest to 25%, 50%, 75% of the way through the game:
        turn_idxs = []
        # for pct in [0.25, 0.5, 0.75]:
        for pct in [0.5, 0.75]:
            turn_idxs.append(
                np.argmin(np.abs(np.array(prefix_word_counts) - pct * total_word_count))
            )
        # Get index of final proposal:
        proposal_idx = None
        for turn in range(0, len(data["action_log"])):
            if data["action_log"][turn]["type"] == "proposal":
                proposal_idx = turn
        if proposal_idx is None:
            raise ValueError("Game doesn't include a proposal")
        turn_idxs.append(proposal_idx)

        #pdb.set_trace()
        # for turn in range(0, len(data["action_log"])):
        names = ["50", "75", "end"]
        # for turn in range(2, len(data["action_log"]) // 2):
        #     end = 2 * (turn + 1)
        for name, end in zip(names, turn_idxs):
            # if end >= len(game["games"][0]["action_log"]): continue
            data["action_log"] = original_log[:end]
            metadata = {
                "initialized_turns": len(data["action_log"]),
                "initialized_words": count_words(data["action_log"]),
                "hh_turns": len(original_log),
                "hh_words": count_words(original_log),
                "hh_score": score,
                "hh_score_norm": data["result"]["norm"],
            }
            for sidx in range(samples_per_game):
                name = f"{game_idx + resume}_start{len(data['action_log'])}_{name}_{sidx}"
                yield data, name, metadata



@retry(allowed_exceptions=[OutOfContextError, RateLimitError, ResampleError])
def run(
    game_cls,
    data,
    metadata,
    player_ctor,
    env_ctor,
    logfile,
    use_word_limit=False,
    max_length=3
):
    # Create players.
    #pdb.set_trace()
    players = player_ctor()
    # Create env.
    env = env_ctor()
    extracted_features = []
    # TODO: make api the same
    if use_word_limit:
        obss = env.reset(word_limit=metadata["hh_words"],
                         game_state=data)
    else:
        #pdb.set_trace()
        obss = env.reset(game_state=data)

    # Log initial info.
    log = Logger(logfile)
    for pname, player in players.items():
        log.write(
            f"{pname} params",
            json.dumps(getattr(player, 'model_kwargs', {}), indent=2))
        log.write(f"{pname} prompt", player.prompt)
    if game_cls == PlanningEnv:
        if env.query_executor == "gpt4":
            log.write(f"Query Executor Prompt", env.search.prompt)
        else:
            log.write("Using deterministic query executor.")

    # Env loop.
    t = 0
    player_cycle = cycle(players.keys())
    if game_cls == MediationEnv:
        while not obss["done"] and t < max_length:
            console.rule("environment obs")
            console.print(obss)
            [player.observe(obss[pname]) for pname, player in players.items()]
            for pname in players:
                log.log(key=pname, value=obss[pname], title=f"obs t={t}")

            # Cycle through players, letting them speak if it's their turn
            next_player = next(player_cycle)
            while next_player not in obss["turn_players"]:
                next_player = next(player_cycle)
            resample = True
            resample_cnt = 0
            while resample and resample_cnt < 3:
                if resample_cnt >= 1:
                    console.print("INVALID: resampling...", style="bold red")
                stepped = False
                while not stepped:
                    try:
                        resp = players[next_player].respond()
                        stepped = True
                    except RateLimitError:
                        print("Rate limited. Sleeping...")
                        time.sleep(30)
                log.log(
                    key=next_player,
                    value=resp,
                    title=f"generate t={t} try={resample_cnt}"
                )
                stepped = False
                while not stepped:
                    try:
                        obss, resample = env.step(resp, next_player)
                        stepped = True
                    except RateLimitError:
                        print("Rate limited during environment step. Sleeping...")
                        time.sleep(30)
                resample_cnt += 1
            t += 1
    else:
        agent_ready = False
        user_ready = False
        user_response_counter = 0
        cap = 2
        propose_ready = False
        agent_pending = False
        not_ready = []
        checkpoint_mgr = CheckpointManager()
        def use_checkpoint_state(checkpoint_name: str):
            """Load and use checkpoint state"""
            checkpoint_mgr = CheckpointManager()
            state = checkpoint_mgr.load_state(checkpoint_name)
            
            # Access player objects
            players = state['players']
            agent = players['agent']
            user = players['user']
            
            # Access important attributes
            print(f"Turn: {state['t']}")
            print(f"Extracted features: {state['extracted_features']}")
            
            # Access player-specific data
            print(f"Agent prompt: {agent.prompt}")
            print(f"User conversation history: {user.user_prompt_obss}")
    
            # Generate responses using loaded state
            responses = n_10_user_personas(
                players=players,
                extracted_features=state['extracted_features']
            )
    
            return responses
        
        def modeling_user_prompt(obss):
            nonlocal extracted_features
            known_preferences = str(extracted_features)
            modeling_prompt = open("/Users/georgiazhou/research_machine/dialop/dialop/RL/benchmark_features.txt").read()
            obss['agent'] += '\n' "Now you need to self evaluate your proposal as a good planning agent. Here are the known preferences: " + known_preferences + modeling_prompt
        def sample_preferences(known_preferences):
            all_prefs_list = json.loads(open("/Users/georgiazhou/research_machine/dialop/dialop/RL/feature_dict.txt").read())
            available_prefs = list(set(all_prefs_list) - set(known_preferences))
            pref_list = random.sample(available_prefs, (10-len(known_preferences)))
            return pref_list
        def n_different_proposals(players):
            resp1 = players['agent'].respond(t, max_length, vary=True, propose = True)
            resp2 = players['agent'].respond(t, max_length, vary=True, propose = True)
            resp3 = players['agent'].respond(t, max_length, vary=True, propose = True)
            return resp1, resp2, resp3
        def n_10_user_personas(players, extracted_features):
            pdb.set_trace()
            real_user = players['user']
            agent_3 = []
            chat_history_user = real_user.user_prompt_obss
            if players['agent'].temp_prompt:
                chat_history_agent = players['agent'].temp_prompt
            else:
                chat_history_agent = players['agent'].prompt
            for p in chat_history_agent.values():
                resp = p.split("--special prompt--")[1]
                agent_3.append(resp)
            user_personas = {}
            responses_per_persona = {}
            features_prompt = 'Here are your preferences: ' + str(extracted_features)
            for resp in agent_3:
                agent_resp = 'agent: ' + str(resp) + "\n"
                for i in range(10):
                    user_personas[i] = player_ctor()
                    total_feature_prompt = features_prompt + sample_preferences(extracted_features)
                    user_personas[i].prompt += total_feature_prompt
                    user_personas[i].prompt += chat_history_user 
                    user_personas[i].prompt += agent_resp
                    responses_per_persona[i] = user_personas[i].respond(vary = True)
                    print(responses_per_persona[i])
                pdb.set_trace()
       
              
            
            
            return user_personas 
        def n_different_responses(players):
            resp1 = players['agent'].respond(t, max_length, vary=True)
            resp2 = players['agent'].respond(t, max_length, vary=True)
            resp3 = players['agent'].respond(t, max_length, vary=True)
            return [resp1, resp2, resp3]
        def user_persona_respond(players1, players2, players3, extracted_features):
            extracted_features = str(extracted_features)
            resp_dic1 = players1['user'].respond_ready_user_planning(extracted_features)
            resp_dic2 = players2['user'].respond_ready_user_planning(extracted_features)
            resp_dic3 = players3['user'].respond_ready_user_planning(extracted_features)
        

            
        while not obss["done"] and t < max_length:
            #pdb.set_trace()
            console.rule("environment obs")
            ##if propose_ready:
            ##    
            ##    modeling_user_prompt(obss1)
            ##    modeling_user_prompt(obss2)
            ##    modeling_user_prompt(obss3)
            ##    players['agent'].observe(obss1['agent'],temporal = 0)
            ##    estimation1 = players['agent'].respond(vary=True, temporal = True)
            ##    players['agent'].observe(obss2['agent'], temporal = 0)
            ##    estimation2 = players['agent'].respond(vary=True, temporal = True)
            ##    players['agent'].observe(obss3['agent'], temporal = 0)
            ##    estimation3 = players['agent'].respond(vary=True, temporal = True)
            ##    pdb.set_trace()
            ##    
            ##elif user_ready:
            ##    players2 = player_ctor(determined = True)
            ##    players3 = player_ctor(determined = True)
            ##    [[player.observe(obss1[pname]) for pname, player in players.items()]]
            ##    [[player2.observe(obss2[pname]) for pname, player2 in players2.items()]]
            ##    [[player3.observe(obss3[pname]) for pname, player3 in players3.items()]]
            
            
            if agent_pending or user_ready:
                ignore_obs = True
                
            else:
                ignore_obs = False
            console.print(obss)
            [player.observe(obss[pname], ignore_obs = ignore_obs) for pname, player in players.items()]
            for pname in players:
                log.log(key=pname, value=obss[pname], title=f"obs t={t}")
            resample = True
            resample_cnt = 0
            while resample and resample_cnt < 3:
                if resample_cnt >= 1:
                    console.print("INVALID: resampling...", style="bold red")
                stepped = False
                while not stepped:
                    #pdb.set_trace()
                    try:
                        
                        if user_response_counter >= cap and obss["turn_player"] == 'agent' and not agent_pending:
                            agent_ready = True
                            state = {
                                'env': env,
                                'players': players,
                                'obss': obss,
                                't': t,
                                'max_length': max_length,
                                'extracted_features': extracted_features
                            }
                            checkpoint_mgr.save_state(state, f"agent_ready_t{t}")
                            pdb.set_trace()
                        if agent_pending:
                            for list_ in not_ready:
                                players['agent'].observe(list_[0]['agent'], temporal = list_[1:])
                                console.print(list_[0]['agent'], "agent's currrent obs")
                            
                            
                            
                            
                        #pdb.set_trace()
                        
                            
                        ##if user_ready:
                        ##    resp1_dict, resp2_dict, resp3_dict = user_persona_respond(players, players2, players3, extracted_features)
                        ##    file_path = '/Users/georgiazhou/research_machine/dialop/RL_data/all/'
                        ##    with open(f'{file_path}{t}user_persona_replies.txt', 'w') as f:
                        ##        for pname, player in players.items():
                        ##            if pname == 'user':
                        ##                f.write(json.dumps(player.prompt, indent=2))
                        ##        f.write(json.dumps(resp1))
                        ##        f.write(json.dumps(resp1_dict, indent=2))
                        ##        f.write(json.dumps(resp2))
                        ##        f.write(json.dumps(resp2_dict, indent=2))
                        ##        f.write(json.dumps(resp3))
                        ##        f.write(json.dumps(resp3_dict, indent=2))
                        ##    pdb.set_trace()
                        if user_ready:
                            pdb.set_trace()
                            n_10_user_personas(players, extracted_features)
                            
                            
                        elif agent_ready:
                            ##proposal1, proposal2, proposal3 = n_different_proposals(players)
                            resp1, resp2, resp3 = n_different_responses(players)
                            pdb.set_trace()
                        elif agent_pending:
                            agent_pending_responses = []
                            for list_ in not_ready:
                                agent_pending_responses.append([players['agent'].respond(list_[0], temporal = list_[1:]), list_[1], list_[2]])
                        else:
                       
                            if players[obss["turn_player"]].role == 'user':
                                resp = players[obss["turn_player"]].respond(vary = True)
                                user_response_counter += 1
                            else:
                                resp = players[obss["turn_player"]].respond()
                        #obss, resample = env.step(resp)
                        
                        stepped = True
                    except RateLimitError:
                        print("Rate limited. Sleeping...")
                        time.sleep(30)
                log.log(
                    key=obss["turn_player"],
                    value=resp,
                    title=f"generate t={t} try={resample_cnt}"
                )
                stepped = False
                while not stepped:
                    try:
                         # make sure a proposal is made
                        #pdb.set_trace()
                          
                   
                            
                            # now there's 3 agent proposals, we need to make 3 estimated ratings 
                        if agent_ready or agent_pending:
                            #pdb.set_trace()
                            
                            if agent_pending:
                                obss_list = []
                                ready_list = []
                                id_list = []
                                count_list = []
                                
                                
                                for resp_with_id in agent_pending_responses:
                                    obss, resample, extracted_feature, ready = env.step(resp_with_id[0], t>=(max_length-5), pause_turn = True, agent_pending = True)
                                    obss_list.append(obss)
                                    ready_list.append(ready)
                                    count_list.append(resp_with_id[1])
                                    id_list.append(resp_with_id[2])
                                obss_ready_tuples = list(zip(obss_list, count_list, id_list, ready_list))
                            elif agent_ready:

                                obss1, resample, extracted_feature1, ready1 = env.step(resp1, t>=(max_length-5), pause_turn = True)
                                obss2, resample, extracted_feature2, ready2 = env.step(resp2, t>=(max_length-5), pause_turn = True)
                                obss3, resample, extracted_feature3, ready3 = env.step(resp3, t>=(max_length-5), pause_turn = True)
                                ready_list = [ready1, ready2, ready3]
                                obss_list = [obss1, obss2, obss3]
                                obss_ready_tuples = list(zip(obss_list, ready_list))
                            if all(ready_list):
                                user_ready = True
                                agent_pending = False
                                pdb.set_trace()
                            else:
                                pdb.set_trace()
                                if not_ready:
                                    current_indices = {x[2] for x in not_ready}
                                    new_indices = {x[2] for i, x in enumerate(obss_ready_tuples) if x[3]}
                                    # Keep only ones that are still not ready
                                    still_not_ready = current_indices ^ new_indices
                                    not_ready = [[x[0], x[1] + 1, x[2]] for x in obss_ready_tuples if x[2] in still_not_ready]
                                else: 
                                    # [obss, how many convs, id]
                                    not_ready =[[x[0], 0, i] for i, x in enumerate(obss_ready_tuples) if not x[1]]
                                agent_pending = True
                                agent_ready = False
                        else:
                            obss, resample, extracted_feature, ready = env.step(resp, t>=(max_length-5))
                            
                        
                        
                      
                            
                     
                          
                        
                                
                        extracted_features.append(extracted_feature)
                        #pdb.set_trace()
                        stepped = True
                        if obss["done"]:
                            if resample_cnt >= 3:
                                print("Resampled too many times.")
                                raise ResampleError()
                            log.flush()
                            for pname, player in players.items():
                                log.flush_key(pname, title=f"{pname} Log")
                                log.write(f"Final {pname} Prompt", player.prompt)
                            result = {**obss, **metadata, "t": t,
                                      "num_turns": len(env.game.action_log),
                                      "num_words": count_words(env.game.action_log)}
                            save_extracted_features(extracted_features)
                            log.write("Result", json.dumps(result))
                            log.flush()
                            log.close()
                            
                            return
                                
                    except RateLimitError:
                        print("Rate limited on step. Sleeping...")
                        time.sleep(30)
                #resample_cnt += 1
            t += 1
    
        

    if resample_cnt >= 3:
        print("Resampled too many times.")
        raise ResampleError()

    
        
        
    log.flush()
    for pname, player in players.items():
        log.flush_key(pname, title=f"{pname} Log")
        log.write(f"Final {pname} Prompt", player.prompt)
    result = {**obss, **metadata, "t": t,
              "num_turns": len(env.game.action_log),
              "num_words": count_words(env.game.action_log)}
    save_extracted_features(extracted_features)
    log.write("Result", json.dumps(result))
    log.flush()
    log.close()

count = 0
def save_extracted_features(extracted_features):
    global count
    extracted_features_path = '/Users/georgiazhou/research_machine/dialop/RL_data/conv_features/'
    with open(f'{extracted_features_path}{count}extracted_features.txt', 'w') as f:
        f.write(str(extracted_features))
    count += 1
def main(
    exp_name: str,
    game: Literal["matching", "planning", "mediation"],
    mode: Literal["selfplay", "prompted_sp"],
    resume: Optional[int]=0,
    end: Optional[int]=1000,
    samples_per_game: Optional[int]=1,
    temperature: Optional[float]=0.1,
    dry_run: Optional[bool]=True,
    use_word_limit: Optional[bool]=False,
):
    #pdb.set_trace()

    game_cls = GAME_CLSS[game]
    EXP_DIR = RESDIR / game
    if game_cls == OptimizationEnv:
        DATA_PATH = DATADIR / "reviewer.jsonl"
    elif game_cls == PlanningEnv:
        DATA_PATH = DATADIR / "planning.jsonl"
    elif game_cls == MediationEnv:
        DATA_PATH = DATADIR / "mediation.jsonl"

    os.makedirs(EXP_DIR / exp_name, exist_ok=True)
    with open(DATA_PATH) as f:
        games = []
        for line in f:
            games.append(json.loads(line))

    # Create generator for eval mode.
    if mode == "selfplay":
        gen = selfplay(game_cls, games, samples_per_game, resume, end)
    elif mode == "prompted_sp":
        gen = prompted_selfplay(game_cls, games, samples_per_game, resume, end)
    else:
        raise NotImplementedError()

    def create_players():
        print("Initializing players...")
        # Create prompts.
        #pdb.set_trace()
        if game_cls == OptimizationEnv:
            with open(FPATH / "prompts" / "matching_prompt.txt") as f:
                matching_prompt = f.read()
        elif game_cls == PlanningEnv:
            # if use_word_limit:
            #     with open(FPATH / "prompts" / "planning_agent_timed.txt") as f:
            #         agent_prompt = f.read()
            #     with open(FPATH / "prompts" / "planning_user_timed.txt") as f:
            #         user_prompt = f.read()
            # else:

            with open(FPATH / "prompts" / "planning_agent.txt") as f:
                agent_prompt = f.read()
            with open(FPATH / "prompts" / "planning_user.txt") as f:
                user_prompt = f.read()
        elif game_cls == MediationEnv:
            with open(FPATH / "prompts" / "mediation_agent.txt") as f:
                agent_prompt = f.read()
            with open(FPATH / "prompts" / "mediation_user0.txt") as f:
                user0_prompt = f.read()
            with open(FPATH / "prompts" / "mediation_user1.txt") as f:
                user1_prompt = f.read()

        if game_cls == OptimizationEnv:
            p1, p2 = "player-1", "player-2"
            p1_prompt, p2_prompt = matching_prompt, matching_prompt
            optional1 = p1_prompt[
                p1_prompt.index("EXAMPLE 1"):]
            optional1 = optional1[:optional1.index("EXAMPLE 2")]
            optional2 = p2_prompt[
                p2_prompt.index("EXAMPLE 2"):]
            optional2 = optional2[:optional2.index("EXAMPLE 2")]
        elif game_cls == PlanningEnv:
            p1, p2 = "agent", "user"
            p1_prompt, p2_prompt = agent_prompt, user_prompt
            optional1, optional2 = None, None
        elif game_cls == MediationEnv:
            p1, p2, p3 = "user0", "user1", "agent"
            optional = agent_prompt[agent_prompt.index("User 0 Information"):]
            optional = optional[:optional.index("\n\n") + 2]

        if dry_run:
            assert game_cls != MediationEnv
            players = {p1: DryRunPlayer(p1_prompt, p1, console),
                       p2:  DryRunPlayer(p2_prompt, p2, console)}
        elif game_cls == MediationEnv:
            players = {p1: LLMPlayer(user0_prompt, p1, console,
                                     model_kwargs={"temperature": temperature}),
                       p2:  LLMPlayer(user1_prompt, p2, console,
                                      model_kwargs={"temperature": temperature}),
                       p3:  LLMPlayer(agent_prompt, p3, console,
                                      prefix="\nYou to",
                                      optional=optional,
                                      model_kwargs={"temperature": temperature})}
        else:
            players = {p1: LLMPlayer(p1_prompt, p1, console,
                                     optional=optional1,
                                     model_kwargs={"temperature": temperature}),
                       p2:  LLMPlayer(p2_prompt, p2, console,
                                      optional=optional2,
                                      model_kwargs={"temperature": temperature})}
        return players

    def create_env():
        #pdb.set_trace()
        print("Initializing envs...")
        if game_cls == OptimizationEnv:
            env = OptimizationEnv()
            if use_word_limit:
                env = ForceProposal(env, ["player-1", "player-2"])
        elif game_cls == PlanningEnv:
            env = PlanningEnv(query_executor="gpt4")
            #pdb.set_trace()# this is where preferences constructed
            if use_word_limit:
                env = AsymmetricForceProposal(env, ["agent"])
        elif game_cls == MediationEnv:
            env = MediationEnv()
            if use_word_limit:
                env = AsymmetricForceProposal(env, ["agent"])
        return env

    if dry_run:
        max_length = 15
    elif game_cls == MediationEnv:
        max_length = 45
    else:
        max_length = 35

    # Evaluate.
    times = []
    for i, (data, fname, metadata) in enumerate(gen):
        #pdb.set_trace()
        if (EXP_DIR / exp_name / f"{fname}.out").exists():
            continue
        if not dry_run and i % 20 == 1:
            print(f"Sleeping... {np.mean(times):.1f}")
            time.sleep(30)
            pass
        print(fname)

        start = time.time()
        run(
            game_cls,
            data,
            metadata,
            create_players,
            create_env,
            EXP_DIR / exp_name /f"{fname}.out",
            use_word_limit=use_word_limit,
            max_length=max_length,
        )
        elapsed = (time.time() - start) / 60
        times.append(elapsed)
        print(f" == Finished {i} {elapsed:.1f} == ")

    exit()

    #with open("query_executor_prompt.txt", "w") as f:
    #    f.write(env.search.prompt)
    #test_queries = [
    #    "Search(fields=[name], text_query='not touristy', filters=[category == restaurant]"
    #]
    #env.search(test_queries[0])

    # Try scoring a proposal
    #proposal = "[The Dive, Saul's, NULL]"
    #proposal_info = env._propose(proposal)
    #print(proposal_info)


if __name__ == "__main__":
    tyro.cli(main)
