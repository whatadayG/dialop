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
        self.last_checkpoint_time = 0
    
    def save_state(self, state, name):
        """Save state while handling unpickleable objects"""
        clean_state = {}
        clean_state['t'] = state.get('t', 0)
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
            
        self.last_checkpoint_time = time.time()
        print(f"DEBUG: Successfully saved checkpoint to {self.checkpoint_dir.absolute()}")
    
    #def load_state(self, name):
    #    """Load state and verify player objects"""
    #    path = self.checkpoint_dir / f"{name}.pkl"
    #    with open(path, 'rb') as f:
    #        state = pickle.load(f)
    #        
    #    # Verify player objects are accessible
    #    if 'players' in state:
    #        players = state['players']
    #        # Verify key player attributes are accessible
    #        try:
    #            for player_name, player in players.items():
    #                assert hasattr(player, 'prompt'), f"Player {player_name} missing prompt"
    #                assert hasattr(player, 'role'), f"Player {player_name} missing role"
    #                if player.role == 'user':
    #                    assert hasattr(player, 'user_prompt_obss'), f"User player missing prompt observations"
    #        except Exception as e:
    #            pdb.set_trace()
    #    return state
    def get_latest_checkpoint(self):
        """Get most recent checkpoint created after last_checkpoint_time"""
        # Ensure we're working with Path objects
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)

        # Get list of checkpoint files as Path objects
        checkpoints = list(self.checkpoint_dir.glob("best_path_*.pkl"))
        if not checkpoints:
            return None

        # Convert to Path objects if they're strings
        checkpoints = [Path(p) if isinstance(p, str) else p for p in checkpoints]

        latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[2]))
        checkpoint_time = latest.stat().st_mtime

        if checkpoint_time > self.last_checkpoint_time:
            return latest.stem
        return None
    
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
    def load_and_continue_from_best_path(checkpoint_name, checkpoint_manager, env, players):
        """Load and continue conversation from the best path checkpoint"""
        # Load the saved state
        state = checkpoint_manager.load_full_state(checkpoint_name)
        
        #pdb.set_trace()
        extra_turn = state.get('turn', 0)
        players = state.get('players', {})
        

        # Restore environment state
        env.reset(game_state=state['env_state'])


        return env, players, extra_turn, 



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
    extracted_features = {}
    
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
             
            
            
            return user_personas 
        def n_different_responses(players):
            resp1 = players['agent'].respond(t, max_length, vary=True)
            resp2 = players['agent'].respond(t, max_length, vary=True)
            resp3 = players['agent'].respond(t, max_length, vary=True)
            return [resp1, resp2, resp3]
        
        
        
        tracker = False
        
        while not obss["done"] and t < max_length:
            #pdb.set_trace()
            console.rule("environment obs")
            console.print(obss)
            if tracker:
                [player.observe(obss[pname], ignore_obs=True) for pname, player in players.items()]
            else:
                [player.observe(obss[pname]) for pname, player in players.items()]
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
                        tracker = False
                        if user_response_counter >= cap and obss["turn_player"] == 'agent':
                            agent_ready = True
                            state = {
                                'env': env,
                                'players': players,
                                'obss': obss,
                                't': t,
                                'max_length': max_length,
                                'extracted_features': extracted_features
                            }
                            #print(t)
                            #pdb.set_trace()
                            checkpoint_mgr.save_state(state, "initial_state")
                            tracker = True
                            print("Waiting for best path checkpoint...")
                            pdb.set_trace()
                        if tracker:
                            while True and user_response_counter >= cap: 
                                # Look for most recent best_path checkpoint
                                checkpoint_name = checkpoint_mgr.get_latest_checkpoint()
                                if checkpoint_name:
                                # Get most recent checkpoint
                                    
                                    
                                    print(f"Found checkpoint: {checkpoint_name}")

                                    # Load and restore state
                                    #state = checkpoint_mgr.load_full_state(checkpoint_name)
                                    env, players, extra_turn = CheckpointManager.load_and_continue_from_best_path( checkpoint_name, checkpoint_mgr, env, players)
                              
                                    
                                    #pdb.set_trace()
                                    break
                                time.sleep(3)

                        #pdb.set_trace()
                        if not tracker:
                            resp = players[obss["turn_player"]].respond()
                        else:
                            resp = players['user'].respond()
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

                        #pdb.set_trace()
                        
                        obss, resample, extracted_feature, ready = env.step(resp, t>=(max_length-5))
                        print(obss["turn_player"])
                        if obss["turn_player"] == 'user':
                            user_response_counter += 1
         
                            
                              
                        extracted_features.update(extracted_feature)
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
    pdb.set_trace()

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
        max_length = 60

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
        
        # pdb.set_trace()

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
