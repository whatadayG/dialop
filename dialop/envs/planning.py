from thefuzz import fuzz
import traceback
from pathlib import Path
import pdb
import random
from skills import Agent_tell
from IPython.core.debugger import set_trace
import ast
import copy

from dialop.games import PlanningGame
from dialop.envs import DialogueEnv, GameError
from dialop.templates import (
    PlanningUserPromptTemplate,
    PlanningProposalTemplate,
    PlanningAgentPromptTemplate,
    PlanningPersonaTemplate
)
from planning_query_executor import (
  StaticQueryExecutor, GPT3QueryExecutor, GPT4QueryExecutor, SearchError
)
from dialop.games.planning_data import (
    Event, PREF_NAME_TO_CLS, ALL_PREFS, FeaturePreference, ALL_CONSTRAINTS, DistancePreference,
)


class PlanningEnv(DialogueEnv):
    count = 0

    def __init__(self, query_executor="gpt4"):
        
        self.feature_extractor_prompt = (Path(__file__).parent / "data" / "planning_feature_extraction.txt").read_text()
        
        self.players = ["user", "agent"]
        datapath = Path(__file__).parent / "data"
        self.instructions = [
            (datapath / "planning_user.txt").read_text(),
            (datapath / "planning_agent.txt").read_text()
            ]
        self.query_executor = query_executor
        if query_executor == "static":
            self._search_cls = StaticQueryExecutor
        elif query_executor == "gpt3":
            self._search_cls = GPT3QueryExecutor
        elif query_executor == "gpt4":
            self._search_cls = GPT4QueryExecutor
        else:
            raise NotImplementedError
        
    def create_sampled_user_preferences(self, num_preferences):
        random_prefs = []
        for p in ALL_PREFS:
            random_prefs.extend(p.generate(self.game.game_data))
        random.shuffle(random_prefs)
        random_prefs = random_prefs[:(num_preferences-1)]
        random_prefs.extend(DistancePreference.generate(self.game.game_data))
        output = [(p.readable, p.weight, type(p).__name__, p.__dict__) 
                                       for p in random_prefs]
        
        return output
    
    def create_non_overlapping_preferences(self, num_preferences, existing_prefs):
        # Convert existing prefs keys to set for efficient lookup
        existing_keys = set(existing_prefs.keys())
        
        random_prefs = []
        for p in ALL_PREFS:
            # Generate preferences
            new_prefs = p.generate(self.game.game_data)
            
            # Only add preferences whose type doesn't exist in existing_prefs
            for pref in new_prefs:
                if type(pref).__name__ not in existing_keys:
                    random_prefs.append(pref)
                    
        random.shuffle(random_prefs)
        
        # Add distance preference only if not in existing prefs
        if 'DistancePreference' not in existing_keys:
            random_prefs.extend(DistancePreference.generate(self.game.game_data))
            
        # Take only requested number of preferences
        random_prefs = random_prefs[:num_preferences]
        
        output = [(p.readable, p.weight, type(p).__name__, p.__dict__) 
                  for p in random_prefs]
        
        return output
    
    

    def reset(self, game_state=None, known_user_preferences_num =None, extracted_features = None):
        #import pdb; pdb.set_trace()
        
        if game_state is not None:
            if known_user_preferences_num is not None:
                
                game_state["preferences"] = self.create_non_overlapping_preferences(known_user_preferences_num, extracted_features)

            self.game = PlanningGame.create_from_game_state(game_state)
            self.preferences = game_state["preferences"]
            self.events = game_state["events"]
            self.persona_styles = self.game.persona_styles
        else:
            self.game = PlanningGame({})
            self.game.reset()
            info = self.game.get_game_info()
            self.preferences = info["preferences"]
            self.events = info["events"]
            self.persona_styles = info["persona_styles"]
        self.search = self._search_cls(self.events)
        # Compute score range
        all_scores = self.game._compute_all_solutions(sample=False)
        self.best_score = max(all_scores)
        self.worst_score = min(all_scores)
        self.num_msgs = 0
        self.known_user_preferences = []
        obss = self._init_from_action_log(no_action_log=True)
        #pdb.set_trace()
        return {**obss,
                'extracted_features': {},
                "turn_player": self.players[self.game.turn_player],
                "done": False}

    def step(self, message, last_message = False, ready = False, pause_turn = False, propose = False, agent_pending = False):
        """Step the game state with a message.

        Errors from game logic (e.g. incorrect search syntax, searching by
        fields we don't support, invalid proposal lengths) will be caught
        and added to the prompt, allowing the LM to regenerate. For all other
        errors, resample from the LM.

        Return:
            obss: Dict with obs for each player and game info
            resample: bool indicating whether we need to resample
        """

        done = False
        reward = 0
        info = {"num_msgs": self.num_msgs}
        player = self.players[self.game.turn_player]
        extracted_feature = {}
        
        #import pdb;pdb.set_trace()
        try:
            #import pdb; pdb.set_trace()
            edited_prompt_propose = False
            if propose:
                edited_prompt_propose = True
                
            m = self._parse_message(
                message,
                edited_prompt_propose,
                can_propose=(player == "agent"),
                can_respond=(player != "agent"),
                must_respond=(player != "agent" and self.game.proposal is not None),
            )
            type_ = m["mtype"]
            content = m["msg"]

            #from IPython.core.debugger import set_trace; set_trace()
            #if last_message:
            if pause_turn:
                self.game.pause = True

            if type_ == "message":
                self.num_msgs += 1
                if player == "agent":
                    if ready or agent_pending:
                        if agent_pending:
                            ready = True
                        obss = [f"\nAgent: {message}", message]
                    else:
                        ready = True
                        obss = [f"\nAgent: {message}", ""]
                else:
                    extract_prompt = self.feature_extractor_prompt + "\n" + content
                    finished = False
                    correction_str = ""
                    while finished == False:
                        try:
                            extracted_feature = ast.literal_eval(Agent_tell(extract_prompt + correction_str))
                            finished = True
                        except Exception as e:
                            correction_str = f"Please avoid this syntax error: {e}."
                            
                    obss = [message, f"\nUser: {message}"]
                self.game.message({
                        "data": content,
                        "from_player": self.game.turn_player,
                        "type": "utterance"})
#                self._take_turn()
            elif type_ == "think":
                if player == "agent":
                    obss = ["", message]
                else:
                    obss = [message, ""]
            elif type_ == "tool":
                #import pdb; pdb.set_trace()
                assert player == "agent", "User cannot use tool."
                result = self._call_tool(content)
                obss = ["", f"{message}\n{result}"]
            elif type_ == "propose":
                #import pdb; pdb.set_trace()
                ready = True 
                self.num_msgs += 1
                if player != "agent":
                    raise GameError("Only the agent can make proposals.")
                _, proposal_info = self._propose(content)
                obss = [f"\nAgent: [propose] {proposal_info}", message]
                if self.game.is_full_proposal:
                    obss[0] += (f"\nYou can [think], or output one of these choices:\n"
                                f"(1) [accept] Accept\n(2) [reject] Reject")
#                self._take_turn()
            elif type_ == "accept" or type_ == "reject":
                
                self.num_msgs += 1
                if last_message:
                    type_ = "accept"
    
                done, game_infos = self._proposal_response(
                    type_ == "accept",
                    self.game.turn_player)
                info.update({
                    "best_score": self.best_score,
                    "worst_score": self.worst_score,
                    "reward_normalized": self._normalize(game_infos[0]["reward"])
                })
                reward = game_infos[0]["reward"]
                obss = [f"[{type_}]", f"\nUser: [{type_}]"]
            else:
                raise ValueError(f"Message type not found for: {message}.")
        except (GameError, SearchError) as e:
            obss = ["", ""]
            obss[self.game.turn_player] = f"{message}\nError: {str(e)}"
        except Exception as e:
            print(f"!!! {traceback.format_exc()}")
            import pdb; pdb.set_trace()
            return {
                **{p: "error" for p in self.players},
                "done": False,
                "reward": 0,
                "turn_player": self.players[self.game.turn_player]
            }, True
        
        obss = {self.players[i]: obs for i, obs in enumerate(obss)}
        obss["turn_player"] = self.players[self.game.turn_player]
        obss["done"] = done
        obss["reward"] = reward
        obss["info"] = info
        
        return obss, False, extracted_feature, ready
    def get_feature_weights(self):
        feature_weights = {}
        for pref in self.game.prefs:
            if isinstance(pref, FeaturePreference):
                feature_weights[pref.name] = {
                    'weight': pref.weight,
                    'readable': pref.readable,
                    'value_sets': pref.value_sets  # [disliked_values, liked_values]
                }
        return feature_weights
    def _propose(self, message):
        proposal = self._parse_events(message)
        if len(proposal) < 5:
            raise GameError("You must have 3 events in your proposal. "
                            "Put 'NULL' for a slot to make a partial "
                            "proposal.")
        pinfo = self.game.propose(proposal, self.game.turn_player)
        pinfo_prompt = self._format_proposal(
            pinfo["proposal_data"],
            pinfo["itinerary_scores"],
            pinfo["evt_scores"],
            pinfo["total_score"])
        self.proposal_info = pinfo
        return pinfo, pinfo_prompt

    def _call_tool(self, message: str):
        return self.search(message)

    def _init_from_action_log(self, no_action_log=False):
        #import pdb; pdb.set_trace()
        obss = {}
        if no_action_log:
            action_log = []
        else:
            action_log = self.game.action_log
        #import pdb; pdb.set_trace()
        user_original = PlanningUserPromptTemplate.render(
            travel_doc="\n".join([p[0] for p in self.preferences]),
            preferences=self.game.prefs,
            messages=action_log,
            player_id=0,
            format_proposal=self._format_proposal,
            any=any, # to call any() in template
        ).rstrip()
        user_style = PlanningPersonaTemplate.render(
            persona_styles=self.persona_styles
            ).rstrip()
        obss["user"] = user_original + "\n"  + user_style
            
            
        obss["agent"] = PlanningAgentPromptTemplate.render(
            messages=self.game.action_log,
            player_id=1,
            format_proposal=self._format_proposal_agent,
        ).rstrip()
        obss[self.players[self.game.turn_player]] += "\nYou:"
        return obss

    def _format_proposal(self, proposal, it_scores, evt_scores, tot_score):
        all_scores = evt_scores + [s["score"] for s in it_scores]
        if round(sum(all_scores)) != tot_score:
            import pdb; pdb.set_trace()
        score_calculation = "".join(
            f"+{s}" if s >= 0 else f"{s}" for s in all_scores) + f"={tot_score}"
        proposal_msg = "[" + ", ".join([
            proposal[0]["name"] if proposal[0] else "NULL",
            proposal[2]["name"] if proposal[2] else "NULL",
            proposal[4]["name"] if proposal[4] else "NULL",
        ]) + "]"
        return PlanningProposalTemplate.render(
            proposal_msg=proposal_msg,
            proposal=proposal,
            itinerary_scores=it_scores,
            evt_scores=evt_scores,
            score_calculation=score_calculation,
        )

    def _format_proposal_agent(self, proposal):
        proposal_msg = "[" + ", ".join([
            proposal[0]["name"] if proposal[0] else "NULL",
            proposal[2]["name"] if proposal[2] else "NULL",
            proposal[4]["name"] if proposal[4] else "NULL",
        ]) + "]"
        return proposal_msg

    def _parse_events(self, message):
        """Parses a proposal message to a list of event dicts as a structured
        proposal to pass to the game."""
        names = message[message.index("[") + 1 :
                        message.index("]")].split(",")
        names = [e.strip() for e in names]
        if len(names) > 3:
            raise GameError("You can only propose up to three events."
                            "Try re-proposing with three events.")
        events = []
        site_names = [site["name"] for site in self.events]
        proposal = []
        for e in names:
            found = False
            if e == "NULL":
                proposal.append(None)
                found = True
            elif e in site_names:
                proposal.append({
                    **self.events[site_names.index(e)],
                    "type": "event"
                })
                found = True
            # Fall back to fuzzy search
            else:
                for i, evt in enumerate(site_names):
                    if fuzz.ratio(evt, e) > 80:
                        proposal.append({
                            **self.events[i],
                            "type": "event"
                        })
                        found = True
                        break
            if not found:
                raise GameError(f"{e} is not a valid destination."
                                "Did you type the correct name?")
        # Populate distances
        proposal_with_dists = []
        for i in range(len(proposal) - 1):
            proposal_with_dists.append(proposal[i])
            if proposal[i] and proposal[i + 1]:
                dist = self.search.distance(proposal[i], proposal[i+1])
                proposal_with_dists.append({
                    "data": dist,
                     "name": (f"Travel from {proposal[i]['name']} to "
                              f"{proposal[i+1]['name']}, {dist}mi"),
                     "type": "travel"})
            else:
                proposal_with_dists.append(None)
        if len(proposal) == 0:
            import pdb; pdb.set_trace()
        proposal_with_dists.append(proposal[-1])
        return proposal_with_dists

    def _normalize(self, score):
        return (score - self.worst_score) / (self.best_score - self.worst_score)

