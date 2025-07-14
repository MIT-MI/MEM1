import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
from copy import deepcopy
from .attn_mask_utils import compose_final_output

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
        is_webagent: bool = False
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.is_webagent = is_webagent
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor, is_webagent: bool = False) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        if is_webagent:
            responses_str = [resp.split('</answer>')[0] + '</answer>'
                    if '</answer>' in resp 
                    else resp
                    for resp in responses_str]
        else:
            responses_str = [resp.split('</answer>')[0] + '</answer>'
                    if '</answer>' in resp 
                    else resp.split('</search>')[0] + '</search>'
                    if '</search>' in resp 
                    else resp
                    for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: Optional[torch.Tensor] = None) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        if next_obs_ids is not None:
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses,
                next_obs_ids
            ])
        else:
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses
            ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()
        
        if "stop" in active_batch.meta_info:
            padded_active_batch.meta_info.update(
                {
                    "stop": active_batch.meta_info["stop"]
                }
            )

        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output
    
    def _calculate_kept_lengths(self, next_obs_ids: torch.Tensor, information: bool = False) -> int:
        """Calculate the kept lengths of the next observations."""
        if not information:
            next_obs_str = self.tokenizer.decode(next_obs_ids, skip_special_tokens=True)
            kept_obs_str = next_obs_str.split('</think>')[1] if '</think>' in next_obs_str else next_obs_str
            kept_obs_ids = self.tokenizer(kept_obs_str, add_special_tokens=False, return_tensors='pt')['input_ids']
            return (kept_obs_ids != self.tokenizer.pad_token_id).sum().item()
        else:
            return (next_obs_ids != self.tokenizer.pad_token_id).sum().item()
    
    def _extract_think_and_response(self, response_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract the think and response from the response ids."""
        response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        if '</think>' in response_str:
            think_str = response_str.split('</think>')[0] + '</think>'
            response_str = response_str.split('</think>')[1]
        else:
            think_str = ''
            response_str = response_str
        think_ids = self.tokenizer(think_str, add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze(0)
        new_response_ids = self.tokenizer(response_str, add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze(0)
        return think_ids, new_response_ids

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor, is_validation: bool = False) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        """
        STEP 1: construct the original left side and right side and statistics
        """
        original_left_side = {'input_ids': initial_input_ids.clone()}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        batch_size = gen_batch.batch['input_ids'].shape[0]

        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns_stats = torch.ones(batch_size, dtype=torch.int)
        batch_rewards = torch.zeros(batch_size, dtype=torch.float)
        valid_action_stats = torch.zeros(batch_size, dtype=torch.int)
        valid_search_stats = torch.zeros(batch_size, dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        peak_seq_token_len = torch.zeros(batch_size, dtype=torch.long)
        average_token_len_per_turn = torch.zeros(batch_size, dtype=torch.float)
  
        reconstruction_list = [{} for _ in range(batch_size)]
        # first populate the prompts in the reconstruction_list
        for i in range(batch_size):
            reconstruction_list[i]['q'] = original_left_side['input_ids'][i]

        kept_lengths = [0 for _ in range(batch_size)]
        initial_token_lengths = [0 for _ in range(batch_size)]
        for i in range(batch_size):
            initial_token_lengths[i] = (gen_batch.batch['input_ids'][i] != self.tokenizer.pad_token_id).sum().item()
        
        # step two: Main generation loop
        rollings = gen_batch
        for step in range(self.config.max_turns):
            """
            Preparation: Generate hint text
            """
            num_turns_left = self.config.max_turns - step - 1
            if num_turns_left > 1:
                hint = f"[HINT]You have {num_turns_left} turns left.[/HINT]"
            elif num_turns_left == 1:
                hint = f"[HINT]You have 1 turn left. You must answer the question in the next turn.[/HINT]"
            else:
                hint = ""


            """
            ==========================
            First loop: generate responses
            ==========================
            """

            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })       

            # filter by active_mask
            active_kept_lengths = [kept_lengths[i] for i in range(len(kept_lengths)) if active_mask[i]]
            active_initial_token_lengths = [initial_token_lengths[i] for i in range(len(initial_token_lengths)) if active_mask[i]]

            # apply padding mask to rollings_active
            # the padding mask should mask everything except the last observations
            padding_mask = self.tensor_fn.create_attention_mask(rollings_active.batch['input_ids'])
            padding_mask = self.tensor_fn.mask_using_kept_lengths(padding_mask, active_kept_lengths, active_initial_token_lengths)
            # change the input_ids accordingly. set all invalid tokens to pad_token_id
            input_ids = rollings_active.batch['input_ids']
            input_ids = torch.where(padding_mask == 0, self.tokenizer.pad_token_id, input_ids)
            rollings_active.batch['input_ids'] = input_ids
            
            # convert the paddings to left
            input_ids, _ = self.tensor_fn.convert_pad_structure(input_ids, pad_to_left=True)
            rollings_active.batch['input_ids'] = input_ids
            rollings_active.batch['attention_mask'] = self.tensor_fn.create_attention_mask(input_ids)
            rollings_active.batch['position_ids'] = self.tensor_fn.create_position_ids(rollings_active.batch['attention_mask'])

            # prepare for second step: stop at answer or search
            rollings_active.meta_info.update(
                {
                    'stop': ["</answer>", " </answer>", "</answer> ", "</search>", " </search>", "</search> "]
                }
            )

            # if step >= 2:
            #     import pdb; pdb.set_trace()

            #     responses_str = self.tokenizer.batch_decode(rollings_active.batch['input_ids'], skip_special_tokens=True)

            gen_output = self._generate_with_gpu_padding(rollings_active)

            # calculate the peak_seq_token_len and average_token_len_per_turn
            seq = torch.cat([rollings_active.batch['input_ids'], gen_output.batch['responses']], dim=1)
            effective_len = (seq != self.tokenizer.pad_token_id).sum(dim=1)
            peak_seq_token_len[active_mask] = torch.max(peak_seq_token_len[active_mask], effective_len)
            average_token_len_per_turn[active_mask] = average_token_len_per_turn[active_mask] * (step / (step + 1)) + effective_len / (step + 1)


            active_responses_ids, active_responses_str = self._postprocess_responses(gen_output.batch['responses'], is_webagent=self.is_webagent)
            responses_ids, responses_str = self.tensor_fn._example_level_pad(active_responses_ids, active_responses_str, active_mask)

            # print the input and responses for step and action 2
            tmp_input_ids = rollings_active.batch['input_ids'].clone()
            tmp_input_ids = tmp_input_ids[0]
            tmp_input_str = self.tokenizer.decode(tmp_input_ids, skip_special_tokens=True)
            print(f"########\n Input for step {step} and action 2: #########\n {tmp_input_str}" + "\n" + f"######## Responses for step {step} and action 2: #########\n {active_responses_str[0]}")
            
            meta_info = gen_output.meta_info  

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search, rewards = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, hint, active_mask, format_reward=True, cur_step=step
            )

            # add the next_obs_ids into the reconstruction_list
            for i in range(len(reconstruction_list)):
                reconstruction_list[i][f't{step}'], reconstruction_list[i][f'r{step}'] = self._extract_think_and_response(responses_ids[i])
                if dones[i] and "num_rounds" not in reconstruction_list[i]:
                    reconstruction_list[i]["num_rounds"] = step + 1

            for i in range(len(kept_lengths)):
                kept_lengths[i] = self._calculate_kept_lengths(responses_ids[i], information=False)
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            batch_rewards += torch.tensor(rewards, dtype=torch.float)
            next_obs_ids = self._process_next_obs(next_obs)

            for i in range(len(reconstruction_list)):
                reconstruction_list[i][f'i{step}'] = next_obs_ids[i]

            # include the next_obs_ids into kept_lengths as well
            for i in range(len(kept_lengths)):
                kept_lengths[i] += self._calculate_kept_lengths(next_obs_ids[i], information=True)
                # (next_obs_ids[i] != self.tokenizer.pad_token_id).sum().item()
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
        
        for i in range(len(reconstruction_list)):
            if "num_rounds" not in reconstruction_list[i]:
                reconstruction_list[i]["num_rounds"] = self.config.max_turns
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        meta_info['batch_rewards'] = batch_rewards.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        # import json
        # with open("reconstruction_list.json", "w") as f:
        #     new_reconstruction_list = [{k: self.tokenizer.decode(v, skip_special_tokens=True) if k != "num_rounds" else v for k, v in item.items()} for item in reconstruction_list]
        #     json.dump(new_reconstruction_list, f)
        

        # print all trajectories
        import random
        print_index = random.randint(0, len(reconstruction_list) - 1)
        print_trajectory = reconstruction_list[print_index]
        print_content = ""
        for k, v in print_trajectory.items():
            if k == "num_rounds":
                continue
            print_content += f"{k}: {self.tokenizer.decode(v, skip_special_tokens=True)}\n"
        print("########\n Full Trajectory ########\n", print_content)

        # import pdb; pdb.set_trace()

        final_output = self._compose_final_output(reconstruction_list, meta_info)
        
        final_output.meta_info['peak_seq_token_len'] = peak_seq_token_len.tolist()
        final_output.meta_info['average_token_len_per_turn'] = average_token_len_per_turn.tolist()

        return final_output

    def _compose_final_output(self, reconstruction_list: List[Dict],
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""

        # - attention_mask: BoolTensor [B, S] (standard causal mask)
        # - info_mask: BoolTensor [B, S] (causal mask + info tokens masked)
        # - position_ids: LongTensor [B, S] (standard position IDs)

        final_output = compose_final_output(reconstruction_list, self.tokenizer.pad_token_id)

        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        

        return final_output
    
    def extract_summary(self, responses: List[str]) -> str:
        pattern = r'<think>(.*?)</think>'
        summaries = []
        dones = []
        valid_action = []

        for response in responses:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                summaries.append("<think>" + match.group(1).strip() + "</think>")
                dones.append(0)
                valid_action.append(1)
            else:
                summaries.append(response)
                dones.append(1)
                valid_action.append(0)
                
        return summaries, dones, valid_action

    def execute_predictions(self, predictions: List[str], pad_token: str, hint: str, active_mask=None, do_search=True, format_reward = True, cur_step=None) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        
        def check_valid_status(prediction, ):
            # Count full tag pairs
            think_matches = re.findall(r"<think>.*?</think>", prediction, re.DOTALL)
            # summary_matches = re.findall(r"<summary>.*?</summary>", prediction, re.DOTALL)
            answer_matches = re.findall(r"<answer>.*?</answer>", prediction, re.DOTALL)
            search_matches = re.findall(r"<search>.*?</search>", prediction, re.DOTALL)

            # Count all tag openings and closings
            all_tags = re.findall(r"</?(\w+)>", prediction)
            tag_counts = {tag: all_tags.count(tag) for tag in set(all_tags)}
            return True

            # Each tag must appear exactly once (open + close = 2), or 0
            # for tag in ['think', 'summary', 'answer', 'search']:
            for tag in ['think', 'answer', 'search']:
                if tag_counts.get(tag, 0) not in [0, 2]:
                    return False

            # Only one of answer/search should be present

            # if len(think_matches) == 1 and len(summary_matches) == 1:
            #     if len(answer_matches) == 1 and len(search_matches) == 0:
            #         return True
            #     if len(search_matches) == 1 and len(answer_matches) == 0:
            #         return True

            if cur_step <= 1:
                # check search
                if len(search_matches) == 1 and len(think_matches) == 1 and len(answer_matches) == 0:
                    return True
                else:
                    return False
            elif cur_step < self.config.max_turns - 1:
                # check answer or search
                if (len(answer_matches) == 1 and len(search_matches) == 0) or (len(answer_matches) == 0 and len(search_matches) == 1):
                    if len(think_matches) == 1:
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                # check answer
                if len(answer_matches) == 1 and len(search_matches) == 0:
                    if len(think_matches) == 1:
                        answer = answer_matches[0].strip()
                        answers = answer.split(';')
                        answer_validity = (not any([a.split() == "" for a in answers])) and len(answers) == 2
                        return answer_validity
                    else:
                        return False
                else:
                    return False
            return False
            
            
        valid_status = [check_valid_status(prediction) for prediction in predictions]
            
        next_obs, dones, valid_action, is_search, rewards = [], [], [], [], []
        
        search_queries = [content for i, (action, content) in enumerate(zip(cur_actions, contents)) if action == 'search' and valid_status[i]]
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for i, (action, content) in enumerate(zip(cur_actions, contents)) if action == 'search' and valid_status[i]])
        else:
            search_results = [''] * sum([1 for i, (action, content) in enumerate(zip(cur_actions, contents)) if action == 'search' and valid_status[i]])
        

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            if (not valid_status[i]) and predictions[i]:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                if format_reward:
                    rewards.append(0)
                    # if cur_step == 0:
                    #     rewards.append(-1)
                    # elif cur_step == 1:
                    #     rewards.append(-0.7)
                    # elif cur_step == 2:
                    #     rewards.append(-0.4)
                    # else:
                    #     rewards.append(-0.2)
                    # if cur_step == 0:
                    #     rewards.append(-1)
                    # elif cur_step == 1:
                    #     rewards.append(-0.5)
                    # else:
                    #     rewards.append(-0.2)
                else:
                    rewards.append(0)
            elif not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                rewards.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{hint}\n\n{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(0)
                    is_search.append(0)
                rewards.append(0)
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search, rewards


    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
        # summaries = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
            #     # match <summary>...</summary>
            #     pattern = r'<summary>(.*?)</summary>'
            #     match = re.search(pattern, prediction, re.DOTALL)
            #     if match:
            #         summary = match.group(1).strip()
            #     else:
            #         summary = None
            #     summaries.append(summary)
                    

                # there are two types of patterns: <summary>...</summary><next_action>...</next_action> and <answer>...</answer>
                # if <answer>...</answer> is found, we do not need to find <next_action>...</next_action>
                # modify the pattern to match both cases
                pattern = r'<(answer|search)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()
                    action = match.group(1).strip()
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
        # return actions, contents, summaries
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        try:
            return requests.post(self.config.search_url, json=payload).json()
        except Exception as e:
            print(f"Error in batch_search: {e}")
            return []

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
