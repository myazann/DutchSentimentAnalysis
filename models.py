import configparser
import os
from threading import Thread
from pathlib import Path
import copy
import warnings
warnings.filterwarnings("ignore")

from llama_cpp import Llama
from huggingface_hub import login, logging, hf_hub_download, snapshot_download
logging.set_verbosity_error()
import tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, logging, BitsAndBytesConfig, AsyncTextIteratorStreamer
logging.set_verbosity_error()

from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai


class LLM:

    def __init__(self, model_name, default_prompt=None, model_params=None, gen_params=None) -> None:
        
        login(token=os.getenv("HF_API_KEY"), new_session=False)
        self.cfg = LLM.get_cfg()[model_name]
        self.model_name = model_name
        self.family = model_name.split("-")[0]
        self.repo_id = self.cfg.get("repo_id")
        self.file_name = self.cfg.get("file_name", None)
        self.context_length = int(self.cfg.get("context_length"))
        self.provider = self.get_provider()
        self.tokenizer = self.init_tokenizer()
        self.model_params = self.get_model_params(model_params)
        self.gen_params = self.get_gen_params(gen_params)
        self.model = self.init_model()
        self.default_prompt = default_prompt if default_prompt is not None else []

    @staticmethod
    def get_cfg():

        config = configparser.ConfigParser()
        config.read(os.path.join(Path(__file__).absolute().parent, "model_config.cfg"))
        return config
        
    def get_provider(self):

        if self.model_name.endswith("GROQ"):
            return "GROQ"
        elif self.model_name.endswith("GGUF"):
            return "GGUF"
        elif self.cfg.get("provider"):
            return self.cfg.get("provider")  
        else:
            return "HF"
        
    def init_tokenizer(self):

        if self.provider in ["GROQ", "GGUF", "DEEPSEEK"]:
            return AutoTokenizer.from_pretrained(self.cfg.get("tokenizer"), use_fast=True)
        elif self.provider in ["ANTHROPIC", "OPENAI", "GOOGLE"]:
            return None
        else:
            return AutoTokenizer.from_pretrained(self.repo_id, use_fast=True)
            
    def get_gen_params(self, gen_params):

        if self.provider == "GOOGLE":
            self.name_token_var = "max_output_tokens"
        elif self.provider in ["OPENAI", "ANTHROPIC", "GGUF", "DEEPSEEK"]:
            if self.family in ["o1", "o3"]:
                self.name_token_var = "max_completion_tokens"
            else:
                self.name_token_var = "max_tokens"
        else:
            self.name_token_var = "max_new_tokens"
        if gen_params is None:
            return {self.name_token_var: 512}
        if "max_new_tokens" in gen_params and self.name_token_var != "max_new_tokens":
            gen_params[self.name_token_var] = gen_params.pop("max_new_tokens")
        elif "max_tokens" in gen_params and self.name_token_var != "max_tokens":
            gen_params[self.name_token_var] = gen_params.pop("max_tokens")
        elif "max_output_tokens" in gen_params and self.name_token_var != "max_output_tokens":
            gen_params[self.name_token_var] = gen_params.pop("max_output_tokens")
        elif "max_completion_tokens" in gen_params and self.name_token_var != "max_completion_tokens":
            gen_params[self.name_token_var] = gen_params.pop("max_completion_tokens")
        return gen_params
    
    def get_model_params(self, model_params):

        if model_params is None:
            if self.provider == "GROQ":
                return {
                    "base_url": "https://api.groq.com/openai/v1",
                    "api_key": os.getenv("GROQ_API_KEY")
                }   
            elif self.provider == "DEEPSEEK":
                return {
                    "base_url": "https://api.deepseek.com",
                    "api_key": os.getenv("DEEPSEEK_API_KEY")
                }                     
            elif self.provider == "ANTHROPIC":
                return {
                    "api_key": os.getenv("ANTHROPIC_API_KEY")
                }
            elif self.provider == "OPENAI":
                return {
                    "api_key": os.getenv("OPENAI_API_KEY")
                }
            elif self.provider == "GOOGLE":
                return {
                    "api_key": os.getenv("GOOGLE_API_KEY")
                }
            elif self.provider == "GGUF":
                return {
                    "n_gpu_layers": -1,
                    "verbose": False,
                    "n_ctx": self.context_length
                }
            else:
                return {}
        else:
            return model_params
    
    def init_model(self):

        if self.provider == "ANTHROPIC":
            return Anthropic(**self.model_params)
        elif self.provider in ["OPENAI", "GROQ", "DEEPSEEK"]:
            return OpenAI(**self.model_params)       
        elif self.provider == "GOOGLE":
            genai.configure(**self.model_params)
            return genai.GenerativeModel(self.repo_id)
        elif self.provider == "GGUF":
            if os.getenv("HF_HOME") is None:
                hf_cache_path = os.path.join(os.path.expanduser('~'), ".cache", "huggingface", "hub")
            else:
                hf_cache_path = os.getenv("HF_HOME")
            model_path = os.path.join(hf_cache_path, self.file_name)
            if not os.path.exists(model_path):
                if self.file_name.endswith("gguf"):
                    hf_hub_download(repo_id=self.repo_id, filename=self.file_name, local_dir=hf_cache_path)
                else:
                    snapshot_download(repo_id=self.repo_id, local_dir=hf_cache_path, allow_patterns = [f"*{self.file_name}*"])
            if not self.file_name.endswith("gguf"):
                len_files = len(os.listdir(model_path))
                model_path = f"{model_path}/{self.file_name}-00001-of-0000{len_files}.gguf"
            return Llama(model_path=model_path, **self.model_params)
        else: 
            bnb_config = None
            if "quantization" in self.model_params:
                quant_params = self.model_params.pop("quantization")
                if isinstance(quant_params, dict):
                    bnb_config = BitsAndBytesConfig(**quant_params)
                elif isinstance(quant_params, BitsAndBytesConfig):
                    bnb_config = quant_params
            return AutoModelForCausalLM.from_pretrained(
                    self.repo_id,
                    **self.model_params,
                    quantization_config=bnb_config,
                    low_cpu_mem_usage=True,
                    device_map="auto")

    def format_prompt(self, prompt, params=None):
        """
        Ensure that the prompt is a list of dictionaries in the format:
        [{"role": <role>, "content": <content>}, ...].
        
        - If prompt is None, use self.default_prompt.
        - If prompt is a string, convert it into a list with one dict using role "user".
        - If prompt is a list, any non-dict element is assumed to be a string and converted accordingly.
        
        If neither prompt nor self.default_prompt is provided, raise an error.
        Then, if params are provided, format each message's content using str.format().
        
        :param prompt: A string, list (of dicts or strings), or None representing the prompt.
        :param params: A dict of parameters to format the content strings.
        :return: A standardized list of dicts with parameters injected.
        """
        # If prompt is None, use default_prompt
        default_prompt = copy.deepcopy(self.default_prompt)
        if prompt is None:
            if default_prompt:
                prompt = default_prompt
            else:
                raise ValueError("No prompt provided and no default prompt available.")

        # Normalize prompt into a list of dicts.
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            new_prompt = []
            for item in prompt:
                if isinstance(item, dict):
                    new_prompt.append(item)
                elif isinstance(item, str):
                    new_prompt.append({"role": "user", "content": item})
                else:
                    raise ValueError("Each item in the prompt list must be either a dict or a string.")
            prompt = new_prompt
        else:
            raise TypeError("Prompt must be either a string, a list, or None.")

        # If a default prompt exists and prompt was not already the default,
        # prepend the default_prompt (avoid duplicating if prompt came from default_prompt).
        if not (prompt == default_prompt) and self.default_prompt:
            prompt = default_prompt + prompt

        # Format each message's content if parameters are provided.
        if params and isinstance(params, dict):
            for message in prompt:
                if "content" in message and isinstance(message["content"], str):
                    try:
                        message["content"] = message["content"].format(**params)
                    except Exception as e:
                        raise ValueError("Error formatting prompt content: " + str(e))

        return prompt

    def get_avail_space(self, prompt):

        avail_space = self.context_length - self.gen_params[self.name_token_var] - self.count_tokens(prompt)
        if avail_space <= 0:
            return None
        else:
            return avail_space   
        
    def trunc_chat_history(self, chat_history, hist_dedic_space=0.2):

        hist_dedic_space = int(self.context_length*0.2)
        total_hist_tokens = sum(self.count_tokens(tm['content']) for tm in chat_history)
        while total_hist_tokens > hist_dedic_space:
            removed_message = chat_history.pop(0)
            total_hist_tokens -= self.count_tokens(removed_message['content'])
        return chat_history 
       
    def count_tokens(self, prompt):

        prompt = self.format_prompt(prompt)
        
        prompt_text = "\n".join([turn["content"] for turn in prompt])
        
        if self.provider == "OPENAI":
            encoding = tiktoken.encoding_for_model(self.repo_id)
            return len(encoding.encode(prompt_text))
        elif self.provider == "GOOGLE":
            return self.model.count_tokens(prompt_text).total_tokens
        elif self.provider == "ANTHROPIC":
            return self.model.count_tokens(prompt_text)
        else:
            return len(self.tokenizer(prompt_text).input_ids)
        
    def prepare_context(self, prompt, context, query=None, chat_history=[]):

        prompt = self.format_prompt(prompt)
        
        if chat_history:
            chat_history = self.trunc_chat_history(chat_history)
        
        query_len = self.count_tokens(query) if query else 0
        avail_space = self.get_avail_space(prompt + chat_history) - query_len  
        if avail_space:         
            while True:
                info = "\n".join([doc for doc in context])
                if self.count_tokens(info) > avail_space:
                    print("Context exceeds context window, removing one document!")
                    context = context[:-1]
                else:
                    break
            return info
        else:
            return -1

    def generate(self, prompt=None, stream=False, gen_params=None, prompt_params=None):

        prompt = self.format_prompt(prompt, prompt_params)

        if not gen_params:
            gen_params = self.gen_params
        else:
            gen_params = self.get_gen_params(gen_params)

        if self.provider in ["GROQ", "DEEPSEEK", "OPENAI"]:
            response = self.model.chat.completions.create(
                model=self.repo_id, messages=prompt, stream=stream, **gen_params
            )
            if stream:
                def stream_response():
                    has_reasoning = False
                    finished_thinking_yielded = False
                    reasoning_content = ""

                    if self.provider == "DEEPSEEK" and self.cfg.get("reason"):
                        yield "**Thinking**...\n\n\n"

                    for chunk in response:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                            has_reasoning = True 
                            reasoning_content += delta.reasoning_content
                            yield delta.reasoning_content
                        elif hasattr(delta, 'content') and delta.content:
                            if has_reasoning and not finished_thinking_yielded:
                                yield "\n\n\n**Finished Thinking!**\n\n\n"
                                finished_thinking_yielded = True
                            yield delta.content
                    if has_reasoning and not finished_thinking_yielded:
                        yield "\n\n\n**Finished Thinking!**\n\n\n"
                return stream_response()
                        
            output = response.choices[0].message.content
            if self.provider == "DEEPSEEK" and self.cfg.get("reason"):
                reasoning_steps = response.choices[0].message.reasoning_content
                output = f"**Thinking**...\n\n\n{reasoning_steps}\n\n\n**Finished thinking!**\n\n\n{output}"

        elif self.provider == "ANTHROPIC":
            if prompt[0]["role"] == "system":
                sys_msg = prompt[0]["content"]
                prompt = prompt[1:]
            else:
                sys_msg = ""
            if stream:
                stream = self.model.messages.stream(
                    model=self.repo_id, messages=prompt, system=sys_msg, **gen_params
                ).__enter__()
                return stream.text_stream
            else:
                response = self.model.messages.create(
                    model=self.repo_id, messages=prompt, system=sys_msg, **gen_params
                )
                output = response.content[0].text   

        elif self.provider == "GOOGLE":
            messages = []
            for turn in prompt:
                role = "user" if turn["role"] in ["user", "system"] else "model"
                messages.append({
                    "role": role,
                    "parts": [turn["content"]]
                })
            response = self.model.generate_content(
                messages, generation_config=genai.types.GenerationConfig(**gen_params)
            )
            output = response.text 

        else:
            if self.family in ["MISTRAL", "GEMMA"]:
                if len(prompt) > 1:
                    prompt = [{"role": "user", "content": "\n".join([turn["content"] for turn in prompt])}]
            if self.provider == "GGUF":
                response = self.model.create_chat_completion(prompt, stream=False, **gen_params)
                output = response["choices"][-1]["message"]["content"]
            else:
                if stream:
                    return self.stream_hf_output(prompt, gen_params)
                else:
                    streamer = None
                pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, streamer=streamer, **gen_params)
                output = pipe(prompt)[0]["generated_text"][-1]["content"]

        return output

    
    async def stream_hf_output(self, prompt, gen_params):

        streamer = AsyncTextIteratorStreamer(self.tokenizer, skip_prompt=True)

        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, streamer=streamer, **gen_params)
        thread = Thread(target=pipe, args=(prompt,))
        thread.start()

        async for token in streamer:
            if token in ["<end_of_turn>", "<eot>", "<eos>", "<|eot_id|>", "<｜end▁of▁sentence｜>"]:
                continue
            elif token.strip() == "<think>":
                yield "**Thinking..\n\n**"
            elif token.strip() == "</think>":
                yield "**\n\nFinished Thinking!**"
            yield token