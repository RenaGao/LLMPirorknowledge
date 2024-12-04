"""
@Time : 2024/12/3 22:35
@Author : Ray
@Email : 1206953809@qq.com
@File : L2model_generation.py
@Purpose
"""


import yaml
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torchsummary import summary


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


class Lingual_Llama_Agent:
    def __init__(self, model_name, device="cuda", system_prompt="You are a helpful assistant.", agent_name="Agent"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16).to(self.device)
        self.system_prompt = system_prompt
        self.agent_name = agent_name
        self.conversation_history = [f"{self.system_prompt}"]

    def generate_response(self, input_text=None, max_length=100):
        if input_text is not None:
            self.conversation_history.append(f"{input_text}")
            full_prompt = "\n".join(self.conversation_history)
        else:
            full_prompt = self.conversation_history
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        # inputs = {key: val.to(self.device) for key, val in inputs.items()}
        outputs = self.model.generate(**inputs, max_length=max_length, num_return_sequences=1)
        # outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.conversation_history.append(f"{response}")
        return response

    def clear_conversation_history(self):
        self.conversation_history = [f"{self.system_prompt}"]


class Agent_Interaction:
    def __init__(self, agent1, agent2):
        self.agent1 = agent1
        self.agent2 = agent2
        self.question = []
        self.respond = []

    def log_message(self, question_save_path=None, respond_save_path=None):
        if question_save_path is not None:
            with open(question_save_path, "w") as file:
                for line in self.question:
                    file.write(line + "\n")
        if respond_save_path is not None:
            with open(respond_save_path, "w") as file:
                for line in self.respond:
                    file.write(line + "\n")

    def clear_qr(self):
        self.question = []
        self.respond = []

    def interaction(self, init_message=None, num_turns=5):
        message = init_message
        for turn in range(num_turns):
            response1 = self.agent1.generate_response(message)
            self.question.append(response1)
            response2 = self.agent2.generate_response(response1)
            self.respond.append(response2)
            message = response2


if __name__ == '__main__':
    # config
    Bilingual_config_file = "finetuned/llama-3-8b-japanese.yaml"
    config = load_config(Bilingual_config_file)

    agent1_system_prompt = ""   ### we need system prompt for initialization
    agent2_system_prompt = ""

    # Bilingual model
    # model agents
    Bi_agent1 = Lingual_Llama_Agent(model_name=config["Bi_model"]["name"],
                                    system_prompt=agent1_system_prompt,
                                    agent_name="Bi_agent1")
    Bi_agent2 = Lingual_Llama_Agent(model_name=config["Bi_model"]["name"],
                                    system_prompt=agent2_system_prompt,
                                    agent_name="Bi_agent2")

    # interaction   ### we need topic prompt list
    num_turns = config["num_turns"]
    num_topic = config["topic"]
    Bi_model_output_path = config["Bi_model_output_dir"]
    Bi_model_conversation = Agent_Interaction(agent1=Bi_agent1, agent2=Bi_agent2)
    for i in range(num_topic):
        Bi_model_conversation.interaction(num_turns=num_turns)
        output_file_path = os.path.join(Bi_model_output_path, f"topic{i}.txt")
        Bi_model_conversation.log_message(respond_save_path=output_file_path)
        Bi_model_conversation.clear_qr()

    # Monolingual model
    # model agents
    Mono_agent1 = Lingual_Llama_Agent(model_name=config["Mono_model"]["name"],
                                      system_prompt=agent1_system_prompt,
                                      agent_name="Mono_agent1")
    Mono_agent2 = Lingual_Llama_Agent(model_name=config["Mono_model"]["name"],
                                      system_prompt=agent2_system_prompt,
                                      agent_name="Mono_agent2")

    # interaction
    Mono_model_output_path = config["Mono_model_output_dir"]
    Mono_conversation = Agent_Interaction(agent1=Mono_agent1, agent2=Mono_agent2)
    for i in range(num_topic):
        Mono_conversation.interaction(num_turns=num_turns)
        output_file_path = os.path.join(Mono_model_output_path, f"topic{i}.txt")
        Mono_conversation.log_message(respond_save_path=output_file_path)
        Mono_conversation.clear_qr()