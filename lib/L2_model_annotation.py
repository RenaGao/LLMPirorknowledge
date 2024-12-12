import yaml
import os
import torch
from torch.xpu import device
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torchsummary import summary


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


class Lingual_Features:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.model_name = self.config['model']['name']
        self.device = self.config['model']['device']
        self.data_dir = self.config['data']['dir_name']
        self.language_code = self.config['language']['code']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16).to(self.device)

    def llama_generate(self, input_text, max_length=500):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        outputs = self.model.generate(**inputs, max_length=max_length, early_stopping=True, num_return_sequences=1, top_p=0.95, top_k=50)
        # outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
        del inputs
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def llama_hidden_states(self, input_text, max_length=500):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        outputs = self.model(**inputs, max_length=max_length, output_hidden_states=True)
        del inputs
        return outputs.hidden_states

    def llama_attentions(self, input_text, max_length=500):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        outputs = self.model(**inputs, max_length=max_length, output_attentions=True)
        # outputs = model(**inputs, max_length=max_length, attn_implementation="eager")
        del inputs
        return outputs.attentions

    def evaluate_grammatical_properties(self, dialogue):

        evaluation = f"Evaluating Dialogue: {dialogue}"
        # print(evaluation)

        hidden_states = self.llama_hidden_states(evaluation)
        attentions = self.llama_attentions(evaluation)

        properties = {}
        # Reference Words
        input_text = f"Resolve the references in this dialogue:\n{dialogue}\nProvide the resolved text."
        properties["Reference Words"] = self.llama_generate(input_text)

        # Noun & Verb Collocations
        input_text = f"Check the noun and verb collocations in this sentence:\n{dialogue}\nProvide the list."
        properties["Noun & Verb Collocations"] = self.llama_generate(input_text)

        # Numbers Agreement
        input_text = f"Analyze the sentence for number agreement:\n{dialogue}"
        properties["Numbers Agreement"] = self.llama_generate(input_text)

        # Tense Agreement
        input_text = f"Check the verb tense alignment in this dialogue:\n{dialogue}"
        properties["Tense Agreement"] = self.llama_generate(input_text)

        # Subject-Verb Agreement
        input_text = f"Check subject-verb agreement in this dialogue:\n{dialogue}"
        properties["Subject-Verb Agreement"] = self.llama_generate(input_text)

        # Speech Acts
        input_text = f"Classify the speech act in this dialogue:\n{dialogue}"
        properties["Speech Acts"] = self.llama_generate(input_text)

        # Modal Verbs and Expressions
        input_text = f"Analyze modal verbs in this dialogue:\n{dialogue}"
        properties["Modal Verbs"] = self.llama_generate(input_text)

        # Quantifiers and Numerals
        input_text = f"Check quantifiers and numerals in this dialogue:\n{dialogue}"
        properties["Quantifiers and Numerals"] = self.llama_generate(input_text)

        return hidden_states, attentions, properties


if __name__ == '__main__':
    Bilingual_config_file = "lib/llama-3-8b-japanese.yaml"
    Bilingual_Model = Lingual_Features(Bilingual_config_file)
    data_dir = Bilingual_Model.data_dir
    language_code = Bilingual_Model.language_code
    Bi_Results = []

    for item in sorted(os.listdir(data_dir)):
        samples_path = os.path.join(data_dir, item)
        for file_name in sorted(os.listdir(samples_path)):
            if file_name.endswith('.txt') and language_code in file_name:
                file_path = os.path.join(samples_path, file_name)
                txt_name = os.path.splitext(file_name)
                with (open(file_path, 'r', encoding='utf-8') as file):
                    sample = file.read()
                    hidden_states, attentions, properties = Bilingual_Model.evaluate_grammatical_properties(sample)
                    Bi_Results.append({"Dialogue": txt_name,
                                       "Properties": properties,
                                       "Hidden_states": hidden_states,
                                       "Attentions": attentions})
                    print("-----")
            torch.cuda.empty_cache()
