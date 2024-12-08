"""
@Time : 2024/12/6 17:34
@Author : Ray
@Email : 1206953809@qq.com
@File : gpt_incontext.py
@Purpose
"""

import openai
import os


def read_instruction(instruction_path):
    with open(instruction_path, 'r', encoding='utf-8') as file:
        instruction = file.read()
    return instruction


def format_prompt(system_instruction, instruction, input):
    # print(input)
    text = input
    # print(text)
    # print(label)
    message = instruction.format(text=text)
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": message},
    ]
    return messages


def request(output_dir_path, system_instruction, assist_instruction, topic_list):  # parameters
    for i in range(len(topic_list)):
        for j in range(len(topic_list[i]["prompts"])):
            messages = format_prompt(system_instruction, assist_instruction, topic_list[i]["prompts"][j])
            response = openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.8,
                max_tokens=2000
            )
            ans_model = response.choices[0].message.content
            ans_model = ans_model[3:-3].strip()
            ans_model = ans_model.lstrip('json').strip()
            file_save_path = os.path.join(output_dir_path, f"{topic_list[i]["category"]}_{j}.txt")
            with open(file_save_path, "w") as file:
                file.write(ans_model)
    return


if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = ""
    openai.organization = ""
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    MODEL = "gpt-4o"

    topic_list = [
        {
            "category": "Daily_Life",
            "prompts": [
                "Discuss your morning routine with a friend.",
                "Plan a weekend picnic with your family.",
                "Explain how you prepare your favorite meal to a new roommate."
            ]
        },
        {
            "category": "Shopping",
            "prompts": [
                "Ask a shop assistant for help finding a product.",
                "Negotiate a price for an item at a local market.",
                "Discuss a product return policy with a store employee."
            ]
        },
        {
            "category": "Travel_and_Transportation",
            "prompts": [
                "Describe your recent trip to a new city to a friend.",
                "Ask for directions to a nearby hotel in an unfamiliar town.",
                "Plan a group vacation, deciding on the destination and activities."
            ]
        },
        {
            "category": "Education",
            "prompts": [
                "Discuss an upcoming exam with a classmate.",
                "Explain a challenging topic to a peer in your study group.",
                "Ask your teacher for clarification on a homework assignment."
            ]
        },
        {
            "category": "Workplace",
            "prompts": [
                "Discuss a project deadline with your manager.",
                "Role-play a job interview, answering common questions.",
                "Explain a task to a new coworker."
            ]
        },
        {
            "category": "Health_and_Wellness",
            "prompts": [
                "Describe symptoms to a doctor during a visit.",
                "Give advice to a friend about staying healthy.",
                "Discuss exercise routines and fitness goals with a trainer."
            ]
        },
        {
            "category": "Social_Interactions",
            "prompts": [
                "Introduce yourself to someone at a party.",
                "Plan a birthday surprise for a friend with your group.",
                "Role-play a phone conversation with a distant relative."
            ]
        },
        {
            "category": "Problem_Solving",
            "prompts": [
                "Resolve a misunderstanding with a friend.",
                "Request assistance from customer service for a technical issue.",
                "Apologize for a mistake and offer a solution."
            ]
        },
        {
            "category": "Entertainment",
            "prompts": [
                "Discuss your favorite movies and TV shows with a friend.",
                "Plan an evening out, deciding between restaurants and activities.",
                "Recommend a book to someone and explain why they should read it."
            ]
        },
        {
            "category": "Cultural_Exchange",
            "prompts": [
                "Explain a tradition or festival from your country to a foreigner.",
                "Discuss cultural differences in dining etiquette with a friend.",
                "Ask a coworker about their favorite holiday in their country."
            ]
        }
    ]

    context_instruction_dir_path = "instructions/context_instructions"
    generation_instruction_dir_path = "instructions/generation_instructions"

    for context_instruction, generation_instruction in zip(sorted(os.listdir(context_instruction_dir_path)), sorted(os.listdir(generation_instruction_dir_path))):
        language = context_instruction.strip(".txt")
        context_instruction_path = os.path.join(context_instruction_dir_path, context_instruction)
        generation_instruction_path = os.path.join(generation_instruction_dir_path, generation_instruction)
        system_instruction = read_instruction(context_instruction_path)
        instruction = read_instruction(generation_instruction_path)
        output_dir_path = f"../data/{language}_dialog"
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        request(output_dir_path, system_instruction, instruction, topic_list)