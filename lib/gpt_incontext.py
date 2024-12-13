import openai
import os
import argparse


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


def request(output_dir_path, system_instruction, assist_instruction, topic_list, repeat_id=0):  # parameters
    for i in range(len(topic_list)):
        for j in range(len(topic_list[i]["prompts"])):
            messages = format_prompt(system_instruction, assist_instruction, topic_list[i]["prompts"][j])
            response = openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=1,
                max_tokens=2000
            )
            ans_model = response.choices[0].message.content
            ans_model = ans_model[3:-3].strip()
            ans_model = ans_model.lstrip('json').strip()
            file_save_path = os.path.join(output_dir_path, f"{topic_list[i]["category"]}_{j+repeat_id*3}.txt")
            with open(file_save_path, "w") as file:
                file.write(ans_model)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Set paths and API credentials for GPT generation.")
    parser.add_argument("--context_instruction_dir_path", required=False, default="instructions/context_instructions",
                        help="Path to the context instruction directory")
    parser.add_argument("--generation_instruction_dir_path", required=False, default="instructions/generation_instructions",
                        help="Path to the generation instruction directory")
    parser.add_argument("--output_dir", required=True, default="../gpt_generation_data/data_0", help="Path to the output directory")
    parser.add_argument("--openai_api_key", required=True, help="OpenAI API key")
    parser.add_argument("--openai_org", required=False, help="OpenAI organization ID")

    args = parser.parse_args()

    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    openai.organization = args.openai_org
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

    new_topic_list = [
    {
      "category": "Daily Life",
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
      "category": "Travel and Transportation",
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
      "category": "Health and Wellness",
      "prompts": [
        "Describe symptoms to a doctor during a visit.",
        "Give advice to a friend about staying healthy.",
        "Discuss exercise routines and fitness goals with a trainer."
      ]
    },
    {
      "category": "Social Interactions",
      "prompts": [
        "Introduce yourself to someone at a party.",
        "Plan a birthday surprise for a friend with your group.",
        "Role-play a phone conversation with a distant relative."
      ]
    },
    {
      "category": "Problem-Solving",
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
      "category": "Cultural Exchange",
      "prompts": [
        "Explain a tradition or festival from your country to a foreigner.",
        "Discuss cultural differences in dining etiquette with a friend.",
        "Ask a coworker about their favorite holiday in their country."
      ]
    },
          {
      "category": "Daily Conversations",
      "prompts": [
        "Talk about your favorite hobby and why you enjoy it.",
        "Discuss your plans for the weekend with a friend.",
        "Describe a recent memorable event and how it made you feel."
      ]
    },
    {
      "category": "Travel and Exploration",
      "prompts": [
        "Plan a trip to a new city with a group of friends.",
        "Describe your favorite place to visit and why it’s special.",
        "Ask for directions to a landmark in a new town."
      ]
    },
    {
      "category": "Workplace Interactions",
      "prompts": [
        "Discuss project deadlines and tasks with a coworker.",
        "Explain a process or task to a new team member.",
        "Plan a team meeting and decide on the agenda."
      ]
    },
    {
      "category": "Educational Settings",
      "prompts": [
        "Ask a teacher for clarification on an assignment.",
        "Discuss an upcoming test with a classmate and share study tips.",
        "Explain a challenging topic to a study group member."
      ]
    },
    {
      "category": "Shopping and Services",
      "prompts": [
        "Ask a shopkeeper for recommendations on a product.",
        "Discuss the features of an item before deciding to buy it.",
        "Explain an issue with a recent purchase to customer service."
      ]
    },
    {
      "category": "Health and Wellness",
      "prompts": [
        "Describe symptoms to a doctor during a consultation.",
        "Discuss exercise routines and goals with a fitness trainer.",
        "Give advice to a friend on staying healthy and active."
      ]
    },
    {
      "category": "Social Events and Gatherings",
      "prompts": [
        "Introduce yourself to someone new at a party.",
        "Plan a surprise party for a friend with a group.",
        "Talk about shared interests with someone you just met."
      ]
    },
    {
      "category": "Problem-Solving and Conflict Resolution",
      "prompts": [
        "Resolve a misunderstanding with a friend through discussion.",
        "Discuss ways to fix a technical issue with a service provider.",
        "Plan steps to address a shared problem with a coworker."
      ]
    },
    {
      "category": "Cultural Exchange and Traditions",
      "prompts": [
        "Describe a tradition or festival from your culture to a friend.",
        "Discuss differences in dining etiquette with someone from another culture.",
        "Plan a cultural event to share your traditions with others."
      ]
    },
    {
      "category": "Personal Goals and Experiences",
      "prompts": [
        "Talk about your goals for the next year and how you plan to achieve them.",
        "Describe a skill you want to learn and why it’s important to you.",
        "Discuss a recent personal achievement and what it means to you."
      ]
    }]

    context_instruction_dir_path = args.context_instruction_dir_path
    generation_instruction_dir_path = args.generation_instruction_dir_path
    output_dir = args.output_dir

    for context_instruction, generation_instruction in zip(sorted(os.listdir(context_instruction_dir_path)), sorted(os.listdir(generation_instruction_dir_path))):
        language = context_instruction.strip(".txt")
        context_instruction_path = os.path.join(context_instruction_dir_path, context_instruction)
        generation_instruction_path = os.path.join(generation_instruction_dir_path, generation_instruction)
        system_instruction = read_instruction(context_instruction_path)
        instruction = read_instruction(generation_instruction_path)
        output_dir_path = output_dir + f"{language}_dialog"
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        request(output_dir_path, system_instruction, instruction, new_topic_list)