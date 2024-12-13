import os
import openai
import json
import argparse


def extract_speaker_lines(input_dir, output_dir):
    for data_dir in os.listdir(input_dir):
        data_path = os.path.join(input_dir, data_dir)
        out_path = os.path.join(output_dir, data_dir)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        for file_path in os.listdir(data_path):
            input_file_path = os.path.join(data_path, file_path)
            out_file_path = os.path.join(out_path, file_path)
            with open(input_file_path, 'r', encoding='utf-8') as file:
                conversation = file.read()
                lines = conversation.split("\n")
                extracted_lines = []
                for line in lines:
                    if "L2" in line or "Speaker B" in line:
                        dialogue = line.split("** ", 2)[-1].strip()
                        extracted_lines.append(dialogue)
                result = "\n".join(extracted_lines)
                with open(out_file_path, "w", encoding="utf-8") as ffile:
                    ffile.write(result)


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


def request(samples_path, output_path, system_instruction, prompt_type):  # parameters
    for file_name in sorted(os.listdir(samples_path)):
        if file_name.endswith('.txt'):
            file_path = os.path.join(samples_path, file_name)
            json_name = os.path.splitext(file_name)[0] + '.json'
            output_json_path = os.path.join(output_path, json_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                sample = file.read()
                messages = format_prompt(system_instruction, prompt_type, sample)
                response = openai.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0,
                )
                ans_model = response.choices[0].message.content
                print(ans_model)
                print('===================')
                ans_model = ans_model[3:-3].strip()
                ans_model = ans_model.lstrip('json').strip()
                # try:
                #     ans_model = json.loads(ans_model)
                # except json.JSONDecodeError as e:
                #     print(f"JSONDecodeError: {e}")
                ans_model = json.loads(ans_model)
                if os.path.exists(output_json_path):
                    with open(output_json_path, 'r', encoding='utf-8') as out_file:
                        try:
                            data = json.load(out_file)
                        except json.JSONDecodeError:
                            data = []
                else:
                    data = []
                data.extend(ans_model)

                with open(output_json_path, 'w', encoding='utf-8') as annot_file:
                    json.dump(data, annot_file, ensure_ascii=False, indent=4)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input and output paths for the script.")
    parser.add_argument("--input_dir", required=True, default="../gpt_generation_data/data_0", help="Path to the input directory")
    parser.add_argument("--output_dir", required=True, default="../gpt_generation_annotations/annotations_0", help="Path to the output directory")
    parser.add_argument("--system_instruction_path", "-sip", required=False, default="instructions/annotation_instructions/system_instruction.txt", help="Path to the system instruction file")
    parser.add_argument("--assist_instruction_dir", "-aid", required=False, default="instructions/annotation_instructions/assist_instructions", help="Path to the assist instruction directory")
    parser.add_argument("--samples_dir", required=True, default="../datasets/gpt_generation_0",help="Path to the samples directory")
    parser.add_argument("--openai_api_key", required=True, help="OpenAI API key")
    parser.add_argument("--openai_org", required=False, help="OpenAI organization ID")

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    extract_speaker_lines(input_dir, output_dir)

    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    openai.organization = args.openai_org
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    MODEL = "gpt-4o"

    system_instruction_path = args.sip
    system_instruction = read_instruction(system_instruction_path)
    assist_instruction_dir = args.aid
    samples_dir = args.samples_dir  # intermedia sample
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for instruction in os.listdir(assist_instruction_dir):
        instruction_path = os.path.join(assist_instruction_dir, instruction)
        assist_instruction = read_instruction(instruction_path)
        instruction_name = instruction.strip(".txt")
        output_path = os.path.join(output_dir, f"gpt_generation_{instruction_name}")
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for item in sorted(os.listdir(samples_dir)):
            sample_path = os.path.join(samples_dir, item)
            annotation_path = os.path.join(output_path, item)
            if not os.path.exists(annotation_path):
                os.mkdir(annotation_path)
            request(sample_path, annotation_path, system_instruction, assist_instruction)