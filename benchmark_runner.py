from model_runner import generate_response

def run_prompt_test(file_path):

    results = []

    with open(file_path) as f:
        prompts = f.readlines()

    for prompt in prompts:

        prompt = prompt.strip()

        print("Prompt:", prompt)

        response = generate_response(prompt)

        print("Response:", response)
        print("---------------")

        results.append((prompt, response))

    return results