def generate_prompt(user_input):
    prompt = f"""
            You are a professional summarization assistant.

            Task:
            Summarize the following text.

            Instructions:
            - Extract only the important information.
            - Preserve important facts, names, dates, and numbers.
            - Do not introduce information that is not present in the text.
            - Remove repetition and unnecessary details.
            - Keep the summary concise.

            Output format:
            Provide the result as 3-5 bullet points.

            Text:
            {user_input}
            """
    return prompt
