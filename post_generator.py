from llm_helper import llm
from few_shot import FewShotPosts
from evaluator import safe_generate_post

few_shot = FewShotPosts()


def get_length_str(length):
    if length == "Short":
        return "1 to 5 lines"
    if length == "Medium":
        return "6 to 10 lines"
    if length == "Long":
        return "11 to 15 lines"


def generate_post(length, language, tag):
    prompt = get_prompt(length, language, tag)
    response = llm.invoke(prompt)
    return response.content

def generate_safe_post(length, language, tag):
    """
    Generates a LinkedIn post with specified length, language, and tag.
    Applies guardrails and evaluation checks to ensure content quality and safety.
    
    Parameters:
        length (str): Desired length of the post ('Short', 'Medium', 'Long').
        language (str): Language of the post ('English', 'Tanglish').
        tag (str): Topic tag for the post.
    
    Returns:
        tuple: Generated post content and evaluation data.
    """
    return safe_generate_post(length, language, tag, generate_post)


def get_prompt(length, language, tag):
    length_str = get_length_str(length)

    prompt = f'''
    Generate a LinkedIn post with the following structure:

    1. **Hook**: Start with a interative quote or statement.
    2. **Body**: Present the main points in short paragraphs or bullet points.
    3. **Call-to-Action (CTA)**: End with a question or prompt to encourage engagement.
    4. **Hashtags**: Include relevant hashtags at the end atleast 5.
    5. **Tone**: Professional yet approachable, suitable for LinkedIn.

    Guidelines:
    - Topic: {tag}
    - Length: {length_str}
    - Language: {language}
    - If Language is Tanglish, use a mix of Tamil and English, but write in the English script.
    - Use emojis where appropriate to enhance engagement.
    - Ensure the post is formatted for readability on LinkedIn.
    - Aim for approximately 500 - 700 words for Medium posts and 700 - 1000 words for Long posts.

    '''

    examples = few_shot.get_filtered_posts(length, language, tag)

    if examples:
        prompt += "\nHere are some examples to guide the tone and style:\n"
        for i, post in enumerate(examples[:2]):
            prompt += f"\nExample {i+1}:\n{post['text']}\n"

    return prompt



if __name__ == "__main__":
    print(generate_post("Medium", "English", "Mental Health"))