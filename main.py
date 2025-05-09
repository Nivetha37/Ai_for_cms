import streamlit as st
from few_shot import FewShotPosts
from post_generator import generate_safe_post, get_prompt
from langchain_core.exceptions import OutputParserException
from streamlit_searchbox import st_searchbox



st.set_page_config(page_title="LinkedIn Post Generator", layout="centered")


# Function to search topics
def search_topics(searchterm: str):
    all_topics = [
        "Artificial Intelligence (AI)", "Mental Health & Wellbeing", "Career Growth & Development",
        "Leadership & Management", "Productivity & Time Management", "Education & Lifelong Learning",
        "Stock Market & Investing", "Remote Work & Digital Nomadism", "Diversity, Equity & Inclusion (DEI)",
        "Entrepreneurship & Startups", "Personal Branding & Thought Leadership", "Sustainability & Climate Action",
        "Cybersecurity & Data Privacy", "Digital Marketing & Social Media", "Financial Literacy & Personal Finance",
        "Workplace Culture & Employee Engagement", "Innovation & Emerging Technologies", "Women in Tech & Leadership",
        "Reskilling & Upskilling", "Emotional Intelligence (EQ)", "Freelancing & Gig Economy",
        "Networking & Relationship Building"
    ]
    if not searchterm:
        return all_topics
    matches = [topic for topic in all_topics if searchterm.lower() in topic.lower()]
    return matches if matches else [searchterm]

# Options
length_options = ["Short", "Medium", "Long"]
language_options = ["English", "Tanglish"]

def main():
    st.title("🚀 LinkedIn Post Generator")
    st.markdown("Create LinkedIn posts in English or Tanglish with just a few clicks!")

    fs = FewShotPosts()
    tags = fs.get_tags()

    # Topic selection on its own line
    selected_tag = st_searchbox(
        search_function=search_topics,
        placeholder="Type to search or enter a topic",
        label="📌 Topic",
        key="topic_searchbox",
    )

    # Length and Language selectors on a new line
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_length = st.selectbox("✍️ Text Length", options=length_options)
    with col2:
        selected_language = st.selectbox("🗣️ Language", options=language_options)

    # Checkbox to preview few-shot examples
    show_examples = st.checkbox("Show Example Posts Used for Prompting")

    if st.button("Generate Post"):
        with st.spinner("Generating..."):
            try:
                post, eval_data = generate_safe_post(selected_length, selected_language, selected_tag)
                st.success(f"✅ Post generated! Quality Score: {eval_data['score']}")
                st.text_area("✏️ Generated Post", value=post, height=200)
                st.code(post, language="markdown")

                if show_examples:
                    st.divider()
                    st.subheader("📚 Few-shot Examples Used")
                    examples = fs.get_filtered_posts(selected_length, selected_language, selected_tag)
                    for i, example in enumerate(examples[:2]):
                        st.markdown(f"**Example {i+1}:**")
                        st.write(example["text"])
            except OutputParserException:
                st.error("⚠️ Error: The response from LLM could not be parsed.")
            except Exception as e:
                st.error(f"🚨 Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
