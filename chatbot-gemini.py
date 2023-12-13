import streamlit as st
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize Vertex AI
vertexai.init(project="static-mediator-380708")

# Setting page title and header
st.set_page_config(page_title="Gemini Pro", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>Gemini Pro Chatbot</h1>", unsafe_allow_html=True)

# Load chat model
@st.cache_resource
def load_model():
    model = GenerativeModel("gemini-pro")
    return model

chat_model = load_model()

# Initialise session state variables
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.sidebar.title("Sidebar")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Reset conversation
if clear_button:
    st.session_state['messages'] = []

# Display previous messages
for message in st.session_state['messages']:
    role = message["role"]
    content = message["content"]
    with st.chat_message(role):
        st.markdown(content)

# Chat input
prompt = st.chat_input("You:")
if prompt:
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response from the new model
    model_response = chat_model.generate_content(prompt, generation_config={"temperature": 0.1})
    text_content = model_response.candidates[0].content.parts[0].text

    st.session_state['messages'].append({"role": "assistant", "content": text_content})
    with st.chat_message("assistant"):
        st.markdown(text_content)
