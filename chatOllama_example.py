from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize the ChatOllama model
model = ChatOllama(model="llama3.2", temperature=0.7)

# Create messages
messages = [
    SystemMessage(content="You are a helpful assistant that summarizes medical reports."),
    HumanMessage(content="Please summarize the following consultation note: Patient presents with persistent cough and fatigue for the past two weeks. No fever or shortness of breath. History of asthma. Recommends chest X-ray and follow-up in one week.")
]

# Generate a response
response = model.invoke(messages)

print(response.content)
