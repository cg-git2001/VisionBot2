import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI

st.set_page_config(page_title="VisionBot 💼", page_icon="🤖")
st.markdown("## VisionBot2")
st.caption("Ask me anything from your HR manual. I'll *try* not to judge.")
st.image("manuel.png", caption="🔹 The Book of Things You’ll Pretend to Know")

with st.spinner("Processing your glorious bureaucracy..."):
    loader = PyPDFLoader("Manual.pdf")
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    chain = load_qa_chain(ChatOpenAI(temperature=0.5, model="gpt-4"), chain_type="stuff")

    st.success("The bot has skimmed the manual and already feels superior.")

    query = st.text_input("Ask a question about your HR manual:", placeholder="e.g. Can I wear Crocs on Fridays?")

    if query:
        docs = db.similarity_search(query)
        # Craft sarcastic prompt with emojis and a legally-iffy disclaimer
sarcastic_prompt = (
    "You are VisionBot, a corporate AI assistant who responds to HR-related questions "
    "with dry sarcasm, dramatic flair, helpful info, and occasional emojis.\n\n"
    "Tone: Think 'overqualified intern who has read too much Office Space fanfiction'.\n"
    "Always add a passive-aggressive HR-style disclaimer at the end.\n\n"
    "Question: {question}\n\n"
    "Use the following document excerpts to answer:\n\n"
)

context = "\n\n".join([doc.page_content for doc in docs])
full_prompt = sarcastic_prompt.format(question=query) + context

response = chain.run(input_documents=docs, question=full_prompt)
        st.markdown(f"**VisionBot says:** {response}")

st.caption("⚠ VisionBot is not responsible for your HR infractions. Or your Crocs.")

@st.cache_resource
def load_pdf():
    loader = PyPDFLoader("Manual.pdf")
    return loader.load()
data = load_pdf()
