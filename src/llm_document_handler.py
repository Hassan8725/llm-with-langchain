from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough


openai_api_key = ""


def query_document(data: str, openai_api_key: str) -> object:
    """Query document using Langchain."""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(data)

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_texts(texts, embeddings)
    retriever = db.as_retriever()

    template = """Answer the question based only on the following context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model_name="gpt-3.5-turbo-16k", openai_api_key=openai_api_key, temperature=0.3)

    def format_docs(docs: str) -> str:
        """Format docs from retriever."""
        return "\n\n".join([d.page_content for d in docs])

    chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | model | StrOutputParser()

    return chain


if __name__ == "__main__":
    data = ""

    chain = query_document(data, openai_api_key)
    query = ""
    chain.invoke(query)
