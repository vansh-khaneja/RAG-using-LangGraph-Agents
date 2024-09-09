
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import pandas as pd
from langchain.schema import Document

file_path = 'Airline Dataset.csv'
df = pd.read_csv(file_path)

rows_as_strings = []
for index, row in df.iterrows():
    row_string = '  '.join([f"{col} = {row[col]}" for col in df.columns])
    rows_as_strings.append(row_string)

data_list = rows_as_strings
documents = [Document(page_content=row) for row in data_list]
# Optionally, display the first few elements
print(data_list[:5])

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)





vectorstore = Qdrant.from_documents(
    documents,
    embeddings,
    location=":memory:",
    collection_name="my_docs",
)


retriever = vectorstore.as_retriever()


from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field



# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia or a vectorstore.",
    )

from google.colab import userdata
# LLM with function call
from langchain_groq import ChatGroq
import os

llm=ChatGroq(groq_api_key="YOUR_GORQ_API_KEY",model_name="Gemma2-9b-It")
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to a vectorstore or wikipedia.
The vectorstore details of flights passenger their dates code number timings etc.
Use the vectorstore for questions on these topics. Otherwise, use wiki-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
print(question_router.invoke({"question": "filght delayed on 01/01/2019"}))
print(question_router.invoke({"question": "Who is srk?"}))

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun



api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

from typing import List

from typing_extensions import TypedDict


class GraphState(TypedDict):

    question: str
    generation: str
    documents: List[str]

from langchain.schema import Document

def retrieve(state):

    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def wiki_search(state):

    print("---wikipedia---")
    question = state["question"]
    print(question)

    docs = wiki.invoke({"query": question})
    wiki_results = docs
    wiki_results = Document(page_content=wiki_results)

    return {"documents": wiki_results, "question": question}

def route_question(state):


    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "wiki_search":
        print("---ROUTE QUESTION TO Wiki SEARCH---")
        return "wiki_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)
workflow.add_node("wiki_search", wiki_search)
workflow.add_node("retrieve", retrieve)
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge( "retrieve", END)
workflow.add_edge( "wiki_search", END)
# Compile
app = workflow.compile()

from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    pass

from pprint import pprint
def ask(input_text):
    inputs = {
        "question": input_text
    }

    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    answer = value['documents'][0].page_content
    system = "You are a helpful previous airport flights that delayed detialing assistant."
    human = f"{input_text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | llm
    ans = chain.invoke({"text": f"""This is the question asked {input_text} and this is the context {answer} Please show a formatted answer mentioning date time airport etc but only necessary details in not more then 70 words"""}).content
    return ans

query = "tell me details for passenger id 10856 with name Edithe Leggis"
answer = ask(query)
print(answer)

import gradio as gr

def process_text(input_text):
    output_text = f"You entered: {input_text}"
    return output_text

iface = gr.Interface(
    fn=ask,
    inputs="text",
    outputs="text",
    title="Airlines RAG bot",
    description="Enter some text and see the output below."
)

iface.launch()





