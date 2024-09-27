
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests

import pandas as pd
from langchain.schema import Document

file_path = '/content/Airline Dataset.csv'
df = pd.read_csv(file_path)

rows_as_strings = []
actual_list = []
for index, row in df.iterrows():
    formatted_row = ""
    for column_name, value in row.items():
        formatted_row += f"{column_name} :  {value}"+" \n"
    row_string = row['First Name']+" " +row['Last Name']
    rows_as_strings.append(row_string)
    actual_list.append(formatted_row)


data_list = rows_as_strings
documents = [Document(page_content=row) for row in data_list]
documents = documents[:50]
# Optionally, display the first few elements
print(actual_list[1])

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)



vectorstore = Qdrant.from_documents(
    documents,
    embeddings,
    location=":memory:",
    collection_name="my_docs",
    ids=[i for i in range(len(documents))]
)

#vectorstore = Qdrant().from_documents(splits, embeddings)

retriever = vectorstore.as_retriever()

data_fetch = retriever.invoke("Please show me the flight details of passenger Dominica Pyle with passenger id 78493")[0]
print(actual_list[data_fetch.metadata["_id"]])

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field



class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "fetchweather"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia or a fetchweather.",
    )

from google.colab import userdata
from langchain_groq import ChatGroq
import os

llm=ChatGroq(groq_api_key="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",model_name="Gemma2-9b-It")
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or fetchweather.
The vectorstore details of flights passenger their dates code number timings etc.
The fetchweather tells abouth the weather condition of particular state use if theres only some country or state name given.
Use the vectorstore for questions on the flight related topics. Otherwise, use fetchweather."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
print(question_router.invoke({"question": "filght delayed on 01/01/2019"}))
print(question_router.invoke({"question": "weather in New Delhi"}))



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
    print(actual_list[data_fetch.metadata["_id"]])

    # Retrieval
    documents = retriever.invoke(question)[0]
    answer = actual_list[documents.metadata["_id"]]
    results = [Document(page_content=answer)]

    return {"documents": results, "question": question}

def fetch_weather(state):

    print("---Fetch_Weather---")
    question = state["question"]

    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": f"you are location extractor from the given query"
        },
        {
            "role": "user",
            "content": f"This is the query {question} only extract the location name from it",
        }
    ],

    model="gemma2-9b-it",
    temperature=0.5,
    )
    location = chat_completion.choices[0].message.content
    print(location)
    api_key = "44282da02f134063973153129242609"
    base_url = "http://api.weatherapi.com/v1/current.json"
    params = {
        'key': api_key,
        'q': "Berlin"
    }

    # Make the GET request to the API
    response = requests.get(base_url, params=params)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Retrieve and format the desired information
        weather_info = f"""Location: {data['location']['name']}, {data['location']['country']}\n
            Temperature (Celsius): {data['current']['temp_c']}°C\n
            Weather Condition: {data['current']['condition']['text']}\n
            Wind Speed (kph): {data['current']['wind_kph']} kph\n
            Dew Point (Celsius): {data['current']['dewpoint_c']}°C\n
            Pressure (mb): {data['current']['pressure_mb']} mb\n"""

    else:
        weather_info = f"Failed to retrieve data: {response.status_code}"

    results = [Document(page_content=weather_info)]

    return {"documents": results, "question": question}

def route_question(state):


    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "fetchweather":
        print("---ROUTE QUESTION TO Wiki SEARCH---")
        return "fetchweather"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)
workflow.add_node("fetch_weather", fetch_weather)
workflow.add_node("retrieve", retrieve)
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "fetchweather": "fetch_weather",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge( "retrieve", END)
workflow.add_edge( "fetch_weather", END)
# Compile
app = workflow.compile()

from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    pass

from pprint import pprint
from groq import Groq

client = Groq(api_key="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")

def ask(input_text):
    inputs = {
        "question": input_text
    }

    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Node '{key}':")

        pprint("\n---\n")
    print(value)
    answer = value['documents'][0].page_content
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": f"you are flight and weather guide"
        },
        {
            "role": "user",
            "content": f"This is the question {input_text} given by user and this the database response {answer} rephrase the answer based on the question and dont start with something heres rephrased answer just give the answer what you formed.",
        }
    ],

    model="gemma2-9b-it",
    temperature=0.5,
    )
    return chat_completion.choices[0].message.content



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






