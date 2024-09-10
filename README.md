# Multi-Stage-Queries-with-MRL
This project implements a Agentic RAG application using  `LangGraph` and `Qdrant`. The embeddings are stored and queried using the [Qdrant](https://qdrant.tech/) vector database. To learn more about the project please refer this [article](j).

![Alt Text - description of the image](https://github.com/vansh-khaneja/RAG-using-LangGraph-Agents/blob/main/workflow.png?raw=true)


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Execution](#execution)
- [Contact](#contact)

## Introduction

In this project we are building a RAG application that uses agents to answer the question based on the query given by the user.

## Features

- Fast and efficient way for data retrieval
- Wide queries support
- Multi agentic RAG
- Scalable and high-performance retrieval system

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/vansh-khaneja/RAG-using-LangGraph-Agents
    cd RAG-using-LangGraph-Agents
    ```

2. Set up the Python environment and install dependencies:

    ```sh
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. Set up Qdrant:

    Follow the [Qdrant documentation](https://qdrant.tech/documentation/) to install and configure Qdrant on your system.

## Execution

1.Download the dataset for this project [here](https://www.kaggle.com/datasets/iamsouravbanerjee/airline-dataset) or you can try with your own dataset. Just change the path of the file here.

```sh
    file_path = '/content/Airline Dataset.csv'
```


2.Execute the ```main.py``` file by running this command in terminal.

```sh
    python main.py
```


## Contact

For any questions or issues, feel free to open an issue on this repository or contact me at vanshkhaneja2004@gmail.com.

Happy coding!
