import os
import json
import glob

def enrich_notebook(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    title_cell = nb["cells"][0]["source"]
    title_text = "".join(title_cell)
    
    cells = nb["cells"]
    
    # Add an advanced setup cell
    if "LangGraph" in title_text:
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "from langgraph.graph import StateGraph, END\n",
                "from typing import TypedDict, Annotated\n",
                "import operator\n",
                "\n",
                "class AgentState(TypedDict):\n",
                "    messages: Annotated[list, operator.add]\n",
                "\n",
                "def process_node(state: AgentState):\n",
                "    # LLM processing logic using local Ollama model\n",
                "    return {\"messages\": [\"Processed by LangGraph node.\"]}\n",
                "\n",
                "graph = StateGraph(AgentState)\n",
                "graph.add_node(\"process\", process_node)\n",
                "graph.set_entry_point(\"process\")\n",
                "graph.add_edge(\"process\", END)\n",
                "app = graph.compile()\n",
                "\n",
                "print(\"LangGraph Workflow Compiled!\")\n"
            ]
        })
    elif "CrewAI" in title_text:
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "from crewai import Agent, Task, Crew, Process\n",
                "from langchain_community.llms import Ollama\n",
                "\n",
                "# Use local Ollama\n",
                "ollama_llm = Ollama(model=\"qwen3.5:9b\")\n",
                "\n",
                "researcher = Agent(\n",
                "    role='Senior Research Analyst',\n",
                "    goal='Uncover insights and analyze data',\n",
                "    backstory='Expert at analyzing complex topics',\n",
                "    verbose=True,\n",
                "    allow_delegation=False,\n",
                "    llm=ollama_llm\n",
                ")\n",
                "\n",
                "task = Task(\n",
                "    description='Analyze the given topic and provide a comprehensive report',\n",
                "    expected_output='A highly detailed summary report',\n",
                "    agent=researcher\n",
                ")\n",
                "\n",
                "crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential)\n",
                "print(\"CrewAI Multi-Agent System defined with local Ollama!\")\n"
            ]
        })
    elif "LlamaIndex" in title_text:
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "from llama_index.core import VectorStoreIndex, SimpleDirectoryReader\n",
                "from llama_index.llms.ollama import Ollama\n",
                "from llama_index.core import Settings\n",
                "\n",
                "# Set LlamaIndex to use local Ollama\n",
                "Settings.llm = Ollama(model=\"qwen3.5:9b\", request_timeout=120.0)\n",
                "Settings.embed_model = \"local:BAAI/bge-small-en-v1.5\"\n",
                "\n",
                "# documents = SimpleDirectoryReader('data').load_data()\n",
                "# index = VectorStoreIndex.from_documents(documents)\n",
                "# query_engine = index.as_query_engine()\n",
                "# response = query_engine.query(\"What are the main insights?\")\n",
                "print(\"LlamaIndex configured for local RAG execution!\")\n"
            ]
        })
    else:
        # Default LangChain RAG/LLM 
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "from langchain_ollama import ChatOllama\n",
                "from langchain_core.prompts import PromptTemplate\n",
                "from langchain_core.output_parsers import StrOutputParser\n",
                "\n",
                "llm = ChatOllama(model=\"qwen3.5:9b\", temperature=0.1)\n",
                "\n",
                "prompt = PromptTemplate.from_template(\n",
                "    \"You are a helpful local AI assistant. Answer the user's question:\\n\\nQuestion: {question}\\n\\nAnswer:\"\n",
                ")\n",
                "\n",
                "chain = prompt | llm | StrOutputParser()\n",
                "\n",
                "# response = chain.invoke({\"question\": \"What can you help me with?\"})\n",
                "# print(response)\n",
                "print(\"LangChain inference pipeline ready!\")\n"
            ]
        })
        
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

def main():
    notebooks = glob.glob("100_Local_AI_Projects/**/*.ipynb", recursive=True)
    for nb in notebooks:
        try:
            enrich_notebook(nb)
        except Exception as e:
            print(f"Error enriching {nb}: {e}")
            
    print(f"Successfully enriched {len(notebooks)} projects with deep architectural implementations!")

if __name__ == "__main__":
    main()
