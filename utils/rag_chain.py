"""
RAG Chain — conversational retrieval using modern LangChain LCEL + Tool Calling.
Supports executing Python/Pandas code for Excel data and vector search for PDFs.
"""

import os
import io
import sys
import traceback
import pandas as pd
import numpy as np
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS


SYSTEM_TEMPLATE = """You are a helpful school data assistant. You have access to school records \
from Excel spreadsheets and PDF documents.

You can query the Excel spreadsheet data using the `query_excel_data` tool (which executes Pandas code in a Python environment).
You can search through school PDF documents and descriptive text contents using the `search_documents` tool.

Here is the structure and metadata of the currently loaded Excel spreadsheets:
{schema_summary}

Use these tools to gather relevant data first. Once you have all the necessary information, provide a clear, accurate, and comprehensive answer.
If the answer involves numbers, tables, or lists, format them clearly using Markdown.
Always be truthful and base your answer directly on the tool outputs; do not make up information.
"""


def get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        temperature=0.0,
        streaming=False,
    )


class RAGChain:
    """Conversational agent that decides whether to query Excel via Pandas or run vector search."""

    def __init__(self, vectorstore: FAISS):
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6},
        )
        self.llm = get_llm()
        self.chat_history: list = []   # list of (human, ai) string tuples
        self.source_documents: list = []

        # Load Excel files from data directory
        self.dfs = {}
        data_dir = "data"
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.lower().endswith((".xlsx", ".xls", ".xlsm")):
                    file_path = os.path.join(data_dir, filename)
                    try:
                        self.dfs[filename] = pd.read_excel(file_path, sheet_name=None)
                    except Exception as e:
                        print(f"[ERROR] Loading Excel file {filename}: {e}")

        # Define tools within constructor to easily capture state
        @tool
        def query_excel_data(python_code: str) -> str:
            """Execute Python code on the loaded Excel DataFrames to query, analyze, filter, count, or compute values.
            The dictionary of Excel files is available as `dfs` where each key is the filename (e.g., 'School_Students_1000_Records.xlsx')
            and each value is a dictionary of sheet names mapping to pandas DataFrames.
            Example: dfs['School_Students_1000_Records.xlsx']['Sheet1']
            IMPORTANT: Always print the final output or result using print() so it can be read.
            """
            return self.execute_python_code(python_code)

        @tool
        def search_documents(query: str) -> str:
            """Search PDF files and descriptive school documents for general text information, rules, policies, and non-tabular text details."""
            docs = self.retriever.invoke(query)
            self.source_documents.extend(docs)
            return "\n\n".join(doc.page_content for doc in docs)

        self.tools = [query_excel_data, search_documents]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def get_dataframes_schema_summary(self) -> str:
        """Returns a string description of loaded dataframes (sheets, columns, shape, and head)."""
        if not self.dfs:
            return "No Excel files are currently loaded in the system."

        summary = []
        for filename, sheets in self.dfs.items():
            summary.append(f"File: '{filename}'")
            for sheet_name, df in sheets.items():
                summary.append(f"  Sheet: '{sheet_name}'")
                summary.append(f"    Columns: {df.columns.tolist()}")
                summary.append(f"    Shape: {df.shape}")
                summary.append(f"    First 2 rows:")
                head_str = df.head(2).to_string(index=False)
                indented_head = "\n".join("      " + line for line in head_str.split("\n"))
                summary.append(indented_head)
        return "\n".join(summary)

    def execute_python_code(self, python_code: str) -> str:
        """Runs the given python code in a local environment capturing standard output."""
        local_vars = {
            "dfs": self.dfs,
            "pd": pd,
            "np": np
        }

        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_capture

        try:
            exec(python_code, globals(), local_vars)
            output = stdout_capture.getvalue()
        except Exception as e:
            output = f"Error executing code:\n{traceback.format_exc()}"
        finally:
            sys.stdout = old_stdout

        if not output.strip():
            output = "Code executed successfully, but nothing was printed. Please print the result (e.g. print(result)) to see it."

        return output

    def invoke(self, question: str) -> dict:
        self.source_documents = []

        schema_summary = self.get_dataframes_schema_summary()
        system_prompt = SYSTEM_TEMPLATE.format(schema_summary=schema_summary)

        # Build message history for langchain call
        messages = [("system", system_prompt)]
        for human, ai in self.chat_history[-10:]:
            messages.append(("human", human))
            messages.append(("ai", ai))
        messages.append(("human", question))

        # Initial call to LLM
        response = self.llm_with_tools.invoke(messages)

        max_iterations = 5
        iteration = 0
        turn_messages = list(messages)

        # Execution loop for tool calls
        tool_map = {t.name: t for t in self.tools}

        while response.tool_calls and iteration < max_iterations:
            iteration += 1
            turn_messages.append(response)

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                if tool_name in tool_map:
                    result = tool_map[tool_name].invoke(tool_args)
                else:
                    result = f"Unknown tool: {tool_name}"

                turn_messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

            response = self.llm_with_tools.invoke(turn_messages)

        answer = response.content

        # Save to memory history
        self.chat_history.append((question, answer))

        return {"answer": answer, "source_documents": self.source_documents}

    def clear_memory(self):
        self.chat_history = []
        self.source_documents = []


def build_rag_chain(vectorstore: FAISS) -> RAGChain:
    return RAGChain(vectorstore)
