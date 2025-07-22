import asyncio
import logging
import os
import json
from typing import Dict, Any, List, Callable, Optional
from typing_extensions import TypedDict

import httpx
from openai import AsyncOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_together import ChatTogether



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# model = "gpt-3.5-turbo"
# model =  "gemini-2.0-flash"
# model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# Define the state
class State(TypedDict):
    messages: List[Any]

# LangGraph ChatBot class
class LangGraphChatBot:
    def __init__(self, system_message: str, tools_schema: List[Dict], tool_functions: Dict[str, Callable], model: str):
        self.system_message = system_message
        self.tools_schema = tools_schema
        self.tool_functions = tool_functions
        self.exclude_functions = ["plot_chart"]
        self.model = model
        self.steps = []

        logger.info(f"Initializing LangGraphChatBot with model: {model}")
        self.llm = self._initialize_llm(model)

        self.graph = self._create_graph()
        try:
            graph_png = self.graph.get_graph().draw_mermaid_png()
            with open("graph.png", "wb") as f:
                f.write(graph_png)
            logger.info("Graph visualization saved as graph.png")
        except Exception as e:
            logger.warning(f"Could not save graph visualization: {e}")

        self.conversation_history = []
        if system_message:
            self.conversation_history.append(SystemMessage(content=system_message))
        logger.info("Conversation history initialized.")

    def clear_conversation_history(self):
        logger.info("Clearing conversation history.")
        self.conversation_history = []
        self.steps = []
        if self.system_message:
            self.conversation_history.append(SystemMessage(content=self.system_message))

    def _convert_tools_to_langchain(self):
        """Convert OpenAI tools format to LangChain format."""
        langchain_tools = []
        for tool in self.tools_schema:
            if tool["type"] == "function":
                func_info = tool["function"]
                langchain_tools.append({
                    "name": func_info["name"],
                    "description": func_info["description"],
                    "parameters": func_info["parameters"]
                })
        logger.debug(f"Converted {len(langchain_tools)} tools for LangChain.")
        return langchain_tools

    def _initialize_llm(self, model: str):
        """Initialize the LLM based on the model name/prefix."""
        logger.info(f"Selecting LLM provider for model: {model}")
        if model.startswith("gpt-") or model.startswith("openai"):
            logger.info("Using OpenAI provider.")
            return ChatOpenAI(
                model=model,
                api_key=os.environ.get("OPENAI_API_KEY"),
                temperature=0
            ).bind_tools(self._convert_tools_to_langchain())
        elif model.startswith("gemini") or model.startswith("google"):
            logger.info("Using Google Gemini provider.")
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=os.environ.get("GOOGLE_API_KEY"),
                temperature=0,
            ).bind_tools(self._convert_tools_to_langchain())
        elif model.startswith("meta-") or model.startswith("llama") or model.startswith("together"):
            logger.info("Using Together provider.")
            return ChatTogether(
                model=model,
                together_api_key=os.environ.get("TOGETHER_API_KEY"),
                temperature=0,
            ).bind_tools(self._convert_tools_to_langchain())
        else:
            logger.error(f"Unknown or unsupported model: {model}")
            raise ValueError(f"Unknown or unsupported model: {model}")

    def _create_graph(self):
        workflow = StateGraph(State)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self._call_tools)

        def should_continue(state):
            messages = state["messages"]
            if not messages:
                logger.debug("No messages in state; ending workflow.")
                return END
            last_message = messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                logger.debug("Tool calls detected; routing to tools node.")
                return "tools"
            logger.debug("No tool calls; ending workflow.")
            return END

        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")
        workflow.set_entry_point("agent")
        logger.info("LangGraph workflow created and compiled.")
        return workflow.compile()

    async def _call_model(self, state: State):
        messages = state["messages"]
        all_messages = self.conversation_history + messages
        logger.info(f"Calling model with {len(all_messages)} messages.")
        try:
            response = await self.llm.ainvoke(all_messages)
            result = {"messages": [response]}
            self.steps.append(response)
            logger.debug("Model call successful.")
            return result
        except Exception as e:
            logger.error(f"Error during model call: {e}")
            raise

    async def _call_tools(self, state: State):
        messages = state["messages"]
        last_message = messages[-1]
        tool_responses = []
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                function_name = tool_call["name"]
                function_args = tool_call["args"]
                if function_name in self.tool_functions:
                    logger.info(f"Calling tool function '{function_name}' with args: {function_args}")
                    try:
                        function_response = await self.tool_functions[function_name](**function_args)
                        tool_responses.append(
                            ToolMessage(
                                content=function_response,
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"]
                            )
                        )
                        logger.debug(f"Tool '{function_name}' executed successfully.")
                    except Exception as e:
                        logger.error(f"Error calling tool '{function_name}': {e}")
                        tool_responses.append(
                            ToolMessage(
                                content=f"Error: {str(e)}",
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"]
                            )
                        )
        logger.info(f"Tool responses generated: {len(tool_responses)}")
        result = {"messages": [last_message, *tool_responses]}
        self.steps.extend(tool_responses)
        return result

    async def __call__(self, message: str):
        user_message = HumanMessage(content=message)
        current_state = {"messages": [user_message]}
        logger.info(f"Processing user message: {message}")
        try:
            final_state = await self.graph.ainvoke(current_state)
            logger.info(f"Graph execution complete. Steps: {len(self.steps)}")
            self.conversation_history.extend(final_state["messages"])
            assistant_message = None
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage):
                    assistant_message = msg
                    break
            logger.info(f"Assistant response: {assistant_message.content if assistant_message else 'No response'}")
            return assistant_message, self.steps
        except Exception as e:
            logger.error(f"Error during conversation: {e}")
            raise

    async def call_functions(self, tool_calls):
        """Legacy method for compatibility with existing code."""
        function_responses = []
        tool_messages = []
        for tool_call in tool_calls:
            function_name = tool_call["name"]
            function_args = tool_call["args"]
            if function_name in self.tool_functions:
                logger.info(f"Calling legacy tool function '{function_name}' with args: {function_args}")
                try:
                    function_response = await self.tool_functions[function_name](**function_args)
                    tool_message = ToolMessage(
                        content=str(function_response),
                        tool_call_id=tool_call["id"]
                    )
                    tool_messages.append(tool_message)
                    response_obj = {
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                    function_responses.append(response_obj)
                    logger.debug(f"Legacy tool '{function_name}' executed successfully.")
                except Exception as e:
                    logger.error(f"Error calling legacy tool '{function_name}': {e}")
                    tool_message = ToolMessage(
                        content=f"Error: {str(e)}",
                        tool_call_id=tool_call["id"]
                    )
                    tool_messages.append(tool_message)
                    error_response = {
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "name": function_name,
                        "content": f"Error: {str(e)}",
                    }
                    function_responses.append(error_response)
        self.conversation_history.extend(tool_messages)
        try:
            response_message = await self.llm.ainvoke(self.conversation_history)
            self.conversation_history.append(response_message)
            logger.info("Legacy tool call completed and response appended to history.")
            return response_message, function_responses
        except Exception as e:
            logger.error(f"Error during legacy tool model call: {e}")
            raise
