from langchain.agents import initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI



openai_api_key = ""
SERPAPI_API_KEY = ""


def run_agent_query(query: str) -> str:
    """Search and llm chain."""
    # Initialize the language model
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", openai_api_key=openai_api_key, temperature=0.3)

    # Load the required tools
    tool_names = ["serpapi"]
    tools = load_tools(tool_names)

    # Initialize the agent with the specified tools and language model
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

    # Run the query using the agent
    response = agent.run(query)

    return response


if __name__ == "__main__":
    query = "What is the temperature today in Erlangen, Germany in celsius?"
    response = run_agent_query(query)
