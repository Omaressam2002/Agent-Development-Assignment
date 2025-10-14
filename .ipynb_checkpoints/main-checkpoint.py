from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, List, Dict
from langchain_groq import ChatGroq
from pydantic import BaseModel, ValidationError, Field
from langchain.prompts import PromptTemplate
from langchain.tools import tool,Tool
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
import os, json, uuid
import argparse

parser = argparse.ArgumentParser(description="Run Shopping Assistant Graph")

parser.add_argument(
    "--user_path",
    type=str,
    default="./user_profile.json",
    help="Path to the user profile JSON file"
)
parser.add_argument(
    "--product_path",
    type=str,
    default="./catalog1.json",
    help="Path to the product catalog JSON file"
)
parser.add_argument(
    "--log_path",
    type=str,
    default="./transaction_history.json",
    help="Path to the log JSON file"
)

parser.add_argument(
    "--verbose",
    type=bool,
    default=False,
    help="Verbose to the Agent"
)
args = parser.parse_args()

user_path = args.user_path
product_path = args.product_path
log_path = args.log_path
verbose = args.verbose

print("User Path:", user_path)
print("Product Path:", product_path)
print("Log Path:", log_path)


class AgentState(TypedDict):
    user_profile_path: str
    product_catalog_path: str
    unique_categories: List[str]
    product_catalogue: List[dict]
    user_profile: dict
    justification: str
    recommended_category: str
    top3_products: List[dict]
    chosen_product: dict
    history_log_file: str
    error: str


groq_api_keys = ["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", "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", "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","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", "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", "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"]



llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.6,
    api_key = groq_api_keys[0]
)



# PROMPTS
category_prompt = PromptTemplate.from_template("""
You are an intelligent shopping recommendation assistant.

You will receive:
1. The user's purchase history as a JSON list.
2. A list of unique product categories available in the catalog.

Your task:
- Analyze the user's purchase history.
- Identify categories the user has **not purchased from** before.
- If all categories have been purchased from, choose one the user has **purchased from the least**.
- Choose **one category** that the user might be most interested in exploring next.
- Generate a **catchy, persuasive justification** in natural language explaining *why* this category fits the user’s preferences or lifestyle.

Return your final answer **strictly** in the following JSON format:

{{
  "recommended_category": "<category_name>",
  "justification": "<short persuasive message>"
}}

### USER HISTORY:
{user_history}

### AVAILABLE CATEGORIES:
{available_categories}

Now respond only with the JSON object described above — no extra text.
""")



recommendation_prompt = PromptTemplate.from_template("""
You are a smart shopping assistant.

You are given:
- The user's purchase history: {user_history}
- A list of available products: {products}
- justification message to convince the user to explore the category : {justification}

Steps to follow:
1. Calculate the user's average spending based on their purchase history.
2. Rank the products based on how close their price is to this average and any inferred preferences of the user.
3. Choose the **top 3 closest products** to recommend.
4. Write a short, friendly **message** summarizing your recommendations for the user and mix it with the justification message and also dont forget to include the price of each product in the message.
5. Also, return a **JSON object** containing the selected products.


⚠️ Important Instructions:
- When you have completed your analysis and are ready to give your final recommendation, you must return your answer exactly in the format below.
- Once you output your final answer, stop reasoning and do not take any further actions.

Your response must be a **valid JSON object only**, with no extra text, markdown, or explanations,
Your response **must follow this exact format**:

Final Answer:
{{
  "average_price": "<computed_average_as_number>",
  "message":"<Your friendly, natural language message to the user>"
  "top_3_products": [
    {{"product_id": "<id1>", "product_name": "<name1>", "price": <price1>}},
    {{"product_id": "<id2>", "product_name": "<name2>", "price": <price2>}},
    {{"product_id": "<id3>", "product_name": "<name3>", "price": <price3>}}
  ]
}}

Now respond **strictly** following this format and stop after your final answer.
""")




match_prompt = PromptTemplate.from_template("""
You are an intelligent assistant helping a user choose a product.

You are given:
- The user's message: "{user_input}"
- The top 3 recommended products as JSON:
{products}

Your task:
1. Determine which product the user most likely wants to buy.
2. Be flexible — the user might refer to it by number ("first", "second", "3rd"), description ("cheap one", "the perfume"), or approximate price.
3. If you are **unsure** which product they mean, respond with "None" and do not guess.

Return your response strictly in this JSON format:
{{
  "chosen_product_id": "<product_id or None>",
  "confidence": <a number between 0 and 1 representing your confidence>
}}

only return the JSON do not return any other explaination.
""")




# TOOLS
@tool("average_calculator", return_direct=True)
def average_calculator(numbers: str) -> str:
    """
    Calculates the average of a comma-separated list of numbers.
    Example input: "120, 80, 200"
    """
    try:
        nums = [float(n.strip()) for n in numbers.split(",") if n.strip()]
        if not nums:
            return "No valid numbers provided."
        avg = sum(nums) / len(nums)
        return f"The average is {avg:.2f}"
    except Exception as e:
        return f"Error calculating average: {str(e)}"

@tool("rank_products_by_difference", return_direct=True)
def rank_products_by_difference(differences: str) -> str:
    """
    Rank products based on their absolute differences in price.

    Args:
        differences (str): A JSON string or comma-separated key-value pairs like:
                           '{"Perfume Elegant Rose": 6.20, "Electric Shaver": 3.80}'
    Returns:
        str: A sorted list of tuples (product_name, difference) in ascending order.
    """

    try:
        if isinstance(differences, str):
            differences = json.loads(differences)

        if not isinstance(differences, dict):
            return """❌ Invalid input. 
            Args:
            differences (str): A JSON string or comma-separated key-value pairs like:
                               '{"Perfume Elegant Rose": 6.20, "Electric Shaver": 3.80}'
            Do not forget that the values are the difference between the average and the product price
            """

        ranked = sorted(differences.items(), key=lambda x: x[1])

        return json.dumps(ranked, indent=2)

    except json.JSONDecodeError:
        return """❌ Invalid JSON format in input.
        Args:
        differences (str): A JSON string or comma-separated key-value pairs like:
                           '{"Perfume Elegant Rose": 6.20, "Electric Shaver": 3.80}'
        Do not forget that the values are the difference between the average and the product price
        """
    except Exception as e:
        return f"❌ Error: {str(e)}"



# PARSERS
class CategoryRecommendation(BaseModel):
    recommended_category: str = None
    justification: str = None


# CHAINS AND AGENTS
category_chain = LLMChain(llm=llm, prompt=category_prompt)

tools = [
    Tool(
        name="Average Calculator",
        func=average_calculator,
        description="Use this tool to calculate the average of the prices of the user's history purchased products, Give the tool a the list of prices in that format: 100.00, 200.99, ..., 50.98 "
    ),
    Tool(
        name="Product Ranking",
        func=rank_products_by_difference,
        description="Use this tool to rank the products based on the absolute difference"
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=verbose
)

match_chain = LLMChain(llm=llm, prompt=match_prompt)



# NODE 1
def category_extraction(state):
    # read user profile
    state["user_profile_path"] = user_path
    state["product_catalog_path"] = product_path
    state["history_log_file"] = log_path
    try:
        with open(state["user_profile_path"], "r") as f:
            state["user_profile"] = json.load(f)
            print(f"User_profile : {state['user_profile']} \n\n")
    except FileNotFoundError:
        state["error"] = f"❌ File path: {state['user_profile_path']} not found. Please check the path."
        return state
    except json.JSONDecodeError:
        state["error"] = f"❌ Invalid JSON format in {state['user_profile_path']}."
        return state

    # read product catalogue
    try:
        with open(state["product_catalog_path"], "r") as f:
            state["product_catalogue"] = json.load(f)
            state["unique_categories"] = sorted({item["category"] for item in state["product_catalogue"]})
            print(f"unique categories : {state['unique_categories']} \n\n")
    except FileNotFoundError:
        state["error"] = f"❌ File path: {state['product_catalog_path']} found. Please check the path."
        return state
    except json.JSONDecodeError:
        state["error"] = f"❌ Invalid JSON format in {state['product_catalog_path']}."
        return state
    return state

# NODE 2
def category_recommendation(state):
    try:
        # Run the LLM chain with provided user data
        print()
        output = category_chain.run(
            user_history=state["user_profile"]["history"],
            available_categories=state["unique_categories"]
        )

        # Parse JSON
        data = json.loads(output.strip())

        # Validate using your Pydantic model
        validated = CategoryRecommendation(**data)

        # Additional safeguard: ensure both fields are not empty or whitespace
        if not validated.recommended_category.strip() or not validated.justification.strip():
            state["error"] = "❌ One or more fields are empty or invalid in LLM output."
            return state

        # Assign validated values to state
        state["recommended_category"] = validated.recommended_category.strip()
        state["justification"] = validated.justification.strip()

        print(f"Recommended Category : {state['recommended_category']} \n\n")


    except json.JSONDecodeError:
        state["error"] = "❌ Invalid JSON format in LLM output."
        return state
    except ValidationError as e:
        state["error"] = f"❌ Invalid LLM output: {str(e)}"
        return state
    except Exception as e:
        state["error"] = f"❌ Unexpected error: {str(e)}"
        return state


    return state
# NODE 3, 4, ERROR
def product_recommendation(state):
    available_products = [item for item in state["product_catalogue"] if state["recommended_category"] == item["category"]]
    input_prompt = recommendation_prompt.format(
    user_history=state["user_profile"]["history"],
    products=available_products,
    justification = state["justification"]
    )
    result = agent.run(input_prompt) 
    try:
        output = json.loads(result)
        state['top3_products'] = output['top_3_products']
        state['justification'] = output['message']
    except json.JSONDecodeError:
        state['error'] = "❌ Output JSON INVALID in Product Recommender Node."
        return state

    return state

def select_and_purchase(state):
    print(state['justification'])
    print("Which of these products would like to try next?")
    trials = 3
    while trials:
        for i,product in enumerate(state['top3_products']):
            print(f"{i+1}. {product['product_name']} : {product['price']}")
        user_input = input("Which of these products would like to try next?")
        try:
            result = match_chain.run(user_input=user_input, products=state["top3_products"])
            data = json.loads(result)
            if data["chosen_product_id"] != "None" and data["confidence"] >= 0.6:
                print(f"✅ User selected: {data['chosen_product_id']}")
                # update user profile in the file and update the
                for product in state['product_catalogue']:
                    if product["id"] == data['chosen_product_id']:
                        break
                new_log_entry = {"transaction_id": str(uuid.uuid4()) ,"user_id": state["user_profile"]["user_id"], "product_id": product["id"]}
            
                # Load existing data if the file exists
                log_file_path = state["history_log_file"]
                if os.path.exists(log_file_path):
                    try:
                        with open(log_file_path, "r") as f:
                            log_file_content = json.load(f)
                            if not isinstance(log_file_content, list):
                                log_file_content = []  # Reset if file content isn't a list
                    except json.JSONDecodeError:
                        log_file_content = []  # Handle corrupted or empty JSON
                else:
                    log_file_content = []
            
                # Append the new record
                log_file_content.append(new_log_entry)
            
                # Save back to file
                with open(log_file_path, "w") as f:
                    json.dump(log_file_content, f, indent=4)

                # success message
                print(f"✅ Purchase saved: {new_log_entry}, Product Name: {product['name']}")

                user_file_path = state["user_profile_path"]
                new_product_entry = {
                    "product": product["name"],
                    "category": state["recommended_category"],
                    "price": product["price"]
                }
                state["user_profile"]["history"].append(new_product_entry)
                with open(user_file_path, "w") as f:
                    json.dump(state["user_profile"], f, indent=2)
                print(f"✅ Added {new_product_entry['product']} to user {state['user_profile']['user_id']}'s history.")
                return state
            else:
                print("⚠️ Not sure which product you meant to buy")
                trials -= 1
                print("Please enter the product's Name, Id, Price, or any clear hint about it.")
                
        except json.JSONDecodeError:
            state['error'] = "❌ Invalid JSON from LLM in Select and Purchase Node"
            return state
    
    state["error"] = "User's Input doesn't match any of the proposed products"
    return state

def error_node(state):
    print(f"Error : {state['error']} \n\n")
    # end after it
    return state


# Create the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("category_extraction", category_extraction)
graph.add_node("category_recommendation", category_recommendation)
graph.add_node("product_recommendation", product_recommendation)
graph.add_node("select_and_purchase", select_and_purchase)
graph.add_node("error_node", error_node)

graph.add_conditional_edges("category_extraction",lambda state: "error" if state.get("error") else "next",
    {
        "error": "error_node",
        "next": "category_recommendation"
    }
)

graph.add_conditional_edges("category_recommendation",lambda state: "error" if state.get("error") else "next",
    {
        "error": "error_node",
        "next": "product_recommendation"
    }
)

graph.add_conditional_edges("product_recommendation", lambda state: "error" if state.get("error") else "next",
    {
        "error": "error_node",
        "next": "select_and_purchase"
    }
)

graph.add_conditional_edges("select_and_purchase", lambda state: "error" if state.get("error") else "done",
    {
        "error": "error_node",
        "done": END
    }
)

graph.add_edge("error_node", END)

graph.set_entry_point("category_extraction")

# Compile the graph
shopping_assistant_graph = graph.compile()

state = AgentState()
result = shopping_assistant_graph.invoke({"state": state})