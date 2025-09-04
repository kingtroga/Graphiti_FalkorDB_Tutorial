"""
Complete ManyBirds Shoe Sales Chatbot with Knowledge Graph Integration
====================================================================

This script demonstrates:
1. Product data ingestion into a knowledge graph
2. Conversational AI that uses the knowledge graph for context
3. A complete sales flow with memory and tool usage
4. Interactive chat interface using Jupyter widgets

Learning Goals:
- Understanding conversation state management
- Graph-based context retrieval for AI responses
- Asynchronous conversation processing
- Tool integration in conversational AI
"""

import asyncio 
import json 
import logging
import os
import sys
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Dict, List, Any

import ipywidgets as widgets
from dotenv import load_dotenv
from IPython.display import Image, display
from typing_extensions import TypedDict

# Graphiti imports for knowledge graph functionality
from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_EPISODE_MENTIONS

# LangGraph imports for conversation flow management
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode


load_dotenv()



def setup_logging():
    """Configure logging to track the application's behavior"""
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

def extract_essential_product_info(product):
    """
    Extract clean product information that won't cause search syntax errors.
    
    This function prevents issues like currency symbols causing FQL parsing errors
    by formatting prices as readable text instead of symbols.
    """
    essential_info = {
        'title': product.get('title', 'Unknown Product'),
        'vendor': product.get('vendor', ''),
        'product_type': product.get('product_type', ''),
        'description': '',
        'price_info': '',
        'sizes': [],
        'key_features': []
    }
    
    # Clean the description to prevent LLM JSON parsing errors
    description = product.get('body_html', '')
    if description:
        # Remove HTML tags and problematic characters
        description = description.replace('<p>', '').replace('</p>', '')
        description = description.replace('<br>', ' ').replace('<br/>', ' ')
        description = description.replace('"', "'").replace('\n', ' ').replace('\r', ' ')
        if len(description) > 150:
            description = description[:150] + "..."
        essential_info['description'] = description.strip()
    
    # Handle pricing in a search-friendly way (avoid currency symbols)
    if 'variants' in product and product['variants']:
        prices = []
        sizes = []
        
        for variant in product['variants']:
            if variant.get('price'):
                try:
                    price_num = float(variant['price'])
                    prices.append(price_num)
                except (ValueError, TypeError):
                    pass
            
            if variant.get('title'):
                size = variant['title'].strip()
                if size and size not in sizes:
                    sizes.append(size)
        
        # Create price description without problematic characters
        if prices:
            min_price = min(prices)
            max_price = max(prices)
            if min_price == max_price:
                essential_info['price_info'] = f"priced at {min_price:.0f} dollars"
            else:
                essential_info['price_info'] = f"ranging from {min_price:.0f} to {max_price:.0f} dollars"
        
        essential_info['sizes'] = sizes[:5]  # Limit to prevent overwhelming
    
    # Extract meaningful features, avoiding metadata that causes search issues
    if 'tags' in product and product['tags']:
        meaningful_tags = []
        for tag in product['tags']:
            # Skip internal metadata and problematic characters
            if (tag.startswith('Manybirds::') or '=' in tag or 
                any(char in tag for char in ['$', '(', ')', '[', ']', '\\', '/'])):
                continue
            meaningful_tags.append(tag.strip())
            if len(meaningful_tags) >= 3:
                break
        essential_info['key_features'] = meaningful_tags
    
    return essential_info

# Database connection setup
falkordb_driver = FalkorDriver(
    host=os.getenv("FALKORDB_HOST"),
    port=int(os.getenv("FALKORDB_PORT")),
    username=os.getenv("FALKORDB_USERNAME"),
    password=os.getenv("FALKORDB_PASSWORD"),
)

client = Graphiti(graph_driver=falkordb_driver)

async def ingest_products_data(client: Graphiti):
    """
    Load and ingest product data into the knowledge graph.
    This creates the knowledge base that the chatbot will use for recommendations.
    """
    script_dir = Path.cwd()
    json_file_path = script_dir / 'manybirds_products.json'

    with open(json_file_path) as file:
        products = json.load(file)['products']

    print(f"Processing {len(products)} products...")

    for i, product in enumerate(products):
        try:
            essential_product = extract_essential_product_info(product)
            
            # Create structured, readable product description
            episode_body = f"""
Product Name: {essential_product['title']}
Brand: {essential_product['vendor']}
Category: {essential_product['product_type']}
Description: {essential_product['description']}
Pricing: {essential_product['price_info']}
Available Sizes: {', '.join(essential_product['sizes'])}
Features: {', '.join(essential_product['key_features'])}
""".strip()
            
            print(f"Ingesting product {i+1}/{len(products)}: {essential_product['title']}")
            
            await client.add_episode(
                name=essential_product['title'],
                episode_body=episode_body,
                source_description='ManyBirds Product Catalog',
                source=EpisodeType.json,
                reference_time=datetime.now(timezone.utc),
            )
            
        except Exception as e:
            print(f"Failed to ingest product {i+1}: {str(e)}")
            continue

def edges_to_facts_string(entities: list[EntityEdge]) -> str:
    """
    Convert knowledge graph edges (relationships) into readable facts.
    This transforms the graph data into natural language for the AI.
    """
    if not entities:
        return "No relevant information found."
    return '- ' + '\n- '.join([edge.fact for edge in entities])


async def terminal_chat_interface(graph, user_name, user_node_uuid):
    """
    Simple terminal-based chat interface that works in any Python environment.
    This replaces the Jupyter widgets for command-line usage.
    """
    print("\n" + "="*60)
    print("🛍️  MANYBIRDS SHOE SALES CHATBOT")
    print("="*60)
    print("Hello! I'm your ManyBirds shoe sales assistant.")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("Ask me about shoes, sizes, colors, or anything else!")
    print("-" * 60)
    
    # Create conversation config
    config = {'configurable': {'thread_id': uuid.uuid4().hex}}
    
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n🤖 Assistant: Thank you for chatting! Have a great day finding the perfect shoes! 👟")
                break
            
            if not user_input:
                print("Please enter a message or type 'quit' to exit.")
                continue
            
            # Prepare the graph state
            graph_state = {
                'messages': [{'role': 'user', 'content': user_input}],
                'user_name': user_name,
                'user_node_uuid': user_node_uuid,
            }
            
            # Show that the assistant is thinking
            print("\n🤖 Assistant: ", end="", flush=True)
            
            # Process through the conversation graph
            assistant_response = ""
            async for event in graph.astream(graph_state, config=config):
                for value in event.values():
                    if 'messages' in value:
                        last_message = value['messages'][-1]
                        if isinstance(last_message, AIMessage) and isinstance(last_message.content, str):
                            # Print response as it streams
                            print(last_message.content, end="", flush=True)
                            assistant_response = last_message.content
            
            # Add to conversation history for display
            conversation_history.append({
                'user': user_input,
                'assistant': assistant_response
            })
            
            print()  # New line after response
            
        except KeyboardInterrupt:
            print("\n\n🤖 Assistant: Goodbye! Thanks for using ManyBirds shoe assistant! 👟")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")

# ============================================================================
# CONVERSATIONAL AI SETUP
# ============================================================================

class State(TypedDict):
    """
    State management for the conversation.
    
    This TypedDict defines the structure of data that flows through
    the conversation system, including:
    - messages: The conversation history
    - user_name: The user's identifier  
    - user_node_uuid: The user's node in the knowledge graph
    """
    messages: Annotated[list, add_messages]
    user_name: str
    user_node_uuid: str

async def chatbot(state: State):
    """
    The main chatbot function that processes user messages and generates responses.
    
    Flow:
    1. Check if there are previous messages in the conversation
    2. If yes, search the knowledge graph for relevant context
    3. Create a system message with sales instructions and context
    4. Get AI response using the LLM
    5. Log the conversation to the knowledge graph for future use
    6. Return the AI response
    
    This is where the magic happens - the AI gets context from the knowledge graph
    to provide informed, personalized responses about shoes.
    """
    facts_string = None
    
    # Only search for context if there are previous messages
    if len(state['messages']) > 0:
        last_message = state['messages'][-1]
        
        # Create a search query from the conversation
        # This identifies who said what for better context retrieval
        if isinstance(last_message, AIMessage):
            graphiti_query = f'SalesBot: {last_message.content}'
        else:
            graphiti_query = f'{state["user_name"]}: {last_message.content}'
        
        print(f"Searching knowledge graph for: {graphiti_query}")
        
        # Search the knowledge graph using the user's node as the center
        # This ensures we get information most relevant to this specific user
        edge_results = await client.search(
            graphiti_query,
            center_node_uuid=state['user_node_uuid'],
            num_results=5
        )
        
        facts_string = edges_to_facts_string(edge_results)
        print(f"Found {len(edge_results)} relevant facts")

    # Create system instructions for the AI
    # This defines the AI's personality and sales objectives
    system_message = SystemMessage(
        content=f"""You are a skillful shoe salesperson working for ManyBirds. 
Review information about the user and their prior conversation below and respond accordingly. 
Keep responses short and concise. And remember, always be selling (and helpful!)

Things you'll need to know about the user in order to close a sale:
- the user's shoe size
- any other shoe needs? maybe for wide feet?
- the user's preferred colors and styles  
- their budget

Ensure that you ask the user for the above if you don't already know.

Facts about the user and their conversation:
{facts_string or 'No facts about the user and their conversation'}"""
    )

    # Combine system message with conversation history
    messages = [system_message] + state['messages']
    
    # Get response from the language model
    response = await llm.ainvoke(messages)
    print(f"AI Response: {response.content[:100]}...")

    # Log this exchange to the knowledge graph asynchronously
    # This doesn't block the conversation flow but ensures future context
    asyncio.create_task(
        client.add_episode(
            name='Chatbot Conversation',
            episode_body=f'{state["user_name"]}: {state["messages"][-1].content}\nSalesBot: {response.content}',
            source=EpisodeType.message,
            reference_time=datetime.now(timezone.utc),
            source_description='Sales Chatbot',
        )
    )

    return {'messages': [response]}

# ============================================================================
# CONVERSATION FLOW SETUP
# ============================================================================

# Create the conversation flow graph
graph_builder = StateGraph(State)

# Memory to track conversation history across interactions
memory = MemorySaver()

async def should_continue(state, config):
    """
    Determine if the conversation should continue to tool usage or end.
    
    This function checks if the AI's last message included tool calls.
    - If there are tool calls, continue to the tools node
    - If no tool calls, end the conversation turn
    """
    messages = state['messages']
    last_message = messages[-1]
    
    if not last_message.tool_calls:
        print("No tool calls detected, ending turn")
        return 'end'
    else:
        print(f"Tool calls detected: {len(last_message.tool_calls)}")
        return 'continue'

# Build the conversation graph
# This creates a flow: START -> agent -> (tools if needed) -> agent -> END
graph_builder.add_node('agent', chatbot)
# Note: We'll add the tools node later after we define the actual tools

graph_builder.add_edge(START, 'agent')
# Note: We'll add conditional edges after setting up tools

async def main():
    """
    Main function that sets up the entire system:
    1. Sets up database and ingests product data
    2. Creates user profile in the knowledge graph
    3. Sets up tools and conversation system
    4. Provides interactive chat interface
    """
    
    # Step 1: Setup database and ingest products
    try:
        await client.build_indices_and_constraints()
        print("Successfully created indices")
    except Exception as e:
        print(f"Failed to create indices: {e}")
        return

    print("Starting product data ingestion...")
    await ingest_products_data(client)
    print("Product ingestion completed")

    # Step 2: Create user profile
    user_name = 'jess'
    await client.add_episode(
        name='User Profile',
        episode_body=f'{user_name} is interested in buying a pair of shoes',
        source=EpisodeType.text,
        reference_time=datetime.now(timezone.utc),
        source_description='Customer Management System',
    )

    # Step 3: Find user and brand nodes in the knowledge graph
    try:
        # Search for the user's node
        user_search = await client._search(user_name, NODE_HYBRID_SEARCH_EPISODE_MENTIONS)
        if not user_search.nodes:
            print(f"Could not find user node for {user_name}")
            return
        user_node_uuid = user_search.nodes[0].uuid
        print(f"Found user node: {user_node_uuid}")

        # Search for the ManyBirds brand node  
        brand_search = await client._search('ManyBirds', NODE_HYBRID_SEARCH_EPISODE_MENTIONS)
        if not brand_search.nodes:
            print("Could not find ManyBirds node")
            return
        manybirds_node_uuid = brand_search.nodes[0].uuid
        print(f"Found ManyBirds node: {manybirds_node_uuid}")
        
    except Exception as e:
        print(f"Failed to find required nodes: {e}")
        return

    # Step 4: Define tools for the AI to use
    @tool
    async def get_shoe_data(query: str) -> str:
        """Search the knowledge graph for information about shoes and products"""
        try:
            print(f"Tool called with query: {query}")
            edge_results = await client.search(
                query,
                center_node_uuid=manybirds_node_uuid,
                num_results=10,
            )
            result = edges_to_facts_string(edge_results)
            print(f"Tool returning {len(edge_results)} facts")
            return result
        except Exception as e:
            return f"Search failed: {str(e)}"

    # Step 5: Set up LLM and tools
    global llm
    tools = [get_shoe_data]
    tool_node = ToolNode(tools)
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0).bind_tools(tools)
    
    # Now add the tools node and complete the graph connections
    graph_builder.add_node('tools', tool_node)
    graph_builder.add_conditional_edges('agent', should_continue, {'continue': 'tools', 'end': END})
    graph_builder.add_edge('tools', 'agent')
    
    # Compile the conversation graph with memory
    graph = graph_builder.compile(checkpointer=memory)
    
    # Optional: Display the conversation flow diagram
    with suppress(Exception):
        display(Image(graph.get_graph().draw_mermaid_png()))

    # Step 6: Test the system with a sample interaction
    print("\n" + "="*50)
    print("Testing the conversation system...")
    print("="*50)
    
    test_result = await graph.ainvoke(
        {
            'messages': [
                {
                    'role': 'user',
                    'content': 'What sizes do the TinyBirds Wool Runners in Natural Black come in?',
                }
            ],
            'user_name': user_name,
            'user_node_uuid': user_node_uuid,
        },
        config={'configurable': {'thread_id': uuid.uuid4().hex}},
    )
    
    print(f"Test completed. AI responded with: {test_result['messages'][-1].content}")

    # Step 7: Set up terminal chat interface
    print("\n" + "="*50)
    print("Starting terminal chat interface...")
    print("="*50)

    # Run the terminal chat
    await terminal_chat_interface(graph, user_name, user_node_uuid)
    

if __name__ == "__main__":
    asyncio.run(main())