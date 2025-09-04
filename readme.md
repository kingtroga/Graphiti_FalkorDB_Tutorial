# ManyBirds Shoe Sales Chatbot

A smart chatbot that remembers conversations and gets better over time using Graphiti + FalkorDB.

## What This Does

* **Remembers customers** : Never forgets preferences, sizes, or past conversations
* **Learns continuously** : Gets smarter with every interaction
* **Personalizes responses** : Uses conversation history for better recommendations
* **Searches dynamically** : Can look up fresh product information when needed

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key
FALKORDB_HOST=localhost
FALKORDB_PORT=6379
FALKORDB_USERNAME=your_username
FALKORDB_PASSWORD=your_password
```

### 3. Start FalkorDB

```bash
docker run -p 6379:6379 falkordb/falkordb:latest
```

### 4. Add Product Data

Make sure you have `manybirds_products.json` in the same directory.

### 5. Run the Chatbot

```bash
python your_script.py
```

## How It Works

1. **Loads products** into a knowledge graph
2. **Creates user profiles** that remember preferences
3. **Stores conversations** for future context
4. **Searches intelligently** based on user history
5. **Learns from every interaction**

## What Makes It Smart

* Uses **Graphiti** for temporal knowledge graphs
* Stores data in **FalkorDB** for fast relationship queries
* **LangGraph** manages conversation flow
* **Asynchronous** memory updates don't slow down chats

## Example Conversation

**First time:**

```
User: "I need running shoes"
Bot: "What size do you wear and what's your budget?"
```

**Next time:**

```
User: "Any new arrivals?"
Bot: "I remember you wanted running shoes in size 9. We just got new TinyBirds that might interest you!"
```

## Key Features

* ✅ Persistent memory across sessions
* ✅ Personalized product recommendations
* ✅ Dynamic knowledge updates
* ✅ Conversation history tracking
* ✅ Intelligent tool usage
* ✅ Error-resistant data processing

## Files You Need

* Your main Python script
* `manybirds_products.json` (product catalog)
* `.env` (environment variables)

That's it! The chatbot will remember everything and get smarter with every conversation.
