# ai_assistant.py
# Groq AI integration for data insights

import groq
import json
import streamlit as st
from config import GROQ_API_KEY
from data_utils import get_detailed_data_summary


def ask_groq_question(question, data):
    """Ask a question to Groq AI about the data with comprehensive context"""
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        
        # Get detailed data summary
        data_summary = get_detailed_data_summary(data)
        
        # Format the context for the AI
        context = f"""
        DATASET INFORMATION:
        
        Shape: {data_summary['shape'][0]} rows, {data_summary['shape'][1]} columns
        Columns: {', '.join(data_summary['columns'])}
        
        Column Types:
        {json.dumps(data_summary['dtypes'], indent=2)}
        
        Missing Values:
        {json.dumps(data_summary['missing_values'], indent=2)}
        
        Numeric Statistics:
        {json.dumps(data_summary['numeric_stats'], indent=2) if data_summary['numeric_stats'] else 'No numeric columns'}
        
        Categorical Information:
        {json.dumps(data_summary['categorical_info'], indent=2) if data_summary['categorical_info'] else 'No categorical columns'}
        
        Sample Data (first 10 rows):
        {json.dumps(data_summary['sample_data'], indent=2)}
        """
        
        # Add correlations if available
        if 'correlations' in data_summary:
            context += f"\n\nCorrelations between numeric columns:\n{json.dumps(data_summary['correlations'], indent=2)}"
        
        # Create the prompt
        prompt = f"""You are a data science expert assistant with access to the complete dataset information above.

DATASET CONTEXT:
{context}

USER QUESTION: {question}

Please provide a detailed, accurate answer based on the dataset information provided. If the user asks about specific values, patterns, or statistics, refer to the actual data in the context. If the question requires calculations not directly in the data, explain what you can determine from the available information."""
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a data science expert assistant. Use the provided dataset context to answer questions accurately. If the question can be answered from the data, provide specific details and statistics. Be helpful and clear."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,  # Lower temperature for more factual answers
            max_tokens=1500
        )
        
        response = chat_completion.choices[0].message.content
        return response
    except Exception as e:
        return f"Error: {str(e)}"