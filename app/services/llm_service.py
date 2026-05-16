import os
import google.generativeai as genai
from dotenv import load_dotenv
from app.utils.prompts import SYSTEM_PROMPT
#from google.generativeai import generative_model

load_dotenv()

genai.configure(  # type: ignore
    api_key=os.getenv("GEMINI_API_KEY")
) # type: ignore

model = genai.GenerativeModel(   # type: ignore
    "gemini-1.5-flash"
)


def generate_reply(user_query, recommendations):

    prompt = f"""
    {SYSTEM_PROMPT}

    User Query:
    {user_query}

    Recommendations:
    {recommendations}

    Generate a conversational recruiter-friendly response.
    """

    response = model.generate_content(prompt)

    return response.text