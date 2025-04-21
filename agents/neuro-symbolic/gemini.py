from google import genai
from google.genai import types
import base64

import base64
import vertexai
from vertexai.preview.generative_models import GenerativeModel, SafetySetting, Part, Tool
from vertexai.preview.generative_models import grounding
from vertexai.generative_models import GenerativeModel, SafetySetting, Part


neuro_generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.7,
    "top_p": 0.95,
}

neuro_safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]

tools = [
    Tool.from_google_search_retrieval(
        google_search_retrieval=grounding.GoogleSearchRetrieval(
        )
    ),
]

def multiturn_generate_content(input_prompt):
    vertexai.init(
        project="warren-440921",
        location="us-central1",
        api_endpoint="us-central1-aiplatform.googleapis.com"
    )
    model = GenerativeModel(
        "gemini-1.5-pro-002",
        system_instruction=["""You are an expert career support agent and you give advice based on questions of Berkeley-Haas MBA students. You are given a user question and some context. You answer the question by referring the given context but using your own intelligence.
                            Make sure your answers are detailed and comprehensive. You directly address the user and use the terms You, Your etc."""],
        tools=tools,
    )
    chat = model.start_chat()
    neuro_response = chat.send_message(
        [input_prompt],
        generation_config=neuro_generation_config,
        safety_settings=neuro_safety_settings
    )
    print("Response from neuro: ", neuro_response)
    return neuro_response

def get_symbolic_data(user_input):
  client = genai.Client(
      vertexai=True,
      project="981638136728",
      location="us-central1",
  )

  model = "projects/981638136728/locations/us-central1/endpoints/627571550320590848"
  contents = [
    types.Content(
      role="user",
      parts=[
        types.Part.from_text(text=user_input)
      ]
    ),
  ]
  generate_content_config = types.GenerateContentConfig(
    temperature = 0,
    top_p = 0.95,
    max_output_tokens = 8192,
    response_modalities = ["TEXT"],
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    system_instruction=[types.Part.from_text(text="""You are a highly skilled career support agent dedicated to assisting Haas MBA students with expert guidance and advice. Your expertise spans resume building, networking strategies, career path management, interview preparation, and other career-related topics. Your advice is personalized and tailored to the unique needs and goals of each individual student.""")],
  )

  final_response = ""
  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    final_response += chunk.text
  print(final_response)
  return final_response

# get_symbolic_data()

def get_user_input():
  while True:
    user_input = input("Hi, I am Bearly! I am here to help Berkeley-Haas students with MBA interview advice: ")
    if user_input == "exit":
      break
    print("Bearly says: Getting symbolic data ...")
    symbolic_data = get_symbolic_data(user_input)
    print("Bearly says: Getting neuro response ...")
    form_prompt = f""" Identify the intent of the User question and refer the context provided to give a detailed answer.
    User question: {user_input}
    background context: {symbolic_data}
    """
    # print(form_prompt)
    neuro_response = multiturn_generate_content(form_prompt)
    print("Bearly final advice: ", neuro_response.candidates[0].content.parts[0].text)

get_user_input()


# I am a first year MBA student looking to pivot from Edtech to Consulting. How should I plan for this?
