from flask import Flask, request, jsonify
from pydub import AudioSegment
import openai
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RefineDocumentsChain, LLMChain, SimpleSequentialChain
from transformers import AutoModelForCausalLM

app = Flask(__name__)

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    audio_file = request.files.get('audio_file')
    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400
    
    # Saving the uploaded file
    file_path = os.path.join("uploads", audio_file.filename)
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    audio_file.save(file_path)
    
    # Invoke your existing function to process the audio file
    result = select_audio_file(file_path)
    return jsonify({"result": result})

# Your existing functions
def select_audio_file(file_path):
    if file_path:
        file_size = os.path.getsize(file_path)
        full_transcript = ""
        if file_size > 25 * 1024 * 1024:
            split_audio_files = split_audio_file(file_path)
            for split_file in split_audio_files:
                text = transcribe_audio(split_file)
                full_transcript += text + "\n"
        else:
            full_transcript = transcribe_audio(file_path)

        summary = summarize_transcript(full_transcript)
        return {"transcript": full_transcript, "summary": summary}

def split_audio_file(file_path):
    audio = AudioSegment.from_file(file_path)
    file_size = len(audio)
    chunk_length = 600000
    chunks = [audio[i:i + chunk_length] for i in range(0, file_size, chunk_length)]
    
    split_files = []
    for i, chunk in enumerate(chunks):
        split_file_path = f"{file_path}_part_{i}.mp3"
        chunk.export(split_file_path, format="mp3")
        split_files.append(split_file_path)
        
    return split_files

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript['text']
class CustomDocument:
    def __init__(self, page_content, metadata={}):
        self.page_content = page_content
        self.metadata = metadata

def summarize_transcript(transcript):
    # Your existing setup code
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=15000, chunk_overlap=500
    )
    llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4")
    llm_long_context = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-3.5-turbo-16k")
    #llm_long_context = AutoModelForCausalLM.from_pretrained("Yukang/Llama-2-13b-longlora-32k-ft" )

    docs = text_splitter.create_documents([transcript])
    
    # New setup code for RefineDocumentsChain
    document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="{page_content}"
    )

    document_variable_name = "context"
    initial_response_name = "prev_response"
    
    initial_prompt = PromptTemplate.from_template(
        """
        Het volgende is een deel van de vergadering:
        "{context}"
        Gebruik deze data om direct een uitgebreide notule mee te maken, hou de volgorde in stand en geef aan welke acties tot personen behoren.
        Probeer daarbij dan ook om de aanwezigen te identificeren.
        Gebruik zoveel mogelijk tokens.
        NOTULE:
        """
    )

    initial_llm_chain = LLMChain(
        llm=llm_long_context, 
        prompt=initial_prompt, 
        verbose=True
        )
    
    refine_prompt = PromptTemplate.from_template(
        """
        Het is jouw taak om een uitgebreide definitieve notule te maken.
        We hebben tot op zekere hoogte een bestaande notule gegeven: {prev_response}.
        We hebben de mogelijkheid om de bestaande notule opnieuw te controleren.
        Pas aan waar meer context nodig is wanneer het in de bestaande notule niet duidelijk is.
        ------------
        {context}
        ------------
        Gezien de nieuwe context, verfijn en verrijk de oorspronkelijke notule.
        Gebruik zoveel mogelijk tokens.
        """
    )
    refine_llm_chain = LLMChain(
        llm=llm_long_context, 
        prompt=refine_prompt, 
        verbose=True
        )
    
    refine_chain = RefineDocumentsChain(
        initial_llm_chain=initial_llm_chain,
        refine_llm_chain=refine_llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
        initial_response_name=initial_response_name,
        return_intermediate_steps=False,
        verbose=True
    )

    # Aanmaken van een nieuwe prompt voor het sorteren van de samenvatting
    sort_prompt = """
    Gegeven de volgende notulen van een vergadering:
    {summary}
    Sorteer, categorizeer en verfijn deze notulen zodat wat opgeleverd wordt een overzichtelijk, informatief en duidelijke notule is.
    Probeer door middel van kopjes de notule te structureren.
    gebruik zoveel mogelijk tokens, maar zorg ervoor dat er geen data dubbel in de notule staat.
    """
    sort_prompt_template = PromptTemplate(template=sort_prompt, input_variables=["summary"])

    # Aanmaken van een nieuwe LLMChain om het sorteren te behandelen
    sort_llm_chain = LLMChain(
        llm=llm_long_context, 
        prompt=sort_prompt_template, 
        verbose=True
        )
    
    # Definieer de SequentialChain
    overall_chain = SimpleSequentialChain(
        chains=[refine_chain, sort_llm_chain],  # De ketens die we willen uitvoeren
        verbose=True
    )
    
    # Voer de SequentialChain uit
    results = overall_chain.run(docs)
    
    return results

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

