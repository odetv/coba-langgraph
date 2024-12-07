import re
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict, Annotated, Sequence, Set
from operator import add
from dotenv import load_dotenv



load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")



class AnswerState(TypedDict):
    answer = None
class AgentState(TypedDict):
    question: str
    question_type: str
    foodQuestion: str
    destinationQuestion: str
    outOfContextQuestion: str
    totalAgents: int
    finishedAgents: Set[str]
    answerAgents: Annotated[Sequence[AnswerState], add]
    responseFinal: str



def chat_llm(question: str):
    openai = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0, streaming=True)
    result = openai.invoke(question).content if hasattr(openai.invoke(question), "content") else openai.invoke(question)
    return result



def questionIdentifierAgent(state: AgentState):
    info = "\n--- QUESTION IDENTIFIER ---"
    print(info)

    promptTypeQuestion = """
        Anda adalah seoarang pemecah pertanyaan pengguna.
        Tugas Anda sangat penting. Klasifikasikan atau parsing pertanyaan dari pengguna untuk dimasukkan ke variabel sesuai konteks.
        Tergantung pada jawaban Anda, akan mengarahkan ke agent yang tepat.
        Ada 3 konteks diajukan:
        - DESTINATION_AGENT - Pertanyaan yang menyebutkan segala informasi mengenai destinasi wisata.
        - FOOD_AGENT - Pertanyaan yang menyebutkan segala informasi mengenai makanan.
        - OUTOFCONTEXT_AGENT - Pertanyaan yang tidak sesuai dengan konteks.
        Kemungkinan pertanyaannya berisi lebih dari 1 variabel konteks yang berbeda (jika hanya 1, maka 1 saja), buat yang sesuai dengan konteks saja (jika tidak ada jangan dibuat).
        Jawab pertanyaan dan sertakan pertanyaan pengguna yang sesuai dengan kategori dengan contoh seperti ({"DESTINATION_AGENT": "pertanyaan relevan terkait destinasi wisata", "FOOD_AGENT": "pertanyaan relevan terkait makanan", "OUTOFCONTEXT_AGENT": "pertanyaan diluar konteks"}).
        Buat dengan format data JSON tanpa membuat key baru.
    """
    messagesTypeQuestion = [
        SystemMessage(content=promptTypeQuestion),
        HumanMessage(content=state["question"]),
    ]
    responseTypeQuestion = chat_llm(messagesTypeQuestion).strip().lower()
    
    state["question_type"] = responseTypeQuestion
    print("\nPertanyaan:", state["question"])

    total_agents = 0
    if "destination_agent" in state["question_type"]:
        total_agents += 1
    if "food_agent" in state["question_type"]:
        total_agents += 1
    if "outofcontext_agent" in state["question_type"]:
        total_agents += 1

    state["totalAgents"] = total_agents
    print(f"DEBUG: Total agents bertugas: {state['totalAgents']}")

    pattern = r'"(.*?)":\s*"(.*?)"'
    matches = re.findall(pattern, responseTypeQuestion)
    result_dict = {key: value for key, value in matches}

    state["destinationQuestion"] = result_dict.get("destination_agent", None)
    state["foodQuestion"] = result_dict.get("food_agent", None)
    state["outOfContextQuestion"] = result_dict.get("outofcontext_agent", None)
    
    print(f"DEBUG: destinationQuestion: {state['destinationQuestion']}")
    print(f"DEBUG: foodQuestion: {state['foodQuestion']}")
    print(f"DEBUG: outOfContextQuestion: {state['outOfContextQuestion']}")

    return state



def destinationAgent(state: AgentState):
    print("\n--- DESTINATION AGENT ---")
    prompt = f"""
        Anda seorang yang memiliki pengetahuan yang sangat luas dan hebat tentang destinasi wisata.
    """
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state["destinationQuestion"])
    ]
    response = chat_llm(messages)
    agentOpinion = {
        "answer": response
    }
    print("\n\nDESTINATION ANSWER:::", response)
    state["finishedAgents"].add("answerDestination_agent")
    return {"answerAgents": [agentOpinion]}



def foodAgent(state: AgentState):
    print("\n--- FOOD AGENT ---")
    prompt = f"""
        Anda seorang yang memiliki pengetahuan yang sangat luas dan hebat tentang makanan.
    """
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state["foodQuestion"])
    ]
    response = chat_llm(messages)
    agentOpinion = {
        "answer": response
    }
    print("\n\nFOOD ANSWER:::", response)
    state["finishedAgents"].add("answerFood_agent")
    return {"answerAgents": [agentOpinion]}



def outOfContextAgent(state: AgentState):
    print("\n--- OUTOFCONTEXT AGENT ---")
    prompt = f"""
        Anda seorang pengarah terkait pertanyaan yang tidak relevan, tawarkan untuk menanyakan terkait makanan dan destinasi wisata.
    """
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state["outOfContextQuestion"])
    ]
    response = chat_llm(messages)
    agentOpinion = {
        "answer": response
    }
    print("\n\nOUTOFCONTEXT ANSWER:::", response)
    state["finishedAgents"].add("answerOutofcontext_agent")
    return {"answerAgents": [agentOpinion]}



def resultWriterAgent(state: AgentState):
    if len(state["finishedAgents"]) < state["totalAgents"]:
        print("\nMenunggu agent lain menyelesaikan tugas...")
        return None
    
    elif len(state["finishedAgents"]) == state["totalAgents"]:
        info = "\n--- RESULT WRITER AGENT ---"
        print(info)
        prompt = f"""
            Berikut pedoman yang harus diikuti untuk menulis ulang informasi:
            - Berikan informasi secara lengkap dan jelas apa adanya sesuai informasi yang diberikan.
            - Urutan informasi sesuai dengan urutan pertanyaan.
            - Jangan menyebut ulang pertanyaan secara eksplisit.
            - Jangan menjawab selain menggunakan informasi pada informasi yang diberikan, sampaikan dengan apa adanya jika Anda tidak mengetahui jawabannya.
            - Jangan tawarkan informasi lainnya selain informasi yang diberikan yang didapat saja.
            - Hasilkan response dalam format Markdown.
            Berikut adalah informasinya:
            {state["answerAgents"]}
        """
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=state["question"])
        ]
        response = chat_llm(messages)
        print(response)
        state["responseFinal"] = response

        return {"responseFinal": state["responseFinal"]}



def run(question):
    workflow = StateGraph(AgentState)
    initial_state = questionIdentifierAgent({"question": question, "finishedAgents": set()})
    context = initial_state["question_type"]
    workflow.add_node("questionIdentifier_agent", lambda x: initial_state)
    workflow.add_edge(START, "questionIdentifier_agent")

    if "destination_agent" in context:
        workflow.add_node("destination_agent", destinationAgent)
        workflow.add_edge("questionIdentifier_agent", "destination_agent")
        workflow.add_edge("destination_agent", "resultWriter_agent")
    if "food_agent" in context:
        workflow.add_node("food_agent", foodAgent)
        workflow.add_edge("questionIdentifier_agent", "food_agent")
        workflow.add_edge("food_agent", "resultWriter_agent")
    if "outofcontext_agent" in context:
        workflow.add_node("outofcontext_agent", outOfContextAgent)
        workflow.add_edge("questionIdentifier_agent", "outofcontext_agent")
        workflow.add_edge("outofcontext_agent", "resultWriter_agent")

    workflow.add_node("resultWriter_agent", resultWriterAgent)
    workflow.add_edge("resultWriter_agent", END)

    graph = workflow.compile()
    result = graph.invoke({"question": question})



# DEBUG QUESTION
run("aku ingin liburan ke bali dan makanan apa yang enak?")