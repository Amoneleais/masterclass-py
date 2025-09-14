import os
from dotenv import load_dotenv
from typing import TypedDict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import List, Literal, Dict
from IPython.display import display, Image
from pathlib import Path
import re, pathlib

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Importação do modelo Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature=0, api_key=GEMINI_API_KEY
)

# Prompt para triagem inicial
TRIAGEM_PROMPT = (
    "Você é um triador de Service Desk para políticas internas da empresa Carraro Desenvolvimento. "
    "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
    "{\n"
    '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
    '  "urgencia": "BAIXA" | "MEDIA" | "ALTA",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n"
    "Regras:\n"
    '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas (Ex: "Posso reembolsar a internet do meu home office?", "Como funciona a política de alimentação em viagens?").\n'
    '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema ou contexto (Ex: "Preciso de ajuda com uma política", "Tenho uma dúvida geral").\n'
    '- **ABRIR_CHAMADO**: Pedidos de exceção, liberação, aprovação ou acesso especial, ou quando o usuário explicitamente pede para abrir um chamado (Ex: "Quero exceção para trabalhar 5 dias remoto.", "Solicito liberação para anexos externos.", "Por favor, abra um chamado para o RH.").'
    "Analise a mensagem e decida a ação mais apropriada."
)


# Definição do modelo de saída estruturada
class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"]
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)


triagem_chain = llm.with_structured_output(TriagemOut)


# Função para realizar a triagem
def triagem(mensagem: str) -> Dict:
    saida: TriagemOut = triagem_chain.invoke(
        [SystemMessage(content=TRIAGEM_PROMPT), HumanMessage(content=mensagem)]
    )

    return saida.model_dump()


testes = [
    "Gostaria de saber se posso trabalhar de casa 3 dias por semana.",
    "Qual é a política de reembolso para despesas de viagem?",
    "Preciso de ajuda com uma dúvida sobre férias.",
    "Por favor, abra um chamado para o RH.",
]

for t in testes:
    print(triagem(t))

docs = []

# Carregando documentos PDF da pasta 'content'
for n in Path("content").glob("*.pdf"):
    try:
        docs.extend(PyMuPDFLoader(str(n)).load())
        print(f"Carregado {n}")
    except Exception as e:
        print(f"Erro ao carregar {n}: {e}")

print(f"Total de documentos carregados: {len(docs)}")

# Dividindo os documentos em chunks
chunks = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=30
).split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY)

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3, "k": 4})

# Prompt para RAG
prompt_rag = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um Assistente de Políticas Internas (RH/IT) da empresa Carraro Desenvolvimento. "
     "Responda SOMENTE com base no contexto fornecido. "
     "Se não houver base suficiente, responda apenas 'Não sei'."),

    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])

document_chain = create_stuff_documents_chain(llm, prompt_rag)

# Função RAG
def rag(mensagem: str) -> Dict:
    docs_relevantes = retriever.invoke(mensagem)
    if not docs_relevantes:
        return {"resposta": "Desculpe, não consegui encontrar informações relevantes para sua pergunta.", "docs": [], "contexto_encontrado": False}
    resposta = document_chain.invoke({"context": docs_relevantes, "input": mensagem})
    txt = (resposta or "").strip()
    if txt.rstrip(".!?") == "Não sei":
        return {"resposta": "Desculpe, não consegui encontrar informações relevantes para sua pergunta.", "docs": [], "contexto_encontrado": False}
    return {"resposta": txt, "docs": docs_relevantes, "contexto_encontrado": True}

def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def extrair_trecho(texto: str, query: str, janela: int = 240) -> str:
    txt = _clean_text(texto)
    termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
    pos = -1
    for t in termos:
        pos = txt.lower().find(t)
        if pos != -1: break
    if pos == -1: pos = 0
    ini, fim = max(0, pos - janela//2), min(len(txt), pos + janela//2)
    return txt[ini:fim]

def formatar_citacoes(docs_rel: List, query: str) -> List[Dict]:
    cites, seen = [], set()
    for d in docs_rel:
        src = pathlib.Path(d.metadata.get("source","")).name
        page = int(d.metadata.get("page", 0)) + 1
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        cites.append({"documento": src, "pagina": page, "trecho": extrair_trecho(d.page_content, query)})
    return cites[:3]

for t in testes:
    resposta = rag(t)
    print(f"Pergunta: {t}")
    print(f"Resposta: {resposta['resposta']}")
    if resposta["contexto_encontrado"]:
        print("Citações:")
        for c in formatar_citacoes(resposta["docs"], t):
            print(f"- {c['documento']} (pág. {c['pagina']}): ...{c['trecho']}...")
    print(f"Contexto encontrado: {resposta['contexto_encontrado']}")
    print("---")

class AgentState(TypedDict):
    mensagem: str
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool
    acao_final: str

def node_triagem(state: AgentState) -> AgentState:
    print("Executando nó de triagem...")
    return {"triagem": triagem(state["mensagem"])}

def node_auto_resolver(state: AgentState) -> AgentState:
    print("Executando nó de auto-resolver...")
    resposta_rag = rag(state["mensagem"])
    update: AgentState = {
        "resposta": resposta_rag["resposta"],
        "citacoes": formatar_citacoes(resposta_rag["docs"], state["mensagem"]) if resposta_rag["contexto_encontrado"] else [],
        "rag_sucesso": resposta_rag["contexto_encontrado"]
    }
    if resposta_rag["contexto_encontrado"]:
        update["acao_final"] = "RESPONDER_USUARIO"
    return update

def node_pedir_info(state: AgentState) -> AgentState:
    print("Executando nó de pedir info...")
    campos = state["triagem"].get("campos_faltantes", [])
    if not campos:
        campos = ["detalhes sobre a dúvida ou pedido"]
    pergunta = "Por favor, forneça as seguintes informações para que eu possa ajudar melhor: " + ", ".join(campos) + "."
    return {"resposta": pergunta, "acao_final": "PEDIR_INFO_USUARIO"}

def node_abrir_chamado(state: AgentState) -> AgentState:
    print("Executando nó de abrir chamado...")
    triagem = state["triagem"]
    return {
        "resposta": f"Sua solicitação foi encaminhada para abertura de chamado com urgência {triagem['urgencia']}. Descrição: {state['mensagem'][:140]}...",
        "acao_final": "ABRIR_CHAMADO"
    }

KEYWORDS_ABRIR_TICKET = ["abrir chamado", "solicito liberação", "quero exceção", "por favor, abra um chamado", "preciso de aprovação", "necessito acesso especial"]

def decidir_principal(state: AgentState) -> str:
    print("Decidindo próximo nó...")
    decisao = state["triagem"]["decisao"]
    if decisao == "AUTO_RESOLVER":
        return "AUTO_RESOLVER"
    elif decisao == "PEDIR_INFO":
        return "PEDIR_INFO"
    elif decisao == "ABRIR_CHAMADO":
        return "ABRIR_CHAMADO"
    else:
        msg_lower = state["mensagem"].lower()
        if any(kw in msg_lower for kw in KEYWORDS_ABRIR_TICKET):
            return "ABRIR_CHAMADO"
        elif len(state["mensagem"].split()) < 5:
            return "PEDIR_INFO"
        else:
            return "AUTO_RESOLVER"

def decidir_pos_auto_resolver(state: AgentState) -> str:
    print("Decidindo pós auto-resolver...")
    if state.get("rag_sucesso"):
        return "FIM"
    else:
        return "PEDIR_INFO"
    
workflow = StateGraph(AgentState)

workflow.add_node('triagem', node_triagem)
workflow.add_node('auto_resolver', node_auto_resolver)
workflow.add_node('pedir_info', node_pedir_info)
workflow.add_node('abrir_chamado', node_abrir_chamado)

workflow.add_edge(START, 'triagem')
workflow.add_conditional_edges('triagem', decidir_principal, {
    "AUTO_RESOLVER": 'auto_resolver',
    "PEDIR_INFO": 'pedir_info',
    "ABRIR_CHAMADO": 'abrir_chamado'})

workflow.add_conditional_edges('auto_resolver', decidir_pos_auto_resolver, {
    "PEDIR_INFO": 'pedir_info',
    "CHAMADO": 'abrir_chamado',
    "FIM": END
})

grafo = workflow.compile()

graph_bytes = grafo.get_graph().draw_mermaid_png()

def salvar_grafo_imagem(bytes_img: bytes, nome_arquivo: str = "grafo.png"):
    with open(nome_arquivo, "wb") as f:
        f.write(bytes_img)
    print(f"Grafo salvo em {nome_arquivo}")

salvar_grafo_imagem(graph_bytes)

for t in testes:
    resposta_final = grafo.invoke({"mensagem": t})

    triag = resposta_final.get("triagem", {})
    print(f"Pergunta: {t}")
    print(f"Decisão de triagem: {triag.get('decisao','N/A')}, Urgência: {triag.get('urgencia','N/A')}, Campos faltantes: {triag.get('campos_faltantes',[])}")
    print(f"Ação final: {resposta_final.get('acao_final','N/A')}")
    print(f"Resposta ao usuário: {resposta_final.get('resposta','N/A')}")
    if resposta_final.get("citacoes"):
        print("Citações:")
        for c in resposta_final["citacoes"]:
            print(f"- {c['documento']} (pág. {c['pagina']}): ...{c['trecho']}...")
    print("---")