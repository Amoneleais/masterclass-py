import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from typing import List, Literal, Dict
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