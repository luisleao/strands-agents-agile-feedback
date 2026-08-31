import json
import os
import unicodedata

from strands import Agent, tool
from strands.models.ollama import OllamaModel


SYSTEM_PROMPT="""Você é um agente de análise de produto.

Seu trabalho:
1. Use 'extrair_pontos_de_dor' para identificar os problemas reportados
2. Para cada ponto extraído, use 'classificar_severidade' para avaliar impacto
3. Com todas as classificações prontas, use 'gerar_resumo_executivo' para priorizar

Responda em português, de forma direta e estruturada.
Ao final, apresente o resumo com os top 3 itens e ações recomendadas.
Não formate em markdown.
"""

COMANDO="""
Analise o seguinte feedback coletado de usuários esta semana:

"O app trava toda vez que aplico cupom no checkout pelo celular. 
A dashboard demora uma eternidade pra carregar, deve ter uns 8 segundos. 
Recebi o mesmo email de confirmação 3 vezes. 
Ocorre um erro quando tento abrir a nota fiscal.
A busca por SKU não retorna nada, tenho que buscar pelo nome completo."

Extraia os pontos de dor, classifique por severidade e me dê um plano de ação priorizado.
"""





# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
modelo = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.1"
    #temperature=0.1
)

# Agente auxiliar (sem tools) usado pelas tools que precisam de raciocínio de LLM.
# Manter separado do agente principal evita recursão de tool calls.
_llm_auxiliar = Agent(model=modelo)


def _consultar_llm(prompt: str) -> str:
    """Envia um prompt ao modelo e devolve a resposta como texto puro."""
    return str(_llm_auxiliar(prompt)).strip()


# ---------------------------------------------------------------------------
# Base de pontos de dor conhecidos (arquivo externo)
# ---------------------------------------------------------------------------
CAMINHO_BASE_CONHECIDA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pontos_de_dor_conhecidos.json"
)


def _normalizar(texto: str) -> str:
    """Remove acentos e coloca em minúsculas para facilitar o matching."""
    texto = texto.lower()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c))


def _carregar_base_conhecida() -> list:
    """Carrega o catálogo de pontos de dor já reconhecidos do arquivo JSON."""
    try:
        with open(CAMINHO_BASE_CONHECIDA, "r", encoding="utf-8") as f:
            dados = json.load(f)
        return dados.get("pontos_conhecidos", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"\033[31mErro ao carregar base de pontos de dor: {e}\033[0m")
        return []


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def extrair_pontos_de_dor(feedback_texto: str) -> str:
    """Extrai pontos de dor de um texto de feedback de usuários.
    Compara o feedback com um catálogo externo de erros já reconhecidos
    (pontos_de_dor_conhecidos.json) usando as palavras-chave de cada item.
    Retorna quais problemas conhecidos foram identificados no texto.
    """
    base = _carregar_base_conhecida()
    feedback_norm = _normalizar(feedback_texto)

    reconhecidos = []
    for item in base:
        for chave in item.get("palavras_chave", []):
            if _normalizar(chave) in feedback_norm:
                reconhecidos.append(item)
                break  # basta uma palavra-chave casar

    if not reconhecidos:
        return (
            "Nenhum ponto de dor reconhecido no catálogo foi identificado. "
            "Pode ser um problema novo, ainda não catalogado."
        )

    linhas = []
    for item in reconhecidos:
        linhas.append(
            f"- [{item['id']}] {item['descricao']} (categoria: {item.get('categoria', 'n/d')})"
        )
    return "\n".join(linhas)





@tool
def classificar_severidade(ponto_de_dor: str) -> str:
    """Classifica um ponto de dor por severidade e tipo de ação recomendada usando uma LLM."""
    print("\033[33mSEVERIDADE:", ponto_de_dor, "\033[0m")

    prompt = f"""Você é um especialista em priorização de bugs e produto.

Classifique o ponto de dor abaixo.

Ponto de dor: "{ponto_de_dor}"

Responda EXATAMENTE neste formato de uma linha, sem markdown e sem explicações extras:
Ponto: <ponto de dor> | Severidade: <Crítica|Alta|Média|Baixa> | Impacto: <frase curta> | Ação: <ação recomendada objetiva>

Considere impacto no usuário, no negócio (receita, retenção) e urgência ao definir a severidade."""

    return _consultar_llm(prompt)


@tool
def gerar_resumo_executivo(analises: str) -> str:
    """Gera um resumo executivo com os top 3 itens priorizados para ação imediata usando uma LLM."""
    print("\033[31mRESUMO EXECUTIVO:", analises, "\033[0m")

    prompt = f"""Você é um líder de produto escrevendo um resumo executivo.

Com base nas análises de severidade abaixo, produza um resumo executivo em português.
Análises:
{analises}

Requisitos da resposta:
- Não use markdown.
- Liste os TOP 3 itens mais prioritários, do mais crítico ao menos crítico.
- Para cada item traga: problema, severidade e ação recomendada.
- Finalize com uma seção curta "Plano de ação" com a ordem de execução sugerida.
- Seja direto e objetivo."""

    return _consultar_llm(prompt)


_after_tool = False

def callback_handler(**kwargs):
    global _after_tool
    if "reasoningText" in kwargs:
        print(f"💭 {kwargs['reasoningText']}", end="", flush=True)
    if "data" in kwargs:
        if _after_tool:
            print("\n")
            _after_tool = False
        print(kwargs["data"], end="", flush=True)
    if "current_tool_use" in kwargs:
        _after_tool = True
        t = kwargs["current_tool_use"]
        if t.get("name"):
            print(f"\n\n🔧 Ferramenta: {t['name']}")
        if t.get("input"):
            print(f"   Parâmetros: {t['input']}")


# ---------------------------------------------------------------------------
# Agente principal
# ---------------------------------------------------------------------------
agent = Agent(
    model=modelo,
    system_prompt=SYSTEM_PROMPT,
    tools=[
        extrair_pontos_de_dor,
        classificar_severidade,
        gerar_resumo_executivo,
    ],
    callback_handler=callback_handler

)


if __name__ == "__main__":
    resposta = agent(COMANDO)
    print(resposta)

