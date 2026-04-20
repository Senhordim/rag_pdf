# RAG PDF

Pipeline de **Retrieval-Augmented Generation (RAG)** para consulta de documentos PDF usando LangChain, ChromaDB e OpenAI.

## Como funciona

1. **Carregamento** — lê todos os PDFs da pasta `assets/`
2. **Chunking** — divide os documentos em pedaços de 2.000 caracteres (com overlap de 500) usando `RecursiveCharacterTextSplitter`
3. **Vetorização** — gera embeddings via `OpenAIEmbeddings` e persiste no banco vetorial ChromaDB (pasta `db/`)
4. **Consulta** — recebe uma pergunta do usuário, busca os 3 chunks mais relevantes (score ≥ 0.7) e gera uma resposta com `ChatOpenAI`

## Estrutura

```
rag_pdf/
├── assets/          # PDFs de entrada
├── db/              # Banco vetorial ChromaDB (gerado automaticamente)
├── src/
│   └── db/
│       └── vector_db.py   # Carregamento, chunking e vetorização
├── main.py          # Interface de consulta (perguntas ao banco vetorial)
├── pyproject.toml
└── .env             # Variáveis de ambiente (não versionado)
```

## Pré-requisitos

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (gerenciador de pacotes)
- Chave de API da OpenAI

## Instalação

```bash
# Instalar dependências
uv sync

# Configurar variáveis de ambiente
cp .env.example .env
# Edite .env e adicione sua OPENAI_API_KEY
```

## Uso

### 1. Indexar PDFs

Coloque os arquivos PDF na pasta `assets/` e execute:

```bash
uv run src/db/vector_db.py
```

Carrega os PDFs, cria os chunks e persiste os embeddings no ChromaDB.

### 2. Consultar documentos

```bash
uv run main.py
```

Digite sua pergunta no terminal. O sistema busca os trechos mais relevantes e gera uma resposta baseada no conteúdo dos PDFs indexados. Caso nenhum trecho com relevância ≥ 0.7 seja encontrado, informa que não há informação suficiente.

## Dependências principais

| Pacote | Papel |
|---|---|
| `langchain` + `langchain-community` | Orquestração do pipeline RAG |
| `langchain-chroma` | Integração com ChromaDB |
| `langchain-openai` | Embeddings e LLM via OpenAI |
| `chromadb` | Banco de dados vetorial local |
| `pypdf` | Leitura de arquivos PDF |
| `python-dotenv` | Carregamento de variáveis de ambiente |
| `rich` | Output formatado no terminal |
