import os
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time

# APIs e LLMs
import openai
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.llms import OpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.llms import OpenAI
        from langchain_community.chat_models import ChatOpenAI
        from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
    except ImportError:
        from langchain.llms import OpenAI
        from langchain.chat_models import ChatOpenAI
        from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks import StreamingStdOutCallbackHandler

# RAG Components
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import DirectoryLoader, CSVLoader, JSONLoader
except ImportError:
    from langchain.vectorstores import FAISS
    from langchain.document_loaders import DirectoryLoader, CSVLoader, JSONLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain_community.chains import RetrievalQA

# Tools
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.utilities import GoogleSearchAPIWrapper
except ImportError:
    from langchain.tools import DuckDuckGoSearchRun
    from langchain.utilities import GoogleSearchAPIWrapper

try:
    from langchain_experimental.tools import PythonREPLTool
except ImportError:
    from langchain.tools import PythonREPLTool
import requests

# RL Components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from rl.rl_agent import EcoTravelRLAgent
except ImportError:
    # Fallback se não conseguir importar
    EcoTravelRLAgent = None

# Configurar APIs de forma segura
if os.getenv('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
else:
    print("⚠️ OPENAI_API_KEY não configurada. Configure antes de usar o sistema completo.")

if os.getenv('GOOGLE_CSE_ID'):
    os.environ['GOOGLE_CSE_ID'] = os.getenv('GOOGLE_CSE_ID')

if os.getenv('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

class EcoTravelAgentWithRL:
    """
    Agente EcoTravel aprimorado com Reinforcement Learning
    Integra LangChain com política RL para seleção inteligente de ferramentas
    """
    
    def __init__(self, 
                 rl_model_path: Optional[str] = None,
                 use_gpt4: bool = True,
                 use_deepseek: bool = False,
                 enable_rl: bool = True):
        
        self.enable_rl = enable_rl
        self.metrics = {
            "queries_processed": 0,
            "tools_used": {},
            "average_response_time": 0,
            "co2_saved": 0,
            "user_satisfaction": []
        }
        
        # Configurar LLM principal
        if use_gpt4:
            self.llm = ChatOpenAI(
                model="gpt-4-turbo-preview",
                temperature=0.7,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()]
            )
        elif use_deepseek:
            # DeepSeek integration
            self.llm = ChatOpenAI(
                model="deepseek-chat",
                openai_api_base="https://api.deepseek.com/v1",
                openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
                temperature=0.7
            )
        else:
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        
        # Configurar embeddings de alta qualidade
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            chunk_size=1000
        )
        
        # Inicializar componente RL
        if self.enable_rl:
            self.rl_agent = EcoTravelRLAgent(
                use_advanced_embeddings=True,
                load_checkpoint=rl_model_path
            )
            print("Agente RL carregado e pronto para uso!")
        
        # Configurar RAG avançado
        self._setup_advanced_rag()
        
        # Configurar ferramentas
        self._setup_tools()
        
        # Configurar memória
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,
            return_messages=True
        )
        
        # Criar agente principal
        self._create_main_agent()
    
    def _setup_advanced_rag(self):
        """Configura sistema RAG com estratégias avançadas"""
        
        # Carregar documentos
        loaders = [
            DirectoryLoader("data/guias", glob="**/*.pdf", loader_cls=PDFLoader),
            DirectoryLoader("data/emissoes", glob="**/*.csv", loader_cls=CSVLoader),
            DirectoryLoader("data/avaliacoes", glob="**/*.json", loader_cls=JSONLoader)
        ]
        
        documents = []
        for loader in loaders:
            try:
                documents.extend(loader.load())
            except Exception as e:
                print(f"Erro ao carregar documentos: {e}")
        
        # Chunking inteligente
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Criar vector store com FAISS
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Configurar retriever híbrido (semantic + keyword)
        semantic_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # BM25 para busca por palavras-chave
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 5
        
        # Ensemble retriever combina ambos
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever],
            weights=[0.3, 0.7]  # Maior peso para busca semântica
        )
        
        # Multi-query retriever para melhorar recall
        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.ensemble_retriever,
            llm=self.llm
        )
        
        # Chain RAG com verificação anti-alucinação
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.multi_query_retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self._create_rag_prompt_with_verification()
            }
        )
    
    def _create_rag_prompt_with_verification(self):
        """Cria prompt RAG com verificação anti-alucinação"""
        from langchain.prompts import PromptTemplate
        
        template = """Você é um assistente especializado em viagens sustentáveis.
        
        Contexto recuperado:
        {context}
        
        Pergunta: {question}
        
        Instruções:
        1. Responda APENAS com base nas informações do contexto fornecido
        2. Se a informação não estiver no contexto, diga claramente que não tem essa informação
        3. Cite as fontes quando possível
        4. Focalize em aspectos de sustentabilidade e redução de CO2
        5. Seja preciso e evite inventar informações
        
        Resposta:"""
        
        return PromptTemplate(template=template, input_variables=["context", "question"])
    
    def _setup_tools(self):
        """Configura ferramentas disponíveis para o agente"""
        
        # Tool 1: RAG System
        def rag_search(query: str) -> str:
            """Busca informações na base de conhecimento sobre viagens sustentáveis"""
            result = self.rag_chain({"query": query})
            sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
            return f"{result['result']}\n\nFontes: {', '.join(set(sources))}"
        
        # Tool 2: Weather API
        def get_weather(location: str) -> str:
            """Obtém previsão do tempo via Open-Meteo API"""
            try:
                # Geocoding primeiro
                geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=pt&format=json"
                geo_response = requests.get(geo_url).json()
                
                if geo_response.get("results"):
                    lat = geo_response["results"][0]["latitude"]
                    lon = geo_response["results"][0]["longitude"]
                    
                    # Buscar previsão
                    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=America/Sao_Paulo"
                    weather_response = requests.get(weather_url).json()
                    
                    current = weather_response["current_weather"]
                    return f"Clima em {location}: {current['temperature']}°C, vento: {current['windspeed']} km/h"
                else:
                    return f"Localização {location} não encontrada"
            except Exception as e:
                return f"Erro ao buscar clima: {str(e)}"
        
        # Tool 3: Web Search (DuckDuckGo + Google como fallback)
        search_tool = DuckDuckGoSearchRun()
        
        # Tool 4: Python Calculator
        python_tool = PythonREPLTool()
        
        # Tool 5: Carbon Calculator
        def calculate_carbon(transport: str, distance: float) -> str:
            """Calcula emissões de CO2 para diferentes meios de transporte"""
            emissions = {
                "aviao": 0.255,
                "carro": 0.171,
                "onibus": 0.089,
                "trem": 0.041,
                "bicicleta": 0,
                "caminhada": 0
            }
            
            if transport.lower() in emissions:
                co2 = emissions[transport.lower()] * distance
                return f"Emissões de CO2 para {distance}km de {transport}: {co2:.2f}kg CO2"
            else:
                return f"Transporte '{transport}' não reconhecido. Opções: {', '.join(emissions.keys())}"
        
        # Criar lista de ferramentas
        self.tools = [
            Tool(
                name="RAG_Search",
                func=rag_search,
                description="Busca informações sobre viagens sustentáveis, hotéis eco-friendly, guias de destinos"
            ),
            Tool(
                name="Weather_API",
                func=get_weather,
                description="Obtém previsão do tempo e condições climáticas para qualquer localização"
            ),
            Tool(
                name="Web_Search",
                func=search_tool.run,
                description="Busca informações atuais na web sobre eventos, notícias, informações gerais"
            ),
            Tool(
                name="Python_Calculator",
                func=python_tool.run,
                description="Executa cálculos Python, análises de dados, comparações numéricas"
            ),
            Tool(
                name="Carbon_Calculator",
                func=calculate_carbon,
                description="Calcula emissões de CO2 para diferentes meios de transporte"
            )
        ]
    
    def _create_main_agent(self):
        """Cria o agente principal com ou sem RL"""
        
        if self.enable_rl:
            # Agente com seleção de ferramentas via RL
            self.agent = self._create_rl_enhanced_agent()
        else:
            # Agente padrão LangChain
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.CHAT_CONVERSATIONAL_REACT,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
    
    def _create_rl_enhanced_agent(self):
        """Cria agente que usa RL para seleção de ferramentas"""
        from langchain.agents import AgentExecutor
        from langchain.agents.conversational_chat.base import ConversationalChatAgent
        
        # Prompt customizado que integra recomendações do RL
        rl_prompt = """Você é o EcoTravel Agent, especializado em planejar viagens sustentáveis.
        
        Você tem acesso às seguintes ferramentas:
        {tools}
        
        Para responder ao usuário, siga este processo:
        1. Analise a query cuidadosamente
        2. Consulte o sistema de recomendação para escolher a melhor ferramenta
        3. Use a ferramenta recomendada primeiro
        4. Se necessário, use ferramentas adicionais
        5. Sempre focalize em sustentabilidade e redução de CO2
        
        Histórico da conversa:
        {chat_history}
        
        Usuário: {input}
        {agent_scratchpad}
        """
        
        # Criar agente customizado
        agent = ConversationalChatAgent.from_llm_and_tools(
            llm=self.llm,
            tools=self.tools,
            system_message=rl_prompt
        )
        
        # Wrapper que integra RL
        class RLAgentExecutor(AgentExecutor):
            def __init__(self, rl_agent, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.rl_agent = rl_agent
            
            def _take_next_step(self, *args, **kwargs):
                # Extrair query atual
                if args and hasattr(args[0], 'input'):
                    query = args[0].input
                    
                    # Obter recomendação do RL
                    rl_recommendation = self.rl_agent.predict_tool(query)
                    
                    # Log da recomendação
                    print(f"\n🤖 RL recomenda: {rl_recommendation['recommended_tool']} "
                          f"(confiança: {rl_recommendation['confidence']:.2%})")
                    
                    # Adicionar recomendação ao contexto
                    self.agent.llm_prefix = f"\nRecomendação do sistema: Use {rl_recommendation['recommended_tool']} primeiro.\n"
                
                return super()._take_next_step(*args, **kwargs)
        
        return RLAgentExecutor(
            rl_agent=self.rl_agent,
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )
    
    def process_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Processa uma query do usuário
        
        Returns:
            Tuple com (resposta, métricas)
        """
        start_time = time.time()
        
        # Obter recomendação RL se habilitado
        rl_info = None
        if self.enable_rl:
            rl_info = self.rl_agent.predict_tool(query)
            print(f"\n🎯 Análise RL da query:")
            print(f"   Ferramenta recomendada: {rl_info['recommended_tool']}")
            print(f"   Confiança: {rl_info['confidence']:.2%}")
            print(f"   Probabilidades: {rl_info['all_probabilities']}")
        
        # Processar com o agente
        try:
            response = self.agent.run(query)
            success = True
        except Exception as e:
            response = f"Desculpe, ocorreu um erro: {str(e)}"
            success = False
        
        # Calcular métricas
        elapsed_time = time.time() - start_time
        
        metrics = {
            "query": query,
            "response_time": elapsed_time,
            "success": success,
            "rl_recommendation": rl_info,
            "timestamp": datetime.now().isoformat()
        }
        
        # Atualizar estatísticas
        self.metrics["queries_processed"] += 1
        self.metrics["average_response_time"] = (
            (self.metrics["average_response_time"] * (self.metrics["queries_processed"] - 1) + elapsed_time) 
            / self.metrics["queries_processed"]
        )
        
        return response, metrics
    
    def provide_feedback(self, query: str, tool_used: str, satisfaction: float):
        """
        Fornece feedback para aprendizado online do RL
        
        Args:
            query: Query processada
            tool_used: Ferramenta que foi usada
            satisfaction: Satisfação do usuário (-1 a 1)
        """
        if self.enable_rl:
            tool_mapping = {"RAG": 0, "API": 1, "Search": 2, "Python": 3}
            tool_idx = tool_mapping.get(tool_used, 0)
            
            self.rl_agent.online_learning(
                query=query,
                tool_used=tool_idx,
                success=satisfaction > 0,
                latency=self.metrics["average_response_time"],
                user_feedback=satisfaction
            )
            
            self.metrics["user_satisfaction"].append(satisfaction)
            print(f"✅ Feedback registrado para aprendizado contínuo")
    
    def save_session_metrics(self, filepath: str = None):
        """Salva métricas da sessão"""
        if filepath is None:
            filepath = f"metrics/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        session_data = {
            "metrics": self.metrics,
            "rl_metrics": self.rl_agent.get_metrics() if self.enable_rl else None,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"📊 Métricas salvas em {filepath}")


# Exemplo de uso
if __name__ == "__main__":
    # Criar agente com RL
    agent = EcoTravelAgentWithRL(
        enable_rl=True,
        use_gpt4=True
    )
    
    # Queries de teste
    test_queries = [
        "Quero planejar uma viagem sustentável de São Paulo para o Rio de Janeiro",
        "Calcule as emissões de CO2 se eu for de avião vs trem",
        "Qual a previsão do tempo no Rio para próxima semana?",
        "Encontre hotéis eco-friendly no Rio com bom custo-benefício"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
        response, metrics = agent.process_query(query)
        
        print(f"\nResposta: {response}")
        print(f"\nMétricas:")
        print(f"  Tempo de resposta: {metrics['response_time']:.2f}s")
        if metrics.get('rl_recommendation'):
            print(f"  RL recomendou: {metrics['rl_recommendation']['recommended_tool']}")
        
        # Simular feedback do usuário
        satisfaction = 0.8  # Usuário satisfeito
        agent.provide_feedback(query, "RAG", satisfaction)
    
    # Salvar métricas
    agent.save_session_metrics() 