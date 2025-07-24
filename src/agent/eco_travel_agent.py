"""
EcoTravel Agent - Agente principal para planejamento de viagens sustentáveis
Implementa padrão ReAct com integração RAG e múltiplas ferramentas
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from dotenv import load_dotenv

# Adicionar src ao path para imports
sys.path.append(str(Path(__file__).parent.parent))

# Imports do LangChain
from langchain.agents import initialize_agent, AgentType
from langchain.agents.react.base import DocstoreExplorer
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage
from langchain.callbacks.base import BaseCallbackHandler

# Imports locais
from rag.rag_system import AdvancedRAGSystem
from tools.carbon_calculator import create_carbon_calculator_tool
from tools.weather_api import create_weather_tools
from tools.web_search import create_web_search_tools

# LLM imports (tentativa de múltiplas opções)
try:
    from langchain.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain.llms import HuggingFacePipeline
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


class EcoTravelCallbackHandler(BaseCallbackHandler):
    """Handler personalizado para logging do agente"""
    
    def __init__(self):
        self.steps = []
        self.current_step = {}
    
    def on_agent_action(self, action, **kwargs):
        """Registra ações do agente"""
        self.current_step = {
            "action": action.tool,
            "input": action.tool_input,
            "log": action.log
        }
    
    def on_agent_finish(self, finish, **kwargs):
        """Registra conclusão do agente"""
        if self.current_step:
            self.current_step["result"] = finish.return_values
            self.steps.append(self.current_step.copy())
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Registra início de uso de ferramenta"""
        if self.current_step:
            self.current_step["tool_start"] = {
                "tool": serialized.get("name", "unknown"),
                "input": input_str
            }
    
    def on_tool_end(self, output, **kwargs):
        """Registra fim de uso de ferramenta"""
        if self.current_step:
            self.current_step["tool_output"] = output
            self.steps.append(self.current_step.copy())
            self.current_step = {}
    
    def get_execution_steps(self):
        """Retorna passos da execução"""
        return self.steps


class EcoTravelAgent:
    def __init__(
        self,
        data_path: str = "data",
        model_name: str = "auto",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        memory_window: int = 10,
        verbose: bool = True
    ):
        # Carregar variáveis de ambiente
        load_dotenv()
        
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        
        # Configurar componentes
        self.llm = self._setup_llm()
        self.rag_system = None
        self.tools = []
        self.agent = None
        self.callback_handler = EcoTravelCallbackHandler()
        
        # Configurar memória
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        # Inicializar sistemas
        self._initialize_rag()
        self._initialize_tools()
        self._initialize_agent()
    
    def _setup_llm(self):
        """Configura o LLM baseado na disponibilidade"""
        
        # Tentar Ollama primeiro (local)
        if self.model_name == "auto" or self.model_name.startswith("ollama"):
            if OLLAMA_AVAILABLE:
                try:
                    model = "llama3" if self.model_name == "auto" else self.model_name.replace("ollama:", "")
                    llm = Ollama(
                        model=model,
                        temperature=self.temperature,
                        num_predict=self.max_tokens
                    )
                    # Testar conexão
                    llm.invoke("Test")
                    print(f"Usando Ollama com modelo: {model}")
                    return llm
                except Exception as e:
                    print(f"Ollama não disponível: {e}")
        
        # Tentar OpenAI
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                print("Usando OpenAI GPT-3.5-turbo")
                return llm
            except Exception as e:
                print(f"OpenAI não disponível: {e}")
        
        # Fallback para HuggingFace local
        if HUGGINGFACE_AVAILABLE:
            try:
                from transformers import pipeline
                pipe = pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-medium",
                    max_length=self.max_tokens,
                    temperature=self.temperature
                )
                llm = HuggingFacePipeline(pipeline=pipe)
                print("Usando HuggingFace local")
                return llm
            except Exception as e:
                print(f"HuggingFace não disponível: {e}")
        
        # Último recurso: LLM mock para desenvolvimento
        print("AVISO: Usando LLM mock para desenvolvimento. Instale Ollama, configure OpenAI ou HuggingFace.")
        return MockLLM()
    
    def _initialize_rag(self):
        """Inicializa sistema RAG"""
        try:
            self.rag_system = AdvancedRAGSystem(data_path=str(self.data_path))
            
            # Verificar se já existem dados
            if not self._check_existing_data():
                print("Criando dados de exemplo...")
                from rag.rag_system import create_sample_data
                create_sample_data()
            
            print("Construindo índices RAG...")
            self.rag_system.build_index()
            print("Sistema RAG inicializado com sucesso!")
            
        except Exception as e:
            print(f"Erro ao inicializar RAG: {e}")
            self.rag_system = None
    
    def _check_existing_data(self) -> bool:
        """Verifica se existem dados no diretório"""
        data_files = [
            self.data_path / "guias" / "guia_sustentavel_brasil.txt",
            self.data_path / "emissoes" / "emissoes_transporte.csv",
            self.data_path / "avaliacoes" / "hoteis_sustentaveis.json"
        ]
        return all(f.exists() for f in data_files)
    
    def _initialize_tools(self):
        """Inicializa todas as ferramentas"""
        self.tools = []
        
        # Ferramenta RAG
        if self.rag_system:
            self.tools.append(self._create_rag_tool())
        
        # Ferramentas de cálculo de carbono
        try:
            carbon_tools = create_carbon_calculator_tool()
            self.tools.extend(carbon_tools)
            print(f"Adicionadas {len(carbon_tools)} ferramentas de carbono")
        except Exception as e:
            print(f"Erro ao inicializar ferramentas de carbono: {e}")
        
        # Ferramentas de clima
        try:
            weather_tools = create_weather_tools()
            self.tools.extend(weather_tools)
            print(f"Adicionadas {len(weather_tools)} ferramentas de clima")
        except Exception as e:
            print(f"Erro ao inicializar ferramentas de clima: {e}")
        
        # Ferramentas de busca web
        try:
            web_tools = create_web_search_tools()
            self.tools.extend(web_tools)
            print(f"Adicionadas {len(web_tools)} ferramentas de busca")
        except Exception as e:
            print(f"Erro ao inicializar ferramentas de busca: {e}")
        
        print(f"Total de ferramentas inicializadas: {len(self.tools)}")
    
    def _create_rag_tool(self):
        """Cria ferramenta RAG"""
        from langchain.tools import Tool
        
        def rag_search(query: str) -> str:
            """Busca informações na base de conhecimento sustentável"""
            try:
                context = self.rag_system.get_context_for_query(query)
                if context:
                    return f"Informações encontradas na base de conhecimento:\n\n{context}"
                else:
                    return "Nenhuma informação relevante encontrada na base de conhecimento."
            except Exception as e:
                return f"Erro na busca RAG: {str(e)}"
        
        return Tool(
            name="SustainableTravelKnowledge",
            func=rag_search,
            description="Busca informações sobre viagens sustentáveis na base de conhecimento"
        )
    
    def _initialize_agent(self):
        """Inicializa o agente ReAct"""
        if not self.tools:
            raise ValueError("Nenhuma ferramenta disponível para o agente")
        
        # Prompt personalizado para o agente
        system_prompt = self._create_system_prompt()
        
        try:
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.memory,
                verbose=self.verbose,
                callbacks=[self.callback_handler],
                handle_parsing_errors=True,
                max_iterations=10,
                early_stopping_method="generate"
            )
            
            # Configurar prompt personalizado se possível
            if hasattr(self.agent, 'agent') and hasattr(self.agent.agent, 'llm_chain'):
                self.agent.agent.llm_chain.prompt.template = system_prompt
            
            print("Agente inicializado com sucesso!")
            
        except Exception as e:
            print(f"Erro ao inicializar agente: {e}")
            raise
    
    def _create_system_prompt(self) -> str:
        """Cria prompt do sistema para o agente"""
        return """Você é o EcoTravel Agent, um assistente especializado em planejamento de viagens sustentáveis.

OBJETIVO: Ajudar usuários a planejar viagens considerando sustentabilidade, redução de pegada de carbono e experiências locais autênticas.

FERRAMENTAS DISPONÍVEIS:
- SustainableTravelKnowledge: Base de conhecimento sobre viagens sustentáveis
- CarbonFootprintCalculator: Calcula emissões de CO2 de diferentes transportes
- TransportModeComparison: Compara emissões entre modais de transporte
- SustainabilityRecommendation: Gera recomendações sustentáveis
- CityWeather: Informações meteorológicas de cidades
- TravelWeatherAnalysis: Análise de clima para viagens
- WebSearch: Busca informações atualizadas na web
- TravelInfoSearch: Busca informações específicas de viagem
- LocalEventsSearch: Busca eventos locais
- SustainableOptionsSearch: Busca opções sustentáveis

PROCESSO DE RACIOCÍNIO:
1. ANÁLISE: Entenda exatamente o que o usuário está pedindo
2. PESQUISA: Use as ferramentas para coletar informações relevantes
3. CÁLCULO: Calcule pegadas de carbono quando aplicável
4. COMPARAÇÃO: Compare diferentes opções de transporte/hospedagem
5. RECOMENDAÇÃO: Forneça recomendações sustentáveis específicas
6. CONTEXTO: Inclua informações sobre clima, eventos locais e cultura

DIRETRIZES:
- SEMPRE priorize sustentabilidade nas recomendações
- Calcule e compare pegadas de carbono quando relevante
- Considere fatores como clima, eventos locais e cultura
- Forneça alternativas práticas e viáveis
- Seja específico com números, custos e emissões
- Explique o "porquê" das recomendações sustentáveis

Responda de forma estruturada, didática e sempre com foco na sustentabilidade.

{chat_history}

Pergunta: {input}

{agent_scratchpad}"""
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Executa uma consulta no agente
        
        Args:
            query: Pergunta do usuário
            
        Returns:
            Dicionário com resposta e metadados
        """
        try:
            # Limpar steps anteriores
            self.callback_handler.steps = []
            
            # Executar agente
            response = self.agent.run(input=query)
            
            # Coletar metadados
            execution_steps = self.callback_handler.get_execution_steps()
            
            return {
                "response": response,
                "query": query,
                "execution_steps": execution_steps,
                "tools_used": list(set([step.get("action") for step in execution_steps if step.get("action")])),
                "success": True
            }
            
        except Exception as e:
            return {
                "response": f"Erro na execução: {str(e)}",
                "query": query,
                "execution_steps": [],
                "tools_used": [],
                "success": False,
                "error": str(e)
            }
    
    def get_sustainability_score(self, travel_plan: Dict) -> Dict:
        """
        Calcula score de sustentabilidade para um plano de viagem
        
        Args:
            travel_plan: Dicionário com detalhes do plano
            
        Returns:
            Score e análise de sustentabilidade
        """
        score = 0
        factors = []
        
        # Avaliar transporte (peso: 40%)
        transport_mode = travel_plan.get("transport_mode", "").lower()
        if transport_mode in ["trem", "onibus"]:
            score += 40
            factors.append("Transporte de baixa emissão")
        elif transport_mode == "carro" and travel_plan.get("passengers", 1) > 1:
            score += 25
            factors.append("Transporte compartilhado")
        elif transport_mode == "aviao":
            score += 10
            factors.append("Transporte de alta emissão")
        
        # Avaliar hospedagem (peso: 30%)
        if travel_plan.get("eco_hotel", False):
            score += 30
            factors.append("Hospedagem sustentável")
        elif travel_plan.get("local_accommodation", False):
            score += 20
            factors.append("Hospedagem local")
        
        # Avaliar atividades (peso: 20%)
        if travel_plan.get("local_activities", False):
            score += 20
            factors.append("Atividades locais")
        elif travel_plan.get("eco_activities", False):
            score += 15
            factors.append("Ecoturismo")
        
        # Avaliar duração vs distância (peso: 10%)
        distance = travel_plan.get("distance_km", 0)
        duration = travel_plan.get("duration_days", 1)
        if distance > 0 and duration > 0:
            if distance / duration < 100:  # Menos de 100km por dia
                score += 10
                factors.append("Viagem com ritmo sustentável")
        
        # Categorizar score
        if score >= 80:
            category = "Muito Sustentável"
        elif score >= 60:
            category = "Sustentável"
        elif score >= 40:
            category = "Moderadamente Sustentável"
        else:
            category = "Pouco Sustentável"
        
        return {
            "score": score,
            "category": category,
            "factors": factors,
            "recommendations": self._get_sustainability_recommendations(score, travel_plan)
        }
    
    def _get_sustainability_recommendations(self, score: int, travel_plan: Dict) -> List[str]:
        """Gera recomendações para melhorar sustentabilidade"""
        recommendations = []
        
        if score < 80:
            if travel_plan.get("transport_mode", "").lower() == "aviao":
                recommendations.append("Considere compensar as emissões de carbono do voo")
                recommendations.append("Para próximas viagens, avalie alternativas como trem ou ônibus")
            
            if not travel_plan.get("eco_hotel", False):
                recommendations.append("Busque hospedagens com certificações sustentáveis")
            
            if not travel_plan.get("local_activities", False):
                recommendations.append("Inclua atividades que beneficiem a comunidade local")
            
            recommendations.append("Prefira produtos e serviços locais durante a viagem")
        
        return recommendations
    
    def chat(self):
        """Interface de chat interativo"""
        print("=== EcoTravel Agent - Assistente de Viagens Sustentáveis ===")
        print("Digite 'sair' para encerrar ou 'ajuda' para ver exemplos\n")
        
        while True:
            try:
                user_input = input("Você: ").strip()
                
                if user_input.lower() in ['sair', 'exit', 'quit']:
                    print("Obrigado por usar o EcoTravel Agent! Viaje sustentável! 🌱")
                    break
                
                if user_input.lower() == 'ajuda':
                    self._show_help()
                    continue
                
                if not user_input:
                    continue
                
                print("\nEcoTravel Agent: Analisando sua solicitação...\n")
                
                result = self.run(user_input)
                
                print(f"EcoTravel Agent: {result['response']}\n")
                
                if self.verbose and result['tools_used']:
                    print(f"Ferramentas utilizadas: {', '.join(result['tools_used'])}\n")
                
            except KeyboardInterrupt:
                print("\n\nEncerrando EcoTravel Agent...")
                break
            except Exception as e:
                print(f"\nErro: {str(e)}\n")
    
    def _show_help(self):
        """Mostra exemplos de uso"""
        examples = [
            "Como viajar de São Paulo para o Rio de Janeiro de forma sustentável?",
            "Qual a pegada de carbono de uma viagem de avião São Paulo - Brasília?",
            "Hotéis sustentáveis em Salvador",
            "Eventos culturais em Recife na próxima semana",
            "Compare emissões entre avião, ônibus e carro para 500km",
            "Clima em Florianópolis para os próximos 3 dias"
        ]
        
        print("\n=== Exemplos de Perguntas ===")
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")
        print("")


class MockLLM:
    """LLM mock para desenvolvimento quando nenhum modelo real está disponível"""
    
    def __init__(self):
        self.responses = [
            "Como um assistente de viagens sustentáveis, recomendo considerar o transporte público ou compartilhado para reduzir a pegada de carbono.",
            "Para uma viagem mais sustentável, sugiro buscar hospedagens locais e atividades que beneficiem a comunidade.",
            "É importante calcular a pegada de carbono de diferentes modais de transporte antes de decidir.",
            "Considere compensar as emissões de carbono da sua viagem através de programas certificados."
        ]
        self.call_count = 0
    
    def invoke(self, prompt):
        """Simula resposta do LLM"""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response
    
    def __call__(self, prompt):
        return self.invoke(prompt)


def main():
    """Função principal para testar o agente"""
    try:
        # Criar agente
        agent = EcoTravelAgent(verbose=True)
        
        # Testar com uma consulta
        test_query = "Quero viajar de São Paulo para o Rio de Janeiro de forma sustentável. Qual a melhor opção?"
        
        print(f"Testando consulta: {test_query}\n")
        result = agent.run(test_query)
        
        print("=== RESULTADO ===")
        print(f"Resposta: {result['response']}")
        print(f"Ferramentas usadas: {result['tools_used']}")
        print(f"Sucesso: {result['success']}")
        
        # Iniciar chat interativo
        print("\n" + "="*50)
        agent.chat()
        
    except Exception as e:
        print(f"Erro ao inicializar agente: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()