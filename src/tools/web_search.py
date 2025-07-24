"""
Ferramenta de busca na web usando DuckDuckGo para informações atualizadas
"""

import json
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta

try:
    from duckduckgo_search import DDGS
except ImportError:
    print("duckduckgo_search não instalado. Execute: pip install duckduckgo-search")
    DDGS = None

from bs4 import BeautifulSoup
import requests


class WebSearchTool:
    def __init__(self):
        self.ddgs = DDGS() if DDGS else None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search_web(
        self,
        query: str,
        num_results: int = 5,
        region: str = "br-pt",
        time_filter: str = None
    ) -> List[Dict]:
        """
        Busca informações na web usando DuckDuckGo
        
        Args:
            query: Consulta de busca
            num_results: Número de resultados
            region: Região da busca (padrão: Brasil em português)
            time_filter: Filtro de tempo (d=dia, w=semana, m=mês, y=ano)
            
        Returns:
            Lista de resultados da busca
        """
        if not self.ddgs:
            return [{"error": "DuckDuckGo search não disponível"}]
        
        try:
            results = []
            search_params = {
                "keywords": query,
                "region": region,
                "max_results": num_results
            }
            
            if time_filter:
                search_params["timelimit"] = time_filter
            
            for result in self.ddgs.text(**search_params):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "source": self._extract_domain(result.get("href", ""))
                })
            
            return results
            
        except Exception as e:
            return [{"error": f"Erro na busca: {str(e)}"}]
    
    def search_news(
        self,
        query: str,
        num_results: int = 5,
        time_filter: str = "w"  # última semana
    ) -> List[Dict]:
        """
        Busca notícias recentes
        
        Args:
            query: Consulta de busca
            num_results: Número de resultados
            time_filter: Filtro temporal
            
        Returns:
            Lista de notícias
        """
        if not self.ddgs:
            return [{"error": "DuckDuckGo search não disponível"}]
        
        try:
            results = []
            
            for result in self.ddgs.news(
                keywords=query,
                region="br-pt",
                max_results=num_results,
                timelimit=time_filter
            ):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("body", ""),
                    "source": result.get("source", ""),
                    "date": result.get("date", "")
                })
            
            return results
            
        except Exception as e:
            return [{"error": f"Erro na busca de notícias: {str(e)}"}]
    
    def search_travel_info(
        self,
        destination: str,
        travel_type: str = "sustentável"
    ) -> Dict:
        """
        Busca informações específicas de viagem
        
        Args:
            destination: Destino da viagem
            travel_type: Tipo de viagem (sustentável, ecológica, etc.)
            
        Returns:
            Dicionário com informações de viagem
        """
        queries = [
            f"{destination} turismo {travel_type}",
            f"{destination} hotéis ecológicos sustentáveis",
            f"{destination} transporte público como chegar",
            f"{destination} atrações locais cultura",
            f"{destination} eventos atuais 2024"
        ]
        
        all_results = {}
        
        for i, query in enumerate(queries):
            category = ["turismo", "hotéis", "transporte", "atrações", "eventos"][i]
            all_results[category] = self.search_web(query, num_results=3)
        
        return {
            "destination": destination,
            "search_results": all_results,
            "summary": self._summarize_travel_info(all_results)
        }
    
    def search_local_events(
        self,
        city: str,
        date_range: str = "próxima semana"
    ) -> List[Dict]:
        """
        Busca eventos locais em uma cidade
        
        Args:
            city: Nome da cidade
            date_range: Período de busca
            
        Returns:
            Lista de eventos encontrados
        """
        queries = [
            f"{city} eventos {date_range}",
            f"{city} shows concertos {date_range}",
            f"{city} exposições museus",
            f"{city} festivais locais",
            f"{city} atividades fim de semana"
        ]
        
        all_events = []
        
        for query in queries:
            results = self.search_web(query, num_results=3, time_filter="w")
            for result in results:
                if not result.get("error"):
                    all_events.append({
                        "title": result["title"],
                        "url": result["url"],
                        "description": result["snippet"],
                        "source": result["source"],
                        "query_type": query.split()[1]  # tipo de evento
                    })
        
        return all_events
    
    def search_sustainable_options(
        self,
        location: str,
        category: str = "geral"
    ) -> List[Dict]:
        """
        Busca opções sustentáveis em uma localização
        
        Args:
            location: Localização
            category: Categoria (transporte, hospedagem, alimentação, etc.)
            
        Returns:
            Lista de opções sustentáveis
        """
        category_queries = {
            "transporte": [
                f"{location} transporte público sustentável",
                f"{location} bike sharing bicicletas",
                f"{location} carona solidária"
            ],
            "hospedagem": [
                f"{location} hotéis sustentáveis eco",
                f"{location} pousadas ecológicas",
                f"{location} albergues verdes"
            ],
            "alimentação": [
                f"{location} restaurantes orgânicos",
                f"{location} comida local sustentável", 
                f"{location} mercados locais"
            ],
            "geral": [
                f"{location} turismo sustentável",
                f"{location} ecoturismo",
                f"{location} atividades ecológicas"
            ]
        }
        
        queries = category_queries.get(category, category_queries["geral"])
        results = []
        
        for query in queries:
            search_results = self.search_web(query, num_results=3)
            results.extend(search_results)
        
        return results
    
    def _extract_domain(self, url: str) -> str:
        """Extrai domínio de uma URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return url
    
    def _summarize_travel_info(self, search_results: Dict) -> Dict:
        """Gera resumo das informações de viagem"""
        summary = {
            "total_results": 0,
            "categories_found": [],
            "key_findings": []
        }
        
        for category, results in search_results.items():
            if results and not results[0].get("error"):
                summary["categories_found"].append(category)
                summary["total_results"] += len(results)
                
                # Extrair insights básicos
                for result in results[:2]:  # Primeiros 2 resultados por categoria
                    if "sustentável" in result.get("snippet", "").lower():
                        summary["key_findings"].append(f"Opções sustentáveis encontradas em {category}")
                        break
        
        return summary
    
    def search_carbon_offset_programs(self, region: str = "Brasil") -> List[Dict]:
        """
        Busca programas de compensação de carbono
        
        Args:
            region: Região para buscar programas
            
        Returns:
            Lista de programas encontrados
        """
        queries = [
            f"{region} compensação carbono programas",
            f"{region} créditos carbono comprar",
            f"offset carbono {region} sustentabilidade"
        ]
        
        results = []
        for query in queries:
            search_results = self.search_web(query, num_results=3)
            results.extend(search_results)
        
        return results
    
    def get_travel_advisories(self, destination: str) -> List[Dict]:
        """
        Busca avisos e recomendações de viagem
        
        Args:
            destination: Destino da viagem
            
        Returns:
            Lista de avisos encontrados
        """
        queries = [
            f"{destination} avisos viagem segurança",
            f"{destination} recomendações turismo",
            f"{destination} situação atual turistas"
        ]
        
        advisories = []
        for query in queries:
            results = self.search_news(query, num_results=2, time_filter="w")
            advisories.extend(results)
        
        return advisories


def create_web_search_tools():
    """Cria ferramentas do LangChain para busca web"""
    from langchain.tools import Tool
    
    search_tool = WebSearchTool()
    
    def web_search_tool(query: str) -> str:
        """
        Busca informações na web.
        
        Input: Consulta de busca
        """
        try:
            results = search_tool.search_web(query, num_results=5)
            
            if results and results[0].get("error"):
                return f"Erro na busca: {results[0]['error']}"
            
            search_text = f"Resultados da busca para '{query}':\n\n"
            
            for i, result in enumerate(results[:3], 1):
                search_text += f"{i}. {result['title']}\n"
                search_text += f"   {result['snippet'][:200]}...\n"
                search_text += f"   Fonte: {result['source']}\n\n"
            
            return search_text
            
        except Exception as e:
            return f"Erro na busca: {str(e)}"
    
    def travel_info_search_tool(input_str: str) -> str:
        """
        Busca informações específicas de viagem.
        
        Input: "destino:tipo_viagem" ou apenas "destino"
        """
        try:
            parts = input_str.split(":")
            destination = parts[0]
            travel_type = parts[1] if len(parts) > 1 else "sustentável"
            
            result = search_tool.search_travel_info(destination, travel_type)
            
            info_text = f"Informações de viagem para {destination}:\n\n"
            
            for category, results in result["search_results"].items():
                if results and not results[0].get("error"):
                    info_text += f"{category.title()}:\n"
                    for res in results[:2]:
                        info_text += f"- {res['title']}: {res['snippet'][:150]}...\n"
                    info_text += "\n"
            
            summary = result["summary"]
            info_text += f"Resumo: {len(summary['categories_found'])} categorias encontradas"
            if summary["key_findings"]:
                info_text += f"\nDestaques: {'; '.join(summary['key_findings'])}"
            
            return info_text
            
        except Exception as e:
            return f"Erro na busca de informações de viagem: {str(e)}"
    
    def local_events_search_tool(input_str: str) -> str:
        """
        Busca eventos locais em uma cidade.
        
        Input: "cidade:período" ou apenas "cidade"
        """
        try:
            parts = input_str.split(":")
            city = parts[0]
            date_range = parts[1] if len(parts) > 1 else "próxima semana"
            
            events = search_tool.search_local_events(city, date_range)
            
            if not events:
                return f"Nenhum evento encontrado para {city}"
            
            events_text = f"Eventos em {city} ({date_range}):\n\n"
            
            for event in events[:5]:  # Primeiros 5 eventos
                events_text += f"• {event['title']}\n"
                events_text += f"  {event['description'][:100]}...\n"
                events_text += f"  Tipo: {event['query_type']}\n\n"
            
            return events_text
            
        except Exception as e:
            return f"Erro na busca de eventos: {str(e)}"
    
    def sustainable_options_search_tool(input_str: str) -> str:
        """
        Busca opções sustentáveis em uma localização.
        
        Input: "localização:categoria" ou apenas "localização"
        """
        try:
            parts = input_str.split(":")
            location = parts[0]
            category = parts[1] if len(parts) > 1 else "geral"
            
            options = search_tool.search_sustainable_options(location, category)
            
            if not options or options[0].get("error"):
                return f"Nenhuma opção sustentável encontrada para {location}"
            
            options_text = f"Opções sustentáveis em {location} ({category}):\n\n"
            
            for option in options[:4]:  # Primeiras 4 opções
                if not option.get("error"):
                    options_text += f"• {option['title']}\n"
                    options_text += f"  {option['snippet'][:120]}...\n\n"
            
            return options_text
            
        except Exception as e:
            return f"Erro na busca de opções sustentáveis: {str(e)}"
    
    return [
        Tool(
            name="WebSearch",
            func=web_search_tool,
            description="Busca informações gerais na web. Input: consulta de busca"
        ),
        Tool(
            name="TravelInfoSearch",
            func=travel_info_search_tool,
            description="Busca informações específicas de viagem. Input: 'destino:tipo' ou 'destino'"
        ),
        Tool(
            name="LocalEventsSearch",
            func=local_events_search_tool,
            description="Busca eventos locais. Input: 'cidade:período' ou 'cidade'"
        ),
        Tool(
            name="SustainableOptionsSearch",
            func=sustainable_options_search_tool,
            description="Busca opções sustentáveis. Input: 'localização:categoria' ou 'localização'"
        )
    ]


if __name__ == "__main__":
    # Teste básico
    search = WebSearchTool()
    
    # Teste busca geral
    results = search.search_web("Rio de Janeiro turismo sustentável", num_results=3)
    print("Busca geral:")
    for result in results:
        if not result.get("error"):
            print(f"- {result['title']}: {result['snippet'][:100]}...")
    
    # Teste informações de viagem
    travel_info = search.search_travel_info("Salvador", "sustentável")
    print(f"\nInformações de viagem: {travel_info['summary']}")
    
    # Teste eventos locais
    events = search.search_local_events("São Paulo")
    print(f"\nEventos encontrados: {len(events)}")