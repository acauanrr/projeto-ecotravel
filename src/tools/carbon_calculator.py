"""
Calculadora de Pegada de Carbono para diferentes modais de transporte
Baseada em dados reais do IPCC 2023
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json


class CarbonCalculator:
    """Calculadora de emissões de CO2 para transporte sustentável"""
    
    def __init__(self, emissions_data_path: str = "data/emissoes/emissoes_transporte_completo.csv"):
        self.emissions_data_path = Path(emissions_data_path)
        self.emissions_data = self._load_emissions_data()
        
        # Dados reais do IPCC 2023 como fallback
        self.emissions_factors_ipcc = {
            "aviao_domestico": 0.158,
            "aviao_internacional": 0.255,
            "carro_gasolina": 0.171,
            "carro_etanol": 0.142,
            "carro_flex": 0.156,
            "onibus_urbano": 0.089,
            "onibus_rodoviario": 0.082,
            "trem_eletrico": 0.041,
            "trem_diesel": 0.084,
            "metro": 0.028,
            "vlt": 0.035,
            "moto": 0.113,
            "bicicleta": 0.0,
            "caminhada": 0.0,
            "barco_fluvial": 0.267,
            "ferry": 0.195
        }
        
        # Mapeamento de nomes alternativos
        self.transport_aliases = {
            "aviao": "aviao_domestico",
            "avião": "aviao_domestico",
            "plane": "aviao_domestico",
            "airplane": "aviao_domestico",
            "carro": "carro_flex",
            "car": "carro_flex",
            "automóvel": "carro_flex",
            "onibus": "onibus_urbano",
            "ônibus": "onibus_urbano",
            "bus": "onibus_urbano",
            "trem": "trem_eletrico",
            "train": "trem_eletrico",
            "metrô": "metro",
            "subway": "metro",
            "bike": "bicicleta",
            "bicycle": "bicicleta",
            "walk": "caminhada",
            "walking": "caminhada",
            "barco": "barco_fluvial",
            "boat": "barco_fluvial",
            "balsa": "ferry"
        }
        
        # Fatores de ocupação média para cálculo mais preciso
        self.avg_occupancy = {
            "carro": 1.5,
            "onibus": 40,
            "trem": 100,
            "metro": 150,
            "aviao": 0.8  # 80% de ocupação
        }
        
        # Dados de rotas específicas do Brasil
        self.brazil_routes = self._load_brazil_routes()
    
    def _load_emissions_data(self) -> Optional[pd.DataFrame]:
        """Carrega dados de emissões do arquivo CSV"""
        try:
            if self.emissions_data_path.exists():
                df = pd.read_csv(self.emissions_data_path)
                # Criar índice por modal de transporte
                df.set_index('modal_transporte', inplace=True)
                return df
        except Exception as e:
            print(f"Aviso: Usando dados de emissões integrados. Erro ao carregar arquivo: {e}")
        return None
    
    def _load_brazil_routes(self) -> Dict:
        """Carrega dados de rotas específicas do Brasil"""
        routes = {
            "sao_paulo_rio": {"distance": 430, "best_modal": "onibus_rodoviario"},
            "rio_salvador": {"distance": 1660, "best_modal": "aviao_domestico"},
            "brasilia_goiania": {"distance": 209, "best_modal": "onibus_rodoviario"},
            "curitiba_florianopolis": {"distance": 300, "best_modal": "trem_eletrico"},
            "recife_fernando_noronha": {"distance": 545, "best_modal": "aviao_domestico"},
            "manaus_belem": {"distance": 1292, "best_modal": "barco_fluvial"},
            "porto_alegre_gramado": {"distance": 120, "best_modal": "carro_flex"},
            "fortaleza_jericoacoara": {"distance": 300, "best_modal": "onibus_rodoviario"},
            "cuiaba_pantanal": {"distance": 150, "best_modal": "carro_flex"},
            "rio_buzios": {"distance": 170, "best_modal": "onibus_rodoviario"}
        }
        return routes
    
    def get_emission_factor(self, transport_mode: str) -> float:
        """Obtém fator de emissão para um modal de transporte"""
        # Normalizar nome do modal
        transport_mode = transport_mode.lower().strip()
        
        # Verificar aliases
        if transport_mode in self.transport_aliases:
            transport_mode = self.transport_aliases[transport_mode]
        
        # Primeiro tenta obter dos dados carregados do CSV
        if self.emissions_data is not None:
            try:
                if transport_mode in self.emissions_data.index:
                    return float(self.emissions_data.loc[transport_mode, 'emissao_co2_kg_km'])
            except:
                pass
        
        # Fallback para dados integrados do IPCC
        return self.emissions_factors_ipcc.get(transport_mode, 0.171)  # Default para carro
    
    def calculate_carbon_footprint(
        self,
        transport_mode: str,
        distance_km: float,
        passengers: int = 1,
        round_trip: bool = False,
        occupancy_rate: Optional[float] = None
    ) -> Dict[str, Union[float, str, Dict]]:
        """
        Calcula pegada de carbono para uma viagem usando dados reais
        
        Args:
            transport_mode: Modal de transporte
            distance_km: Distância em km
            passengers: Número de passageiros
            round_trip: Se é viagem de ida e volta
            occupancy_rate: Taxa de ocupação (opcional)
            
        Returns:
            Dicionário com detalhes do cálculo
        """
        # Normalizar modal
        transport_mode_normalized = transport_mode.lower().strip()
        if transport_mode_normalized in self.transport_aliases:
            transport_mode_normalized = self.transport_aliases[transport_mode_normalized]
        
        # Obter fator de emissão
        emission_factor = self.get_emission_factor(transport_mode_normalized)
        
        # Ajustar distância para ida e volta
        total_distance = distance_km * (2 if round_trip else 1)
        
        # Calcular emissões totais
        total_emissions = emission_factor * total_distance * passengers
        
        # Calcular emissões por passageiro considerando ocupação média
        if occupancy_rate is None and transport_mode_normalized in ["carro_gasolina", "carro_etanol", "carro_flex"]:
            # Para carros, dividir pelas pessoas no veículo
            emissions_per_passenger = (emission_factor * total_distance) / max(passengers, 1)
        else:
            emissions_per_passenger = emission_factor * total_distance
        
        # Comparar com outros modais
        comparisons = self._calculate_comparisons(distance_km, round_trip)
        
        # Sugestões de redução
        suggestions = self._generate_suggestions(transport_mode_normalized, distance_km)
        
        return {
            "transport_mode": transport_mode,
            "transport_mode_normalized": transport_mode_normalized,
            "distance_km": distance_km,
            "round_trip": round_trip,
            "total_distance_km": total_distance,
            "passengers": passengers,
            "emission_factor_kg_per_km": emission_factor,
            "total_emissions_kg": round(total_emissions, 2),
            "emissions_per_passenger_kg": round(emissions_per_passenger, 2),
            "emissions_tonnes": round(total_emissions / 1000, 3),
            "comparisons": comparisons,
            "suggestions": suggestions,
            "data_source": "IPCC 2023"
        }
    
    def _calculate_comparisons(self, distance_km: float, round_trip: bool = False) -> Dict[str, float]:
        """Calcula emissões para diferentes modais para comparação"""
        total_distance = distance_km * (2 if round_trip else 1)
        
        comparisons = {}
        for modal, factor in self.emissions_factors_ipcc.items():
            emissions = factor * total_distance
            comparisons[modal] = round(emissions, 2)
        
        # Ordenar do menor para maior
        return dict(sorted(comparisons.items(), key=lambda x: x[1]))
    
    def _generate_suggestions(self, transport_mode: str, distance_km: float) -> List[str]:
        """Gera sugestões para reduzir emissões"""
        suggestions = []
        
        # Obter emissão atual
        current_emissions = self.get_emission_factor(transport_mode)
        
        # Sugestões baseadas na distância
        if distance_km < 5:
            suggestions.append("Para distâncias curtas (< 5km), considere caminhar ou usar bicicleta - emissão zero!")
        elif distance_km < 50:
            suggestions.append("Para distâncias médias, o transporte público (ônibus, metrô) pode reduzir emissões em até 70%")
        elif distance_km < 500:
            suggestions.append("Para viagens regionais, ônibus rodoviários emitem 50% menos que carros individuais")
        else:
            suggestions.append("Para longas distâncias, combine diferentes modais (ex: trem + metrô) para otimizar emissões")
        
        # Sugestões específicas por modal
        if "carro" in transport_mode:
            suggestions.append("Compartilhe o veículo: cada passageiro adicional reduz significativamente as emissões per capita")
            suggestions.append("Considere veículos flex com etanol, que emitem 17% menos CO2 que gasolina")
        elif "aviao" in transport_mode:
            suggestions.append("Voos diretos emitem menos que voos com conexões")
            suggestions.append("Compense suas emissões através de programas de carbono certificados")
        
        # Verificar rotas específicas do Brasil
        for route, info in self.brazil_routes.items():
            if abs(distance_km - info["distance"]) < 50:  # Margem de 50km
                best_modal = info["best_modal"]
                best_emissions = self.get_emission_factor(best_modal)
                if best_emissions < current_emissions:
                    reduction = ((current_emissions - best_emissions) / current_emissions) * 100
                    suggestions.append(
                        f"Para esta rota, {best_modal.replace('_', ' ')} "
                        f"pode reduzir emissões em {reduction:.0f}%"
                    )
                break
        
        return suggestions[:3]  # Retornar top 3 sugestões
    
    def calculate_route_emissions(
        self,
        origin: str,
        destination: str,
        transport_mode: Optional[str] = None
    ) -> Dict[str, Union[float, str, Dict]]:
        """
        Calcula emissões para rotas específicas do Brasil
        
        Args:
            origin: Cidade de origem
            destination: Cidade de destino
            transport_mode: Modal de transporte (opcional)
            
        Returns:
            Dicionário com cálculo de emissões
        """
        # Normalizar nomes das cidades
        origin_lower = origin.lower().replace(" ", "_")
        destination_lower = destination.lower().replace(" ", "_")
        
        # Buscar rota
        route_key = f"{origin_lower}_{destination_lower}"
        route_key_reverse = f"{destination_lower}_{origin_lower}"
        
        route_info = None
        if route_key in self.brazil_routes:
            route_info = self.brazil_routes[route_key]
        elif route_key_reverse in self.brazil_routes:
            route_info = self.brazil_routes[route_key_reverse]
        
        if route_info:
            distance = route_info["distance"]
            recommended_modal = route_info["best_modal"]
            
            # Usar modal recomendado se não especificado
            if transport_mode is None:
                transport_mode = recommended_modal
            
            result = self.calculate_carbon_footprint(
                transport_mode=transport_mode,
                distance_km=distance,
                round_trip=True
            )
            
            result["route"] = f"{origin} - {destination}"
            result["recommended_transport"] = recommended_modal
            
            return result
        else:
            return {
                "error": f"Rota {origin} - {destination} não encontrada no banco de dados",
                "suggestion": "Por favor, forneça a distância em km para calcular as emissões"
            }
    
    def compare_transport_modes(
        self,
        distance_km: float,
        modes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compara emissões entre diferentes modais de transporte
        
        Args:
            distance_km: Distância da viagem
            modes: Lista de modais para comparar (None = todos)
            
        Returns:
            DataFrame com comparação
        """
        if modes is None:
            modes = list(self.emissions_factors_ipcc.keys())
        
        data = []
        for mode in modes:
            emissions = self.calculate_carbon_footprint(mode, distance_km)
            data.append({
                "Modal": mode.replace("_", " ").title(),
                "Emissões (kg CO2)": emissions["total_emissions_kg"],
                "Fator (kg/km)": emissions["emission_factor_kg_per_km"],
                "Redução vs Avião (%)": 0  # Será calculado depois
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values("Emissões (kg CO2)")
        
        # Calcular redução em relação ao avião
        aviao_emissions = df[df["Modal"].str.contains("Aviao")]["Emissões (kg CO2)"].max()
        if aviao_emissions > 0:
            df["Redução vs Avião (%)"] = ((aviao_emissions - df["Emissões (kg CO2)"]) / aviao_emissions * 100).round(1)
        
        return df
    
    def get_sustainability_score(self, transport_mode: str, distance_km: float) -> Dict[str, Union[int, str]]:
        """
        Calcula score de sustentabilidade (0-100) para uma viagem
        
        Args:
            transport_mode: Modal de transporte
            distance_km: Distância da viagem
            
        Returns:
            Score e classificação
        """
        emissions = self.calculate_carbon_footprint(transport_mode, distance_km)
        emissions_per_km = emissions["emission_factor_kg_per_km"]
        
        # Score baseado em emissões (invertido - menor emissão = maior score)
        # Bicicleta/Caminhada = 100, Avião internacional = 0
        max_emission = 0.267  # Barco fluvial
        
        if emissions_per_km == 0:
            score = 100
        else:
            score = max(0, 100 - (emissions_per_km / max_emission * 100))
        
        # Classificação
        if score >= 80:
            classification = "Excelente - Transporte muito sustentável"
        elif score >= 60:
            classification = "Bom - Transporte sustentável"
        elif score >= 40:
            classification = "Regular - Considere alternativas mais verdes"
        elif score >= 20:
            classification = "Ruim - Alto impacto ambiental"
        else:
            classification = "Muito Ruim - Impacto ambiental crítico"
        
        return {
            "score": round(score),
            "classification": classification,
            "emissions_kg_per_km": emissions_per_km,
            "transport_mode": transport_mode
        }


# Função de teste
def test_carbon_calculator():
    """Testa a calculadora com dados reais"""
    calc = CarbonCalculator()
    
    print("=== Teste da Calculadora de Carbono ===\n")
    
    # Teste 1: Cálculo simples
    result = calc.calculate_carbon_footprint("aviao_domestico", 500, round_trip=True)
    print(f"1. Voo doméstico SP-RJ (ida e volta):")
    print(f"   - Emissões totais: {result['total_emissions_kg']} kg CO2")
    print(f"   - Sugestões: {result['suggestions'][0]}")
    
    # Teste 2: Comparação de modais
    print("\n2. Comparação de modais para 300km:")
    df = calc.compare_transport_modes(300, ["carro_flex", "onibus_rodoviario", "trem_eletrico", "aviao_domestico"])
    print(df.to_string(index=False))
    
    # Teste 3: Rota específica
    print("\n3. Rota específica:")
    route_result = calc.calculate_route_emissions("São Paulo", "Rio")
    print(f"   - Rota: {route_result.get('route', 'N/A')}")
    print(f"   - Modal recomendado: {route_result.get('recommended_transport', 'N/A')}")
    print(f"   - Emissões: {route_result.get('total_emissions_kg', 'N/A')} kg CO2")
    
    # Teste 4: Score de sustentabilidade
    print("\n4. Scores de sustentabilidade:")
    for modal in ["bicicleta", "metro", "onibus_urbano", "carro_flex", "aviao_internacional"]:
        score_info = calc.get_sustainability_score(modal, 100)
        print(f"   - {modal}: Score {score_info['score']}/100 - {score_info['classification']}")


if __name__ == "__main__":
    test_carbon_calculator()