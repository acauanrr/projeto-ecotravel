"""
Calculadora de Pegada de Carbono para diferentes modais de transporte
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class CarbonCalculator:
    def __init__(self, emissions_data_path: str = "data/emissoes/emissoes_transporte.csv"):
        self.emissions_data_path = Path(emissions_data_path)
        self.emissions_data = self._load_emissions_data()
        
        # Fatores de emissão padrão (kg CO2 por km por passageiro)
        self.default_emissions = {
            "aviao": 0.255,
            "carro": 0.171, 
            "onibus": 0.089,
            "trem": 0.041,
            "moto": 0.113,
            "bicicleta": 0.0,
            "caminhada": 0.0
        }
        
        # Fatores de correção para diferentes tipos de viagem
        self.correction_factors = {
            "aviao": {
                "domestico_curto": 1.3,    # < 500km
                "domestico_medio": 1.0,    # 500-1500km  
                "domestico_longo": 0.9,    # > 1500km
                "internacional": 0.8
            },
            "carro": {
                "sozinho": 1.0,
                "2_pessoas": 0.5,
                "3_pessoas": 0.33,
                "4_pessoas": 0.25
            }
        }
    
    def _load_emissions_data(self) -> Optional[pd.DataFrame]:
        """Carrega dados de emissões do arquivo CSV"""
        try:
            if self.emissions_data_path.exists():
                return pd.read_csv(self.emissions_data_path)
        except Exception as e:
            print(f"Erro ao carregar dados de emissões: {e}")
        return None
    
    def get_emission_factor(self, transport_mode: str) -> float:
        """Obtém fator de emissão para um modal de transporte"""
        transport_mode = transport_mode.lower()
        
        # Primeiro tenta obter dos dados carregados
        if self.emissions_data is not None:
            try:
                row = self.emissions_data[
                    self.emissions_data['modal_transporte'] == transport_mode
                ]
                if not row.empty:
                    return float(row['emissao_co2_kg_km'].iloc[0])
            except:
                pass
        
        # Fallback para valores padrão
        return self.default_emissions.get(transport_mode, 0.0)
    
    def calculate_carbon_footprint(
        self,
        transport_mode: str,
        distance_km: float,
        trip_type: str = "domestico_medio",
        passengers: int = 1,
        round_trip: bool = False
    ) -> Dict:
        """
        Calcula pegada de carbono para uma viagem
        
        Args:
            transport_mode: Modal de transporte
            distance_km: Distância em km
            trip_type: Tipo de viagem (para correções)
            passengers: Número de passageiros (para carro)
            round_trip: Se é viagem de ida e volta
            
        Returns:
            Dicionário com detalhes do cálculo
        """
        transport_mode = transport_mode.lower()
        
        # Obter fator base de emissão
        base_emission = self.get_emission_factor(transport_mode)
        
        # Aplicar correções
        correction_factor = 1.0
        
        if transport_mode == "aviao":
            correction_factor = self.correction_factors["aviao"].get(trip_type, 1.0)
        elif transport_mode == "carro":
            if passengers == 1:
                correction_factor = self.correction_factors["carro"]["sozinho"]
            elif passengers == 2:
                correction_factor = self.correction_factors["carro"]["2_pessoas"]
            elif passengers == 3:
                correction_factor = self.correction_factors["carro"]["3_pessoas"]
            elif passengers >= 4:
                correction_factor = self.correction_factors["carro"]["4_pessoas"]
        
        # Calcular emissão
        emission_per_km = base_emission * correction_factor
        total_distance = distance_km * (2 if round_trip else 1)
        total_emission = emission_per_km * total_distance
        
        return {
            "transport_mode": transport_mode,
            "distance_km": distance_km,
            "total_distance_km": total_distance,
            "passengers": passengers,
            "round_trip": round_trip,
            "base_emission_kg_co2_km": base_emission,
            "correction_factor": correction_factor,
            "emission_per_km": emission_per_km,
            "total_emission_kg_co2": total_emission,
            "trip_type": trip_type
        }
    
    def compare_transport_modes(
        self,
        distance_km: float,
        modes: List[str] = None,
        round_trip: bool = False
    ) -> List[Dict]:
        """
        Compara emissões entre diferentes modais de transporte
        
        Args:
            distance_km: Distância da viagem
            modes: Lista de modais para comparar
            round_trip: Se é viagem de ida e volta
            
        Returns:
            Lista de dicionários com comparação
        """
        if modes is None:
            modes = ["aviao", "carro", "onibus", "trem"]
        
        results = []
        
        for mode in modes:
            result = self.calculate_carbon_footprint(
                transport_mode=mode,
                distance_km=distance_km,
                round_trip=round_trip
            )
            results.append(result)
        
        # Ordenar por emissão
        results.sort(key=lambda x: x["total_emission_kg_co2"])
        
        return results
    
    def get_sustainability_recommendation(
        self,
        origin: str,
        destination: str,
        distance_km: float,
        available_modes: List[str] = None
    ) -> Dict:
        """
        Gera recomendação de viagem sustentável
        
        Args:
            origin: Origem da viagem
            destination: Destino da viagem  
            distance_km: Distância estimada
            available_modes: Modais disponíveis
            
        Returns:
            Dicionário com recomendação
        """
        if available_modes is None:
            available_modes = ["aviao", "carro", "onibus", "trem"]
        
        # Comparar modais
        comparison = self.compare_transport_modes(distance_km, available_modes)
        
        # Modal mais sustentável
        most_sustainable = comparison[0]
        least_sustainable = comparison[-1]
        
        # Economia de CO2
        emission_savings = (
            least_sustainable["total_emission_kg_co2"] - 
            most_sustainable["total_emission_kg_co2"]
        )
        
        # Categorizar distância
        distance_category = self._categorize_distance(distance_km)
        
        # Gerar recomendação textual
        recommendation = self._generate_recommendation_text(
            most_sustainable, distance_category, emission_savings
        )
        
        return {
            "origin": origin,
            "destination": destination,
            "distance_km": distance_km,
            "distance_category": distance_category,
            "most_sustainable_mode": most_sustainable,
            "least_sustainable_mode": least_sustainable,
            "emission_savings_kg_co2": emission_savings,
            "emission_savings_percentage": (
                emission_savings / least_sustainable["total_emission_kg_co2"] * 100
                if least_sustainable["total_emission_kg_co2"] > 0 else 0
            ),
            "comparison": comparison,
            "recommendation": recommendation
        }
    
    def _categorize_distance(self, distance_km: float) -> str:
        """Categoriza distância da viagem"""
        if distance_km < 50:
            return "local"
        elif distance_km < 200:
            return "regional"
        elif distance_km < 500:
            return "nacional_curto"
        elif distance_km < 1500:
            return "nacional_medio"
        else:
            return "nacional_longo"
    
    def _generate_recommendation_text(
        self,
        best_option: Dict,
        distance_category: str,
        savings: float
    ) -> str:
        """Gera texto de recomendação"""
        mode = best_option["transport_mode"]
        emission = best_option["total_emission_kg_co2"]
        
        recommendations = {
            "local": {
                "bicicleta": "Para distâncias curtas, a bicicleta é a opção mais sustentável e saudável.",
                "caminhada": "Para distâncias muito curtas, caminhar é a opção mais sustentável.",
                "onibus": "Para distâncias locais, o transporte público é mais sustentável que o carro."
            },
            "regional": {
                "onibus": "Para viagens regionais, o ônibus oferece boa relação sustentabilidade-tempo.",
                "trem": "O trem é a opção mais sustentável para viagens regionais quando disponível.",
                "carro": "Se usar carro, considere compartilhar com outros passageiros."
            },
            "nacional_curto": {
                "onibus": "Para distâncias médias, o ônibus é mais sustentável que avião ou carro.",
                "trem": "O trem é ideal para distâncias médias quando disponível.",
                "aviao": "Evite voos para distâncias que podem ser feitas por terra."
            },
            "nacional_medio": {
                "trem": "O trem é a opção mais sustentável para longas distâncias.",
                "onibus": "O ônibus é uma alternativa sustentável para longas viagens.",
                "aviao": "Se necessário voar, prefira voos diretos."
            },
            "nacional_longo": {
                "trem": "Para distâncias muito longas, o trem ainda é mais sustentável.",
                "aviao": "Se o voo for necessário, prefira voos diretos e compense as emissões."
            }
        }
        
        base_text = recommendations.get(distance_category, {}).get(
            mode, 
            f"O {mode} é a opção mais sustentável para esta viagem."
        )
        
        if savings > 0:
            base_text += f" Esta escolha economiza {savings:.1f} kg de CO2 comparado à opção menos sustentável."
        
        return base_text
    
    def calculate_offset_cost(
        self,
        emission_kg_co2: float,
        price_per_ton: float = 25.0  # USD por tonelada de CO2
    ) -> Dict:
        """
        Calcula custo de compensação de carbono
        
        Args:
            emission_kg_co2: Emissão em kg de CO2
            price_per_ton: Preço por tonelada de CO2
            
        Returns:
            Dicionário com custos de compensação
        """
        emission_tons = emission_kg_co2 / 1000
        offset_cost = emission_tons * price_per_ton
        
        return {
            "emission_kg_co2": emission_kg_co2,
            "emission_tons_co2": emission_tons,
            "price_per_ton_usd": price_per_ton,
            "offset_cost_usd": offset_cost,
            "offset_cost_brl": offset_cost * 5.0  # Conversão aproximada
        }


def create_carbon_calculator_tool():
    """Cria ferramenta do LangChain para cálculo de carbono"""
    from langchain.tools import Tool
    
    calculator = CarbonCalculator()
    
    def calculate_carbon_footprint_tool(input_str: str) -> str:
        """
        Calcula pegada de carbono para uma viagem.
        
        Input esperado: "modal:distancia:tipo_viagem:passageiros:ida_volta"
        Exemplo: "aviao:400:domestico_curto:1:true"
        """
        try:
            parts = input_str.split(":")
            
            if len(parts) < 2:
                return "Erro: Formato inválido. Use 'modal:distancia' no mínimo."
            
            transport_mode = parts[0]
            distance_km = float(parts[1])
            trip_type = parts[2] if len(parts) > 2 else "domestico_medio"
            passengers = int(parts[3]) if len(parts) > 3 else 1
            round_trip = parts[4].lower() == "true" if len(parts) > 4 else False
            
            result = calculator.calculate_carbon_footprint(
                transport_mode=transport_mode,
                distance_km=distance_km,
                trip_type=trip_type,
                passengers=passengers,
                round_trip=round_trip
            )
            
            return f"""
Cálculo de Pegada de Carbono:
- Modal: {result['transport_mode']}
- Distância: {result['total_distance_km']} km
- Emissão total: {result['total_emission_kg_co2']:.2f} kg CO2
- Emissão por km: {result['emission_per_km']:.3f} kg CO2/km
- Passageiros: {result['passengers']}
- Fator de correção: {result['correction_factor']}
"""
        except Exception as e:
            return f"Erro no cálculo: {str(e)}"
    
    def compare_transport_modes_tool(input_str: str) -> str:
        """
        Compara emissões entre diferentes modais.
        
        Input esperado: "distancia:modal1,modal2,modal3:ida_volta"
        Exemplo: "400:aviao,onibus,carro:false"
        """
        try:
            parts = input_str.split(":")
            distance_km = float(parts[0])
            modes = parts[1].split(",") if len(parts) > 1 else None
            round_trip = parts[2].lower() == "true" if len(parts) > 2 else False
            
            results = calculator.compare_transport_modes(distance_km, modes, round_trip)
            
            comparison_text = "Comparação de Emissões de CO2:\n"
            for i, result in enumerate(results, 1):
                comparison_text += f"{i}. {result['transport_mode'].title()}: {result['total_emission_kg_co2']:.2f} kg CO2\n"
            
            return comparison_text
            
        except Exception as e:
            return f"Erro na comparação: {str(e)}"
    
    def get_sustainability_recommendation_tool(input_str: str) -> str:
        """
        Gera recomendação de viagem sustentável.
        
        Input esperado: "origem:destino:distancia:modais_disponiveis"
        Exemplo: "São Paulo:Rio de Janeiro:400:aviao,onibus,carro"
        """
        try:
            parts = input_str.split(":")
            origin = parts[0]
            destination = parts[1] 
            distance_km = float(parts[2])
            available_modes = parts[3].split(",") if len(parts) > 3 else None
            
            recommendation = calculator.get_sustainability_recommendation(
                origin, destination, distance_km, available_modes
            )
            
            return f"""
Recomendação de Viagem Sustentável:
Rota: {recommendation['origin']} → {recommendation['destination']}
Distância: {recommendation['distance_km']} km

Opção Mais Sustentável: {recommendation['most_sustainable_mode']['transport_mode'].title()}
- Emissão: {recommendation['most_sustainable_mode']['total_emission_kg_co2']:.2f} kg CO2

Economia vs. Opção Menos Sustentável: {recommendation['emission_savings_kg_co2']:.2f} kg CO2 ({recommendation['emission_savings_percentage']:.1f}%)

Recomendação: {recommendation['recommendation']}
"""
        except Exception as e:
            return f"Erro na recomendação: {str(e)}"
    
    # Retornar lista de ferramentas
    return [
        Tool(
            name="CarbonFootprintCalculator",
            func=calculate_carbon_footprint_tool,
            description="Calcula pegada de carbono para viagens. Input: 'modal:distancia:tipo:passageiros:ida_volta'"
        ),
        Tool(
            name="TransportModeComparison", 
            func=compare_transport_modes_tool,
            description="Compara emissões entre modais de transporte. Input: 'distancia:modal1,modal2:ida_volta'"
        ),
        Tool(
            name="SustainabilityRecommendation",
            func=get_sustainability_recommendation_tool,
            description="Gera recomendação sustentável. Input: 'origem:destino:distancia:modais_disponiveis'"
        )
    ]


if __name__ == "__main__":
    # Teste básico
    calc = CarbonCalculator()
    
    # Teste cálculo individual
    result = calc.calculate_carbon_footprint("aviao", 400, round_trip=True)
    print("Teste individual:")
    print(f"Avião 400km ida/volta: {result['total_emission_kg_co2']:.2f} kg CO2")
    
    # Teste comparação
    comparison = calc.compare_transport_modes(400)
    print("\nComparação 400km:")
    for comp in comparison:
        print(f"{comp['transport_mode']}: {comp['total_emission_kg_co2']:.2f} kg CO2")
    
    # Teste recomendação
    rec = calc.get_sustainability_recommendation("São Paulo", "Rio de Janeiro", 400)
    print(f"\nRecomendação: {rec['recommendation']}")