"""
API de Clima usando Open-Meteo para informações meteorológicas
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd


class WeatherAPI:
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1"
        self.geocoding_url = "https://geocoding-api.open-meteo.com/v1"
        
        # Códigos de tempo do Open-Meteo
        self.weather_codes = {
            0: "Céu limpo",
            1: "Principalmente limpo", 
            2: "Parcialmente nublado",
            3: "Nublado",
            45: "Neblina",
            48: "Neblina com geada",
            51: "Garoa leve",
            53: "Garoa moderada", 
            55: "Garoa intensa",
            56: "Garoa gelada leve",
            57: "Garoa gelada intensa",
            61: "Chuva leve",
            63: "Chuva moderada",
            65: "Chuva intensa",
            66: "Chuva gelada leve",
            67: "Chuva gelada intensa",
            71: "Neve leve",
            73: "Neve moderada",
            75: "Neve intensa",
            77: "Granizo",
            80: "Pancadas de chuva leves",
            81: "Pancadas de chuva moderadas",
            82: "Pancadas de chuva intensas",
            85: "Pancadas de neve leves",
            86: "Pancadas de neve intensas",
            95: "Tempestade",
            96: "Tempestade com granizo leve",
            99: "Tempestade com granizo intenso"
        }
    
    def get_coordinates(self, city_name: str, country: str = "BR") -> Optional[Tuple[float, float]]:
        """
        Obtém coordenadas de uma cidade
        
        Args:
            city_name: Nome da cidade
            country: Código do país (padrão: BR)
            
        Returns:
            Tupla (latitude, longitude) ou None se não encontrado
        """
        try:
            params = {
                "name": city_name,
                "count": 1,
                "language": "pt",
                "format": "json"
            }
            
            response = requests.get(
                f"{self.geocoding_url}/search",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    result = data["results"][0]
                    return (result["latitude"], result["longitude"])
            
        except Exception as e:
            print(f"Erro ao obter coordenadas para {city_name}: {e}")
        
        return None
    
    def get_current_weather(self, latitude: float, longitude: float) -> Dict:
        """
        Obtém clima atual para coordenadas específicas
        
        Args:
            latitude: Latitude
            longitude: Longitude
            
        Returns:
            Dicionário com dados meteorológicos atuais
        """
        try:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": [
                    "temperature_2m",
                    "relative_humidity_2m", 
                    "apparent_temperature",
                    "precipitation",
                    "weather_code",
                    "wind_speed_10m",
                    "wind_direction_10m"
                ],
                "timezone": "America/Sao_Paulo"
            }
            
            response = requests.get(
                f"{self.base_url}/forecast",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                current = data["current"]
                
                return {
                    "timestamp": current["time"],
                    "temperature_celsius": current["temperature_2m"],
                    "feels_like_celsius": current["apparent_temperature"],
                    "humidity_percent": current["relative_humidity_2m"],
                    "precipitation_mm": current["precipitation"],
                    "wind_speed_kmh": current["wind_speed_10m"],
                    "wind_direction_degrees": current["wind_direction_10m"],
                    "weather_code": current["weather_code"],
                    "weather_description": self.weather_codes.get(
                        current["weather_code"], "Desconhecido"
                    )
                }
        
        except Exception as e:
            print(f"Erro ao obter clima atual: {e}")
        
        return {}
    
    def get_weather_forecast(
        self,
        latitude: float,
        longitude: float,
        days: int = 7
    ) -> Dict:
        """
        Obtém previsão do tempo para os próximos dias
        
        Args:
            latitude: Latitude
            longitude: Longitude  
            days: Número de dias de previsão (máximo 14)
            
        Returns:
            Dicionário com previsão meteorológica
        """
        try:
            days = min(days, 14)  # Limite da API
            
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "daily": [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "wind_speed_10m_max",
                    "wind_direction_10m_dominant"
                ],
                "timezone": "America/Sao_Paulo",
                "forecast_days": days
            }
            
            response = requests.get(
                f"{self.base_url}/forecast",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                daily = data["daily"]
                
                forecast = []
                for i in range(len(daily["time"])):
                    day_data = {
                        "date": daily["time"][i],
                        "max_temperature_celsius": daily["temperature_2m_max"][i],
                        "min_temperature_celsius": daily["temperature_2m_min"][i],
                        "precipitation_mm": daily["precipitation_sum"][i],
                        "max_wind_speed_kmh": daily["wind_speed_10m_max"][i],
                        "wind_direction_degrees": daily["wind_direction_10m_dominant"][i],
                        "weather_code": daily["weather_code"][i],
                        "weather_description": self.weather_codes.get(
                            daily["weather_code"][i], "Desconhecido"
                        )
                    }
                    forecast.append(day_data)
                
                return {
                    "location": {"latitude": latitude, "longitude": longitude},
                    "forecast_days": days,
                    "daily_forecast": forecast
                }
        
        except Exception as e:
            print(f"Erro ao obter previsão: {e}")
        
        return {}
    
    def get_city_weather(self, city_name: str, days: int = 3) -> Dict:
        """
        Obtém clima atual e previsão para uma cidade
        
        Args:
            city_name: Nome da cidade
            days: Dias de previsão
            
        Returns:
            Dicionário com clima atual e previsão
        """
        coordinates = self.get_coordinates(city_name)
        
        if not coordinates:
            return {"error": f"Cidade '{city_name}' não encontrada"}
        
        latitude, longitude = coordinates
        
        current = self.get_current_weather(latitude, longitude)
        forecast = self.get_weather_forecast(latitude, longitude, days)
        
        return {
            "city": city_name,
            "coordinates": {"latitude": latitude, "longitude": longitude},
            "current_weather": current,
            "forecast": forecast.get("daily_forecast", [])
        }
    
    def analyze_travel_weather(
        self,
        origin_city: str,
        destination_city: str,
        travel_date: str = None,
        days_ahead: int = 7
    ) -> Dict:
        """
        Analisa condições meteorológicas para uma viagem
        
        Args:
            origin_city: Cidade de origem
            destination_city: Cidade de destino
            travel_date: Data da viagem (YYYY-MM-DD) ou None para hoje
            days_ahead: Dias de previsão a partir da data
            
        Returns:
            Análise meteorológica da viagem
        """
        # Obter dados meteorológicos das duas cidades
        origin_weather = self.get_city_weather(origin_city, days_ahead)
        destination_weather = self.get_city_weather(destination_city, days_ahead)
        
        if "error" in origin_weather or "error" in destination_weather:
            return {
                "error": "Erro ao obter dados meteorológicos",
                "origin_error": origin_weather.get("error"),
                "destination_error": destination_weather.get("error")
            }
        
        # Analisar dados
        analysis = {
            "origin": origin_weather,
            "destination": destination_weather,
            "travel_date": travel_date or datetime.now().strftime("%Y-%m-%d"),
            "recommendations": self._generate_weather_recommendations(
                origin_weather, destination_weather
            )
        }
        
        return analysis
    
    def _generate_weather_recommendations(
        self,
        origin_weather: Dict,
        destination_weather: Dict
    ) -> List[str]:
        """Gera recomendações baseadas no clima"""
        recommendations = []
        
        # Analisar clima atual no destino
        dest_current = destination_weather.get("current_weather", {})
        
        if dest_current.get("precipitation_mm", 0) > 0:
            recommendations.append("Leve guarda-chuva - há previsão de chuva no destino")
        
        if dest_current.get("temperature_celsius", 20) < 15:
            recommendations.append("Leve roupas quentes - temperatura baixa no destino")
        elif dest_current.get("temperature_celsius", 20) > 30:
            recommendations.append("Leve roupas leves - temperatura alta no destino")
        
        if dest_current.get("wind_speed_kmh", 0) > 30:
            recommendations.append("Ventos fortes previstos - cuidado com voos e atividades ao ar livre")
        
        # Analisar previsão
        dest_forecast = destination_weather.get("forecast", [])
        if dest_forecast:
            rain_days = sum(1 for day in dest_forecast if day.get("precipitation_mm", 0) > 2)
            if rain_days > len(dest_forecast) // 2:
                recommendations.append("Período chuvoso esperado - planeje atividades internas")
        
        if not recommendations:
            recommendations.append("Condições meteorológicas favoráveis para a viagem")
        
        return recommendations
    
    def get_weather_travel_impact(self, weather_data: Dict) -> Dict:
        """
        Avalia impacto do clima na viagem
        
        Args:
            weather_data: Dados meteorológicos
            
        Returns:
            Análise de impacto
        """
        current = weather_data.get("current_weather", {})
        forecast = weather_data.get("forecast", [])
        
        # Calcular scores de impacto (0-10, onde 0 = sem impacto, 10 = alto impacto)
        temperature_impact = 0
        temp = current.get("temperature_celsius", 20)
        if temp < 5 or temp > 35:
            temperature_impact = 8
        elif temp < 10 or temp > 30:
            temperature_impact = 5
        
        precipitation_impact = 0
        rain = current.get("precipitation_mm", 0)
        if rain > 10:
            precipitation_impact = 8
        elif rain > 2:
            precipitation_impact = 4
        
        wind_impact = 0
        wind = current.get("wind_speed_kmh", 0)
        if wind > 40:
            wind_impact = 9
        elif wind > 25:
            wind_impact = 5
        
        # Score geral
        overall_impact = max(temperature_impact, precipitation_impact, wind_impact)
        
        return {
            "temperature_impact": temperature_impact,
            "precipitation_impact": precipitation_impact, 
            "wind_impact": wind_impact,
            "overall_impact": overall_impact,
            "impact_level": self._categorize_impact(overall_impact),
            "travel_suitable": overall_impact < 6
        }
    
    def _categorize_impact(self, score: int) -> str:
        """Categoriza nível de impacto"""
        if score <= 2:
            return "Baixo"
        elif score <= 5:
            return "Moderado"
        elif score <= 7:
            return "Alto"
        else:
            return "Muito Alto"


def create_weather_tools():
    """Cria ferramentas do LangChain para clima"""
    from langchain.tools import Tool
    
    weather_api = WeatherAPI()
    
    def get_city_weather_tool(city_name: str) -> str:
        """
        Obtém clima atual e previsão para uma cidade.
        
        Input: Nome da cidade
        """
        try:
            result = weather_api.get_city_weather(city_name, days=3)
            
            if "error" in result:
                return f"Erro: {result['error']}"
            
            current = result["current_weather"]
            forecast = result["forecast"][:3]  # Próximos 3 dias
            
            weather_text = f"""
Clima em {result['city']}:

Atual:
- Temperatura: {current['temperature_celsius']:.1f}°C (sensação: {current['feels_like_celsius']:.1f}°C)
- Condição: {current['weather_description']}
- Umidade: {current['humidity_percent']}%
- Vento: {current['wind_speed_kmh']:.1f} km/h
- Precipitação: {current['precipitation_mm']} mm

Previsão próximos dias:
"""
            
            for day in forecast:
                weather_text += f"- {day['date']}: {day['min_temperature_celsius']:.1f}°C - {day['max_temperature_celsius']:.1f}°C, {day['weather_description']}\n"
            
            return weather_text
            
        except Exception as e:
            return f"Erro ao obter clima: {str(e)}"
    
    def analyze_travel_weather_tool(input_str: str) -> str:
        """
        Analisa clima para viagem entre duas cidades.
        
        Input: "origem:destino" ou "origem:destino:data"
        """
        try:
            parts = input_str.split(":")
            origin = parts[0]
            destination = parts[1]
            travel_date = parts[2] if len(parts) > 2 else None
            
            result = weather_api.analyze_travel_weather(origin, destination, travel_date)
            
            if "error" in result:
                return f"Erro: {result['error']}"
            
            analysis_text = f"""
Análise Meteorológica da Viagem:
{origin} → {destination}

Clima em {origin}:
- Atual: {result['origin']['current_weather']['temperature_celsius']:.1f}°C, {result['origin']['current_weather']['weather_description']}

Clima em {destination}:
- Atual: {result['destination']['current_weather']['temperature_celsius']:.1f}°C, {result['destination']['current_weather']['weather_description']}

Recomendações:
"""
            
            for rec in result['recommendations']:
                analysis_text += f"- {rec}\n"
            
            return analysis_text
            
        except Exception as e:
            return f"Erro na análise: {str(e)}"
    
    return [
        Tool(
            name="CityWeather",
            func=get_city_weather_tool,
            description="Obtém clima atual e previsão para uma cidade. Input: nome da cidade"
        ),
        Tool(
            name="TravelWeatherAnalysis",
            func=analyze_travel_weather_tool,
            description="Analisa clima para viagem. Input: 'origem:destino' ou 'origem:destino:data'"
        )
    ]


if __name__ == "__main__":
    # Teste básico
    weather = WeatherAPI()
    
    # Teste clima atual
    sp_weather = weather.get_city_weather("São Paulo", days=3)
    print("Clima São Paulo:")
    print(json.dumps(sp_weather, indent=2, ensure_ascii=False))
    
    # Teste análise de viagem
    analysis = weather.analyze_travel_weather("São Paulo", "Rio de Janeiro")
    print("\nAnálise de viagem:")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))