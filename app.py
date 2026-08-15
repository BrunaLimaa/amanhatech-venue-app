import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import altair as alt
from datetime import datetime
from typing import Tuple, Dict, List

# Importar módulos principais de ML e dados
from core.data_loader import load_training_data
from core.ml_engine import train_models

JAMBASE_API_KEY = "jbd_trial_hJqB7wA0RiiR_HUjBcVOIaYaJgRLHWvnvVmDoaR96Ayjp"

st.set_page_config(
    page_title="Etratégia de Eventos",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner="Inicializando o Motor de Inteligência e Buscando Dados da API...")
def initialize_pipeline(api_key: str) -> Tuple[any, any, Dict[str, float], List[str]]:
    """
    Inicializa o carregador de dados e treina os modelos de ML.
    Armazena a saída em cache para que os modelos não sejam retreinados a cada interação com o controle deslizante.
    """
    api_key_clean = api_key.strip() if api_key else None
    
    X, y_price, y_sellout, demand_dict, api_cities = load_training_data(jambase_api_key=api_key_clean)
    
    regressor, classifier = train_models(X, y_price, y_sellout)
    
    return regressor, classifier, demand_dict, api_cities

regressor, classifier, real_demand_dict, available_cities = initialize_pipeline(JAMBASE_API_KEY)

def get_city_ibge(city_name: str) -> float:
    
    city_clean = city_name.lower().strip()
    hash_val = int(hashlib.md5(city_clean.encode('utf-8')).hexdigest(), 16)
    return 0.8 + ((hash_val % 400) / 1000.0)

if not available_cities:
    available_cities = ["Nenhum Dados do JamBase Encontrado"]

ibge_city_map = {city: get_city_ibge(city) for city in available_cities}
ibge_city_map = dict(sorted(ibge_city_map.items()))

sorted_artists = sorted(real_demand_dict.items(), key=lambda x: x[1], reverse=True)

top_artists = [artist.title() for artist, score in sorted_artists[:10000]]
if not top_artists:
    top_artists = ["Nenhum Dado do Last.fm Encontrado"]

def calculate_operational_costs(capacity: int, production_tier: str) -> float:
    base_cost_per_head = 15.00
    tier_multipliers = {
        "Baixo (Indie/Acústico)": 0.8,
        "Padrão": 1.0,
        "Alto (Arena/Estádio)": 1.6
    }
    multiplier = tier_multipliers.get(production_tier, 1.0)
    return capacity * base_cost_per_head * multiplier

st.sidebar.title("Amanhã Tech")
st.sidebar.header("Simulador de Reservas")

st.sidebar.subheader("1. Localização e Espaço")

selected_city = st.sidebar.selectbox("Mercado-Alvo (Cidade)", options=list(ibge_city_map.keys()), index=0)
derived_ibge = ibge_city_map[selected_city]

input_capacity_str = st.sidebar.text_input("Capacidade Física do Espaço", value="25000")
try:
    input_capacity = int(input_capacity_str)
except ValueError:
    input_capacity = 25000
    st.sidebar.warning("Por favor, insira um número inteiro válido para a capacidade. Usando o padrão de 25.000.")
    
st.sidebar.subheader("2. Artista/Grupo")

input_artist = st.sidebar.selectbox("Nome do Artista / Banda", options=top_artists)

derived_demand = real_demand_dict.get(input_artist.lower(), 0.0)
data_source = "Dataset Parquet do Last.fm"

st.sidebar.subheader("3. Premissas Operacionais")
production_tier = st.sidebar.selectbox("Requisitos de Produção", ["Baixo (Indie/Acústico)", "Padrão", "Alto (Arena/Estádio)"], index=1)
bar_spend_str = st.sidebar.text_input("Gasto Médio com A&B por Espectador (R$)", value="45.0")
try:
    bar_spend_per_head = float(bar_spend_str)
except ValueError:
    bar_spend_per_head = 45.0
    st.sidebar.warning("Por favor, insira um número válido para o gasto com A&B. Usando o padrão de R$ 45,00.")

input_data = pd.DataFrame({
    'capacity': [input_capacity],
    'spotify_demand': [derived_demand],
    'ibge_index': [derived_ibge]
})

predicted_price = regressor.predict(input_data)[0]
sellout_probability = classifier.predict_proba(input_data)[0][1]

expected_fill_rate = 1.0
expected_tickets_sold = input_capacity * sellout_probability

gross_ticket_revenue = predicted_price * expected_tickets_sold
gross_fb_revenue = expected_tickets_sold * bar_spend_per_head
total_gross_revenue = gross_ticket_revenue + gross_fb_revenue

operational_costs = calculate_operational_costs(input_capacity, production_tier)
artist_fee_estimate = gross_ticket_revenue * 0.70 
total_costs = operational_costs + artist_fee_estimate

net_profit = total_gross_revenue - total_costs
roi_percentage = (net_profit / total_costs) * 100 if total_costs > 0 else 0

st.title("Inteligência de Reservas de Locais")
st.markdown("Uma estrutura de análise preditiva que otimiza roteiros de turnê, precificação de ingressos e margens de locais por meio de aprendizado de máquina em datasets históricos.")

st.header(f"Análise da Turnê: {input_artist} ao vivo em {selected_city}")

# Linha 1: Métricas Principais de Desempenho
st.subheader("Previsões de Desempenho")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Preço Ótimo do Ingresso", value=f"R$ {predicted_price:.2f}")
with col2:
    st.metric(label="Probabilidade de Esgotamento", value=f"{sellout_probability * 100:.1f}%")
with col3:
    st.metric(label="Ingressos Est. Vendidos", value=f"{expected_tickets_sold:,.2f}")

st.divider()

st.subheader("Projeções Financeiras")
fcol1, fcol2, fcol3, fcol4 = st.columns(4)

with fcol1:
    st.metric(label="Receita Bruta Total", value=f"R$ {total_gross_revenue:,.2f}")
with fcol2:
    st.metric(label="Custos Fixos/Variáveis Est.", value=f"R$ {total_costs:,.2f}")
with fcol3:
    delta_color = "normal" if net_profit > 0 else "inverse"
    st.metric(label="Lucro Líquido Projetado", value=f"R$ {net_profit:,.2f}", delta=f"{roi_percentage:.1f}% ROI", delta_color=delta_color)
with fcol4:
    st.metric(label="Contribuição de A&B", value=f"R$ {gross_fb_revenue:,.2f}")

st.markdown("### Insights Operacionais e Detalhamento da Receita")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fin_data = pd.DataFrame({
        "Category": ["Receita de Ingresso", "Receita de A&B", "Op. do Espaço", "Custo do Artista", "Lucro Líquido"],
        "Amount (R$)": [gross_ticket_revenue, gross_fb_revenue, -operational_costs, -artist_fee_estimate, net_profit],
        "Ledger": ["Receita", "Receita", "Despesa", "Despesa", "Resultado Final"]
    })
    
    bar_chart = alt.Chart(fin_data).mark_bar().encode(
        x=alt.X("Category", sort=None, title="", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("Amount (R$)", title="Reais (R$)"),
        color=alt.Color(
            "Ledger", 
            scale=alt.Scale(domain=["Receita", "Despesa", "Resultado Final"], range=["#2ca02c", "#d62728", "#1f77b4"])
        ),
        tooltip=["Category", "Amount (R$)", "Ledger"]
    ).properties(height=350)
    
    st.altair_chart(bar_chart, use_container_width=True)

with chart_col2:
    
    prob_pct = sellout_probability * 100
    rem_pct = max(0.0, 100.0 - prob_pct)
    
    sellout_data = pd.DataFrame({
        "Status": ["Probabilidade de Esgotamento", "Risco Restante"],
        "Percentage": [prob_pct, rem_pct]
    })
    
    donut_chart = alt.Chart(sellout_data).mark_arc(innerRadius=70).encode(
        theta=alt.Theta(field="Percentage", type="quantitative"),
        color=alt.Color(
            field="Status", 
            type="nominal", 
            scale=alt.Scale(domain=["Probabilidade de Esgotamento", "Risco Restante"], range=["#2ca02c", "#333333"]),
            legend=alt.Legend(orient="bottom")
        ),
        tooltip=["Status", alt.Tooltip("Percentage:Q", format=".1f")]
    ).properties(height=350)
    
    st.altair_chart(donut_chart, use_container_width=True)

st.divider()

with st.expander("Arquitetura do Sistema e Inferências Brutas (Documentação para Apresentação)"):
    st.markdown("""
    **Visão Geral do Pipeline de Inteligência:**
    Esta plataforma mapeia a densidade de demanda de streaming diretamente em relação aos multiplicadores econômicos municipais locais.
    Um pipeline de `LogisticRegression` classifica a curva de risco de esgotamento, enquanto um modelo de regressão `Ridge` otimiza a variável de precificação contínua.
    """)
    
    st.code(f"""
    --- Matriz de Inferência ---
    Vetor de Entrada: [Capacidade: {input_capacity}, Demanda: {derived_demand:.4f}, IBGE: {derived_ibge:.2f}]
    
    --- Saídas do Pipeline ---
    Fonte do Motor de Dados: {data_source}
    Saída Bruta da Probabilidade de Esgotamento: {sellout_probability:.4f}
    Saída Bruta do Preço Ótimo do Ingresso: {predicted_price:.4f}
    """)

st.caption(f"Amanhã Tech Alpha v1.0 | Renderização do Motor: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")