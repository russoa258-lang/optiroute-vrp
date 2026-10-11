import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ==========================================
# 1. CONFIGURAZIONE UI CORPORATE
# ==========================================
st.set_page_config(page_title="OptiRoute | Enterprise Routing", layout="wide")

# CSS Custom per un look professionale e minimale
st.markdown("""
    <style>
    .reportview-container .main .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #1E3A8A; font-family: 'Inter', sans-serif; }
    .stButton>button { background-color: #1E3A8A; color: white; border-radius: 4px; font-weight: bold; }
    .stButton>button:hover { background-color: #152C69; border-color: #152C69; }
    .css-1d391kg { background-color: #F8FAFC; } /* Sfondo Sidebar leggero */
    .metric-container { background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

st.title("OptiRoute Dynamic VRP Optimizer")
st.caption("Decision Support System - Operational Expenditure (OPEX) Reduction Tool")

# Inizializzazione Stato
if 'optimized' not in st.session_state: st.session_state.optimized = False
if 'routes' not in st.session_state: st.session_state.routes = []
if 'total_km' not in st.session_state: st.session_state.total_km = 0
if 'ai_reports' not in st.session_state: st.session_state.ai_reports = []

# ==========================================
# 2. MOTORE DATI E CACHING AVANZATO
# ==========================================
# Il caching evita di richiamare l'API per file già elaborati nella sessione
@st.cache_data(show_spinner=False)
def geocode_addresses(df):
    geolocator = Nominatim(user_agent="optiroute_enterprise_v1")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    lats, lons = [], []
    progress_bar = st.progress(0, text="Sincronizzazione dati geospaziali in corso...")
    
    for i, address in enumerate(df['Indirizzo']):
        try:
            full_address = address if "italy" in address.lower() or "italia" in address.lower() else f"{address}, Italy"
            location = geocode(full_address)
            if location:
                lats.append(location.latitude)
                lons.append(location.longitude)
            else:
                lats.append(None)
                lons.append(None)
        except:
            lats.append(None)
            lons.append(None)
        progress_bar.progress((i + 1) / len(df))
    
    progress_bar.empty()
    df['Lat'] = lats
    df['Lon'] = lons
    return df.dropna(subset=['Lat', 'Lon']).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def get_mock_data(n_clients):
    depot_lat, depot_lon = 44.6982, 10.6312 
    np.random.seed(42)
    records = [{"ID": 0, "Nome": "HUB Centrale", "Lat": depot_lat, "Lon": depot_lon, "Kg": 0}]
    for i in range(1, n_clients + 1):
        records.append({
            "ID": i, "Nome": f"Cliente {i:02d}",
            "Lat": depot_lat + np.random.uniform(-0.08, 0.08),
            "Lon": depot_lon + np.random.uniform(-0.09, 0.09),
            "Kg": int(np.random.randint(20, 150))
        })
    return pd.DataFrame(records)

@st.cache_data(show_spinner=False)
def build_dist_matrix(df):
    n = len(df)
    coords = df[['Lat', 'Lon']].to_numpy()
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = math.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1]) * 111000 * 1.3
                matrix[i, j] = int(d)
    return matrix.tolist()

# ==========================================
# 3. SOLVER MATEMATICO
# ==========================================
def run_solver(matrix, demands, n_vehicles, cap):
    manager = pywrapcp.RoutingIndexManager(len(matrix), n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    transit_idx = routing.RegisterTransitCallback(lambda f, t: matrix[manager.IndexToNode(f)][manager.IndexToNode(t)])
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    demand_idx = routing.RegisterUnaryTransitCallback(lambda f: demands[manager.IndexToNode(f)])
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [cap] * n_vehicles, True, 'Capacity')

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 2  # Ottimizzazione ultra-rapida

    solution = routing.SolveWithParameters(params)
    if not solution: return [], 0

    routes, tot_dist = [], 0
    for v in range(n_vehicles):
        idx = routing.Start(v)
        route, route_dist = [], 0
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            route.append(node)
            prev_idx = idx
            idx = solution.Value(routing.NextVar(idx))
            route_dist += routing.GetArcCostForVehicle(prev_idx, idx, v)
        route.append(manager.IndexToNode(idx))
        if len(route) > 2:
            routes.append(route)
            tot_dist += route_dist

    return routes, tot_dist / 1000

# ==========================================
# 4. PANNELLO DI CONTROLLO (SIDEBAR)
# ==========================================
with st.sidebar:
    st.header("Parametri di Input")
    st.markdown("Carica dataset in formato .csv o .xlsx. Colonne richieste: **Nome, Indirizzo, Kg**.")
    uploaded_file = st.file_uploader("Dataset Destinazioni", type=["csv", "xlsx"])
    
    st.markdown("---")
    st.header("Vincoli Operativi")
    n_vehicles = st.number_input("Dimensione Flotta", min_value=1, max_value=50, value=5, step=1)
    capacity = st.number_input("Capacità Veicolo (Kg)", min_value=10, max_value=10000, value=1500, step=50)
    fuel_cost = st.number_input("Costo Medio Carburante/Km", min_value=0.10, max_value=5.0, value=1.65, step=0.05)
    
    st.markdown("---")
    if st.button("ESEGUI OTTIMIZZAZIONE", use_container_width=True):
        with st.spinner("Elaborazione matrice distanze e calcolo percorsi..."):
            if uploaded_file:
                raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                if 'Indirizzo' in raw_df.columns:
                    df_nodes = geocode_addresses(raw_df)
                elif 'Lat' in raw_df.columns and 'Lon' in raw_df.columns:
                    df_nodes = raw_df
                else:
                    st.error("Formato file non valido. Colonne mancanti.")
                    st.stop()
            else:
                df_nodes = get_mock_data(30)
            
            st.session_state.df_nodes = df_nodes
            matrix = build_dist_matrix(df_nodes)
            
            routes, total_km = run_solver(matrix, df_nodes['Kg'].tolist(), n_vehicles, capacity)
            if routes:
                st.session_state.routes = routes
                st.session_state.total_km = total_km
                st.session_state.optimized = True
            else:
                st.error("Capacità flotta insufficiente per il volume di carico richiesto.")
                st.session_state.optimized = False

# ==========================================
# 5. DASHBOARD CENTRALE (MAPPA E KPI)
# ==========================================
df = st.session_state.get('df_nodes', get_mock_data(30))

# Sezione KPI Superiori
if st.session_state.optimized:
    tot_km = st.session_state.total_km
    manual_km = tot_km * 1.35
    km_saved = manual_km - tot_km
    eur_saved_month = km_saved * fuel_cost * 22
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Chilometri Totali", f"{tot_km:.1f} km", f"-{km_saved:.1f} km vs Manuale", delta_color="inverse")
    kpi2.metric("Saving Mensile Stimato", f"€ {eur_saved_month:.0f}")
    kpi3.metric("Veicoli Impiegati", f"{len(st.session_state.routes)} / {n_vehicles}")
    kpi4.metric("Carico Totale (Kg)", f"{df['Kg'].sum()}")
    st.markdown("---")

# Layout Principale: Mappa a Sinistra, Dettagli a Destra
col_map, col_details = st.columns([2.5, 1])
PALETTE = ['#1E3A8A', '#047857', '#B91C1C', '#D97706', '#4C1D95', '#0891B2', '#BE185D']

with col_map:
    st.markdown("### Controllo Flotta in Tempo Reale")
    centro_lat, centro_lon = df['Lat'].mean(), df['Lon'].mean()
    m = folium.Map(location=[centro_lat, centro_lon], zoom_start=11, tiles='CartoDB positron') # Stile mappa corporate
    
    folium.Marker([df.iloc[0]['Lat'], df.iloc[0]['Lon']], popup="HUB", icon=folium.Icon(color='black', icon='building', prefix='fa')).add_to(m)

    if not st.session_state.optimized:
        for _, row in df.iloc[1:].iterrows():
            folium.CircleMarker(location=[row['Lat'], row['Lon']], radius=4, color='#64748B', fill=True).add_to(m)
    else:
        for v_idx, route in enumerate(st.session_state.routes):
            color = PALETTE[v_idx % len(PALETTE)]
            coords = []
            for node in route:
                lat, lon = df.iloc[node]['Lat'], df.iloc[node]['Lon']
                coords.append([lat, lon])
                if node != 0:
                    folium.CircleMarker(location=[lat, lon], radius=5, color=color, fill=True, fill_opacity=1, popup=df.iloc[node]['Nome']).add_to(m)
            folium.PolyLine(coords, color=color, weight=3, opacity=0.9).add_to(m)
            
    folium_static(m, width=900, height=600)

with col_details:
    st.markdown("### Distinta di Carico")
    if st.session_state.optimized:
        for v_idx, route in enumerate(st.session_state.routes):
            color = PALETTE[v_idx % len(PALETTE)]
            tot_kg = sum(df.iloc[node]['Kg'] for node in route)
            
            with st.expander(f"Veicolo {v_idx+1} ({tot_kg} Kg)"):
                st.markdown(f"**Fermate: {len(route)-2}**")
                for step, node in enumerate(route):
                    if node != 0:
                        st.markdown(f"{step}. {df.iloc[node]['Nome']} - {df.iloc[node]['Kg']}Kg")
    else:
        st.info("Carica i dati ed esegui l'ottimizzazione per visualizzare la distinta di carico per ciascun veicolo.")
