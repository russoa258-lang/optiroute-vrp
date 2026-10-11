import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ==========================================
# 1. SETUP UI PROFESSIONALE
# ==========================================
st.set_page_config(page_title="OptiRoute | Enterprise VRP", layout="wide")

st.markdown("""
    <style>
    .reportview-container .main .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #0F172A; font-family: 'Inter', sans-serif; font-weight: 700; }
    .stButton>button { background-color: #0F172A; color: white; border-radius: 6px; font-weight: bold; transition: all 0.3s;}
    .stButton>button:hover { background-color: #334155; border-color: #334155; transform: translateY(-1px); }
    .metric-container { background-color: #FFFFFF; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border-top: 4px solid #0F172A;}
    </style>
""", unsafe_allow_html=True)

st.title("OptiRoute | Dynamic Routing Engine")
st.caption("Advanced Decision Support System for Fleet Operations & OPEX Minimization")

# Inizializzazione Stato
for key in ['optimized', 'routes', 'total_km', 'df_nodes']:
    if key not in st.session_state: st.session_state[key] = None if 'df' in key else (False if type(key) == bool else [])

# ==========================================
# 2. MOTORE DATI (VETTORIZZATO E CACHATO)
# ==========================================
@st.cache_data(show_spinner=False)
def geocode_addresses(df):
    geolocator = Nominatim(user_agent="optiroute_enterprise_v2")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    lats, lons = [], []
    progress_bar = st.progress(0, text="Sincronizzazione API Geospaziale...")
    
    for i, address in enumerate(df['Indirizzo']):
        try:
            full_address = address if "italy" in address.lower() or "italia" in address.lower() else f"{address}, Italy"
            location = geocode(full_address)
            if location:
                lats.append(location.latitude)
                lons.append(location.longitude)
            else:
                lats.append(None); lons.append(None)
        except:
            lats.append(None); lons.append(None)
        progress_bar.progress((i + 1) / len(df))
    
    progress_bar.empty()
    df['Lat'] = lats
    df['Lon'] = lons
    return df.dropna(subset=['Lat', 'Lon']).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def get_mock_data(n_clients):
    depot_lat, depot_lon = 44.6982, 10.6312  # Reggio Emilia Default
    np.random.seed(42)
    records = [{"ID": 0, "Nome": "HUB LOGISTICO", "Lat": depot_lat, "Lon": depot_lon, "Kg": 0}]
    for i in range(1, n_clients + 1):
        records.append({
            "ID": i, "Nome": f"Destinazione {i:02d}",
            "Lat": depot_lat + np.random.uniform(-0.08, 0.08),
            "Lon": depot_lon + np.random.uniform(-0.09, 0.09),
            "Kg": int(np.random.randint(20, 150))
        })
    return pd.DataFrame(records)

# SUPER OTTIMIZZAZIONE: Matrice Distanze con NumPy (Haversine Vettorializzato)
@st.cache_data(show_spinner=False)
def build_dist_matrix_fast(df):
    coords = np.radians(df[['Lat', 'Lon']].to_numpy())
    lat, lon = coords[:, 0], coords[:, 1]
    
    dlat = lat[:, np.newaxis] - lat
    dlon = lon[:, np.newaxis] - lon
    
    a = np.sin(dlat / 2)**2 + np.cos(lat)[:, np.newaxis] * np.cos(lat) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    R = 6371000 # Raggio terrestre in metri
    matrix = np.round((R * c) * 1.3).astype(int) # 1.3 fattore di correzione stradale
    return matrix.tolist()

# ==========================================
# 3. SOLVER OR-TOOLS (HIGH PERFORMANCE)
# ==========================================
def run_solver(matrix, demands, n_vehicles, cap):
    manager = pywrapcp.RoutingIndexManager(len(matrix), n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    transit_idx = routing.RegisterTransitCallback(lambda f, t: matrix[manager.IndexToNode(f)][manager.IndexToNode(t)])
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    demand_idx = routing.RegisterUnaryTransitCallback(lambda f: demands[manager.IndexToNode(f)])
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [cap] * n_vehicles, True, 'Capacity')

    params = pywrapcp.DefaultRoutingSearchParameters()
    # Strategie super-veloci per trovare subito la soluzione
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
    params.time_limit.seconds = 1  # Massimo 1 secondo di elaborazione!

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
# 4. PANNELLO DI CONTROLLO
# ==========================================
with st.sidebar:
    st.header("1. Input Dati")
    uploaded_file = st.file_uploader("Dataset (.csv / .xlsx)", type=["csv", "xlsx"])
    
    st.markdown("---")
    st.header("2. Vincoli Flotta")
    n_vehicles = st.number_input("Mezzi Disponibili", min_value=1, value=5)
    capacity = st.number_input("Capacità Max (Kg)", min_value=10, value=1500, step=50)
    fuel_cost = st.number_input("Costo €/Km", min_value=0.10, value=1.65, step=0.05)
    
    st.markdown("---")
    if st.button("ESEGUI MOTORE OTTIMIZZAZIONE", use_container_width=True):
        with st.spinner("Elaborazione tensoriale e routing in corso..."):
            if uploaded_file:
                raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                if 'Indirizzo' in raw_df.columns:
                    df_nodes = geocode_addresses(raw_df)
                elif 'Lat' in raw_df.columns and 'Lon' in raw_df.columns:
                    df_nodes = raw_df
                else:
                    st.error("Colonne mancanti. Richiesto: Nome, Indirizzo, Kg.")
                    st.stop()
            else:
                df_nodes = get_mock_data(40)
            
            st.session_state.df_nodes = df_nodes
            matrix = build_dist_matrix_fast(df_nodes)
            routes, total_km = run_solver(matrix, df_nodes['Kg'].tolist(), n_vehicles, capacity)
            
            if routes:
                st.session_state.routes = routes
                st.session_state.total_km = total_km
                st.session_state.optimized = True
            else:
                st.error("Infeasible: La flotta non ha capacità sufficiente per il carico.")
                st.session_state.optimized = False

# ==========================================
# 5. DASHBOARD & MAPPA DINAMICA
# ==========================================
df = st.session_state.get('df_nodes')
if df is None: df = get_mock_data(40)

if st.session_state.optimized:
    tot_km = st.session_state.total_km
    manual_km = tot_km * 1.35
    eur_saved = (manual_km - tot_km) * fuel_cost * 22
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Km Ottimizzati", f"{tot_km:.1f} km", f"-{manual_km - tot_km:.1f} km (Delta)")
    c2.metric("EBITDA Saving (Mese)", f"€ {eur_saved:.0f}")
    c3.metric("Flotta Saturata", f"{len(st.session_state.routes)} / {n_vehicles} mezzi")
    c4.metric("Volume Movimentato", f"{df['Kg'].sum():,} Kg")
    st.markdown("---")

col_map, col_details = st.columns([2.5, 1])
PALETTE = ['#0F172A', '#059669', '#DC2626', '#D97706', '#4F46E5', '#0891B2']

with col_map:
    st.markdown("### Tracking Operativo Dinamico")
    
    # MAPPA DINAMICA: Calcola i bound per auto-zoon e centratura perfetta
    sw = df[['Lat', 'Lon']].min().values.tolist()
    ne = df[['Lat', 'Lon']].max().values.tolist()
    
    m = folium.Map(tiles='CartoDB positron')
    m.fit_bounds([sw, ne]) # La mappa si adatta automaticamente a qualsiasi città!
    
    folium.Marker([df.iloc[0]['Lat'], df.iloc[0]['Lon']], popup="HUB", icon=folium.Icon(color='black', icon='building', prefix='fa')).add_to(m)

    if not st.session_state.optimized:
        for _, row in df.iloc[1:].iterrows():
            folium.CircleMarker(location=[row['Lat'], row['Lon']], radius=4, color='#94A3B8', fill=True).add_to(m)
    else:
        for v_idx, route in enumerate(st.session_state.routes):
            color = PALETTE[v_idx % len(PALETTE)]
            coords = []
            for node in route:
                lat, lon = df.iloc[node]['Lat'], df.iloc[node]['Lon']
                coords.append([lat, lon])
                if node != 0:
                    folium.CircleMarker(location=[lat, lon], radius=5, color=color, fill=True, fill_opacity=1, popup=df.iloc[node]['Nome']).add_to(m)
            folium.PolyLine(coords, color=color, weight=3, opacity=0.8).add_to(m)
            
    folium_static(m, width=900, height=550)

with col_details:
    st.markdown("### Manifesti di Carico")
    if st.session_state.optimized:
        for v_idx, route in enumerate(st.session_state.routes):
            tot_kg = sum(df.iloc[node]['Kg'] for node in route)
            with st.expander(f"🚛 Mezzo {v_idx+1} | {tot_kg} Kg | {len(route)-2} Stop"):
                for step, node in enumerate(route):
                    if node != 0:
                        st.markdown(f"**{step}.** {df.iloc[node]['Nome']} *( {df.iloc[node]['Kg']}Kg )*")
    else:
        st.info("In attesa di elaborazione dati...")
