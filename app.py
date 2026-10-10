import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math

# ==========================================
# 1. SETUP PAGINA E STATO
# ==========================================
st.set_page_config(page_title="OptiRoute - VRP Enterprise", layout="wide", page_icon="🚛")
st.title("🚛 OptiRoute: Dynamic VRP Optimizer (Enterprise Edition)")
st.caption("Decision Support System per l'abbattimento dei costi logistici e l'ottimizzazione del routing.")

if 'optimized' not in st.session_state:
    st.session_state.optimized = False
if 'routes' not in st.session_state:
    st.session_state.routes = []
if 'total_km' not in st.session_state:
    st.session_state.total_km = 0
if 'df_nodes' not in st.session_state:
    st.session_state.df_nodes = None

# ==========================================
# 2. GENERAZIONE DATI (MOCK O REALI)
# ==========================================
def get_mock_data(n_clients):
    depot_lat, depot_lon = 44.6982, 10.6312
    np.random.seed(42)
    records = [{"ID": 0, "Nome": "HUB Centrale", "Lat": depot_lat, "Lon": depot_lon, "Kg": 0}]
    for i in range(1, n_clients + 1):
        records.append({
            "ID": i,
            "Nome": f"Cliente #{i:02d}",
            "Lat": depot_lat + np.random.uniform(-0.08, 0.08),
            "Lon": depot_lon + np.random.uniform(-0.09, 0.09),
            "Kg": int(np.random.randint(20, 150))
        })
    return pd.DataFrame(records)

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
# 3. SOLVER OR-TOOLS 
# ==========================================
def run_solver(matrix, demands, n_vehicles, cap):
    manager = pywrapcp.RoutingIndexManager(len(matrix), n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_idx, to_idx):
        return matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]
    transit_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demand_cb(from_idx):
        return demands[manager.IndexToNode(from_idx)]
    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, [cap] * n_vehicles, True, 'Capacity'
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 3

    solution = routing.SolveWithParameters(params)

    if not solution:
        return [], 0

    routes = []
    tot_dist = 0
    for v in range(n_vehicles):
        idx = routing.Start(v)
        route = []
        route_dist = 0
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
# 4. SIDEBAR - DATI E VINCOLI SENZA LIMITI
# ==========================================
with st.sidebar:
    st.header("📁 1. Dati Aziendali")
    uploaded_file = st.file_uploader("Carica File Consegne (CSV/Excel)", type=["csv", "xlsx"], help="Formato richiesto: Colonne 'Nome', 'Lat', 'Lon', 'Kg'. La prima riga deve essere il Deposito.")
    
    if not uploaded_file:
        st.info("Nessun file caricato. Uso dati demo.")
        n_clients = st.slider("Numero Consegne Demo", 5, 100, 25)
    
    st.markdown("---")
    st.header("⚙️ 2. Vincoli di Flotta")
    # Qui abbiamo tolto i limiti rigidi. Minimo 1 veicolo, massimo 1000.
    n_vehicles = st.number_input("Furgoni Disponibili", min_value=1, max_value=1000, value=5, step=1)
    
    # Capacità da 10 kg (es. droni/bici) fino a 50 tonnellate.
    capacity = st.number_input("Capacità Veicolo (Kg)", min_value=10, max_value=50000, value=1500, step=50)
    
    fuel_cost = st.number_input("Costo Medio/Km (€)", min_value=0.10, max_value=10.0, value=1.65, step=0.05)
    
    st.markdown("---")
    if st.button("🚀 AVVIA MOTORE DI OTTIMIZZAZIONE", type="primary", use_container_width=True):
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df_nodes = pd.read_csv(uploaded_file)
            else:
                st.session_state.df_nodes = pd.read_excel(uploaded_file)
        else:
            st.session_state.df_nodes = get_mock_data(n_clients)
            
        matrix = build_dist_matrix(st.session_state.df_nodes)
        demands = st.session_state.df_nodes['Kg'].tolist()
        
        with st.spinner("Macinazione dati in corso (Google OR-Tools)..."):
            routes, total_km = run_solver(matrix, demands, n_vehicles, capacity)
            if routes:
                st.session_state.routes = routes
                st.session_state.total_km = total_km
                st.session_state.optimized = True
            else:
                st.error("Capacità superata! I pacchi pesano più di quanto la flotta possa trasportare. Aumenta il numero di furgoni o la capacità.")
                st.session_state.optimized = False

df = st.session_state.df_nodes if st.session_state.df_nodes is not None else get_mock_data(25)

# ==========================================
# 5. DASHBOARD & MAPPA VISIVA
# ==========================================
col_map, col_metrics = st.columns([2.5, 1])
PALETTE = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00', '#000000']

with col_map:
    st.subheader("Mappa Tattica di Consegna")
    m = folium.Map(location=[df.iloc[0]['Lat'], df.iloc[0]['Lon']], zoom_start=11, tiles="CartoDB positron")
    
    folium.Marker(
        [df.iloc[0]['Lat'], df.iloc[0]['Lon']],
        popup="<b>HUB DI PARTENZA</b>",
        icon=folium.Icon(color='red', icon='star')
    ).add_to(m)

    if not st.session_state.optimized:
        for _, row in df.iloc[1:].iterrows():
            folium.CircleMarker(
                location=[row['Lat'], row['Lon']], radius=5, color='#333', fill=True, popup=f"{row['Nome']}: {row['Kg']} Kg"
            ).add_to(m)
        st_folium(m, width="100%", height=550, key="map_init")
    else:
        for v_idx, route in enumerate(st.session_state.routes):
            color = PALETTE[v_idx % len(PALETTE)]
            coords = []
            for stop_order, node in enumerate(route):
                lat, lon = df.iloc[node]['Lat'], df.iloc[node]['Lon']
                coords.append([lat, lon])
                if node != 0:
                    folium.CircleMarker(
                        location=[lat, lon], radius=6, color=color, fill=True, fill_color=color, fill_opacity=0.9,
                        popup=f"Veicolo {v_idx+1} | Stop {stop_order}: {df.iloc[node]['Nome']}"
                    ).add_to(m)
            folium.PolyLine(coords, color=color, weight=4, opacity=0.8).add_to(m)
        st_folium(m, width="100%", height=550, key="map_opt")

with col_metrics:
    st.subheader("Impatto Finanziario")
    if st.session_state.optimized:
        tot_km = st.session_state.total_km
        manual_km = tot_km * 1.28 # Benchmark 28% di spreco
        km_saved = manual_km - tot_km
        eur_saved_day = km_saved * fuel_cost
        eur_saved_month = eur_saved_day * 22
        
        st.metric("Chilometri Totali Ottimizzati", f"{tot_km:.1f} km", f"-{km_saved:.1f} km a vuoto evitati", delta_color="inverse")
        st.metric("Recupero Cassa (Mensile)", f"€ {eur_saved_month:.0f}", "Sul Conto Economico")
        st.metric("Flotta Attiva", f"{len(st.session_state.routes)} mezzi su {n_vehicles} disp.")
        
        st.divider()
        st.markdown("### 🖨️ Esporta Documenti")
        
        # Generazione CSV per gli autisti
        report_data = []
        for v_idx, route in enumerate(st.session_state.routes):
            for stop_order, node in enumerate(route):
                report_data.append({
                    "Veicolo": f"Furgone {v_idx+1}",
                    "Ordine_Tappa": stop_order,
                    "Cliente": df.iloc[node]['Nome'],
                    "Kg_da_Scaricare": df.iloc[node]['Kg']
                })
        df_report = pd.DataFrame(report_data)
        csv = df_report.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Scarica Piano di Carico (CSV)",
            data=csv,
            file_name="piano_di_consegna.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Carica i dati e premi Avvia per generare il piano di carico.")

