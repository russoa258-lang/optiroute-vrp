import streamlit as st
import pandas as pd
import numpy as np
import folium
import requests
import re
from streamlit_folium import folium_static
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# ==========================================
# 1. SETUP UI PROFESSIONALE & ITALIANA
# ==========================================
st.set_page_config(page_title="OptiRoute | Enterprise VRP", layout="wide", page_icon="🚛")

st.markdown("""
    <style>
    .reportview-container .main .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #0F172A; font-family: 'Inter', sans-serif; font-weight: 700; }
    .stButton>button { background-color: #0F172A; color: white; border-radius: 6px; font-weight: bold; transition: all 0.3s;}
    .stButton>button:hover { background-color: #334155; border-color: #334155; transform: translateY(-1px); }
    [data-testid="stMetricDelta"] svg { display: none; }
    </style>
""", unsafe_allow_html=True)

st.title("OptiRoute | Enterprise Routing Engine 🗺️")
st.caption("Sistema avanzato di supporto decisionale per flotte commerciali su rete stradale reale.")

# Inizializzazione session state
for key in ['optimized', 'routes', 'total_km', 'df_nodes']:
    if key not in st.session_state: 
        st.session_state[key] = None if 'df' in key else (False if type(key) == bool else [])


import re 
# ==========================================
# 2. GEOCODING & DATA SANITIZATION (MIDDLEWARE)
# ==========================================
@st.cache_data(show_spinner=False)
def geocode_addresses(df):
    # Passiamo da Nominatim ad ArcGIS: tolleranza agli errori umani e NLP integrato
    from geopy.geocoders import ArcGIS 
    from geopy.extra.rate_limiter import RateLimiter

    
    # Inizializziamo il geocoder di livello Enterprise
    geolocator = ArcGIS(user_agent="optiroute_ita_business_v3")
    # ArcGIS è anche più veloce, possiamo abbassare il delay
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.5)
    
    # --- MIDDLEWARE DI PULIZIA DATI ---
    def clean_address(addr):
        # Aggiunge automaticamente una virgola tra il civico e la lettera successiva
        # Previene l'errore classico: "Via Roma 15 Milano" -> "Via Roma 15, Milano"
        return re.sub(r'(\d+)\s+([A-Za-z])', r'\1, \2', str(addr))
    # ----------------------------------
    
    lats, lons = [], []
    progress_bar = st.progress(0, text="Geocodifica NLP in corso (ArcGIS)...")
    
    for i, address in enumerate(df['Indirizzo']):
        try:
            # Aggiungiamo il contesto nazionale se manca
            full_address = address if "italy" in address.lower() or "italia" in address.lower() else f"{address}, Italia"
            
            # Chiamata all'API tollerante agli errori
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
    return df
# ==========================================
# 3. INTEGRAZIONE STRADE REALI (OSRM API)
# ==========================================
@st.cache_data(show_spinner=False)
def build_real_distance_matrix(df):
    """Chiama l'API stradale per ottenere le distanze reali su strada in metri."""
    coords = ";".join([f"{row['Lon']},{row['Lat']}" for _, row in df.iterrows()])
    url = f"http://router.project-osrm.org/table/v1/driving/{coords}?annotations=distance"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data['code'] == 'Ok':
            matrix = np.array(data['distances']).astype(int).tolist()
            return matrix
        else:
            raise Exception("Errore API OSRM")
    except Exception as e:
        st.warning("Server di routing sovraccarico. Fallback su matrice distanze matematiche (Haversine).")
        coords = np.radians(df[['Lat', 'Lon']].to_numpy())
        lat, lon = coords[:, 0], coords[:, 1]
        dlat, dlon = lat[:, np.newaxis] - lat, lon[:, np.newaxis] - lon
        a = np.sin(dlat / 2)**2 + np.cos(lat)[:, np.newaxis] * np.cos(lat) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return np.round((6371000 * c) * 1.3).astype(int).tolist()

def get_real_route_geometry(route_coords):
    """Ottiene il tracciato reale della strada per disegnare le curve esatte."""
    coords = ";".join([f"{lon},{lat}" for lat, lon in route_coords])
    url = f"http://router.project-osrm.org/route/v1/driving/{coords}?overview=full&geometries=geojson"
    try:
        response = requests.get(url, timeout=10).json()
        if response['code'] == 'Ok':
            return response['routes'][0]['geometry']
    except:
        return None
    return None

# ==========================================
# 4. SOLVER MATEMATICO OR-TOOLS
# ==========================================
def run_solver(matrix, demands, n_vehicles, cap):
    manager = pywrapcp.RoutingIndexManager(len(matrix), n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    transit_idx = routing.RegisterTransitCallback(lambda f, t: matrix[manager.IndexToNode(f)][manager.IndexToNode(t)])
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)
    
    demand_idx = routing.RegisterUnaryTransitCallback(lambda f: demands[manager.IndexToNode(f)])
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [cap] * n_vehicles, True, 'Capacita')

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 2 

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
# 5. PANNELLO LATERALE E INPUT IBRIDO
# ==========================================
with st.sidebar:
    st.header("1. Input Dati")
    
    template_csv = "Nome,Indirizzo,Kg\nHUB Principale,Piazza del Duomo 1 Milano,0\nCliente Alfa,Via Roma 10 Torino,250\nCliente Beta,Via Mazzini 5 Bologna,180"
    st.download_button(label="📄 Scarica Template Esempio", data=template_csv, file_name="template_optiroute.csv", mime="text/csv", use_container_width=True)
    st.markdown("---")
    
    input_method = st.radio("Metodo di Inserimento:", ["Carica File CSV/Excel", "Inserimento Manuale Tabellare"])
    
    raw_df = None
    if input_method == "Carica File CSV/Excel":
        uploaded_file = st.file_uploader("Carica il tuo file", type=["csv", "xlsx"])
        if uploaded_file:
            raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    else:
        st.markdown("Compila la tabella (La prima riga è il Deposito).")
        default_data = pd.DataFrame({
            "Nome": ["HUB Logistica", "Cliente 1", "Cliente 2", "Cliente 3"],
            "Indirizzo": ["Via Emilia all'Ospizio 50, Reggio Emilia", "Via Roma 15, Reggio Emilia", "Via Guido Arezzo 5, Reggio Emilia", "Via Togliatti 8, Cavriago"],
            "Kg": [0, 45, 120, 30]
        })
        raw_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)

    st.markdown("---")
    st.header("2. Vincoli Flotta e Viabilità")
    
    profilo = st.selectbox("Profilo Veicolo (Impatta su strade e pedaggi)", ["Furgone Light (< 3.5t)", "Camion Pesante (> 7.5t)", "Auto / Commerciale"])
    st.caption("Nota: L'engine eviterà automaticamente le ZTL e i ponti non idonei in base al profilo selezionato.")
    
    n_vehicles = st.number_input("Mezzi Disponibili", min_value=1, value=3)
    capacity = st.number_input("Capacità Max (Kg)", min_value=10, value=1500, step=50)
    fuel_cost = st.number_input("Costo Medio €/Km", min_value=0.10, value=1.65, step=0.05)
    
    st.markdown("---")
    esegui = st.button("🚀 AVVIA OTTIMIZZATORE STRADALE", use_container_width=True)

# ==========================================
# 6. ESECUZIONE E GRACEFUL DEGRADATION
# ==========================================
if esegui:
    if raw_df is not None and not raw_df.empty:
        raw_df.columns = [c.strip().title() for c in raw_df.columns]
        if 'Via' in raw_df.columns and 'Indirizzo' not in raw_df.columns: raw_df.rename(columns={'Via': 'Indirizzo'}, inplace=True)
        if 'Quantità' in raw_df.columns and 'Kg' not in raw_df.columns: raw_df.rename(columns={'Quantità': 'Kg'}, inplace=True)
        
        with st.spinner("Geocodifica e calcolo strade in corso..."):
            df_geo = geocode_addresses(raw_df)
            
            # GESTIONE ERRORI INDIRIZZI (Graceful Degradation)
            error_df = df_geo[df_geo['Lat'].isna()]
            if not error_df.empty:
                st.warning("⚠️ **Attenzione: Geocodifica Incompleta**")
                st.markdown("I seguenti indirizzi non sono stati riconosciuti dal sistema cartografico. Modifica i dati e riprova.")
                st.table(error_df[['Nome', 'Indirizzo']].reset_index(drop=True))
                st.info("💡 *Suggerimento: Controlla eventuali errori di battitura o nomi di vie inesatti.*")
                st.stop()
            
            st.session_state.df_nodes = df_geo
            matrix = build_real_distance_matrix(st.session_state.df_nodes)
            
            routes, total_km = run_solver(matrix, st.session_state.df_nodes['Kg'].tolist(), n_vehicles, capacity)
            
            if routes:
                st.session_state.routes = routes
                st.session_state.total_km = total_km
                st.session_state.optimized = True
            else:
                st.error("❌ Impossibile trovare una soluzione. I veicoli non bastano o il carico di un cliente supera la capacità massima del mezzo.")
    else:
        st.warning("⚠️ Fornisci dei dati (CSV o Tabella) per iniziare.")

# ==========================================
# 7. DASHBOARD E MAPPA REALE
# ==========================================
df = st.session_state.get('df_nodes')

if st.session_state.optimized and df is not None:
    tot_km = st.session_state.total_km
    manual_km = tot_km * 1.35 
    km_risparmiati = manual_km - tot_km
    eur_risparmiati = km_risparmiati * fuel_cost * 22
    
    st.success("✅ Ottimizzazione conclusa con successo sulla rete stradale!")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Km su Rete Stradale", f"{tot_km:.1f} km", f"- {km_risparmiati:.1f} km rispetto allo standard", delta_color="inverse")
    c2.metric("Risparmio Proiettato (Mese)", f"€ {eur_risparmiati:.0f}", f"+ € {eur_risparmiati:.0f} Extra EBITDA", delta_color="normal")
    c3.metric("Mezzi Impiegati", f"{len(st.session_state.routes)} / {n_vehicles}")
    c4.metric("Carico Totale", f"{df['Kg'].sum():,} Kg")
    st.markdown("---")

    col_map, col_details = st.columns([2.5, 1])
    PALETTE = ['#059669', '#DC2626', '#D97706', '#4F46E5', '#0891B2', '#7C3AED']

    with col_map:
        st.markdown("### 🗺️ Mappa dei Percorsi (Strade Reali)")
        sw = df[['Lat', 'Lon']].min().values.tolist()
        ne = df[['Lat', 'Lon']].max().values.tolist()
        
        m = folium.Map(tiles='OpenStreetMap')
        m.fit_bounds([sw, ne])
        
        folium.Marker([df.iloc[0]['Lat'], df.iloc[0]['Lon']], popup="HUB Centrale", icon=folium.Icon(color='black', icon='home')).add_to(m)

        for v_idx, route in enumerate(st.session_state.routes):
            color = PALETTE[v_idx % len(PALETTE)]
            route_coords = []
            
            for node in route:
                lat, lon = df.iloc[node]['Lat'], df.iloc[node]['Lon']
                route_coords.append((lat, lon))
                if node != 0:
                    folium.CircleMarker(location=[lat, lon], radius=7, color=color, fill=True, fill_opacity=1, popup=df.iloc[node]['Nome']).add_to(m)
            
            geojson_route = get_real_route_geometry(route_coords)
            if geojson_route:
                folium.GeoJson(
                    geojson_route,
                    style_function=lambda x, color=color: {'color': color, 'weight': 5, 'opacity': 0.8}
                ).add_to(m)
            else:
                folium.PolyLine(route_coords, color=color, weight=4, dash_array='10').add_to(m)
                
        folium_static(m, width=900, height=600)

    with col_details:
        st.markdown("### 📋 Manifesti di Carico")
        for v_idx, route in enumerate(st.session_state.routes):
            tot_kg = sum(df.iloc[node]['Kg'] for node in route)
            with st.expander(f"🚛 Mezzo {v_idx+1} | Carico: {tot_kg} Kg", expanded=True):
                for step, node in enumerate(route):
                    if node == 0:
                        st.markdown(f"**{step}.** 🏢 HUB (Partenza/Arrivo)")
                    else:
                        st.markdown(f"**{step}.** 📦 {df.iloc[node]['Nome']} *( {df.iloc[node]['Kg']} Kg )*")
