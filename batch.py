import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import io
import pulp
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Référentiel intégré des gares ===
REF_GASTES = [
    {"nomgare": "AEROPMdV", "ligne": "LIN09", "pk": 0.00},
    {"nomgare": "AINSBIT", "ligne": "LIN02", "pk": 0.00},
    {"nomgare": "AINSEBAA", "ligne": "LIN01", "pk": 0.00},
    {"nomgare": "ALAMIR", "ligne": "LIN03", "pk": 0.00},
    {"nomgare": "AOUDA", "ligne": "LIN03", "pk": 1.00},
    {"nomgare": "ARBAOUA", "ligne": "LIN03", "pk": 2.00},
    {"nomgare": "ASILAH", "ligne": "LIN03", "pk": 3.00},
    {"nomgare": "AZACAN", "ligne": "LIN07", "pk": 0.00},
    {"nomgare": "AZEMMOUR", "ligne": "LIN08", "pk": 0.00},
    {"nomgare": "BENIANSAR", "ligne": "LIN05", "pk": 0.00},
    {"nomgare": "BENIOUKIL", "ligne": "LIN02", "pk": 1.00},
    {"nomgare": "BERKANE", "ligne": "LIN05", "pk": 1.00},
    {"nomgare": "BERRECHID", "ligne": "LIN06", "pk": 0.00},
    {"nomgare": "BGR", "ligne": "LIN06", "pk": 1.00},
    {"nomgare": "BIDANE", "ligne": "LIN07", "pk": 1.00},
    {"nomgare": "BOUFARROUJ", "ligne": "LIN06", "pk": 2.00},
    {"nomgare": "BOUGUEDRA", "ligne": "LIN07", "pk": 2.00},
    {"nomgare": "BOUSKOURA", "ligne": "LIN06", "pk": 3.00},
    {"nomgare": "BOUZNIKA", "ligne": "LIN01", "pk": 1.00},
    {"nomgare": "CASAPORT", "ligne": "LIN01", "pk": 2.00},
    {"nomgare": "CHEBABAT", "ligne": "LIN02", "pk": 2.00},
    {"nomgare": "DALIA", "ligne": "LIN03", "pk": 4.00},
    {"nomgare": "ELAGREB", "ligne": "LIN02", "pk": 3.00},
    {"nomgare": "ELAIDI", "ligne": "LIN09", "pk": 1.00},
    {"nomgare": "ELAIOUN", "ligne": "LIN02", "pk": 4.00},
    {"nomgare": "ELARIA", "ligne": "LIN07", "pk": 3.00},
    {"nomgare": "ELGOUFAF", "ligne": "LIN09", "pk": 2.00},
    {"nomgare": "ELGUEDARI", "ligne": "LIN01", "pk": 3.00},
    {"nomgare": "ELJADIDA", "ligne": "LIN08", "pk": 1.00},
    {"nomgare": "ELOUED", "ligne": "LIN09", "pk": 3.00},
    {"nomgare": "ENNASSIM", "ligne": "LIN06", "pk": 4.00},
    {"nomgare": "FACULTES", "ligne": "LIN06", "pk": 5.00},
    {"nomgare": "FES", "ligne": "LIN03", "pk": 5.00},
    {"nomgare": "GTETER", "ligne": "LIN02", "pk": 5.00},
    {"nomgare": "GUERCIF", "ligne": "LIN02", "pk": 6.00},
    {"nomgare": "H", "ligne": "LIN08", "pk": 2.00},
    {"nomgare": "HAJJAJ", "ligne": "LIN09", "pk": 4.00},
    {"nomgare": "HRAZEM", "ligne": "LIN02", "pk": 7.00},
    {"nomgare": "KENITRA", "ligne": "LIN01", "pk": 4.00},
    {"nomgare": "KHOURIBGA", "ligne": "LIN09", "pk": 5.00},
    {"nomgare": "KSARKEBIR", "ligne": "LIN03", "pk": 6.00},
    {"nomgare": "KSIRI", "ligne": "LIN03", "pk": 7.00},
    {"nomgare": "L'OASIS", "ligne": "LIN06", "pk": 6.00},
    {"nomgare": "LALLAYTO", "ligne": "LIN01", "pk": 5.00},
    {"nomgare": "LARBAA", "ligne": "LIN03", "pk": 8.00},
    {"nomgare": "LHAMRA", "ligne": "LIN03", "pk": 9.00},
    {"nomgare": "LYAMANI", "ligne": "LIN03", "pk": 10.00},
    {"nomgare": "M'RIZIG", "ligne": "LIN09", "pk": 6.00},
    {"nomgare": "M'SOUN", "ligne": "LIN02", "pk": 8.00},
    {"nomgare": "MAKHAZIN", "ligne": "LIN03", "pk": 11.00},
    {"nomgare": "MARRAKECH", "ligne": "LIN06", "pk": 7.00},
    {"nomgare": "MATMATA", "ligne": "LIN02", "pk": 9.00},
    {"nomgare": "MEKNES", "ligne": "LIN03", "pk": 12.00},
    {"nomgare": "MERSULTAN", "ligne": "LIN06", "pk": 8.00},
    {"nomgare": "MERZOUKA", "ligne": "LIN02", "pk": 10.00},
    {"nomgare": "METLILI", "ligne": "LIN02", "pk": 11.00},
    {"nomgare": "MOHAMMEDIA", "ligne": "LIN01", "pk": 6.00},
    {"nomgare": "NADOR", "ligne": "LIN05", "pk": 2.00},
    {"nomgare": "NADORSUD", "ligne": "LIN05", "pk": 3.00},
    {"nomgare": "NAIMA", "ligne": "LIN02", "pk": 12.00},
    {"nomgare": "NOUASSER", "ligne": "LIN06", "pk": 9.00},
    {"nomgare": "OLADRAHOU", "ligne": "LIN05", "pk": 4.00},
    {"nomgare": "OUEDAMLIL", "ligne": "LIN02", "pk": 13.00},
    {"nomgare": "OUEDZEM", "ligne": "LIN09", "pk": 7.00},
    {"nomgare": "OUIDANE", "ligne": "LIN05", "pk": 5.00},
    {"nomgare": "OUJDA", "ligne": "LIN02", "pk": 14.00},
    {"nomgare": "RABATAGD", "ligne": "LIN01", "pk": 7.00},
    {"nomgare": "RABATVILLE", "ligne": "LIN01", "pk": 8.00},
    {"nomgare": "RASELAIN", "ligne": "LIN09", "pk": 8.00},
    {"nomgare": "RHAZOUANI", "ligne": "LIN09", "pk": 9.00},
    {"nomgare": "RISSANA", "ligne": "LIN03", "pk": 13.00},
    {"nomgare": "SAFI", "ligne": "LIN07", "pk": 4.00},
    {"nomgare": "SALE", "ligne": "LIN01", "pk": 9.00},
    {"nomgare": "SBAAAIOUN", "ligne": "LIN03", "pk": 14.00},
    {"nomgare": "SELOUANE", "ligne": "LIN05", "pk": 6.00},
    {"nomgare": "SETTAT", "ligne": "LIN06", "pk": 10.00},
    {"nomgare": "SIDIDAOUI", "ligne": "LIN09", "pk": 10.00},
    {"nomgare": "SIDIKACEM", "ligne": "LIN03", "pk": 15.00},
    {"nomgare": "SKHIRAT", "ligne": "LIN01", "pk": 10.00},
    {"nomgare": "SLIMANMED", "ligne": "LIN01", "pk": 11.00},
    {"nomgare": "TABRIQUET", "ligne": "LIN01", "pk": 12.00},
    {"nomgare": "TAMDROST", "ligne": "LIN09", "pk": 11.00},
    {"nomgare": "TAOUJDATE", "ligne": "LIN03", "pk": 16.00},
    {"nomgare": "TAOURIRT", "ligne": "LIN02", "pk": 15.00},
    {"nomgare": "TAZA", "ligne": "LIN02", "pk": 16.00},
    {"nomgare": "TEMARA", "ligne": "LIN01", "pk": 13.00},
    {"nomgare": "TOUABAA", "ligne": "LIN02", "pk": 17.00},
    {"nomgare": "VILLE", "ligne": "LIN03", "pk": 17.00},
    {"nomgare": "VOYAGEUR", "ligne": "LIN01", "pk": 14.00},
    {"nomgare": "YAHIA", "ligne": "LIN01", "pk": 15.00},
    {"nomgare": "YOUSSOUFIA", "ligne": "LIN07", "pk": 5.00}
]

# === Données par défaut ===
DEFAULT_PLAN_TRANSPORT = pd.DataFrame({
    'numtrain': ['TR001', 'TR002', 'TR003', 'TR004', 'TR005'],
    'nomgare': ['CASAPORT', 'RABATAGD', 'FES', 'MARRAKECH', 'KENITRA'],
    'arrivee': ['08:00', '09:30', '11:15', '14:20', '16:45'],
    'depart': ['08:05', '09:35', '11:20', '14:25', '16:50'],
    'DateDebut': ['2023-01-01'] * 5,
    'DateFin': ['2023-12-31'] * 5
})

DEFAULT_ESSAIS = pd.DataFrame({
    'type_essai': ['Essai_Vitesse', 'Essai_Freinage', 'Essai_Confort'],
    'duree': [120, 90, 60],
    'j/n': ['j', 'n', 'j'],
    'ligne': ['LIN01', 'LIN02', 'LIN03'],
    'pk': [5.0, 12.0, 8.0],
    '2023-06-15': ['x', '', 'x'],
    '2023-06-16': ['', 'x', ''],
    '2023-06-17': ['x', '', 'x']
})

DEFAULT_INCIDENTS = pd.DataFrame({
    'Date incident': ['2023-06-01', '2023-06-02', '2023-06-03'],
    'GareAvant': ['CASAPORT', 'RABATAGD', 'FES'],
    'GareApres': ['RABATAGD', 'FES', 'MEKNES'],
    'Retard': [15, 25, 10],
    'Cause': ['Problème technique', 'Conditions météo', 'Problème d\'infrastructure']
})

# === Paramètres ===
JOUR_START = time(6, 0)
JOUR_END = time(21, 0)
PAS_MINUTES = 10
FORMAT_HEURE = "%H:%M"
DELTA_RETARD = 120  # Marge de sécurité par défaut

# === Fonctions utilitaires ===
def est_dans_plage(mode, s, e):
    if mode == 'jour':
        return JOUR_START <= s and e <= JOUR_END
    else:
        return (s < JOUR_START or s >= JOUR_END) and (e < JOUR_START or e >= JOUR_END)

def chevauche(s1, e1, s2, e2):
    m1s = s1.hour * 60 + s1.minute
    m1e = e1.hour * 60 + e1.minute
    m2s = s2.hour * 60 + s2.minute
    m2e = e2.hour * 60 + e2.minute
    return not (m1e <= m2s or m1s >= m2e)

def chevauche_occupation(s, e, occupations):
    return any(chevauche(s, e, arr, dep) for (arr, dep) in occupations)

def _safe_parse_time(val):
    """Retourne un datetime.time ou None"""
    if pd.isna(val):
        return None
    if isinstance(val, time):
        return val
    try:
        tt = pd.to_datetime(str(val), errors='coerce')
        if pd.isna(tt):
            return None
        return tt.time()
    except Exception:
        return None

def analyser_retards_par_gare(df_incidents, df_gares):
    """Analyse les retards par segment entre gares avec gestion des erreurs"""
    try:
        df_retards = df_incidents.copy()
        
        # Standardisation des noms de gares
        def standardiser_nom_gare(nom):
            if pd.isna(nom):
                return nom
            nom = str(nom).strip().upper()
            replacements = {
                'AÉROPORT MED V': 'AEROPMDV',
                'AIN SEBAA': 'AINSEBAA',
                'RABAT AGDAL': 'RABATAGD',
                'CASA PORT': 'CASAPORT',
                'MOHAMMEDIA': 'MOHAMMEDIA',
                'KENITRA': 'KENITRA',
                'FES': 'FES',
                'MEKNES': 'MEKNES',
                'MARRAKECH': 'MARRAKECH',
            }
            for old, new in replacements.items():
                nom = nom.replace(old, new)
            return nom

        df_retards['GareAvant'] = df_retards['GareAvant'].apply(standardiser_nom_gare)
        df_retards['GareApres'] = df_retards['GareApres'].apply(standardiser_nom_gare)
        
        # Calcul des statistiques par segment (entre deux gares)
        retards_par_segment = {}
        segments_stats = {}
        
        for _, row in df_retards.iterrows():
            if pd.notna(row['GareAvant']) and pd.notna(row['GareApres']) and row['GareAvant'] != row['GareApres']:
                segment = f"{row['GareAvant']}-{row['GareApres']}"
                if segment not in retards_par_segment:
                    retards_par_segment[segment] = []
                retards_par_segment[segment].append(row['Retard'])
        
        # Calcul des percentiles par segment
        marges_par_segment = {}
        for segment, retards in retards_par_segment.items():
            if retards:
                marges_par_segment[segment] = {
                    'marge_95': int(np.percentile(retards, 95)),
                    'marge_90': int(np.percentile(retards, 90)),
                    'moyenne': int(np.mean(retards)),
                    'ecart_type': int(np.std(retards)),
                    'nb_incidents': len(retards)
                }
        
        # Calcul des marges par gare (basé sur les segments qui incluent la gare)
        marges_par_gare = {}
        for gare in df_gares['nomgare'].unique():
            retards_gare = []
            for segment, stats in marges_par_segment.items():
                if gare in segment:
                    retards_gare.extend([stats['marge_95']] * stats['nb_incidents'])
            
            if retards_gare:
                marges_par_gare[gare] = int(np.percentile(retards_gare, 95))
            else:
                # Valeur par défaut si pas de données
                marges_par_gare[gare] = DELTA_RETARD
        
        return marges_par_gare, marges_par_segment
    except Exception as e:
        st.error(f"Erreur lors de l'analyse des retards: {str(e)}")
        logger.error(f"Erreur analyse retards: {str(e)}")
        return {}, {}

def obtenir_trains_retardes(df_plan, gare, date_essai, marge):
    try:
        date_essai = pd.to_datetime(date_essai)
        trains_gare = df_plan[df_plan['nomgare'] == gare].copy()
        
        trains_gare['arrivee_time'] = trains_gare['arrivee'].apply(_safe_parse_time)
        trains_gare['depart_time'] = trains_gare['depart'].apply(_safe_parse_time)
        
        trains_retardes = []
        for _, train in trains_gare.iterrows():
            if train['arrivee_time'] and train['depart_time']:
                arr_minutes = train['arrivee_time'].hour * 60 + train['arrivee_time'].minute
                dep_minutes = train['depart_time'].hour * 60 + train['depart_time'].minute
                
                arr_elargi = int(max(0, arr_minutes - marge))
                dep_elargi = int(min(23*60 + 59, dep_minutes + marge))
                
                trains_retardes.append({
                    'numtrain': train.get('numtrain', 'N/A'),
                    'arrivee_original': train['arrivee_time'].strftime(FORMAT_HEURE),
                    'depart_original': train['depart_time'].strftime(FORMAT_HEURE),
                    'arrivee_elargi': f"{arr_elargi//60:02d}:{arr_elargi%60:02d}",
                    'depart_elargi': f"{dep_elargi//60:02d}:{dep_elargi%60:02d}",
                    'marge_appliquee': marge
                })
        
        return trains_retardes
    except Exception as e:
        st.error(f"Erreur lors de l'obtention des trains retardés: {str(e)}")
        return []

def generer_creneaux_valides_robuste(df, date_essai, nom_gare, duree_minutes, mode, marges_par_gare):
    try:
        # Marge de base basée sur les retards historiques
        marge_retard = marges_par_gare.get(nom_gare, DELTA_RETARD)
        
        # Marge opérationnelle (5-10% de la durée)
        marge_operationnelle = max(5, duree_minutes * 0.1)
        
        # Marge totale
        marge_totale = int(marge_retard + marge_operationnelle)
        
        date_essai = pd.to_datetime(date_essai)
        duree_td = timedelta(minutes=duree_minutes)

        if 'DateDebut' in df.columns and 'DateFin' in df.columns:
            df = df.copy()
            df['DateDebut'] = pd.to_datetime(df['DateDebut'], errors='coerce')
            df['DateFin'] = pd.to_datetime(df['DateFin'], errors='coerce')
            actifs = df[(df['DateDebut'].dt.date <= date_essai.date()) & (df['DateFin'].dt.date >= date_essai.date())]
        else:
            actifs = df.copy()

        actifs_gare = actifs[actifs.get('nomgare', pd.Series('', index=actifs.index)) == nom_gare].copy()

        if actifs_gare.empty:
            occupations_elargies = []
        else:
            actifs_gare['arrivee_t'] = actifs_gare['arrivee'].apply(_safe_parse_time)
            actifs_gare['depart_t'] = actifs_gare['depart'].apply(_safe_parse_time)

            occupations = list(zip(
                actifs_gare['arrivee_t'].fillna(time(0,0)),
                actifs_gare['depart_t'].fillna(time(0,0)),
                actifs_gare.get('numtrain', pd.Series(index=actifs_gare.index, data=[None]*len(actifs_gare)))
            ))

            occupations_elargies = []
            for arr, dep, num in occupations:
                if arr is None or dep is None:
                    continue
                arr_minutes = arr.hour * 60 + arr.minute
                dep_minutes = dep.hour * 60 + dep.minute
                arr_elargi = int(max(0, arr_minutes - marge_totale))
                dep_elargi = int(min(23*60 + 59, dep_minutes + marge_totale))
                arr_time = time(arr_elargi // 60, arr_elargi % 60)
                dep_time = time(dep_elargi // 60, dep_elargi % 60)
                occupations_elargies.append((arr_time, dep_time, num))

        current = datetime.combine(datetime.today(), time(0, 0))
        end_day = datetime.combine(datetime.today(), time(23, 59))
        step = timedelta(minutes=PAS_MINUTES)

        creneaux = []
        while current + duree_td <= end_day:
            s = current.time()
            e = (current + duree_td).time()
            occs_simple = [(a, d) for (a, d, n) in occupations_elargies]
            if est_dans_plage(mode, s, e) and not chevauche_occupation(s, e, occs_simple):
                creneaux.append((s.strftime(FORMAT_HEURE), e.strftime(FORMAT_HEURE)))
            current += step

        return creneaux, marge_totale
    except Exception as e:
        st.error(f"Erreur lors de la génération des créneaux: {str(e)}")
        logger.error(f"Erreur génération créneaux: {str(e)}")
        return [], DELTA_RETARD

def trouver_gare_proche(df_gares, ligne, pk):
    try:
        gares_ligne = df_gares.loc[df_gares['ligne'] == ligne].copy()
        if gares_ligne.empty:
            return None
        gares_ligne['distance'] = (gares_ligne['pk'] - float(pk)).abs()
        return gares_ligne.loc[gares_ligne['distance'].idxmin(), 'nomgare']
    except Exception as e:
        st.error(f"Erreur lors de la recherche de gare proche: {str(e)}")
        return None

def optimiser_planification(essais_planifies, df_blancs):
    try:
        prob = pulp.LpProblem("PlanificationRobuste", pulp.LpMaximize)
        variables = {}
        
        for i, essai in enumerate(essais_planifies):
            for j, _ in enumerate(essai['creneaux']):
                var_name = f"x_{i}_{j}"
                variables[var_name] = pulp.LpVariable(var_name, cat='Binary')

        if variables:
            prob += pulp.lpSum(list(variables.values()))
        else:
            return []

        for i1, essai1 in enumerate(essais_planifies):
            for j1, c1 in enumerate(essai1['creneaux']):
                s1, e1 = c1
                t1s = datetime.strptime(s1, FORMAT_HEURE).time()
                t1e = datetime.strptime(e1, FORMAT_HEURE).time()
                for i2 in range(i1 + 1, len(essais_planifies)):
                    essai2 = essais_planifies[i2]
                    if essai1['gare'] == essai2['gare'] and essai1['date'] == essai2['date']:
                        for j2, c2 in enumerate(essai2['creneaux']):
                            s2, e2 = c2
                            t2s = datetime.strptime(s2, FORMAT_HEURE).time()
                            t2e = datetime.strptime(e2, FORMAT_HEURE).time()
                            if chevauche(t1s, t1e, t2s, t2e):
                                prob += variables[f"x_{i1}_{j1}"] + variables[f"x_{i2}_{j2}"] <= 1

        prob.solve()
        solution = []
        
        for i, essai in enumerate(essais_planifies):
            for j, creneau in enumerate(essai['creneaux']):
                if pulp.value(variables[f"x_{i}_{j}"]) == 1:
                    solution.append({
                        "essai": essai['essai'],
                        "date": essai['date'],
                        "gare": essai['gare'],
                        "ligne": essai['ligne'],
                        "pk": essai['pk'],
                        "duree": essai['duree'],
                        "mode": essai['mode'],
                        "creneau": creneau,
                        "marge_appliquee": essai.get('marge_appliquee', DELTA_RETARD)
                    })
        return solution
    except Exception as e:
        st.error(f"Erreur lors de l'optimisation: {str(e)}")
        return []

def visualiser_marges_securite(marges_par_gare):
    """Crée une visualisation des marges de sécurité par gare"""
    if not marges_par_gare:
        return None
        
    df_marges = pd.DataFrame({
        'Gare': list(marges_par_gare.keys()),
        'Marge (min)': list(marges_par_gare.values())
    }).sort_values('Marge (min)', ascending=False)
    
    fig = px.bar(df_marges.head(20), x='Gare', y='Marge (min)',
                 title="Top 20 des marges de sécurité par gare (percentile 95 des retards)",
                 color='Marge (min)', color_continuous_scale='Viridis')
    
    fig.update_layout(xaxis_tickangle=-45, height=500)
    return fig

# === Fonctions de visualisation pour chaque page ===
def visualiser_plan_transport(df_plan):
    """Crée des visualisations pour le plan de transport"""
    if df_plan.empty:
        return None
    
    # Statistiques de base
    st.subheader("📊 Statistiques du plan de transport")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nombre de trains", len(df_plan['numtrain'].unique()))
    col2.metric("Nombre de gares", len(df_plan['nomgare'].unique()))
    
    # Distribution des gares les plus fréquentées
    st.subheader("🏭 Gares les plus fréquentées")
    gares_count = df_plan['nomgare'].value_counts().head(10)
    fig_gares = px.bar(x=gares_count.index, y=gares_count.values,
                       title="Top 10 des gares par nombre de trains",
                       labels={'x': 'Gare', 'y': 'Nombre de trains'})
    st.plotly_chart(fig_gares, use_container_width=True)
    
    # Distribution des horaires
    st.subheader("🕐 Distribution des horaires d'arrivée")
    try:
        df_plan_copy = df_plan.copy()
        df_plan_copy['arrivee_time'] = df_plan_copy['arrivee'].apply(_safe_parse_time)
        df_plan_copy = df_plan_copy.dropna(subset=['arrivee_time'])
        df_plan_copy['heure_arrivee'] = df_plan_copy['arrivee_time'].apply(lambda x: x.hour + x.minute/60)
        
        fig_horaire = px.histogram(df_plan_copy, x='heure_arrivee', nbins=24,
                                  title="Distribution des heures d'arrivée",
                                  labels={'heure_arrivee': 'Heure de la journée'})
        fig_horaire.update_layout(xaxis=dict(tickmode='linear', dtick=1))
        st.plotly_chart(fig_horaire, use_container_width=True)
    except Exception as e:
        st.warning(f"Impossible de créer le graphique des horaires: {e}")

def visualiser_essais(df_essais):
    """Crée des visualisations pour les essais"""
    if df_essais.empty:
        return None
    
    # Statistiques de base
    st.subheader("📊 Statistiques des essais")
    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre d'essais", len(df_essais))
    col2.metric("Types d'essais", len(df_essais['type_essai'].unique()))
    col3.metric("Lignes concernées", len(df_essais['ligne'].unique()))
    
    # Répartition par type d'essai
    st.subheader("🔬 Répartition par type d'essai")
    type_count = df_essais['type_essai'].value_counts()
    fig_type = px.pie(values=type_count.values, names=type_count.index,
                      title="Répartition des essais par type")
    st.plotly_chart(fig_type, use_container_width=True)
    
    # Répartition par ligne
    st.subheader("🛤️ Répartition par ligne")
    ligne_count = df_essais['ligne'].value_counts()
    fig_ligne = px.bar(x=ligne_count.index, y=ligne_count.values,
                       title="Nombre d'essais par ligne",
                       labels={'x': 'Ligne', 'y': "Nombre d'essais"})
    st.plotly_chart(fig_ligne, use_container_width=True)
    
    # Distribution des durées
    st.subheader("⏱️ Distribution des durées d'essais")
    fig_duree = px.histogram(df_essais, x='duree', nbins=10,
                            title="Distribution des durées d'essais (minutes)",
                            labels={'duree': 'Durée (minutes)'})
    st.plotly_chart(fig_duree, use_container_width=True)

def visualiser_incidents(df_incidents):
    """Crée des visualisations pour les incidents"""
    if df_incidents.empty:
        st.warning("Aucune donnée d'incidents à afficher")
        return None
    
    # Vérifier si la colonne Date incident existe
    date_column = None
    for col in df_incidents.columns:
        if 'date' in col.lower() and 'incident' in col.lower():
            date_column = col
            break
    
    if date_column is None:
        # Si on ne trouve pas exactement "date incident", chercher une colonne avec "date"
        for col in df_incidents.columns:
            if 'date' in col.lower():
                date_column = col
                break
    
    if date_column is None:
        st.warning("Colonne de date non trouvée dans les données d'incidents")
        return None
    
    # Statistiques de base
    st.subheader("📊 Statistiques des incidents")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nombre d'incidents", len(df_incidents))
    
    if 'Retard' in df_incidents.columns:
        col2.metric("Retard moyen", f"{df_incidents['Retard'].mean():.1f} min")
        col3.metric("Retard max", f"{df_incidents['Retard'].max()} min")
    else:
        col2.metric("Retard moyen", "N/A")
        col3.metric("Retard max", "N/A")
    
    col4.metric("Jours avec incidents", len(df_incidents[date_column].unique()))
    
    # Évolution des retards dans le temps
    if 'Retard' in df_incidents.columns:
        st.subheader("📈 Évolution des retards dans le temps")
        try:
            df_incidents_copy = df_incidents.copy()
            df_incidents_copy[date_column] = pd.to_datetime(df_incidents_copy[date_column])
            df_incidents_copy = df_incidents_copy.sort_values(date_column)
            
            fig_retards = px.line(df_incidents_copy, x=date_column, y='Retard',
                                  title="Évolution des retards dans le tiempo",
                                  markers=True)
            st.plotly_chart(fig_retards, use_container_width=True)
        except Exception as e:
            st.warning(f"Impossible de créer le graphique d'évolution: {e}")
    
    # Répartition des causes d'incidents
    if 'Cause' in df_incidents.columns:
        st.subheader("🔍 Répartition des causes d'incidents")
        cause_count = df_incidents['Cause'].value_counts()
        fig_cause = px.pie(values=cause_count.values, names=cause_count.index,
                          title="Répartition des incidents par cause")
        st.plotly_chart(fig_cause, use_container_width=True)
    
    # Top des gares à problèmes
    st.subheader("⚠️ Gares les plus concernées par les incidents")
    gares_incidents = []
    if 'GareAvant' in df_incidents.columns:
        gares_incidents.extend(df_incidents['GareAvant'].dropna().tolist())
    if 'GareApres' in df_incidents.columns:
        gares_incidents.extend(df_incidents['GareApres'].dropna().tolist())
    
    if gares_incidents:
        gares_count = pd.Series(gares_incidents).value_counts().head(10)
        fig_gares = px.bar(x=gares_count.index, y=gares_count.values,
                           title="Top 10 des gares concernées par les incidents",
                           labels={'x': 'Gare', 'y': "Nombre d'incidents"})
        st.plotly_chart(fig_gares, use_container_width=True)
    else:
        st.info("Aucune information sur les gares disponibles")

# === Interface Streamlit ===
st.set_page_config(
    page_title="ONCF - Planification des Essais",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS
st.markdown("""
    <style>
    .header { font-size: 36px !important; font-weight: bold !important; color: #0d5fa6 !important;
              border-bottom: 2px solid #0d5fa6; padding-bottom: 10px; margin-bottom: 30px; }
    .subheader { font-size: 24px !important; color: #0d5fa6 !important; margin-top: 25px !important; }
    .stButton>button { background-color: #0d5fa6 !important; color: white !important; font-weight: bold !important;
                       border-radius: 5px; padding: 10px 25px; }
    .info-box { background-color: #e6f0ff; border-radius: 10px; padding: 20px; margin-bottom: 25px; border-left: 4px solid #0d5fa6; }
    .success-box { background-color: #e6ffe6; border-radius: 10px; padding: 15px; margin: 15px 0; border-left: 4px solid #28a745; }
    .warning-box { background-color: #fffae6; border-radius: 10px; padding: 15px; margin: 15px 0; border-left: 4px solid #ffc107; }
    .gare-card { background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin: 15px 0;
                 box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Planification des Essais", "Plan de Transport", "Essais", "Incidents"])

# Initialiser les DataFrames dans session_state
if 'df_plan' not in st.session_state:
    st.session_state.df_plan = DEFAULT_PLAN_TRANSPORT.copy()

if 'df_demande' not in st.session_state:
    st.session_state.df_demande = DEFAULT_ESSAIS.copy()

if 'df_incidents' not in st.session_state:
    st.session_state.df_incidents = DEFAULT_INCIDENTS.copy()

# DataFrame des gares
df_gares = pd.DataFrame(REF_GASTES)

# Pages
if page == "Plan de Transport":
    st.markdown("<div class='header'>📋 Plan de Transport</div>", unsafe_allow_html=True)
    
    st.dataframe(st.session_state.df_plan, use_container_width=True)
    
    # Afficher les visualisations pour le plan de transport
    visualiser_plan_transport(st.session_state.df_plan)
    
    with st.expander("Ajouter une nouvelle entrée au plan de transport"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            new_numtrain = st.text_input("Numéro de train")
        with col2:
            new_nomgare = st.selectbox("Nom de gare", options=[g["nomgare"] for g in REF_GASTES])
        with col3:
            new_arrivee = st.text_input("Heure d'arrivée (HH:MM)")
        with col4:
            new_depart = st.text_input("Heure de départ (HH:MM)")
        
        col5, col6 = st.columns(2)
        with col5:
            new_date_debut = st.date_input("Date de début")
        with col6:
            new_date_fin = st.date_input("Date de fin")
        
        if st.button("Ajouter au plan de transport"):
            new_row = {
                'numtrain': new_numtrain,
                'nomgare': new_nomgare,
                'arrivee': new_arrivee,
                'depart': new_depart,
                'DateDebut': new_date_debut.strftime("%Y-%m-%d"),
                'DateFin': new_date_fin.strftime("%Y-%m-%d")
            }
            st.session_state.df_plan = pd.concat([st.session_state.df_plan, pd.DataFrame([new_row])], ignore_index=True)
            st.success("Entrée ajoutée avec succès!")
            st.rerun()

    uploaded_file = st.file_uploader("Importer un fichier de plan de transport", type=["xlsx", "csv"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                new_df = pd.read_excel(uploaded_file)
            else:
                new_df = pd.read_csv(uploaded_file)
            st.session_state.df_plan = new_df
            st.success("Plan de transport mis à jour!")
            st.rerun()
        except Exception as e:
            st.error(f"Erreur lors de l'importation: {str(e)}")

elif page == "Essais":
    st.markdown("<div class='header'>🧪 Essais à Planifier</div>", unsafe_allow_html=True)
    
    st.dataframe(st.session_state.df_demande, use_container_width=True)
    
    # Afficher les visualisations pour les essais
    visualiser_essais(st.session_state.df_demande)
    
    with st.expander("Ajouter un nouvel essai"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            new_type_essai = st.text_input("Type d'essai")
        with col2:
            new_duree = st.number_input("Durée (min)", min_value=1, value=60)
        with col3:
            new_jn = st.selectbox("Jour/Nuit", options=["j", "n"])
        with col4:
            new_ligne = st.selectbox("Ligne", options=["LIN01", "LIN02", "LIN03", "LIN04", "LIN05", "LIN06", "LIN07", "LIN08", "LIN09"])
        
        new_pk = st.number_input("PK", min_value=0.0, value=0.0, step=0.1)
        
        date_cols = [col for col in st.session_state.df_demande.columns if col not in ['type_essai', 'duree', 'j/n', 'ligne', 'pk']]
        
        st.subheader("Sélectionner les dates existantes")
        selected_existing_dates = st.multiselect("Dates disponibles dans le fichier", options=date_cols)
        
        st.subheader("Ajouter de nouvelles dates")
        new_dates_input = st.text_input(
            "Nouvelles dates (format: YYYY-MM-DD, séparées par des virgules)",
            help="Exemple: 2023-07-15, 2023-07-16, 2023-07-17"
        )
        
        if st.button("Ajouter l'essai"):
            if not new_type_essai:
                st.error("Veuillez saisir un type d'essai")
            else:
                new_row = {
                    'type_essai': new_type_essai,
                    'duree': new_duree,
                    'j/n': new_jn,
                    'ligne': new_ligne,
                    'pk': new_pk
                }
                
                for date in date_cols:
                    new_row[date] = 'x' if date in selected_existing_dates else ''
                
                new_dates = []
                if new_dates_input:
                    for date_str in new_dates_input.split(','):
                        date_str = date_str.strip()
                        try:
                            pd.to_datetime(date_str)
                            new_dates.append(date_str)
                        except:
                            st.warning(f"Date ignorée (format invalide): {date_str}")
                
                for new_date in new_dates:
                    if new_date not in st.session_state.df_demande.columns:
                        st.session_state.df_demande[new_date] = ''
                    new_row[new_date] = 'x'
                
                for col in st.session_state.df_demande.columns:
                    if col not in new_row:
                        new_row[col] = ''
                
                st.session_state.df_demande = pd.concat([st.session_state.df_demande, pd.DataFrame([new_row])], ignore_index=True)
                st.success("Essai ajouté avec succès!")
                st.rerun()

    uploaded_file = st.file_uploader("Importer un fichier d'essais", type=["xlsx", "csv"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                new_df = pd.read_excel(uploaded_file)
            else:
                new_df = pd.read_csv(uploaded_file)
            st.session_state.df_demande = new_df
            st.success("Essais mis à jour!")
            st.rerun()
        except Exception as e:
            st.error(f"Erreur lors de l'importation: {str(e)}")

elif page == "Incidents":
    st.markdown("<div class='header'>⚠️ Incidents et Retards</div>", unsafe_allow_html=True)
    
    st.dataframe(st.session_state.df_incidents, use_container_width=True)
    
    # Afficher les visualisations pour les incidents
    visualiser_incidents(st.session_state.df_incidents)
    
    with st.expander("Ajouter un nouvel incident"):
        col1, col2, col3 = st.columns(3)
        with col1:
            new_date = st.date_input("Date de l'incident")
        with col2:
            new_gare_avant = st.selectbox("Gare avant", options=[g["nomgare"] for g in REF_GASTES])
        with col3:
            new_gare_apres = st.selectbox("Gare après", options=[g["nomgare"] for g in REF_GASTES])
        
        col4, col5 = st.columns(2)
        with col4:
            new_retard = st.number_input("Retard (min)", min_value=0, value=15)
        with col5:
            new_cause = st.text_input("Cause de l'incident")
        
        if st.button("Ajouter l'incident"):
            new_row = {
                'Date incident': new_date.strftime("%Y-%m-%d"),
                'GareAvant': new_gare_avant,
                'GareApres': new_gare_apres,
                'Retard': new_retard,
                'Cause': new_cause
            }
            st.session_state.df_incidents = pd.concat([st.session_state.df_incidents, pd.DataFrame([new_row])], ignore_index=True)
            st.success("Incident ajouté avec succès!")
            st.rerun()

    uploaded_file = st.file_uploader("Importer un fichier d'incidents", type=["xlsx", "csv"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                new_df = pd.read_excel(uploaded_file)
            else:
                new_df = pd.read_csv(uploaded_file)
            st.session_state.df_incidents = new_df
            st.success("Incidents mis à jour!")
            st.rerun()
        except Exception as e:
            st.error(f"Erreur lors de l'importation: {str(e)}")

else:
    # Page principale: Planification des Essais
    col1, col2 = st.columns([1, 5])
    with col2:
        st.markdown("<div class='header'>Planification Automatique des Essais Ferroviaires</div>", unsafe_allow_html=True)

    st.markdown("""
        <div class='info-box'>
        <h3>📋 Guide d'utilisation</h3>
        <ol>
            <li>Consultez et modifiez les données dans les onglets dédiés</li>
            <li>Les données de plan de transport, essais et incidents sont préchargées</li>
            <li>Vous pouvez ajouter de nouvelles entrées dans chaque section</li>
            <li>Paramétrez les options de planification ci-dessous</li>
            <li>Lancez la planification et consultez les résultats</li>
        </ol>
        </div>
    """, unsafe_allow_html=True)

    # Paramètres de robustesse
    st.markdown("<div class='subheader'>⚙️ Paramètres de planification</div>", unsafe_allow_html=True)
    use_optimization = st.checkbox(
        "Utiliser l'optimisation avancée",
        value=True,
        help="Active l'optimisation pour éviter les conflits de planning"
    )

    # Calcul des marges de sécurité basées sur les incidents
    marges_par_gare, marges_par_segment = analyser_retards_par_gare(st.session_state.df_incidents, df_gares)

    # Afficher les marges calculées
    st.subheader("Marges de sécurité calculées")
    if marges_par_gare:
        fig = visualiser_marges_securite(marges_par_gare)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Détail des marges par gare:")
            for gare, marge in list(marges_par_gare.items())[:10]:
                st.text(f"{gare}: {marge} min")
            if len(marges_par_gare) > 10:
                st.text(f"... et {len(marges_par_gare) - 10} autres gares")
        with col2:
            st.metric("Marge moyenne", f"{np.mean(list(marges_par_gare.values())):.1f} min")
            st.metric("Marge maximale", f"{np.max(list(marges_par_gare.values()))} min")
    else:
        st.text(f"Utilisation des marges par défaut ({DELTA_RETARD} min)")

    # Bouton pour lancer la planification
    if st.button("🚀 Lancer la planification des essais", type="primary"):
        essais_planifies = []
        essais_non_trouves = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        total_essais = len(st.session_state.df_demande)
        for idx, (_, row) in enumerate(st.session_state.df_demande.iterrows()):
            type_essai = row['type_essai']
            status_text.text(f"Traitement: {type_essai} ({idx+1}/{total_essais})")
            progress_bar.progress((idx+1)/total_essais)

            ligne_essai = row['ligne']
            pk_essai = row['pk']

            gare_proche = trouver_gare_proche(df_gares, ligne_essai, pk_essai)
            if not gare_proche:
                essais_non_trouves.append(str(type_essai))
                continue

            try:
                duree = int(row['duree'])
            except Exception:
                st.warning(f"⛔ Durée invalide pour l'essai '{type_essai}'")
                continue

            mode_raw = str(row['j/n']).strip().lower()
            if mode_raw == 'j':
                mode = 'jour'
            elif mode_raw == 'n':
                mode = 'nuit'
            else:
                st.warning(f"⛔ Mode invalide (j/n) pour l'essai '{type_essai}'")
                continue

            colonnes_dates = [col for col in st.session_state.df_demande.columns if col not in ['type_essai', 'j/n', 'duree', 'ligne', 'pk']]

            for col in colonnes_dates:
                val = str(row[col]).strip().lower()
                if val in {"1", "x", "true", "oui", "o"}:
                    try:
                        date = pd.to_datetime(col).date()
                    except Exception:
                        continue

                    creneaux, marge_appliquee = generer_creneaux_valides_robuste(
                        st.session_state.df_plan, date, gare_proche, duree, mode, marges_par_gare
                    )
                    if creneaux:
                        essais_planifies.append({
                            "essai": type_essai,
                            "date": date,
                            "gare": gare_proche,
                            "ligne": ligne_essai,
                            "pk": pk_essai,
                            "duree": duree,
                            "mode": mode,
                            "creneaux": creneaux,
                            "marge_appliquee": marge_appliquee
                        })

        progress_bar.empty()
        status_text.empty()

        # Optimisation
        if use_optimization and essais_planifies:
            st.info("🔧 Application de l'optimisation robuste...")
            essais_a_afficher = optimiser_planification(essais_planifies, st.session_state.df_plan)
            st.success(f"✅ Optimisation terminée: {len(essais_a_afficher)} créneau(x) sélectionné(s)")
        else:
            essais_a_afficher = []
            for item in essais_planifies:
                for creneau in item['creneaux']:
                    essais_a_afficher.append({
                        "essai": item['essai'],
                        "date": item['date'],
                        "gare": item['gare'],
                        "ligne": item['ligne'],
                        'pk': item['pk'],
                        "duree": item['duree'],
                        "mode": item['mode'],
                        "creneau": creneau,
                        "marge_appliquee": item.get('marge_appliquee', DELTA_RETARD)
                    })

        # Construction du DataFrame d'affichage
        if essais_a_afficher:
            lignes = []
            for item in essais_a_afficher:
                debut, fin = item["creneau"]
                lignes.append({
                    "Essai": item["essai"],
                    "Date": item["date"],
                    "Gare": item["gare"],
                    "Ligne": item["ligne"],
                    "PK": item["pk"],
                    "Durée (min)": item["duree"],
                    "Période": item["mode"],
                    "Créneau Début": debut,
                    "Créneau Fin": fin,
                    "Marge appliquée (min)": item["marge_appliquee"]
                })
            df_affichage = pd.DataFrame(lignes)
        else:
            df_affichage = pd.DataFrame(columns=[
                "Essai", "Date", "Gare", "Ligne", "PK", "Durée (min)", "Période", "Créneau Début", "Créneau Fin", "Marge appliquée (min)"
            ])

        if not df_affichage.empty:
            # Filtres
            f1, f2, f3 = st.columns(3)
            with f1:
                essais_uniques = ["Tous"] + sorted(df_affichage['Essai'].astype(str).unique().tolist())
                essai_choisi = st.selectbox("Filtrer par type d'essai", essais_uniques, index=0)
            with f2:
                dates_uniques = ["Toutes"] + sorted(df_affichage['Date'].astype(str).unique().tolist())
                date_choisie = st.selectbox("Filtrer par date", dates_uniques, index=0)
            with f3:
                periodes = ["Toutes"] + sorted(df_affichage['Période'].astype(str).unique().tolist())
                periode_choisie = st.selectbox("Filtrer par période", periodes, index=0)

            # Application des filtres
            df_filtre = df_affichage.copy()
            if essai_choisi != "Tous":
                df_filtre = df_filtre[df_filtre["Essai"].astype(str) == essai_choisi]
            if date_choisie != "Toutes":
                df_filtre = df_filtre[df_filtre["Date"].astype(str) == date_choisie]
            if periode_choisie != "Toutes":
                df_filtre = df_filtre[df_filtre["Période"].ast(str) == periode_choisie]

            st.dataframe(df_filtre, use_container_width=True, height=420)

            # Export
            st.subheader("Export des résultats")
            e1, e2 = st.columns(2)
            with e1:
                buffer_xlsx = io.BytesIO()
                with pd.ExcelWriter(buffer_xlsx, engine='xlsxwriter') as writer:
                    df_filtre.to_excel(writer, index=False, sheet_name='Creneaux')
                st.download_button(
                    label="📊 Télécharger en Excel",
                    data=buffer_xlsx.getvalue(),
                    file_name="creneaux_planifies.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with e2:
                csv = df_filtre.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📝 Télécharger en CSV",
                    data=csv,
                    file_name="creneaux_planifies.csv",
                    mime="text/csv"
                )

            # Statistiques
            st.subheader("Statistiques de planification")
            s1, s2, s3, s4 = st.columns(4)
            total_creneaux = len(df_filtre)
            creneaux_jour = len(df_filtre[df_filtre['Période'] == 'jour'])
            creneaux_nuit = total_creneaux - creneaux_jour
            duree_moyenne = float(df_filtre['Durée (min)'].mean()) if total_creneaux > 0 else 0.0
            marge_moyenne = float(df_filtre['Marge appliquée (min)'].mean()) if total_creneaux > 0 else 0.0

            s1.metric("Total créneaux", total_creneaux)
            s2.metric("Créneaux jour", creneaux_jour)
            s3.metric("Créneaux nuit", creneaux_nuit)
            s4.metric("Marge moyenne", f"{marge_moyenne:.1f} min")
            
            # Graphique supplémentaire
            st.subheader("Répartition par gare")
            gares_counts = df_filtre['Gare'].value_counts().reset_index()
            gares_counts.columns = ['Gare', 'Nombre de créneaux']
            fig_gares = px.bar(gares_counts.head(10), x='Gare', y='Nombre de créneaux', 
                               title="Top 10 des gares avec le plus de créneaux")
            st.plotly_chart(fig_gares, use_container_width=True)
        else:
            st.markdown("<div class='warning-box'>⚠️ Aucun créneau valide trouvé pour les essais demandés</div>", unsafe_allow_html=True)

        if essais_non_trouves:
            st.warning(f"⚠️ Gare non trouvée pour {len(essais_non_trouves)} essai(s): {', '.join(essais_non_trouves)}")
