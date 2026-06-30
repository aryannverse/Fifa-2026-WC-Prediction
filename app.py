import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson


st.set_page_config(
    page_title="FIFA World Cup 2026 Analysis & Simulator",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    /* Dark theme customizations */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    

    
    /* Champion Card Banner styling */
    .champ-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #fbbf24; /* gold border */
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 25px -5px rgba(251, 191, 36, 0.1);
    }
    .champ-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(251,191,36,0.05) 0%, transparent 70%);
        pointer-events: none;
    }
    .champ-title {
        color: #fbbf24;
        font-size: 0.95rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
    }
    .champ-name {
        font-size: 3rem;
        font-weight: 900;
        color: #ffffff;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
    }
    .champ-prob {
        font-size: 1.5rem;
        color: #fbbf24;
        font-weight: 700;
    }
    
    /* Simulator card wrapper */
    .sim-card {
        background-color: #1e293b;
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid #334155;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Result card styling */
    .result-box {
        background: linear-gradient(135deg, #1e293b 0%, #1e1b4b 100%);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid #4f46e5;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.2);
    }
    .score-title {
        color: #a5b4fc;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .score-display {
        font-size: 4rem;
        font-weight: 900;
        color: #ffffff;
        letter-spacing: 0.1em;
        margin: 0.5rem 0;
        text-shadow: 0 0 12px rgba(165,180,252,0.4);
    }
    .xg-display {
        font-size: 1rem;
        color: #94a3b8;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)


FLAG_MAP = {
    "Algeria": "dz", "Argentina": "ar", "Australia": "au", "Austria": "at", "Belgium": "be",
    "Bosnia & Herzegovina": "ba", "Brazil": "br", "Cabo Verde": "cv", "Canada": "ca", "Colombia": "co",
    "Croatia": "hr", "Curaçao": "cw", "Czech Republic": "cz", "DR Congo": "cd", "Ecuador": "ec",
    "Egypt": "eg", "England": "gb-eng", "France": "fr", "Germany": "de", "Ghana": "gh",
    "Haiti": "ht", "Iran": "ir", "Iraq": "iq", "Ivory Coast": "ci", "Japan": "jp",
    "Jordan": "jo", "Mexico": "mx", "Morocco": "ma", "Netherlands": "nl", "New Zealand": "nz",
    "Norway": "no", "Panama": "pa", "Paraguay": "py", "Portugal": "pt", "Qatar": "qa",
    "Saudi Arabia": "sa", "Scotland": "gb-sct", "Senegal": "sn", "South Africa": "za",
    "South Korea": "kr", "Spain": "es", "Sweden": "se", "Switzerland": "ch", "Türkiye": "tr",
    "United States": "us", "Uruguay": "uy", "Uzbekistan": "uz", "Tunisia": "tn"
}


@st.cache_resource
def load_data():
    try:
        with open("models/dashboard_data.pkl", "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        import subprocess
        import os
        st.info("Pickle mismatch or model missing. Running data pipeline to regenerate model files...")
        os.makedirs("models", exist_ok=True)
        result = subprocess.run(["python", "prepare_data.py"], capture_output=True, text=True)
        if result.returncode != 0:
            result = subprocess.run(["python3", "prepare_data.py"], capture_output=True, text=True)
        with open("models/dashboard_data.pkl", "rb") as f:
            data = pickle.load(f)
        return data

try:
    data = load_data()
except Exception as e:
    st.error(f"Failed to load pickled model data: {e}. Please check project files.")
    st.stop()


clf = data['models']['match_classifier']
poi = data['models']['poisson_regressor']
final_elo_ratings = data['elo']['final_elo_ratings']
elo_history_df = data['elo']['elo_history']
sorted_teams = data['elo']['sorted_teams']
groups_draw = data['tournament']['groups']
win_probabilities = data['tournament']['win_probabilities']
avg_group_points = data['tournament'].get('avg_group_points', {})
goal_trend = data['eda']['goal_trend']
venue_stats = data['eda']['venue_stats']
NAME_MAP_TO_DATASET = data['mapping']['NAME_MAP_TO_DATASET']
teams_2026 = data['mapping']['teams_2026']


def get_team_current_stats_app(team_name):
    t_ds = NAME_MAP_TO_DATASET.get(team_name, team_name)
    
    
    
    
    
    
    
    
    
    
    
    pass



def get_elo(t):
    t_ds = NAME_MAP_TO_DATASET.get(t, t)
    return final_elo_ratings.get(t_ds, 1500.0)


st.sidebar.image("Image/Frame 1.png", use_container_width=True)






tab1, tab2 = st.tabs(["🏆 Tournament Predictions & Simulator", "📊 Historical Insights & EDA"])

with tab1:
    
    
    
    st.markdown("## 🔮 World Cup Tournament Predictions")
    
    
    top_predicted_team = win_probabilities[0][0]
    top_predicted_prob = win_probabilities[0][1]
    top_team_flag = FLAG_MAP.get(top_predicted_team, "un")
    
    col_banner_left, col_banner_center, col_banner_right = st.columns([1, 4, 1])
    with col_banner_center:
        st.markdown(
            f"""
            <div class="champ-card">
                <div class="champ-title">Predicted World Cup Champion</div>
                <div style="display: flex; justify-content: center; align-items: center; gap: 1.5rem; margin-bottom: 0.5rem;">
                    <img src="https://flagcdn.com/w160/{top_team_flag}.png" style="border: 2px solid #ffffff; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); height: 50px;">
                    <div class="champ-name">{top_predicted_team}</div>
                </div>
                <div class="champ-prob">Probability of Winning: {top_predicted_prob:.1f}%</div>
                <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 0.5rem;">
                    Elo Rating: <strong>{get_elo(top_predicted_team):.1f}</strong> | Based on 10,000 simulated tournament runs
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        
    
    col_lead, col_groups = st.columns([1.1, 0.9])
    
    with col_lead:
        st.markdown("### 📈 Top 10 Teams Win Probability")
        
        top_10 = win_probabilities[:10]
        top_10_df = pd.DataFrame(top_10, columns=['Team', 'Probability'])
        top_10_df = top_10_df.sort_values('Probability', ascending=True)
        
        
        fig_lead = px.bar(
            top_10_df,
            x='Probability',
            y='Team',
            orientation='h',
            text='Probability',
            color='Probability',
            color_continuous_scale='tealrose',
            labels={'Probability': 'Win Chance (%)', 'Team': ''}
        )
        fig_lead.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside',
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5,
            opacity=0.9
        )
        fig_lead.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc',
            xaxis=dict(showgrid=True, gridcolor='#334155', range=[0, top_predicted_prob * 1.25]),
            yaxis=dict(showgrid=False),
            showlegend=False,
            height=400,
            margin=dict(l=0, r=40, t=10, b=10),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_lead, use_container_width=True)
        
    with col_groups:
        st.markdown("### 📋 2026 World Cup Group Stage Draw")
        
        selected_grp_letter = st.selectbox("Select Group to View Teams", list(groups_draw.keys()))
        
        group_teams = sorted(
            groups_draw[selected_grp_letter],
            key=lambda t: avg_group_points.get(t, 0.0),
            reverse=True
        )
        
        
        for i, team in enumerate(group_teams):
            t_flag = FLAG_MAP.get(team, "un")
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; justify-content: space-between; background-color: #1e293b; padding: 0.75rem 1.2rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid #3b82f6;">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <span style="font-weight: 800; color: #fbbf24;">#{i+1}</span>
                        <img src="https://flagcdn.com/w40/{t_flag}.png" style="border-radius: 2px; height: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.3);">
                        <strong style="font-size: 1rem; color: #ffffff;">{team}</strong>
                    </div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">
                        Elo: <strong style="color: #60a5fa;">{get_elo(team):.1f}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
            
    st.markdown("---")
    
    
    
    
    st.markdown("## ⚔️ Head-to-Head Match Simulator")
    
    
    with st.container(border=True):
        col_sel1, col_vs, col_sel2 = st.columns([2.2, 0.6, 2.2])
        
        with col_sel1:
            team1 = st.selectbox("Select Team 1", sorted(teams_2026), index=sorted(teams_2026).index("Brazil"))
            t1_flag = FLAG_MAP.get(team1, "un")
            
            st.markdown(
                f"""
                <div style="text-align: center; margin-top: 1rem;">
                    <img src="https://flagcdn.com/w320/{t1_flag}.png" style="border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4); max-width: 180px; height: auto; border: 3px solid #334155;">
                    <h4 style="margin-top: 1rem; font-weight: 700; color: #ffffff;">Elo: {get_elo(team1):.1f}</h4>
                </div>
                """, unsafe_allow_html=True
            )
            
        with col_vs:
            st.markdown(
                """
                <div style="text-align: center; margin-top: 3.5rem;">
                    <h1 style="color: #fbbf24; font-weight: 900; font-style: italic; margin-bottom: 0;">VS</h1>
                    <p style="color: #64748b; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Neutral Venue</p>
                </div>
                """, unsafe_allow_html=True
            )
            
        with col_sel2:
            team2 = st.selectbox("Select Team 2", sorted(teams_2026), index=sorted(teams_2026).index("Argentina"))
            t2_flag = FLAG_MAP.get(team2, "un")
            st.markdown(
                f"""
                <div style="text-align: center; margin-top: 1rem;">
                    <img src="https://flagcdn.com/w320/{t2_flag}.png" style="border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4); max-width: 180px; height: auto; border: 3px solid #334155;">
                    <h4 style="margin-top: 1rem; font-weight: 700; color: #ffffff;">Elo: {get_elo(team2):.1f}</h4>
                </div>
                """, unsafe_allow_html=True
            )
            
        st.markdown('<div style="text-align: center; margin-top: 2rem;">', unsafe_allow_html=True)
        sim_trigger = st.button("🏆 Simulate Match (10,000 Runs)", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
    
    
    if sim_trigger:
        if team1 == team2:
            st.warning("Please select two different teams to simulate a matchup!")
        else:
            with st.spinner("Running 10,000 Monte Carlo and Poisson simulations..."):
                
                
                
                
                
                
                
                
                
                
                elo_t1 = get_elo(team1)
                elo_t2 = get_elo(team2)
                elo_diff = elo_t1 - elo_t2
                
                
                t1_ds = NAME_MAP_TO_DATASET.get(team1, team1)
                t2_ds = NAME_MAP_TO_DATASET.get(team2, team2)
                pair = tuple(sorted([t1_ds, t2_ds]))
                
                
                
                
                
                
                
                
                form_pts_t1 = np.clip(0.5 + 0.15 * (elo_t1 - 1500) / 400.0, 0.2, 0.85)
                form_pts_t2 = np.clip(0.5 + 0.15 * (elo_t2 - 1500) / 400.0, 0.2, 0.85)
                
                
                scored_t1 = np.clip(1.5 + 0.5 * (elo_t1 - 1500) / 400.0, 0.5, 3.0)
                conceded_t1 = np.clip(1.5 - 0.4 * (elo_t1 - 1500) / 400.0, 0.3, 2.5)
                scored_t2 = np.clip(1.5 + 0.5 * (elo_t2 - 1500) / 400.0, 0.5, 3.0)
                conceded_t2 = np.clip(1.5 - 0.4 * (elo_t2 - 1500) / 400.0, 0.3, 2.5)
                
                
                
                h2h_win_t1 = 0.5
                h2h_win_t2 = 0.5
                
                
                
                
                
                feat_clf = [[elo_t1, elo_t2, elo_diff, form_pts_t1, form_pts_t2, h2h_win_t1, h2h_win_t2, 4, 0]]
                probs_clf = clf.predict_proba(feat_clf)[0]
                
                p_loss = probs_clf[0]
                p_draw = probs_clf[1]
                p_win = probs_clf[2]
                
                
                sim_outcomes = np.random.choice(
                    [team1, "Draw", team2],
                    size=10000,
                    p=[p_win, p_draw, p_loss]
                )
                
                t1_win_pct = (sim_outcomes == team1).sum() / 100.0
                draw_pct = (sim_outcomes == "Draw").sum() / 100.0
                t2_win_pct = (sim_outcomes == team2).sum() / 100.0
                
                
                
                feat_poi_t1 = [[elo_t1 / 400.0, elo_t2 / 400.0, elo_diff / 400.0, scored_t1, conceded_t2, 0]]
                feat_poi_t2 = [[elo_t2 / 400.0, elo_t1 / 400.0, -elo_diff / 400.0, scored_t2, conceded_t1, 0]]
                
                lambda_t1 = max(0.01, poi.predict(feat_poi_t1)[0])
                lambda_t2 = max(0.01, poi.predict(feat_poi_t2)[0])
                
                
                max_p = 0.0
                best_score = (0, 0)
                for x in range(6):
                    for y in range(6):
                        p_score = poisson.pmf(x, lambda_t1) * poisson.pmf(y, lambda_t2)
                        if p_score > max_p:
                            max_p = p_score
                            best_score = (x, y)
                
                
                st.markdown("### 📊 Simulation Results")
                col_res1, col_res2 = st.columns([1.1, 0.9])
                
                with col_res1:
                    st.markdown("#### 🍩 Monte Carlo Win/Draw/Loss Probabilities")
                    
                    labels = [team1, "Draw", team2]
                    values = [t1_win_pct, draw_pct, t2_win_pct]
                    colors = ['#3b82f6', '#475569', '#ef4444'] 
                    
                    fig_donut = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=values,
                        hole=.5,
                        marker_colors=colors,
                        textinfo='percent+label',
                        insidetextorientation='radial'
                    )])
                    fig_donut.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#f8fafc',
                        showlegend=False,
                        height=330,
                        margin=dict(l=0, r=0, t=10, b=10)
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)
                    
                with col_res2:
                    st.markdown("#### 🎯 Predicted Final Scoreline (Poisson)")
                    st.markdown(
                        f"""
                        <div class="result-box">
                            <div class="score-title">Most Probable Scoreline</div>
                            <div class="score-display">{best_score[0]} - {best_score[1]}</div>
                            <div class="xg-display" style="margin-bottom: 1rem;">
                                Expected Goals (xG):<br>
                                <strong>{team1}</strong>: {lambda_t1:.2f} xG | <strong>{team2}</strong>: {lambda_t2:.2f} xG
                            </div>
                            <div style="background-color: rgba(255,255,255,0.05); padding: 0.75rem; border-radius: 8px; border: 1px dashed rgba(255,255,255,0.1);">
                                <span style="font-size: 0.85rem; color: #94a3b8;">
                                    This exact scoreline has a <strong>{max_p*100:.1f}%</strong> theoretical likelihood of occurring.
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True
                    )

with tab2:
    
    
    
    st.markdown("## 📊 Historical Data Insights")
    
    
    st.markdown("### 📈 Team Elo Rating Trajectories")
    default_select_teams = ["Brazil", "Argentina", "France", "England", "Germany"]
    selected_teams_elo = st.multiselect(
        "Select Teams to Compare Elo Rating History",
        sorted(teams_2026),
        default=default_select_teams
    )
    
    if selected_teams_elo:
        
        filtered_elo_df = elo_history_df[elo_history_df['team'].isin(selected_teams_elo)]
        
        fig_elo = px.line(
            filtered_elo_df,
            x='year',
            y='elo',
            color='team',
            labels={'year': 'Year', 'elo': 'Elo Rating', 'team': 'Team'},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_elo.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc',
            xaxis=dict(showgrid=True, gridcolor='#334155'),
            yaxis=dict(showgrid=True, gridcolor='#334155'),
            height=450,
            hovermode='x unified',
            margin=dict(l=0, r=0, t=10, b=10)
        )
        st.plotly_chart(fig_elo, use_container_width=True)
    else:
        st.info("Select one or more teams to see their Elo rating trends over the years.")
        
    st.markdown("---")
    
    
    col_trends1, col_trends2 = st.columns([1, 1])
    
    with col_trends1:
        st.markdown("### ⚽ Decadal Goal-Scoring Trends")
        
        fig_goals = px.line(
            goal_trend,
            x='year',
            y='avg_total_goals',
            labels={'year': 'Year', 'avg_total_goals': 'Avg Goals Per Game'},
            color_discrete_sequence=['#fbbf24']
        )
        fig_goals.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc',
            xaxis=dict(showgrid=True, gridcolor='#334155'),
            yaxis=dict(showgrid=True, gridcolor='#334155'),
            height=350,
            margin=dict(l=0, r=0, t=10, b=10)
        )
        st.plotly_chart(fig_goals, use_container_width=True)
        st.markdown(
            "<p style='color:#94a3b8; font-size:0.85rem; font-style:italic;'>Historically, the average number of goals scored per match has stabilized around 2.5 - 2.8 goals in modern football.</p>",
            unsafe_allow_html=True
        )
        
    with col_trends2:
        st.markdown("### 🏠 Home Field Advantage Analysis")
        
        
        
        venue_plot_data = []
        for idx, r in venue_stats.iterrows():
            venue_name = "Neutral Venue" if r['neutral'] == 1 else "Home Team Venue"
            total = r['total_matches']
            venue_plot_data.append({'Venue': venue_name, 'Outcome': 'Win', 'Percentage': (r['home_wins'] / total) * 100})
            venue_plot_data.append({'Venue': venue_name, 'Outcome': 'Draw', 'Percentage': (r['draws'] / total) * 100})
            venue_plot_data.append({'Venue': venue_name, 'Outcome': 'Loss', 'Percentage': (r['away_wins'] / total) * 100})
            
        venue_plot_df = pd.DataFrame(venue_plot_data)
        
        fig_venue = px.bar(
            venue_plot_df,
            x='Venue',
            y='Percentage',
            color='Outcome',
            barmode='group',
            labels={'Percentage': 'Match Share (%)', 'Venue': '', 'Outcome': 'Result'},
            color_discrete_sequence=['#10b981', '#6b7280', '#ef4444'] 
        )
        fig_venue.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#334155', range=[0, 100]),
            height=350,
            margin=dict(l=0, r=0, t=10, b=10)
        )
        st.plotly_chart(fig_venue, use_container_width=True)
        st.markdown(
            "<p style='color:#94a3b8; font-size:0.85rem; font-style:italic;'>Home teams win nearly 48% of the time, while in neutral venues the share is balanced (approximately 38% vs 38%). This verifies our symmetrization and model inputs.</p>",
            unsafe_allow_html=True
        )
        
    st.markdown("---")
    
    
    st.markdown("### 🏆 Final Elo Rating Leaderboard (All 48 Teams)")
    
    leaderboard_data = []
    for idx, (team, elo) in enumerate(sorted_teams):
        t_flag = FLAG_MAP.get(team, "un")
        leaderboard_data.append({
            'Rank': idx + 1,
            'Flag': f"https://flagcdn.com/w40/{t_flag}.png",
            'Team': team,
            'Elo Rating': round(elo, 1)
        })
    
    leaderboard_df = pd.DataFrame(leaderboard_data)
    
    
    
    st.dataframe(
        leaderboard_df,
        column_config={
            "Flag": st.column_config.ImageColumn("Flag", width="small"),
            "Rank": st.column_config.NumberColumn("Rank", format="%d"),
            "Team": st.column_config.TextColumn("Team"),
            "Elo Rating": st.column_config.NumberColumn("Elo Rating", format="%.1f")
        },
        use_container_width=True,
        hide_index=True,
        height=500
    )
