import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')


print("Loading results.csv...")
results_path = "Dataset/results.csv"
df = pd.read_csv(results_path)
df = df.dropna(subset=['home_score', 'away_score']).reset_index(drop=True)
df['home_score'] = df['home_score'].astype(int)
df['away_score'] = df['away_score'].astype(int)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)


teams_2026 = [
    "Algeria", "Argentina", "Australia", "Austria", "Belgium", "Bosnia & Herzegovina",
    "Brazil", "Cabo Verde", "Canada", "Colombia", "Croatia", "Curaçao", "Czech Republic",
    "DR Congo", "Ecuador", "Egypt", "England", "France", "Germany", "Ghana", "Haiti",
    "Iran", "Iraq", "Ivory Coast", "Japan", "Jordan", "Mexico", "Morocco", "Netherlands",
    "New Zealand", "Norway", "Panama", "Paraguay", "Portugal", "Qatar", "Saudi Arabia",
    "Scotland", "Senegal", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland",
    "Türkiye", "United States", "Uruguay", "Uzbekistan", "Tunisia"
]


NAME_MAP_TO_DATASET = {
    'Bosnia & Herzegovina': 'Bosnia and Herzegovina',
    'Cabo Verde': 'Cape Verde',
    'Türkiye': 'Turkey'
}
NAME_MAP_FROM_DATASET = {v: k for k, v in NAME_MAP_TO_DATASET.items()}


teams_2026_dataset = [NAME_MAP_TO_DATASET.get(t, t) for t in teams_2026]

all_teams = set(df['home_team']).union(set(df['away_team']))
print(f"Total unique teams in history: {len(all_teams)}")


def get_weighted_mean(values):
    if not values:
        return 0.5
    n = len(values)
    weights = np.arange(1, n + 1)
    return np.sum(np.array(values) * weights) / np.sum(weights)

def get_weighted_mean_goals(values, default):
    if not values:
        return default
    n = len(values)
    weights = np.arange(1, n + 1)
    return np.sum(np.array(values) * weights) / np.sum(weights)


def categorize_tournament(t_name):
    t_lower = t_name.lower()
    if 'friendly' in t_lower:
        return 'Friendly', 10
    elif 'fifa world cup' in t_lower and 'qualification' not in t_lower:
        return 'World Cup', 60
    elif 'fifa world cup qualification' in t_lower:
        return 'WC Qualifier', 40
    elif any(term in t_lower for term in ['qualification', 'nations league', 'euro', 'copa', 'african cup', 'asian cup', 'gold cup']):
        return 'Major', 30
    else:
        return 'Minor', 20


elo_ratings = {team: 1500.0 for team in all_teams}
recent_results = {team: [] for team in all_teams} 
recent_goals_scored = {team: [] for team in all_teams}
recent_goals_conceded = {team: [] for team in all_teams}
h2h_records = {} 

match_features = []
elo_history = [] 

current_year = df['date'].iloc[0].year

print("Running Elo calculations and extracting match features...")
for idx, row in df.iterrows():
    h_team = row['home_team']
    a_team = row['away_team']
    h_score = int(row['home_score'])
    a_score = int(row['away_score'])
    tourney = row['tournament']
    neutral = row['neutral']
    match_date = row['date']
    match_year = match_date.year
    
    if match_year != current_year:
        for t in teams_2026_dataset:
            display_name = NAME_MAP_FROM_DATASET.get(t, t)
            elo_history.append({
                'year': current_year,
                'team': display_name,
                'elo': elo_ratings[t]
            })
        current_year = match_year
        
    elo_h = elo_ratings[h_team]
    elo_a = elo_ratings[a_team]
    
    H_val = 100.0 if not neutral else 0.0
    w_e_h = 1.0 / (10.0 ** (-((elo_h + H_val) - elo_a) / 400.0) + 1.0)
    w_e_a = 1.0 - w_e_h
    
    if h_score > a_score:
        w_h, w_a = 1.0, 0.0
        outcome = 2
    elif h_score == a_score:
        w_h, w_a = 0.5, 0.5
        outcome = 1
    else:
        w_h, w_a = 0.0, 1.0
        outcome = 0
        
    gd = abs(h_score - a_score)
    G = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11.0 + gd) / 8.0)
    tourney_cat, K = categorize_tournament(tourney)
    
    
    form_h = get_weighted_mean(recent_results[h_team])
    form_a = get_weighted_mean(recent_results[a_team])
    form_goals_scored_h = get_weighted_mean_goals(recent_goals_scored[h_team], 1.5)
    form_goals_conceded_h = get_weighted_mean_goals(recent_goals_conceded[h_team], 1.5)
    form_goals_scored_a = get_weighted_mean_goals(recent_goals_scored[a_team], 1.2)
    form_goals_conceded_a = get_weighted_mean_goals(recent_goals_conceded[a_team], 1.2)
    
    pair = tuple(sorted([h_team, a_team]))
    if pair not in h2h_records:
        h2h_records[pair] = {'matches': 0, 'wins': {h_team: 0, a_team: 0}}
    n_h2h = h2h_records[pair]['matches']
    h2h_win_h = h2h_records[pair]['wins'].get(h_team, 0) / n_h2h if n_h2h > 0 else 0.5
    h2h_win_a = h2h_records[pair]['wins'].get(a_team, 0) / n_h2h if n_h2h > 0 else 0.5
    
    match_features.append({
        'date': match_date,
        'home_team': h_team,
        'away_team': a_team,
        'elo_h': elo_h,
        'elo_a': elo_a,
        'elo_diff': elo_h - elo_a,
        'form_h': form_h,
        'form_a': form_a,
        'form_goals_scored_h': form_goals_scored_h,
        'form_goals_conceded_h': form_goals_conceded_h,
        'form_goals_scored_a': form_goals_scored_a,
        'form_goals_conceded_a': form_goals_conceded_a,
        'h2h_win_h': h2h_win_h,
        'h2h_win_a': h2h_win_a,
        'tourney_cat': tourney_cat,
        'neutral': int(neutral),
        'home_score': h_score,
        'away_score': a_score,
        'outcome': outcome
    })
    
    elo_ratings[h_team] = elo_h + K * G * (w_h - w_e_h)
    elo_ratings[a_team] = elo_a + K * G * (w_a - w_e_a)
    recent_results[h_team] = (recent_results[h_team] + [w_h])[-5:]
    recent_results[a_team] = (recent_results[a_team] + [w_a])[-5:]
    recent_goals_scored[h_team] = (recent_goals_scored[h_team] + [h_score])[-5:]
    recent_goals_conceded[h_team] = (recent_goals_conceded[h_team] + [a_score])[-5:]
    recent_goals_scored[a_team] = (recent_goals_scored[a_team] + [a_score])[-5:]
    recent_goals_conceded[a_team] = (recent_goals_conceded[a_team] + [h_score])[-5:]
    
    h2h_records[pair]['matches'] += 1
    if w_h == 1.0:
        h2h_records[pair]['wins'][h_team] += 1
    elif w_a == 1.0:
        h2h_records[pair]['wins'][a_team] += 1


for t in teams_2026_dataset:
    display_name = NAME_MAP_FROM_DATASET.get(t, t)
    elo_history.append({
        'year': current_year,
        'team': display_name,
        'elo': elo_ratings[t]
    })

elo_history_df = pd.DataFrame(elo_history)
features_df = pd.DataFrame(match_features)


print("Symmetrizing dataset (post-2000)...")
train_subset = features_df[features_df['date'] >= '2000-01-01'].reset_index(drop=True)
tourney_map = {'Friendly': 0, 'Minor': 1, 'Major': 2, 'WC Qualifier': 3, 'World Cup': 4}
train_subset['tourney_code'] = train_subset['tourney_cat'].map(tourney_map)

rows_orig = []
rows_swap = []
for idx, r in train_subset.iterrows():
    rows_orig.append({
        'elo_team': r['elo_h'],
        'elo_opp': r['elo_a'],
        'elo_diff': r['elo_diff'],
        'form_team': r['form_h'],
        'form_opp': r['form_a'],
        'form_team_scored': r['form_goals_scored_h'],
        'form_team_conceded': r['form_goals_conceded_h'],
        'form_opp_scored': r['form_goals_scored_a'],
        'form_opp_conceded': r['form_goals_conceded_a'],
        'h2h_win_team': r['h2h_win_h'],
        'h2h_win_opp': r['h2h_win_a'],
        'tourney_code': r['tourney_code'],
        'is_home': 0 if r['neutral'] == 1 else 1,
        'goals_team': r['home_score'],
        'goals_opp': r['away_score'],
        'outcome': r['outcome']
    })
    swapped_outcome = 2 if r['outcome'] == 0 else (0 if r['outcome'] == 2 else 1)
    rows_swap.append({
        'elo_team': r['elo_a'],
        'elo_opp': r['elo_h'],
        'elo_diff': -r['elo_diff'],
        'form_team': r['form_a'],
        'form_opp': r['form_h'],
        'form_team_scored': r['form_goals_scored_a'],
        'form_team_conceded': r['form_goals_conceded_a'],
        'form_opp_scored': r['form_goals_scored_h'],
        'form_opp_conceded': r['form_goals_conceded_h'],
        'h2h_win_team': r['h2h_win_a'],
        'h2h_win_opp': r['h2h_win_h'],
        'tourney_code': r['tourney_code'],
        'is_home': 0 if r['neutral'] == 1 else -1,
        'goals_team': r['away_score'],
        'goals_opp': r['home_score'],
        'outcome': swapped_outcome
    })

sym_df = pd.concat([pd.DataFrame(rows_orig), pd.DataFrame(rows_swap)], ignore_index=True)



print("Tuning RandomForestClassifier...")
X_clf = sym_df[['elo_team', 'elo_opp', 'elo_diff', 'form_team', 'form_opp', 'h2h_win_team', 'h2h_win_opp', 'tourney_code', 'is_home']]
y_clf = sym_df['outcome']

param_grid_clf = {
    'n_estimators': [100, 150],
    'max_depth': [8, 12],
    'min_samples_split': [2, 5]
}
grid_clf = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid_clf, cv=3, scoring='accuracy')
grid_clf.fit(X_clf, y_clf)
clf = grid_clf.best_estimator_
print("Random Forest optimized. Best parameters:", grid_clf.best_params_)


print("Tuning Poisson Regressor...")
sym_df['elo_team_scaled'] = sym_df['elo_team'] / 400.0
sym_df['elo_opp_scaled'] = sym_df['elo_opp'] / 400.0
sym_df['elo_diff_scaled'] = sym_df['elo_diff'] / 400.0

X_poi = sym_df[['elo_team_scaled', 'elo_opp_scaled', 'elo_diff_scaled', 'form_team_scored', 'form_opp_conceded', 'is_home']]
y_poi = sym_df['goals_team']

param_grid_poi = {
    'alpha': [1e-6, 1e-4, 1e-2]
}
grid_poi = GridSearchCV(PoissonRegressor(max_iter=300), param_grid_poi, cv=3)
grid_poi.fit(X_poi, y_poi)
poi = grid_poi.best_estimator_
print("Poisson Regressor optimized. Best alpha:", grid_poi.best_params_)


poi_coef = poi.coef_
poi_intercept = poi.intercept_


teams_elo = {NAME_MAP_FROM_DATASET.get(t, t): elo_ratings[t] for t in teams_2026_dataset}
sorted_teams = sorted(teams_elo.items(), key=lambda x: x[1], reverse=True)

groups = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'B': ['Canada', 'Bosnia & Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Türkiye'],
    'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cabo Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama']
}
group_letters = list(groups.keys())

def get_team_current_stats(team):
    t_ds = NAME_MAP_TO_DATASET.get(team, team)
    w_pts = get_weighted_mean(recent_results[t_ds])
    s_goals = get_weighted_mean_goals(recent_goals_scored[t_ds], 1.5)
    c_goals = get_weighted_mean_goals(recent_goals_conceded[t_ds], 1.5)
    return elo_ratings[t_ds], w_pts, s_goals, c_goals

def simulate_match_poisson(t1, t2):
    elo_t1, form_pts_t1, scored_t1, conceded_t1 = get_team_current_stats(t1)
    elo_t2, form_pts_t2, scored_t2, conceded_t2 = get_team_current_stats(t2)
    
    log_lambda_t1 = (
        (elo_t1 / 400.0) * poi_coef[0] +
        (elo_t2 / 400.0) * poi_coef[1] +
        ((elo_t1 - elo_t2) / 400.0) * poi_coef[2] +
        scored_t1 * poi_coef[3] +
        conceded_t2 * poi_coef[4] +
        poi_intercept
    )
    lambda_t1 = max(0.01, np.exp(log_lambda_t1))
    
    log_lambda_t2 = (
        (elo_t2 / 400.0) * poi_coef[0] +
        (elo_t1 / 400.0) * poi_coef[1] +
        ((elo_t2 - elo_t1) / 400.0) * poi_coef[2] +
        scored_t2 * poi_coef[3] +
        conceded_t1 * poi_coef[4] +
        poi_intercept
    )
    lambda_t2 = max(0.01, np.exp(log_lambda_t2))
    
    return np.random.poisson(lambda_t1), np.random.poisson(lambda_t2)

def simulate_knockout_match(t1, t2):
    g1, g2 = simulate_match_poisson(t1, t2)
    if g1 > g2:
        return t1
    elif g2 > g1:
        return t2
    else:
        elo_t1, _, _, _ = get_team_current_stats(t1)
        elo_t2, _, _, _ = get_team_current_stats(t2)
        p_t1_win = 1.0 / (10.0 ** (-(elo_t1 - elo_t2) / 400.0) + 1.0)
        return t1 if np.random.rand() < p_t1_win else t2


tournament_wins = {t[0]: 0 for t in sorted_teams}
group_points_sum = {t[0]: 0.0 for t in sorted_teams}

print("Running 10,000x 48-team World Cup simulations...")
for sim_idx in range(10000):
    if (sim_idx + 1) % 2000 == 0:
        print(f"Simulation {sim_idx + 1}/10,000 complete...")
        
    group_results = {}
    
    
    for letter, teams in groups.items():
        points = {t: 0 for t in teams}
        gd = {t: 0 for t in teams}
        goals = {t: 0 for t in teams}
        
        matches_to_play = [
            (teams[0], teams[1]), (teams[0], teams[2]), (teams[0], teams[3]),
            (teams[1], teams[2]), (teams[1], teams[3]), (teams[2], teams[3])
        ]
        
        for t1, t2 in matches_to_play:
            g1, g2 = simulate_match_poisson(t1, t2)
            goals[t1] += g1
            goals[t2] += g2
            gd[t1] += (g1 - g2)
            gd[t2] += (g2 - g1)
            
            if g1 > g2:
                points[t1] += 3
            elif g2 > g1:
                points[t2] += 3
            else:
                points[t1] += 1
                points[t2] += 1
                
        
        for t, pts in points.items():
            group_points_sum[t] += pts
            
        ranked = sorted(
            teams,
            key=lambda t: (points[t], gd[t], goals[t], elo_ratings[NAME_MAP_TO_DATASET.get(t, t)]),
            reverse=True
        )
        
        group_results[letter] = {
            'winner': ranked[0],
            'runner_up': ranked[1],
            'third': ranked[2],
            'third_stats': {
                'team': ranked[2],
                'points': points[ranked[2]],
                'gd': gd[ranked[2]],
                'goals': goals[ranked[2]],
                'elo': elo_ratings[NAME_MAP_TO_DATASET.get(ranked[2], ranked[2])]
            }
        }
        
    
    thirds = [group_results[letter]['third_stats'] for letter in group_letters]
    thirds_ranked = sorted(
        thirds,
        key=lambda x: (x['points'], x['gd'], x['goals'], x['elo']),
        reverse=True
    )
    best_thirds = [x['team'] for x in thirds_ranked[:8]]
    
    
    winners = {letter: group_results[letter]['winner'] for letter in group_letters}
    runners = {letter: group_results[letter]['runner_up'] for letter in group_letters}
    
    r32_matches = [
        (winners['A'], runners['B']),
        (winners['B'], runners['A']),
        (winners['C'], best_thirds[0]),
        (winners['D'], runners['E']),
        (winners['E'], runners['D']),
        (winners['F'], best_thirds[1]),
        (winners['G'], runners['H']),
        (winners['H'], runners['G']),
        (winners['I'], best_thirds[2]),
        (winners['J'], runners['K']),
        (winners['K'], runners['J']),
        (winners['L'], best_thirds[3]),
        (runners['C'], best_thirds[4]),
        (runners['F'], best_thirds[5]),
        (runners['I'], best_thirds[6]),
        (runners['L'], best_thirds[7])
    ]
    
    r16_teams = []
    for t1, t2 in r32_matches:
        winner = simulate_knockout_match(t1, t2)
        r16_teams.append(winner)
        
    
    qf_teams = []
    for i in range(0, 16, 2):
        winner = simulate_knockout_match(r16_teams[i], r16_teams[i+1])
        qf_teams.append(winner)
        
    
    sf_teams = []
    for i in range(0, 8, 2):
        winner = simulate_knockout_match(qf_teams[i], qf_teams[i+1])
        sf_teams.append(winner)
        
    
    final_teams = []
    for i in range(0, 4, 2):
        winner = simulate_knockout_match(sf_teams[i], sf_teams[i+1])
        final_teams.append(winner)
        
    
    champion = simulate_knockout_match(final_teams[0], final_teams[1])
    tournament_wins[champion] += 1


avg_group_points = {team: pts / 10000.0 for team, pts in group_points_sum.items()}


win_probabilities = sorted(
    [(team, wins / 100.0) for team, wins in tournament_wins.items()],
    key=lambda x: x[1],
    reverse=True
)

print("\n--- 2026 WORLD CUP TOURNAMENT SIMULATION WIN PROBABILITIES (10k RUNS) ---")
for team, prob in win_probabilities[:15]:
    print(f"{team}: {prob:.2f}% chance of winning")
print("------------------------------------------------------------\n")


print("Computing EDA statistics...")
features_df['decade'] = (features_df['date'].dt.year // 10) * 10
goal_trend = features_df.groupby(features_df['date'].dt.year).agg(
    avg_home_goals=('home_score', 'mean'),
    avg_away_goals=('away_score', 'mean'),
    total_matches=('outcome', 'count')
).reset_index().rename(columns={'date': 'year'})
goal_trend['avg_total_goals'] = goal_trend['avg_home_goals'] + goal_trend['avg_away_goals']

venue_stats = features_df.groupby('neutral').agg(
    home_wins=('outcome', lambda x: (x == 2).sum()),
    draws=('outcome', lambda x: (x == 1).sum()),
    away_wins=('outcome', lambda x: (x == 0).sum()),
    total_matches=('outcome', 'count')
).reset_index()


output_path = "models/dashboard_data.pkl"
dashboard_data = {
    'models': {
        'match_classifier': clf,
        'poisson_regressor': poi
    },
    'elo': {
        'final_elo_ratings': elo_ratings,
        'elo_history': elo_history_df,
        'sorted_teams': sorted_teams
    },
    'tournament': {
        'groups': groups,
        'win_probabilities': win_probabilities,
        'avg_group_points': avg_group_points
    },
    'eda': {
        'goal_trend': goal_trend,
        'venue_stats': venue_stats
    },
    'mapping': {
        'NAME_MAP_TO_DATASET': NAME_MAP_TO_DATASET,
        'NAME_MAP_FROM_DATASET': NAME_MAP_FROM_DATASET,
        'teams_2026': teams_2026
    }
}

with open(output_path, 'wb') as f:
    pickle.dump(dashboard_data, f)
print(f"Data and models successfully pickled to {output_path}")
print("Background script execution complete!")
