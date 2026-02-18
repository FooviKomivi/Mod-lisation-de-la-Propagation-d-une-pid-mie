"""
Modèle SIR Avancé - Analyses de Sensibilité et Scénarios Multiples
Auteur: Foovi Komivi
Date: 2020-02-15
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd

# -------------------------------
# Fonctions du modèle SIR
# -------------------------------

def sir_equations(y, t, N, beta, gamma):
    """Équations du modèle SIR"""
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def simuler_sir(N, beta, gamma, S0, I0, R0, temps):
    """Simule le modèle SIR"""
    y0 = [S0, I0, R0]
    solution = odeint(sir_equations, y0, temps, args=(N, beta, gamma))
    return solution.T  # retourne S, I, R

# -------------------------------
# Analyse de sensibilité R0
# -------------------------------

def analyse_sensibilite_R0():
    N = 10000
    S0, I0, R0_init = 9990, 10, 0
    gamma = 0.1
    temps = np.linspace(0, 200, 200)
    
    valeurs_R0 = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    plt.figure(figsize=(14,6))
    
    # Courbes des infectés pour différents R0
    plt.subplot(1,2,1)
    for R0_val in valeurs_R0:
        beta = R0_val * gamma
        S, I, R = simuler_sir(N, beta, gamma, S0, I0, R0_init, temps)
        plt.plot(temps, I, linewidth=2, label=f'R₀={R0_val}')
    
    plt.xlabel('Temps (jours)', fontsize=12)
    plt.ylabel('Nombre d\'infectés', fontsize=12)
    plt.title('Impact de R₀ sur la Courbe Épidémique', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Barres : attaque finale
    plt.subplot(1,2,2)
    attaque_finale = []
    for R0_val in valeurs_R0:
        beta = R0_val * gamma
        S, I, R = simuler_sir(N, beta, gamma, S0, I0, R0_init, temps)
        attaque_finale.append((N - S[-1]) / N * 100)
    
    plt.bar([str(r) for r in valeurs_R0], attaque_finale, color='coral', edgecolor='black')
    plt.xlabel('R₀', fontsize=12)
    plt.ylabel('Taux d\'attaque final (%)', fontsize=12)
    plt.title('Taux d\'Attaque en Fonction de R₀', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('analyse_sensibilite_R0.png', dpi=300)
    print("Graphique sauvegardé : analyse_sensibilite_R0.png")
    plt.show()

# -------------------------------
# Scénarios d'interventions
# -------------------------------

def scenarios_interventions():
    N = 10000
    S0, I0, R0_init = 9990, 10, 0
    gamma = 0.1
    beta_base = 0.5
    jours_total = 200
    temps = np.arange(0, jours_total+1)
    
    scenarios = {
        'Sans intervention': (beta_base, 0),
        'Distanciation modérée (-30%)': (beta_base*0.7, 30),
        'Distanciation forte (-60%)': (beta_base*0.4, 30),
        'Confinement (-80%)': (beta_base*0.2, 30)
    }
    
    plt.figure(figsize=(14,6))
    resultats = []
    
    for nom, (beta, jour_intervention) in scenarios.items():
        if jour_intervention == 0:
            S, I, R = simuler_sir(N, beta, gamma, S0, I0, R0_init, temps)
        else:
            # Avant intervention
            temps1 = np.arange(0, jour_intervention+1)
            S1, I1, R1 = simuler_sir(N, beta_base, gamma, S0, I0, R0_init, temps1)
            
            # Après intervention
            temps2 = np.arange(jour_intervention, jours_total+1)
            S2, I2, R2 = simuler_sir(N, beta, gamma, S1[-1], I1[-1], R1[-1], temps2)
            
            # Combiner les résultats
            S = np.concatenate([S1[:-1], S2])
            I = np.concatenate([I1[:-1], I2])
            R = np.concatenate([R1[:-1], R2])
        
        # Tracer la courbe des infectés
        plt.plot(temps, I, linewidth=2, label=nom)
        
        # Statistiques
        pic = np.max(I)
        total_infectes = N - S[-1]
        resultats.append({
            'Scénario': nom,
            'Pic d\'infectés': int(pic),
            'Total infectés': int(total_infectes),
            'Pourcentage': f"{total_infectes/N*100:.1f}%"
        })
    
    plt.xlabel('Temps (jours)', fontsize=12)
    plt.ylabel('Nombre d\'infectés', fontsize=12)
    plt.title('Comparaison des Scénarios d\'Intervention', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Tableau récapitulatif
    plt.figure(figsize=(8,2))
    plt.axis('off')
    df = pd.DataFrame(resultats)
    table = plt.table(cellText=df.values, colLabels=df.columns,
                      cellLoc='center', loc='center', bbox=[0,0,1,1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.title('Résultats des Scénarios', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('scenarios_interventions.png', dpi=300)
    print("Graphique sauvegardé : scenarios_interventions.png")
    plt.show()
    
    print("\n TABLEAU RÉCAPITULATIF DES SCÉNARIOS")
    print(df.to_string(index=False))

# -------------------------------
# Fonction principale
# -------------------------------

def main_advanced():
    print("\n ANALYSES AVANCÉES DU MODÈLE SIR\n")
    print("\n 1️  Analyse de sensibilité R₀...")
    analyse_sensibilite_R0()
    
    print("\n 2️  Comparaison des scénarios d'intervention...")
    scenarios_interventions()
    
    print("\n Toutes les analyses sont terminées!")

# -------------------------------
if __name__ == "__main__":
    main_advanced()

