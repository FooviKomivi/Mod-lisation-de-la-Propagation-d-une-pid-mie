"""
Modèle SIR - Simulation de la Propagation d'une Maladie
Auteur: Foovi Komivi
Date: 2026-02-06
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class ModeleSIR:
    """Classe pour simuler le modèle SIR"""
    
    def __init__(self, N, beta, gamma, S0, I0, R0):
        """
        Initialisation du modèle
        
        Paramètres:
        -----------
        N : int - Population totale
        beta : float - Taux de transmission (jour^-1)
        gamma : float - Taux de guérison (jour^-1)
        S0, I0, R0 : int - Conditions initiales
        """
        self.N = N
        self.beta = beta
        self.gamma = gamma
        self.S0 = S0
        self.I0 = I0
        self.R0_init = R0
        self.R0 = beta / gamma  # Nombre de reproduction de base
        
    def equations_sir(self, y, t):
        """Système d'équations différentielles du modèle SIR"""
        S, I, R = y
        dSdt = -self.beta * S * I / self.N
        dIdt = self.beta * S * I / self.N - self.gamma * I
        dRdt = self.gamma * I
        return dSdt, dIdt, dRdt
    
    def simuler(self, temps):
        """
        Effectue la simulation
        
        Paramètres:
        -----------
        temps : array - Vecteur de temps
        
        Retourne:
        ---------
        S, I, R : arrays - Solutions pour chaque compartiment
        """
        y0 = [self.S0, self.I0, self.R0_init]
        solution = odeint(self.equations_sir, y0, temps)
        S, I, R = solution.T
        return S, I, R
    
    def tracer_resultats(self, temps, S, I, R):
        """Trace les graphiques de la simulation"""
        plt.figure(figsize=(12, 8))
        
        # Graphique principal
        plt.subplot(2, 2, 1)
        plt.plot(temps, S, 'b-', label='Susceptibles (S)', linewidth=2)
        plt.plot(temps, I, 'r-', label='Infectés (I)', linewidth=2)
        plt.plot(temps, R, 'g-', label='Rétablis (R)', linewidth=2)
        plt.xlabel('Temps (jours)', fontsize=12)
        plt.ylabel('Nombre d\'individus', fontsize=12)
        plt.title('Évolution du Modèle SIR', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Graphique des infectés uniquement
        plt.subplot(2, 2, 2)
        plt.plot(temps, I, 'r-', linewidth=2)
        plt.xlabel('Temps (jours)', fontsize=12)
        plt.ylabel('Nombre d\'infectés', fontsize=12)
        plt.title('Courbe Épidémique', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Proportions
        plt.subplot(2, 2, 3)
        plt.plot(temps, S/self.N*100, 'b-', label='S (%)', linewidth=2)
        plt.plot(temps, I/self.N*100, 'r-', label='I (%)', linewidth=2)
        plt.plot(temps, R/self.N*100, 'g-', label='R (%)', linewidth=2)
        plt.xlabel('Temps (jours)', fontsize=12)
        plt.ylabel('Pourcentage de la population', fontsize=12)
        plt.title('Proportions', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Diagramme de phase S-I
        plt.subplot(2, 2, 4)
        plt.plot(S, I, 'purple', linewidth=2)
        plt.xlabel('Susceptibles (S)', fontsize=12)
        plt.ylabel('Infectés (I)', fontsize=12)
        plt.title('Diagramme de Phase S-I', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('resultats_sir.png', dpi=300, bbox_inches='tight')
        print(" Graphique sauvegardé : resultats_sir.png")
        plt.show()
    
    def afficher_statistiques(self, temps, S, I, R):
        """Affiche les statistiques clés de la simulation"""
        pic_infectes = np.max(I)
        jour_pic = temps[np.argmax(I)]
        total_infectes = self.N - S[-1]
        pourcentage_infectes = (total_infectes / self.N) * 100
        
        print("\n" + "="*60)
        print(" STATISTIQUES DE LA SIMULATION")
        print("="*60)
        print(f"Population totale (N)          : {self.N:,}")
        print(f"Taux de transmission (β)       : {self.beta:.3f} jour⁻¹")
        print(f"Taux de guérison (γ)           : {self.gamma:.3f} jour⁻¹")
        print(f"Nombre de reproduction (R₀)    : {self.R0:.2f}")
        print("-"*60)
        print(f"Pic d'infectés                 : {int(pic_infectes):,} personnes")
        print(f"Jour du pic                    : Jour {int(jour_pic)}")
        print(f"Total infectés (final)         : {int(total_infectes):,} personnes")
        print(f"Pourcentage infecté            : {pourcentage_infectes:.1f}%")
        print(f"Susceptibles restants          : {int(S[-1]):,} personnes")
        print("="*60 + "\n")


def main():
    """Fonction principale"""
    print("\n SIMULATION DU MODÈLE SIR  \n")
    
    # Paramètres de la simulation
    N = 10000              # Population totale
    I0 = 10                # Infectés initiaux
    R0_init = 0            # Rétablis initiaux
    S0 = N - I0 - R0_init  # Susceptibles initiaux
    
    beta = 0.5             # Taux de transmission (contacts par jour)
    gamma = 0.1            # Taux de guérison (1/durée maladie)
    
    # Durée de la simulation
    jours = 200
    temps = np.linspace(0, jours, jours)
    
    # Créer et exécuter le modèle
    modele = ModeleSIR(N, beta, gamma, S0, I0, R0_init)
    S, I, R = modele.simuler(temps)
    
    # Afficher les résultats
    modele.afficher_statistiques(temps, S, I, R)
    modele.tracer_resultats(temps, S, I, R)
    
    print(" Simulation terminée avec succès!")


if __name__ == "__main__":
    main()

