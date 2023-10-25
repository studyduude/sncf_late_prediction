# ML project

## Description
Ce projet est effectué dans le cadre du cours d'apprentissage automatique de 3ème année dans la mention IA. Les personnes ayant travaillées sur ce projet sont :
- Cindy Hua
- Rémi Boutonnier
- Romain Ageron

## Contexte
Nous nous plaçons dans le contexte où la SNCF souhaite évaluer un planning des trains (pour les 6 mois à venir par exemple). Nous considérons donc seulement les variables qui caractérisent les liaisons, et cherchons à prédire le retard à l'arrivée (et potentiellement le retard au départ). Les variables prises en compte sont donc :
- date
- service
- gare_depart
- gare_arrivee
- duree_moyenne
- nb_train_prevu
Les variables à prédire sont :
- retard_moyen_arrivee
- prct_cause_externe
- prct_cause_infra
- prct_cause_gestion_trafic
- prct_cause_materiel_roulant
- prct_cause_gestion_gare
- prct_cause_prise_en_charge_voyageurs

Cela ne laisse pas beaucoup de variables pour la prédiction, mais nous avons utilisé des données extérieures à celles fournies (cf rapport).

## Installation
Pour faire tourner le code du projet, faire une copie avec `git clone https://gitlab-student.centralesupelec.fr/2019ageronr/ml-project.git` puis télécharger les dépendances avec `pip install -r requirements.txt` et lancer le fichier __main.py__ dans le dossier src avec `python __main__.py`