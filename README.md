# Dice-Battle

Deux joueurs s'afrontent dans un jeu de dés. Le but est d'être le premier à atteindre au moins N points (généralement N = 100). A chaque tour, on choisit de lancer entre 1 et D dés à six faces (par exemple, D = 10). Le nombre de points marqués est 1 si l'un des dés au moins tombe sur 1, dans le cas contraire c'est la somme des dés. Nous allons étudier deux variantes de ce jeu :

### variante séquentielle :  
 les joueurs jouent à tour de rôle (le premier joueur a donc bien sûr unavantage) ;

### variante simultanée : 

les joueurs jouent simultanément à chaque tour. Dans ce cas si les deux joueurs atteignent N points ou plus lors du même tour, c'est le joueur qui dépasse le plus les N points qui l'emporte ; si les deux joueurs obtiennent le même score, alors ils sont
ex-aequos.

Le but du projet est de déterminer une stratégie de jeu optimale (et de tester cette stratégie par rapport à d'autres). 

On dit qu'un joueur a un gain égal à 1 s'il remporte la partie, un gain égalà 0 si la partie est nulle, un gain égal à -1 s'il perd. C'est donc en particulier un jeu à somme nulle.

## Notions abordées : 

* Calcul de probabilité
* Programation dynamique
* Résolution d'un probleme d'optimisation.
