
## Liste des Baselines

#### 1. TIPP

 - 1. sans variante
 - 1.b. \+ Variante potentielle: avec GWR (gaz neuronal) pour la création du graphe
 - 1.c. \+ Variante avec choix du noeud en fonction du nombre de fois qu'on l'a rencontré 
et non exploré ?

#### 2. STC (2 versions)

 Stc fonctionne comme TIPP à deux différences près :
 * Il n'utilise pas de politique invariante en translation, 
 * il utilise un réseau de neurones externe pour estimer la distance entre deux états.

Il constitue donc deux baselines:
 * 2.a Une version invariante en translation, (on étudie la pertinence de rajouter un NN en plus pour STC)
 * 2.b Une version basique, pas d'invariance en translation : On montre que même STC qui est un algo récent est pas fou 
dans notre cas d'application.

#### 3. Algo de RL basique

 - HAC dans ant maze
 - SAC dans point maze
 - DQN dans grid world

#### 4. SORB ??
 - 4.a. SORB Avec oracle
 - 4.b. SORB Sans orable, avec un algo maison qui construit un graph par dessus 
le replay-buffer

## Avancement

I = Implemented
G = Enough seeds to generate a graph

+------+-------------------------+-------------------------+-------------------------+\
|      |        Grid World       |        Point Maze       |         Ant Maze        |\
+------+--------+------+---------+--------+------+---------+--------+------+---------+\
|      | Medium | Hard | Extreme | Medium | Hard | Extreme | Medium | Hard | Extreme |\
+------+--------+------+---------+--------+------+---------+--------+------+---------+\
|   1  |   IG   |  IG  |    I    |    I   |   I  |    I    |    I   |   I  |    I    |\
+------+--------+------+---------+--------+------+---------+--------+------+---------+\
| 1.b. |    I   |   I  |    I    |    I   |   I  |    I    |    I   |   I  |    I    |\
+------+--------+------+---------+--------+------+---------+--------+------+---------+\
| 1.c. |    I   |   I  |    I    |    I   |   I  |    I    |    I   |   I  |    I    |\
+------+--------+------+---------+--------+------+---------+--------+------+---------+\
| 2.a. |    I   |   I  |    I    |    I   |   I  |    I    |        |      |         |\
+------+--------+------+---------+--------+------+---------+--------+------+---------+\
| 2.b. |    I   |   I  |    I    |    I   |   I  |    I    |        |      |         |\
+------+--------+------+---------+--------+------+---------+--------+------+---------+\
|  3.  |    I   |   I  |    I    |    I   |   I  |    I    |    I   |   I  |    I    |\
+------+--------+------+---------+--------+------+---------+--------+------+---------+\
| 4.a. |    I   |   I  |    I    |    I   |   I  |    I    |    I   |   I  |    I    |\
+------+--------+------+---------+--------+------+---------+--------+------+---------+\
| 4.b. |    I   |   I  |    I    |    I   |   I  |    I    |    I   |   I  |    I    |\
+------+--------+------+---------+--------+------+---------+--------+------+---------+\