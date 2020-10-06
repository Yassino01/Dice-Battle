import numpy as np
import pandas as pd
from pulp import *
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def tab_Pdk(dmax):
        """Prend en argument le nombre de des a lancer.
	Retourne le tableau des P(d,k) pour d et k superieur a 1  """
        kmax = dmax*6  #la somme des des ne peut etre superieur a 6 fois leur nombre
        res = np.ones((dmax, kmax))

	#on met a zero toutes les cases qui sont impossible a completer
        for d in range(dmax):
                for k in range(kmax):
                        if (k+1)<2*(1+d) or (k+1)>6*(d+1):
                                res[d,k] = 0
		            
	#on initialise pour le cas d=1
        for i in range(1,6):
                res[0][i] = 1/5

	#on met les valeurs des Q(d,k) dans toutes les cases non nulles
        for d in range(1,dmax):
                for k in range(kmax):
                        if (res[d,k]==1) :
                                res[d,k] = 0
                                #on fait un for dans les valeurs qui sont realisables. 
                                #le +1 apres le min est la car nous sommes dans un range
                                for i in range(max(k-6,2*(d+1-1)-1) , min(k-2,6*(d+1-1))+1):
                                        res[d,k] += res[d-1,i]/5

	#On multiplie toutes les cases selon la formule pour obtenir les P(d,k)
        for d in range(dmax):
                for k in range(kmax):
                        res[d,k] =  res[d,k]*(5/6)**(d+1)
		        
        for d in range(dmax):
                res[d, 0] = 1-(5/6)**(d+1)
		
        return res   
        
        
nb_dice = 3
pdk = tab_Pdk(nb_dice)
pd.DataFrame(pdk, #les lignes suivantes ne servent qu'à clarifier le tab
             index = [i+1 for i in range(nb_dice)] ,      
             columns=[i+1 for i in range(6*nb_dice)] )
             
def Q2(D) :
    """Prend en argument le nombre de dés D
    Retourne le nombre optimale de dés à jouer pour maximiser l'ésperance d*(D)
    """
    esperance = []
    for d in range(1,D+1):
        esperance.append(4*d*(5/6)**d+1-(5/6)**d)
    return np.argmax(esperance)+1
    
    
D=3
P = tab_Pdk(D)

N=18

def EG1(i,j,cpt):
    if (j>=N):
        return (-1,cpt)
    if (i>=N-1):
        DOPT[i,j]=1
        return (1,cpt)
    if (E1[i,j]!=0):
        return (E1[i,j],cpt)
    m=-1
    for d in range(D):
        sum=0
        for k in range(1,6*D):
            sum+=EG2(i+k,j,cpt+1)[0]*P[d,k-1]
        if (sum>m):
            #print("(i,j): "+str(i)+' '+str(j))
            m=sum
            if (cpt==0):
                #print("d: "+str(d))
                #print("sum: "+str(sum))
                tabdes.append(d)
            DOPT[i,j]=d+1

                
    E1[i,j]=m
    return (m,cpt)
    
de2=-1
def EG2(i,j,cpt):
    if (i>=N):
        return (1,cpt+1)
    if (j>=N-1):
        return (-1,cpt+1)
    if (E2[i,j]!=0):
        return (E2[i,j],cpt+1)
    m=1
    for d2 in range(D):
        sum=0
        for k2 in range(1,6*D):
            sum+=EG1(i,j+k2,cpt+1)[0]*P[d2,k2-1]
        if (sum<m):
            m=sum
    E2[i,j]=m
    return (m,cpt+1)


def EG_J1(i,j):
    return (EG1(i,j,0),tabdes[-1]+1)


def strategie_optimale(score,adversaire):
    global DOPT
    return int(DOPT[score][adversaire])

import random
def simulation(n,de,stratA,stratB):
    sommea=0
    sommeb=0
    while (sommea<n and sommeb<n):
        sommetempa=[]
        sommetempb=[]
        #print("Score a: {}, score b: {}".format(sommea,sommeb))
        if (stratA=="optimale"):
            dicemaxa=strategie_optimale(sommea,sommeb)
        if (stratA=="aveugle"):
            dicemaxa=Q2(de)
        if (stratA=="aléatoire"):
            dicemaxa=random.randint(1,de)
        if (stratB=="optimale"):
            dicemaxb=strategie_optimale(sommeb,sommea)
        if (stratB=="aveugle"):
            dicemaxb=Q2(de)
        if (stratB=="aléatoire"):
            dicemaxb=random.randint(1,de)
        #print("A joue {} dés".format(dicemaxa))
        #print("B joue {} dés".format(dicemaxb))
        for k in range(dicemaxa):
            dice=random.randint(1,6)
            sommetempa.append(dice)
            #print("Le joueur A a tiré un {}".format(dice))
        for k in range(dicemaxb):
            dice=random.randint(1,6)
            #print("Le joueur B a tiré un {}".format(dice))
            sommetempb.append(dice)
        if (1 in sommetempa):
            sommea+=1
        else:
            sommea+=np.sum(sommetempa)
        if (1 in sommetempb):
            sommeb+=1
        else:
            sommeb+=np.sum(sommetempb)
    #print("SCORE FINAL: Score a: {}, score b: {}".format(sommea,sommeb))
    if (sommea>sommeb):
        return 1
    if (sommeb>sommea):
        return 2
    return 0

import matplotlib.pyplot as plt
def evaluationD(Dmax,Nmax,stratA,stratB):
    global N
    global E1
    global E2
    global tabdes
    global P
    global DOPT
    global D
    N=Nmax
    plota=[]
    plotb=[]
    for dice in range(1,Dmax):
        D=dice
        P = tab_Pdk(D)
        tabdes=[0]
        E1=np.zeros((Nmax+1,Nmax+1))
        E1[:,-1]=-1
        E1[-1,:]=1
        E1[-2,:]=1
        E2=np.zeros((Nmax+1,Nmax+1))
        E2[-1,:]=1
        E2[:,-1]=-1
        DOPT=np.zeros((Nmax+1,Nmax+1))
        EG_J1(0,0)
        for i in range(Nmax):
            EG_J1(0,i)
            EG_J1(i,0)
        cpta=0
        cptb=0
        cptc=0
        for k in range(100):
            a=simulation(Nmax,dice,stratA,stratB)
            if (a==1):
                cpta+=1
            if (a==2):
                cptb+=1
            if (a==0):
                cptc+=1
        plota.append(cpta/100)
        plotb.append(cptb/100)
    plt.plot([i for i in range (1,Dmax)],plota,label='Joueur A, stratégie '+stratA)
    plt.plot([i for i in range (1,Dmax)],plotb,label='Joueur B, stratégie '+stratB)
    plt.xlabel("Nombre de dés disponibles")
    plt.ylabel("Probabilité de victoire")
    plt.ylim(0,1)
    plt.title("Probabilité de victoire en fonction du nombre de dés disponibles pour N="+str(Nmax))
    plt.legend()
    plt.show()
    
def evaluationN(Dmax,Nmax,stratA,stratB):
    global N
    global E1
    global E2
    global tabdes
    global P
    global DOPT
    global D
    N=Nmax
    D=Dmax
    plota=[]
    plotb=[]
    xarray=[]
    for score in range(1,Nmax,10):
        xarray.append(score)
        N=score
        P = tab_Pdk(Dmax)
        tabdes=[0]
        E1=np.zeros((Nmax+1,Nmax+1))
        E1[:,-1]=-1
        E1[-1,:]=1
        E1[-2,:]=1
        E2=np.zeros((Nmax+1,Nmax+1))
        E2[-1,:]=1
        E2[:,-1]=-1
        DOPT=np.zeros((Nmax+1,Nmax+1))
        EG_J1(0,0)
        for i in range(Nmax):
            EG_J1(0,i)
            EG_J1(i,0)
        cpta=0
        cptb=0
        cptc=0
        for k in range(100):
            a=simulation(score,Dmax,stratA,stratB)
            if (a==1):
                cpta+=1
            if (a==2):
                cptb+=1
            if (a==0):
                cptc+=1
        plota.append(cpta/100)
        plotb.append(cptb/100)
    plt.plot(xarray,plota,label='Joueur A, stratégie '+stratA)
    plt.plot(xarray,plotb,label='Joueur B, stratégie '+stratB)
    plt.xlabel("Valeur de N")
    plt.ylabel("Probabilité de victoire")
    plt.ylim(0,1)
    plt.title("Probabilité de victoire en fonction de N pour "+str(Dmax)+" dés")
    plt.legend()
    plt.show()
    
def JeuVsMachine():
    global N
    global E1
    global E2
    global tabdes
    global P
    global DOPT
    global D
    sommea=0
    sommeb=0
    strat=input("Quelle est la stratégie de B? Choix possibles: optimale, aveugle, aléatoire\n")
    de=input("Quel est le nombre de dés maximum?\n")
    de=int(de)
    D=de
    obj=input("Quel est le score a atteindre?\n")
    obj=int(obj)
    N=obj
    if (strat=="optimale"):
        print("Veuillez patienter svp...")
        P = tab_Pdk(de)
        tabdes=[0]
        E1=np.zeros((N+1,N+1))
        E1[:,-1]=-1
        E1[-1,:]=1
        E1[-2,:]=1
        E2=np.zeros((N+1,N+1))
        E2[-1,:]=1
        E2[:,-1]=-1
        DOPT=np.zeros((N+1,N+1))
        EG_J1(0,0)
        for i in range(N):
            EG_J1(0,i)
            EG_J1(i,0)
    while (sommea<N and sommeb<N):
        print("Votre score actuel: {}, score actuel de la machine: {}".format(sommea,sommeb))
        sommetempa=[]
        sommetempb=[]
        dicemaxa=input("Combien de dés voulez-vous jouer?")
        dicemaxa=int(dicemaxa)
        if (strat=="optimale"):
            dicemaxb=strategie_optimale(sommeb,sommea) 
        if (strat=="aveugle"):
            dicemaxb=Q2(de)
        if (strat=="aléatoire"):
            dicemaxb=random.randint(1,de)
        #dicemaxa=strategie_optimale(sommea,sommeb)         
        print("Vous avez décidé de jouer {} dés".format(dicemaxa))
        print("La machine a décidé de jouer {} dés".format(dicemaxb))
        for k in range(dicemaxa):
            dice=random.randint(1,6)
            sommetempa.append(dice)
            print("Vous avez tiré un {}".format(dice))
        for k in range(dicemaxb):
            dice=random.randint(1,6)
            print("La machine a tiré un {}".format(dice))
            sommetempb.append(dice)
        if (1 in sommetempa):
            sommea+=1
        else:
            sommea+=np.sum(sommetempa)
        if (1 in sommetempb):
            sommeb+=1
        else:
            sommeb+=np.sum(sommetempb)
    print("SCORE FINAL: Vous: {}, Machine: {}".format(sommea,sommeb))
    if (sommea>sommeb):
        print("Vous avez gagné!")
        return
    if (sommeb>sommea):
        print("La machine a gagné!")
        return
    print("Égalité!")
    print(sommea)
    print(sommeb)
    return

def Jeu1vs1():
    global N
    global E1
    global E2
    global tabdes
    global P
    global DOPT
    global D
    sommea=0
    sommeb=0
    playerA=input("Veuillez entrer le nom du joueur A svp\n")
    playerB=input("Veuillez entrer le nom du joueur B svp\n")
    de=input("Quel est le nombre de dés maximum?\n")
    de=int(de)
    D=de
    obj=input("Quel est le score a atteindre?\n")
    obj=int(obj)
    N=obj
    while (sommea<N and sommeb<N):
        print("Le score actuel de "+playerA+" est de: "+str(sommea)+", le score actuel de "+playerB+" est de: "+str(sommeb))
        sommetempa=[]
        sommetempb=[]
        dicemaxa=input(playerA+", combien de dés voulez-vous jouer?")
        dicemaxb=input(playerB+", combien de dés voulez-vous jouer?")
        dicemaxa=int(dicemaxa)
        dicemaxb=int(dicemaxb)
        print(playerA+" a décidé de jouer {} dés".format(dicemaxa))
        print(playerB+" a décidé de jouer {} dés".format(dicemaxb))
        for k in range(dicemaxa):
            dice=random.randint(1,6)
            sommetempa.append(dice)
            print(playerA+" a tiré un {}".format(dice))
        for k in range(dicemaxb):
            dice=random.randint(1,6)
            print(playerB+" a tiré un {}".format(dice))
            sommetempb.append(dice)
        if (1 in sommetempa):
            sommea+=1
        else:
            sommea+=np.sum(sommetempa)
        if (1 in sommetempb):
            sommeb+=1
        else:
            sommeb+=np.sum(sommetempb)
    print("SCORE FINAL: "+playerA+": {}, ".format(sommea)+" "+playerB+": {}".format(sommeb))
    if (sommea>sommeb):
        print(playerA+" a gagné!")
        return
    if (sommeb>sommea):
        print(playerB+" a gagné!")
        return
    print("Égalité!")
    print(sommea)
    print(sommeb)
    return



def Eg1(d1,d2):
    global pdk
    return (sum([sum([pdk[d1-1,k1-1]*pdk[d2-1,k2-1] for k1 in range(1,6*d1+1) if k1>k2 ])
                for k2 in range(1,6*d2+1)])
            -sum([sum([pdk[d1-1,k1-1]*pdk[d2-1,k2-1] for k1 in range(1,6*d1+1) if k1<k2 ])
                  for k2 in range(1,6*d2+1)])
           )

EG=[[Eg1(i,j) for i in range(1,nb_dice+1)] for j in range(1,nb_dice+1)]

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
        

def opt_simultane(EG,nb_dice):
    var=['z']+[i for i in range(1,nb_dice+1)]

    # Create the 'prob' variable to contain the problem data
    prob = LpProblem("The opt strategy for player 1", LpMaximize)

    # A dictionary called 'dec_vars' is created to contain the referenced Variables
    dec_vars = LpVariable.dicts("Dec",var[1:],0)
    dec_vars.update(LpVariable.dicts("Dec",var[0],None,None))

    # The objective function is added to 'prob' first
    prob += lpSum([dec_vars['z']]),"objective function"
    
    for j in range(1,nb_dice+1):
        prob += lpSum([dec_vars['z']]+[EG[i-1][j-1]*dec_vars[i] for i in var[1:]]) <= 0.0, "{} contrast".format(j)

    prob += lpSum([dec_vars[i] for i in var[1:]]) == 1.0, "conditio proba"
    
    
    # The problem data is written to an .lp file
    prob.writeLP("strategy_player1.lp")

    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    dico=dict()
    
    for v in prob.variables():
        #print(v.name, "=", v.varValue)
        if (v.name[-1]=='z'):
            continue
        if (RepresentsInt(v.name[-3])):
            dico[int(v.name[-3:])]=v.varValue
            continue
        if (RepresentsInt(v.name[-2])):
            dico[int(v.name[-2:])]=v.varValue
            continue
        dico[int(v.name[-1])]=v.varValue

    # The optimised objective function value is printed to the screen    
    #print("Esperance de gain optimal = ", value(prob.objective))
    
    strategy=[v.varValue for v in prob.variables() if v.name != "Dec_z"]
    return max(dico, key=dico.get)


def simulationSimutanee(nb_dice,stratA,stratB):
    sommea=0
    sommeb=0
    sommetempa=[]
    sommetempb=[]
    global pdk
    #print("Score a: {}, score b: {}".format(sommea,sommeb))
    if (stratA=="optimale"):
        pdk = tab_Pdk(nb_dice)
        EG=[[Eg1(i,j) for i in range(1,nb_dice+1)] for j in range(1,nb_dice+1)]
        dicemaxa=opt_simultane(EG,nb_dice)
    if (stratA=="aveugle"):
        dicemaxa=Q2(nb_dice)
    if (stratA=="aléatoire"):
        dicemaxa=random.randint(1,nb_dice)
    if (stratB=="optimale"):
        pdk = tab_Pdk(nb_dice)
        EG=[[Eg1(i,j) for i in range(1,nb_dice+1)] for j in range(1,nb_dice+1)]
        dicemaxb=opt_simultane(EG,nb_dice)
    if (stratB=="aveugle"):
        dicemaxb=Q2(nb_dice)
    if (stratB=="aléatoire"):
        dicemaxb=random.randint(1,nb_dice)
    #print("A joue {} dés".format(dicemaxa))
    #print("B joue {} dés".format(dicemaxb))
    for k in range(dicemaxa):
        dice=random.randint(1,6)
        sommetempa.append(dice)
        #print("Le joueur A a tiré un {}".format(dice))
    for k in range(dicemaxb):
        dice=random.randint(1,6)
        #print("Le joueur B a tiré un {}".format(dice))
        sommetempb.append(dice)
    if (1 in sommetempa):
        sommea+=1
    else:
        sommea+=np.sum(sommetempa)
    if (1 in sommetempb):
        sommeb+=1
    else:
        sommeb+=np.sum(sommetempb)
    #print("SCORE FINAL: Score a: {}, score b: {}".format(sommea,sommeb))
    if (sommea>sommeb):
        return 1
    if (sommeb>sommea):
        return 2
    return 0

def evaluation1coup(Dmax,stratA,stratB):
    xarray=[]
    plota=[]
    plotb=[]
    for d in range(1,Dmax):
        xarray.append(d)
        cpta=0
        cptb=0
        cptc=0
        for k in range(100):
            a=simulationSimutanee(d,stratA,stratB)
            if (a==1):
                cpta+=1
            if (a==2):
                cptb+=1
            if (a==0):
                cptc+=1
        plota.append(cpta/100)
        plotb.append(cptb/100)
    plt.plot(xarray,plota,label='Joueur A, stratégie '+stratA)
    plt.plot(xarray,plotb,label='Joueur B, stratégie '+stratB)
    plt.xlabel("Valeur de D")
    plt.ylabel("Probabilité de victoire")
    plt.ylim(0,1)
    plt.title("Proba de victoire en fonction de D (jeu en un coup)")
    plt.legend()
    plt.show()

    


#evaluationN(8,200,"optimale","aveugle")
