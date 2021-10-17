import random
import pygame
import time
import matplotlib.pyplot as plt
from math import *
import numpy as np

pygame.mixer.init()
z25DO=pygame.mixer.Sound("25DO.wav")
z26REb=pygame.mixer.Sound("26REb.wav")
z27RE=pygame.mixer.Sound("27RE.wav")
z28MIb=pygame.mixer.Sound("28MIb.wav")
z29MI=pygame.mixer.Sound("29MI.wav")
z30FA=pygame.mixer.Sound("30FA.wav")
z31SOLb=pygame.mixer.Sound("31SOLb.wav")
z32SOL=pygame.mixer.Sound("32SOL.wav")
z33LAb=pygame.mixer.Sound("33LAb.wav")
z34LA=pygame.mixer.Sound("34LA.wav")
z35SIb=pygame.mixer.Sound("35SIb.wav")
z36SI=pygame.mixer.Sound("36SI.wav")

repnotes=[z25DO,z26REb,z27RE,z28MIb,z29MI,z30FA,z31SOLb,z32SOL,z33LAb,z34LA,z35SIb,z36SI]

listeTransition=[['P'],['L'],['R'],['P','L'],['P','R'],['L','R'],['L','P'],['R','L'],['R','P'],['R','P','R'],['L','P','L'],['L','R','L'],['P','R','P'],
                 ['P','L','P'],['R','L','R'],['R','P','L'],['L','R','P'],['P','L','R'],[]]


#---------------------------  Programmes de base  ----------------------------------


def composedouble(f,g): #Réalise 2 fonctions à la suite
    def fonc(x):
        return(g(f(x)))
    return(fonc)


def composetriple(f,g,h):   #Réalise 3 fonctions à la suite
    def fonce(x):
        return(h(g(f(x))))
    return(fonce)


def identity(t):    #Fonction identité
    return(t)


def tri(L):     #Tri par insertion
    for k in range(len(L)-1):
        for i in range(len(L)-1):
            if L[i]>L[i+1]:
                L[i],L[i+1]=L[i+1],L[i]
    return L


def triAcc(L):  #Tri les accords parfaits par leur règle de rangement (ex : [7,10,2])
    F=tri(L[:])
    a,b,c=F[0],F[1],F[2]
    if a+7==b+4==c or a+7==b+3==c:
        return([a,b,c])
    elif b+7==c+4==a+12 or b+7==c+3==a+12:
        return([b,c,a])
    else:
        return([c,a,b])

    
#---------------------------  Fonctions des transitions  -------------------------


def P(acc):     #Transition P
    if (acc[1]-acc[0])%12==4:
        acc[1]=(acc[1]-1)%12
    else:
        acc[1]=(acc[1]+1)%12
    return(triAcc(acc))


def L(acc):     #Transition L
    if (acc[1]-acc[0])%12==4:
        acc[0]=(acc[0]+11)%12
    else:
        acc[2]=(acc[2]-11)%12
    return(triAcc(acc))


def R (acc):    #Transition R
    if (acc[1]-acc[0])%12==4:
        acc[2]=(acc[2]-10)%12
    else:
        acc[0]=(acc[0]+10)%12
    return(triAcc(acc))


listeFonction=[P,L,R,composedouble(P,L),composedouble(P,R),composedouble(L,R),composedouble(L,P),composedouble(R,L),composedouble(R,P),
               composetriple(R,P,R),composetriple(L,P,L),composetriple(L,R,L),composetriple(P,R,P),composetriple(P,L,P),composetriple(R,L,R),composetriple(R,P,L),
               composetriple(L,R,P),composetriple(P,L,R),identity]


def pattern(debut,fin): #Recherche la transition entre deux accords (3 pas max)
    for k in range(len(listeFonction)): #MODIFICATION : programme plus court grâce à la liste qu'on a fait
        if listeFonction[k](debut[:])==fin:
            return(listeTransition[k])
    return([])


#-------------------------------  PROGRAMME DE PROJECTION  ----------------------------------

#-------------------------------  Programmes de base  -----------------------------------------

def doublons(L): #Enlève les doublons d'une liste (stable)
    F=[]
    for elt in L:
        if elt not in F:
            F.append(elt)
    return(F)


def trietdoublon(L): #Permet de trier une liste et d'en enlever les doublons
    FIN=[]
    for k in range(len(L)):
        p=0
        while L[p]!=min(L):
            p+=1
        if L[p] not in FIN:
            FIN.append(L[p])
        L=L[:p]+L[p+1:]        
    return(FIN)
        

def pente(P1, P2): #Retourne la pente d'une droite formée par 2 points
    if P1[0]==P2[0]:
        return False #Et False si elle est infini
    else:
        return((P2[1]-P1[1])/(P2[0]-P1[0]))


def pointgauche(L): #Retourne le ou les points avec l'abscisse la plus faible d'un polygone
    MIG=[L[0]]
    for k in range(1,len(L)):
        if L[k][0]<MIG[0][0]:
            MIG=[L[k]]
        elif L[k][0]==MIG[0][0]:
            MIG.append(L[k])
    return(MIG)


def pointdroitpos(L): #Retourne la ou les positions (dans L) du ou des points avec l'abscisse la plus élevée d'un polygone
    MIG=[L[0]]
    pos=[0]
    for k in range(1,len(L)):
        if L[k][0]>MIG[0][0]:
            MIG=[L[k]]
            pos=[k]
        elif L[k][0]==MIG[0][0]:
            MIG.append(L[k])
            pos.append(k)
    return(pos)


def ordcourbe(ab,L): #Retourne la valeur en ordonnée d'une courbe formée avec des droites pour une abscisse donnée
    k=0
    if ab==L[0][0]:
        return(L[0][1])
    while k<len(L) and L[k][0]<ab: #On cherche entre quels points se trouve le notre
        k+=1
    if ab==L[k][0]: #Si le point est connu, on retourne l'ordonné de celui-ci
        return(L[k][1])
    else:           #Sinon, méthode du barycentre
        inf=L[k-1]
        sup=L[k]
        t=(ab-inf[0])/(sup[0]-inf[0])
        hauteur=inf[1]*(1-t)+t*sup[1]
        return(hauteur)


def horloge(L): #Retourne les points d'un polygone CONVEXE dans l'ordre du sens des aiguilles d'une montre
    MIG=pointgauche(L) #On cherche le ou les deux points les plus à gauche
    if len(MIG)==1:
        a=MIG[0]
    else:   #Dans le cas de deux points, on place le plus haut en premier et l'autre à la fin de la liste
        if MIG[0][1]>MIG[1][1]: 
            a=MIG[0]
            last=MIG[1]
        else:
            a=MIG[1]
            last=MIG[0]
    FIN=[a]
    k=0
    while k<len(L): #On enlève ces points à gauche de la liste L
        if L[k] in MIG:
            L=L[:k]+L[k+1:]
        else:
            k+=1
    for p in range(len(L)): #Méthode des tangentes : on tri les points par leur pente avec le premier point (décroissant)
        inf=pente(a,L[0])   #La pente ne peut pas être infinie car on a enlevé le seul point qui pourrait l'engendrer
        rg=0
        for m in range(1,len(L)): 
            if inf<pente(a,L[m]):
                inf=pente(a,L[m])
                rg=m
        FIN.append(L[rg])
        L=L[:rg]+L[rg+1:]
    if len(MIG)==2: #On ajoute le deuxième point en bas à gauche s'il existe
        return(FIN+[last])
    else:
        return(FIN)


def pointdanspoly(I,L): #Cherche si un point appartient à un polygone
    X=I[0]
    Y=I[1]
    L=horloge(L)    #Sens horaire
    pos=pointdroitpos(L)    
    Up=L[:pos[0]+1] #La liste du haut se construit entre le point en haut à gauche et le point en haut à droite
    if Up[0][0]>X or Up[-1][0]<X:
        return(False)
    Down=L[:pos[0]:-1]  #La liste du bas se contruit entre le point en bas à gauche et le point en bas à droite
    DownAbs=[]  #On prélève les abscisses de Down
    for elt in Down:
        DownAbs.append(elt[0])
    if Up[0][0] not in DownAbs: #et on regarde s'il y a dans Down des points ayant les abscisses extrêmes
        Down=[Up[0]]+Down       #sinon, on doit les ajouter pour pouvoir encadrer tous les points de la figure avec
    if Up[-1][0] not in DownAbs:
        Down.append(Up[-1])
    valup=ordcourbe(X,Up)   #On regarde les valeurs de la courbe UP et DOWN pour l'abscisse de I
    valdown=ordcourbe(X,Down)
    return(valup>=Y>=valdown)   #On vérifie si notre point est entre la courbe du haut et la courbe du bas


#--------------------------------  Programme espace préhilbertien  -------------------------------------


def norme(X,Y):
    return(sqrt((X[0]-Y[0])**2+(X[1]-Y[1])**2))


def proj(X,Y,Z):
    if X==Y:
        return(X)
    ax=X[0]
    ay=Y[0]
    az=Z[0]
    bx=X[1]
    by=Y[1]
    bz=Z[1]
    t=((ay-az)*(ay-ax)+(by-bz)*(by-bx))/((ax-ay)**2+(bx-by)**2) #Méthode du barycentre, calcul basique
    if t>=1:
        return(X)
    elif t<=0:
        return(Y)
    else:
        return([t*ax+(1-t)*ay,t*bx+(1-t)*by])


#---------------------------  Programme distance Point Polygone  -------------------------------


def distPP(Z,P):
    P=horloge(P)
    C=proj(P[-1],P[0],Z)    #On regarde le projeté orthogonal de notre point sur tous les segments du polygone
    distmin=norme(Z,C)      #Puis on regarde la norme du segment entre lui et son projeté
    for k in range(1,len(P)):
        C=proj(P[k-1],P[k],Z)
        distmin=min(norme(Z,C),distmin) #Le minimum de toutes ces normes est la distance du point au polygone
    return(distmin)



#--------------------------------  MAIN DISTANCE  ---------------------------------------
        

def DistanceHausdorff(P,Q):
    distmax=0
    for elt in P:
        if not(pointdanspoly(elt,Q)):
            distmax=max(distPP(elt,Q),distmax)
    for elt in Q:
        if not(pointdanspoly(elt,P)):
            distmax=max(distPP(elt,P),distmax)
    return(distmax)


#-------------------------  Programmes pour Projecteur()  -------------------------------


def ConsonantTriads():  #Crée la liste des accords parfaits
    Triads=[]
    for k in range(12):
        Triads.append([k,(k+3)%12,(k+7)%12])
        Triads.append([k,(k+4)%12,(k+7)%12])
    return(Triads)


def AccIntoPoly(acc):   #Transforme un accord en un polygone dans le cercle chromatique (unitaire ici)
    L=[]    #On évite cos() et sin() par peur de mauvaises approximations de python (ex : cos(3*pi/2) != 10^(-17))
    X=[0,0.5,0.86605,1,0.86605,0.5,0,-0.5,-0.86605,-1,-0.86605,-0.5]
    Y=[1,0.86605,0.5,0,-0.5,-0.86605,-1,-0.86605,-0.5,0,0.5,0.86605]
    for elt in acc:
        x=X[elt]
        y=Y[elt]
        L.append([x,y])
    return(L)


def distAcc(acc1,acc2): #Mesure la distance entre deux accords
    A=AccIntoPoly(acc1)
    B=AccIntoPoly(acc2)
    return(DistanceHausdorff(A,B))


#--------------------------  MAIN PROGRAM Projecteur()  -------------------------------


def Projecteur(acc):    #Cherche les accords parfaits les plus proches en distance de l'accord entré en argument
    Triads=ConsonantTriads()
    Fin=[]
    mini=distAcc(acc,Triads[0])
    for elt in Triads:  #Recherche de la distance min 
        test=distAcc(acc,elt)
        if abs(test-mini)<0.001:
            Fin.append(elt)
        elif test<mini:
            Fin=[elt]
            mini=test
    return(Fin) #Attention : retourne une LISTE d'accords


#-------------------------------  FIN PROGRAMME PROJECTION  ---------------------------------


#---------------------------  Programmes pour lectureFichier()  -----------------------------


def InitPartition(partition):   #Enlève les lignes inutiles au texte partition
    while 'on' not in partition[0]:
        partition.pop(0)
    pasbon=[]
    k=0
    while k<len(partition):
        if 'cntl' in partition[k]:
            partition.pop(k)
        else:
            k+=1
    return(partition)


def lire(L):    #Récupère les notes et leur temps d'apparition
    espace=0
    x=0
    OnOff=''
    Note=''
    while espace!=5:
        if L[x]==' ':
            espace+=1
        if espace==3:
           OnOff+=L[x]
        if espace==4:
            Note+=L[x]
        x+=1
    return(Note[1:],OnOff[1:])


def regroupementparligne(partition):    #Crée les accords avec les lettres
    L=[[]]
    for k in range(len(partition)):
        if 'ch' in partition[k] and 'cntl' not in partition[k]:
            Note,OnOff=lire(partition[k])
            if partition[k][0]=='0':
                if OnOff=='on':
                    L[-1].append(Note)
                else:
                    L[-1].remove(Note)
            else:
                L.append(L[-1][:])
                if OnOff=='on':
                    L[-1].append(Note)
                else:
                    L[-1].remove(Note)
    return L


def conversionEngNb(L): #Transforme les notes en chiffres
    eng=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    for k in range(len(L)): #MODIFICATION ICI SUR LA RANGE (len(L) et plus 3)
        c=0
        while L[k][:-1]!=eng[c]:
            c+=1
        L[k]=c
    return(tri(L))


def triAccords(L):
    F=[]
    for elt in L:
        if len(elt)>=3: #MODIFICATION ICI SUR  LA CONDITION (len(elt)>=3 au lieu de len(elt)==3)
            F.append(elt)
    return(F)


def conversionTransNb(L):   #Transforme les transitions en chiffres correspondant à leur emplacement dans listeTransition
    M=[]
    for elt in L:
        for ф in range(len(listeTransition)):
            if elt==listeTransition[ф]:
                M.append(ф)
    return(M)


#--------------------------------  Programme lectureFichier() principal  --------------------------


def lectureFichier(fichier):   #Transforme une partition .txt en liste d'accords
    texte=open(fichier,'r') #Partition.txt à obtenir avec Ruby
    midi=texte.readlines()  

    midi=InitPartition(midi)

    midi=regroupementparligne(midi)

    mod12=[]                
    for elt in midi:
        mod12.append(conversionEngNb(elt))

    noDoublons=[]

    for elt in mod12:
        noDoublons.append(doublons(elt))

    noDoublons=triAccords(noDoublons)
    
    proj=[]
    for elt in noDoublons:
        proj.append(Projecteur(elt))

    ordered=[]
    for k in range(0,len(proj)):
        ordered.append(proj[k][random.randint(0,len(proj[k])-1)])

    acc1=ordered[0]

    trans=[]
    
    for k in range(len(ordered)-1):
        trans.append(pattern(ordered[k],ordered[k+1]))
    
    return(conversionTransNb(trans),acc1)   #Attention : on retourne des nombres dans la liste principale

#--------------------------------  Programme pour Markov  -----------------------------

        
def CreationMatPass(L):
    MatPass=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    for k in range(len(L)-1):
        MatPass[L[k]][L[k+1]]+=1    #+1 à l'endroit concerné
    for i in range(19):     #On crée des probas en divisant par le nombre d'éléments dans la ligne
        S=0
        for j in range(19):
            S+=MatPass[i][j]
        if S!=0:    #Attention à la division par 0
            for j in range(19):
                MatPass[i][j]/=S
    return(MatPass)

        
def CreationPart(M,n,L):
    first=random.randint(0,len(L)-2)    #On prend une transition au hasard dans la liste initiale des transitions
    Trans=[L[first]]    #sauf la dernière 
    c18=0   #Compteur pour éviter trop de répétitions de la transition identité
    while len(Trans)<n-1:
        if c18!=4:
            a=random.random()
            liste=M[Trans[-1]]
            t=0
            S=liste[t]
            while a>S and t<18: #On parcourt notre matrice de passage pour trouver la transition qui correspond
                t+=1            #à notre variable aléatoire
                S+=liste[t]
            if t==18:
                c18+=1
            else:
                c18=0
            Trans.append(t)
        else:
            Trans.pop() #Si on a trop de reps, on vire le dernier 18 créé
            c18-=1
    return(Trans)


#---------------------------------------  Programme JOUER()  ---------------------------------------


def MakeMelody(Transitions,acc1):   #Synthétise la nouvelle partition à partir de la première note
    Partition=[acc1]                #et de la série de transitions
    for k in range(1,len(Transitions)):
        Partition.append(listeFonction[Transitions[k]](Partition[k-1]))
    return(Partition)

        
def JOUER(partition,rythme):        #Joue la partition créée
    for accord in partition:
        repnotes[accord[0]].play(0,rythme)
        repnotes[accord[1]].play(0,rythme)
        repnotes[accord[2]].play(0,rythme)
        while pygame.mixer.get_busy():
            pass


#-------------------------------------  MAIN PROGRAM  -----------------------------------------------


def MAIN(fichier,taille,rythme): # joue une mélodie aléatoire à partir d'une partition sous format '.txt'
    L,acc1=lectureFichier(fichier) # exploitation du fichier '.txt'
    MatPas=CreationMatPass(L) # création de la matrice de passage à partir de la partition initiale
    Trans=CreationPart(MatPas,taille,L) # création de la série de transitions sur laquelle sera basée la nouvelle partition
    part=MakeMelody(Trans,acc1) # création de la nouvelle partition
    JOUER(part,rythme) # sortie sonore de la nouvelle partition


            
