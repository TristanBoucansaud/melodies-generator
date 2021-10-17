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


#-------------------  Programmes de base  ---------------------


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


def doublons(L): #Enlève les doublons d'une liste (stable)
    F=[]
    for elt in L:
        if elt not in F:
            F.append(elt)
    return(F)


#------------------------  Programmes pour AirePolygone()  ------------------------

        
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
        inf=pente(a,L[0])
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

            
def courbefinale(L): #"Applatit" le polygone pour en faire une courbe
    Courbe=[]
    pos=pointdroitpos(L) #On sépare les points en "haut" et en "bas" du polygone qu'on ordonne en abscisse croissant
    Up=L[:pos[0]+1]
    Down=L[:pos[0]:-1]
    DownAbs=[]  #On prélève les abscisses de Down
    for elt in Down:
        DownAbs.append(elt[0])
    if Up[0][0] not in DownAbs: #et on regarde s'il y a dans Down des points ayant les abscisses extrêmes
        Down=[Up[0]]+Down       #sinon, on doit les ajouter pour pouvoir encadrer tous les points de la figure avec
    if Up[-1][0] not in DownAbs:
        Down.append(Up[-1])
    ListAbs=[]
    for elt in L:       #On prélève toutes les abscisses
        ListAbs.append(elt[0])
    ListAbs=trietdoublon(ListAbs)   #et on les tri en enlevant les doublons
    for elt in ListAbs:         #Pour chaque abscisse, on regarde la valeur en ordonné en haut et en bas du polygone
        valup=ordcourbe(elt,Up)     
        valdown=ordcourbe(elt,Down)
        Courbe.append([elt,valup-valdown])  #On obtient alors l'ordonné de la courbe finale pour cet abscisse
    return(Courbe)

        
def AireCourbe(courbe): #Aire entre une courbe faite avec des droites et l'axe des abscisses
    Aire=0
    for k in range(len(courbe)-1): 
        x,y,v,w=courbe[k][0],courbe[k][1],courbe[k+1][0],courbe[k+1][1]
        Aire+=(v-x)*min(y,w)+((v-x)*abs(y-w))/2
    return(Aire)


#--------------------------  Programme AirePolygone() principal  -------------------------


def AirePolygone(L):    #Calcule l'aire d'un polygone CONVEXE
    if len(L)==0:
        return(0)
    L=horloge(L)        #On place les points en horloge
    L=courbefinale(L)   #étape nécessaire pour le fonctionnement de courbefinale()
    Aire=AireCourbe(L)  #Puis on calcule l'aire entre cette courbe et l'axe des abscisses
    return(Aire)


#--------------------------  Programmes pour Distance()  ---------------------------------


def intersection(P1,P2,Q1,Q2):  #Retourne l'intersection entre deux segments défInis par 2 points chacunes, et False sinon
    m1=pente(P1,P2)
    if type(m1)!=bool:  #On vérifie que la pente soit non infinie
        p1=P1[1]-P1[0]*m1
    m2=pente(Q1,Q2)
    if type(m2)!=bool:
        p2=Q1[1]-Q1[0]*m2
    if type(m1)==bool and type(m2)==bool:   #Deux droites du type x=cst
        return(False)
    elif type(m1)!=bool and type(m2)!=bool: #Cas banal
        if m1==m2:  #Si 2 pentes identiques, alors les segments ne se croisent pas
            return(False)
        X=(p2-p1)/(m1-m2)
        Y=m1*X+p1
    elif type(m1)==bool and type(m2)!=bool: #Une des deux droites du type x=cst
        X=P1[0]
        Y=m2*X+p2
    elif type(m2)==bool and type(m1)!=bool:
        X=Q1[0]
        Y=m1*X+p1
    if max(min(Q1[0],Q2[0]),min(P1[0],P2[0]))<=X<=min(max(Q1[0],Q2[0]),max(P1[0],P2[0])) and max(min(Q1[1],Q2[1]),min(P1[1],P2[1]))<=Y<=min(max(Q1[1],Q2[1]),max(P1[1],P2[1])):
        return([X,Y])   #Condition pour vérifier que le point trouvé appartienne aux 2 segments
    else:
        return(False)

    
def pointdanspoly(I,L): #Cherche si un point appartient à un polygone
    X=I[0]
    Y=I[1]
    L=horloge(L)    #Même méthode initiale que ordcourbe()
    pos=pointdroitpos(L)
    Up=L[:pos[0]+1]
    if Up[0][0]>X or Up[-1][0]<X:
        return(False)
    Down=L[:pos[0]:-1]
    if Up[0][0] not in Down:
        Down=[Up[0]]+Down
    if Up[-1][0] not in Down:
        Down.append(Up[-1])
    valup=ordcourbe(X,Up)
    valdown=ordcourbe(X,Down)
    return(valup>=Y>=valdown)   #On vérifie si notre point est entre la courbe du haut et la courbe du bas


def interpoly(I,J): #Retourne le polygone CONVEXE formé par I inter J
    points=[]
    I=horloge(I)
    J=horloge(J)
    for k in range(-1,len(I)-1):
        I1=I[k]
        I2=I[k+1]   #On teste tous les 2 points consécutifs possibles (chaque côté)
        for n in range(-1,len(J)-1):    #Pareil pour J
            J1=J[n]
            J2=J[n+1]
            if I1[0]<I2[0]: #On place les points dans l'ordre des abscisses pour chaque côté pour intersection()
                A=I1
                B=I2
            else:
                B=I1
                A=I2
            if J1[0]<J2[0]:
                C=J1
                D=J2
            else:
                C=J1
                D=J2
            E=intersection(A,B,C,D) #On récupère (ou non) l'intersection des 2 côtés
            if E!=False:
                points.append(E)    #Si ce point existe, on l'ajoute à la liste des points
    for elt in I:               #On regarde maintenant les points d'un polygone appartenant à l'autre car dans ce cas,
        if pointdanspoly(elt,J):#ces points appartiennent à I inter J
            points.append(elt)
    for elt in J:               #Pareil pour l'autre
        if pointdanspoly(elt,I):
            points.append(elt)
    return(doublons(points))    #On préfèrera enlever les doublons plutôt que de trafiquer les "range" des boucles


#-------------------------  Programme Distance() principal  -----------------------------


def Distance(A,B):  #On mesure la distance entre 2 polygones
    AIB=interpoly(A,B)  #On prend A inter B
    AireA=AirePolygone(A)
    AireB=AirePolygone(B)
    AireAIB=AirePolygone(AIB)
    return(AireA+AireB-2*AireAIB)   #et on a d(A,B)=Aire(A union B - A inter B)= Aire(A) + Aire(B) - 2*Aire(A inter B)


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
    return(Distance(A,B))


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


            
