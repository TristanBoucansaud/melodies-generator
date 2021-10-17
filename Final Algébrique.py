import random
import pygame
import time
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

def doublons(L):
    Lfin=[]
    for elt in L:
        if elt not in Lfin:
            Lfin.append(elt)
    return Lfin
        

    
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




#--------------------------  MAIN PROGRAM Projecteur()  -------------------------------

AccPar=[[0, 3, 7], [0, 4, 7], [1, 4, 8], [1, 5, 8], [2, 5, 9], [2, 6, 9], [3, 6, 10], [3, 7, 10], [4, 7, 11], [4, 8, 11], [5, 8, 0], [5, 9, 0], [6, 9, 1], [6, 10, 1], [7, 10, 2], [7, 11, 2], [8, 11, 3], [8, 0, 3], [9, 0, 4], [9, 1, 4], [10, 1, 5], [10, 2, 5], [11, 2, 6], [11, 3, 6]]


def Projecteur (L):
    distmin=distance(L,AccPar[0])
    PlusProche=[]
    for k in range (24):
        a=distance(L,AccPar[k])
        if a==distmin:
            PlusProche.append(AccPar[k])
        if a<distmin:
            distmin=a
            PlusProche=[AccPar[k]]
    return PlusProche




def distancepoint(a,b):
    return(min((a-b)%12,(b-a)%12))



def EcrFact(k,n):

    L=[]
    for i in range(n):
        L.append(k%(n-i))
        k//(n-i)
    return L

def factorielle(n):
    P=1
    for i in range(2,n+1):
        P=P*i
    return P



def distance (L,AccPar):
    N=len(L)-3             #N+3 est la taille de la liste L et AccParDim
    distmin=-1
    for n1 in range(N+1):
        for n2 in range(N-n1+1):
            AccParDim=[AccPar[0]]*(n1+1)+[AccPar[1]]*(n2+1)+[AccPar[2]]*(N-n1-n2+1)
            for i in range(factorielle(N+3)):
                Position=EcrFact(i,N+3)
                APD=AccParDim[:]
                dist=0
                for k in range(N+3):
                    NotePar=APD.pop(Position[k])
                    dist+=distancepoint(NotePar,L[k])
                if distmin==-1 or dist<distmin:
                    distmin=dist
    return distmin       
                
                

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

#--------------------------------  Programme CreationMatPass()  -----------------------------

        
def EcrPuissN(k,N,n):
    #Ecrit un nombre k en puissance de N dans une liste contenant n caracteres (si N=2 le nombre sera ecrit en binaire)
    
    L=[]
    for i in range(n):
        L=[k%N]+L
        k=k//N
    return L


def CreationMatPass(L,n):
    #creation d'une matrice de passage(7*7) a partir de la liste des notes(de la partition) en considerant les n precedentes notes
    #L represente la liste de note en numerique

    MatPass=np.zeros([19]*(n+1))                     #initialisation de la matrice
    
    for i in range(n,len(L)):                      
        MatPass[tuple(L[i-n:i+1])]+=1               #on incremente de 1 dans la matrice l'endroit correspondant au n notes precedentes
        
    for k in range((19**n)):          
        Inter=EcrPuissN(k,19,n)                      #voir programme ci-dessus
        TotalLigne=0

        for j in range(19):                          #on somme le total de la ligne
            TotalLigne+=MatPass[tuple(Inter+[j])]

        if TotalLigne!=0:
            MatPass[tuple(Inter)]/=TotalLigne       #on divise chaque ligne par le total obtenu juste avant
        
        
    return(MatPass)


#----------------------------------  Programme CreationPart()  ------------------------------

    
def CreationPart(M,N,L,n):
    #creation d'une série de transitions de taille N(au maximum) à partir de la matrice de passage(noté M) créé dans le programme qui tient compte des n
    #precedentes notes ci-dessus
    #L represente la partition de base

    alea=random.randint(0,len(L)-1-n)                           #on prend de facon aleatoire un morceau de la partition originale
    Transitions=L[alea:alea+n]
    compteur18=0

    for i in range(N-n):
        while True:
            a=random.random()                                       #on choisit un nombre aleatoirement entre 0 et 1
            note=18
            for k in range(18):                                     #même principe que pour le premier programme:
                if a>M[tuple(Transitions[i:i+n]+[k])]:                #on regarde la partie correspondante à la n-ieme note precedente, (n-1)-ieme note,..., a la note precedente
                    a=a-M[tuple(Transitions[i:i+n]+[k])]
                elif note==18:
                    note=k
            if note==18:
                compteur18+=1
            else:
                compteur18=0
            if compteur18<4:
                break
                
        if note==18 and M[tuple(Transitions[i:i+n]+[18])]==0:     #petite securite dans le cas ou plus aucune note ne peut etre joué (probabilité partout de 0 dans la ligne corresondant)
            print("teminé en avance")
            return(Transitions)
            
        Transitions.append(note)                                  #on ajoute la note choisit aleatoirement plus haut a la partition

    
    return(Transitions)


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
    MatPas=CreationMatPass(L,1) # création de la matrice de passage à partir de la partition initiale
    Trans=CreationPart(MatPas,taille,L,1) # création de la série de transitions sur laquelle sera basée la nouvelle partition
    part=MakeMelody(Trans,acc1) # création de la nouvelle partition
    JOUER(part,rythme) # sortie sonore de la nouvelle partition


            
