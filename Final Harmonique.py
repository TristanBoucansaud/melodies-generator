import random
import pygame
import time
import matplotlib.pyplot as plt
from math import *
import numpy as np

pygame.mixer.init() # fichiers son
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

repnotes=[z25DO,z26REb,z27RE,z28MIb,z29MI,z30FA,z31SOLb,z32SOL,z33LAb,z34LA,z35SIb,z36SI] # liste des fichiers son
nonbémol=[0,2,4,5,7,9,11]#liste des notes bécarres
bémol=[0,1,3,5,6,8,10]#liste des notes bémoles

#listes des altérations respectives aux 24 tonalités
doM=[]
lam=[z33LAb]
solM=[z31SOLb]
mim=[z31SOLb,z28MIb]
réM=[z31SOLb,z26REb]
sim=[z31SOLb,z26REb,z35SIb]
laM=[z31SOLb,z26REb,z33LAb]
solbm=[z31SOLb,z26REb,z33LAb]
miM=[z31SOLb,z26REb,z33LAb,z28MIb]
rébm=[z31SOLb,z26REb,z33LAb,z28MIb]
siM=[z31SOLb,z26REb,z33LAb,z28MIb,z35SIb]
labm=[z26REb,z33LAb,z28MIb,z35SIb]
solbM=[z35SIb,z28MIb,z33LAb,z26REb,z31SOLb]
mibm=[z35SIb,z28MIb,z33LAb,z31SOLb]
rébM=[z35SIb,z28MIb,z33LAb,z26REb,z31SOLb]
sibm=[z35SIb,z28MIb,z26REb,z31SOLb]
labM=[z35SIb,z28MIb,z33LAb,z26REb]
fam=[z35SIb,z33LAb,z26REb]
mibM=[z35SIb,z28MIb,z33LAb]
dom=[z28MIb,z33LAb]
sibM=[z35SIb,z28MIb]
solm=[z35SIb,z28MIb,z31SOLb]
faM=[z35SIb]
rém=[z35SIb,z26REb]

listeTransition=[['P'],['L'],['R'],['P','L'],['P','R'],['L','R'],['L','P'],['R','L'],['R','P'],['R','P','R'],['L','P','L'],['L','R','L'],['P','R','P'],
                 ['P','L','P'],['R','L','R'],['R','P','L'],['L','R','P'],['P','L','R'],[]] # liste des transitions du Tonnetz




#________________________________________Algorithmes de lecture de la partition initiale sous format Midi________________________________________#


def conversionEngNb(L): # traduit les notes en chiffres
    eng=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    for k in range(len(L)):
        c=0
        while L[k][:-1]!=eng[c]:
            c+=1
        L[k]=c
    return(tri(L))


def conversionTransNb(L): # transforme les transitions en chiffres selon leur emplacement respectif dans listeTransition
    M=[]
    for elt in L:
        for ф in range(len(listeTransition)):
            if elt==listeTransition[ф]:
                M.append(ф)
    return(M)


def InitPartition(partition): # enlève les lignes inutiles pour la suite au texte partition
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


def lire(L): # récupère les informations utiles: les notes et leur temps d'apparition
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


def regroupementparligne(partition): # rescence les accords de la partition initiale avec les lettres
    L=[[]]
    for k in range(len(partition)):
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
    return (L)


def lectureFichier(fichier): # transforme une partition sous format '.txt' en liste d'accords
    texte=open(fichier,'r') # obtention de fichier avec Ruby
    midi=texte.readlines()  
    midi=InitPartition(midi)
    midi=regroupementparligne(midi)
    midi=triAccords(midi)
    mod12=[]                
    for elt in midi:
        mod12.append(conversionEngNb(elt))
    mod12=triPerfect(mod12)
    acc1=mod12[0]
    trans=[]
    for k in range(len(mod12)-1):
        trans.append(pattern(mod12[k],mod12[k+1]))
    return(conversionTransNb(trans),acc1)




#________________________________________Algorithmes des fonctions de transitions du Tonnetz________________________________________#


def P(acc): # représente la transition P
    if (acc[1]-acc[0])%12==4:
        acc[1]=(acc[1]-1)%12
    else:
        acc[1]=(acc[1]+1)%12
    return(triAcc(acc))


def L(acc): # représente la transition L
    if (acc[1]-acc[0])%12==4:
        acc[0]=(acc[0]+11)%12
    else:
        acc[2]=(acc[2]-11)%12
    return(triAcc(acc))


def R (acc): # représente la transition R
    if (acc[1]-acc[0])%12==4:
        acc[2]=(acc[2]-10)%12
    else:
        acc[0]=(acc[0]+10)%12
    return(triAcc(acc))


def composedouble(f,g): # effectue la composition de 2 fonctions
    def fonc(x):
        return(g(f(x)))
    return(fonc)


def composetriple(f,g,h): # effectue la composition de 3 fonctions
    def fonce(x):
        return(h(g(f(x))))
    return(fonce)


def identity(t): # simmule la fonction identité
    return(t)


listeFonction=[P,L,R,composedouble(P,L),composedouble(P,R),composedouble(L,R),composedouble(L,P),composedouble(R,L),composedouble(R,P),
               composetriple(R,P,R),composetriple(L,P,L),composetriple(L,R,L),composetriple(P,R,P),composetriple(P,L,P),composetriple(R,L,R),composetriple(R,P,L),
               composetriple(L,R,P),composetriple(P,L,R),identity]


def pattern(debut,fin): # recherche la transition qu'il y a entre deux accords(3 pas max)
    for k in range(len(listeFonction)): 
        if listeFonction[k](debut[:])==fin:
            return(listeTransition[k])
    return([])




#________________________________________Algorithmes de sélection des accords________________________________________#


def tri(L): # trie par insertion
    for k in range(len(L)-1):
        for i in range(len(L)-1):
            if L[i]>L[i+1]:
                L[i],L[i+1]=L[i+1],L[i]
    return (L)


def triAcc(L): # trie les accords parfaits dans le bon ordre 
    F=tri(L[:])
    a,b,c=F[0],F[1],F[2]
    if a+7==b+4==c or a+7==b+3==c:
        return([a,b,c])
    elif b+7==c+4==a+12 or b+7==c+3==a+12:
        return([b,c,a])
    else:
        return([c,a,b])


def triAccords(L):# PROGRAMME DE SUBSTITUTION QU'IL FAUT AMÉLIORER
    F=[]
    for elt in L:
        if len(elt)==3: # MODIFICATION ICI SUR  LA CONDITION (len(elt)>=3 au lieu de len(elt)==3)
            F.append(elt)
    return(F)


def repérageaccordparfait(elt):# reconnaît si l'accord d'entrée est déjà un accord parfait
    i=0
    for k in range(len(elt)):
        if (elt[k]+7)%12==(elt[(k+1)%3]+3)%12==elt[(k+2)%3] or (elt[k]+7)%12==(elt[(k+1)%3]+4)%12==elt[(k+2)%3]:
            i+=1
    if i==3:
        return(True)


def triPerfect(L): #PROGRAMME DE SUBSTITUTION QU'IL FAUT AMÉLIORER
    F=[]
    for elt in L:
        if repérageaccordparfait(elt):
            F.append(triAcc(elt))
        else:
            F.append(accordparfait(elt))
    return(F)




#________________________________________Approche harmonique________________________________________#


def indice(L,note): # renvoie l'indice du terme note dans la liste L
    for j in range(len(L)):
        if L[j]==note:
            return(j)

def présence(L,elt):
    c=0
    for i in range(len(L)):
        if L[i]==elt:
            c+=1
    if c>=1:
        return(True)


def reconnaissancebis(M,L): # vérifie si 2 notes parmi celles de l'accord initial font partie des 3 premières notes de la série
    r=0
    for i in range(len(L)):
        for j in range(len(M)):
            if L[i]==M[j]:
                r+=1
    if r==2:
        return(True)


def reconnaissance (M,L): # fait la reconnaissance de l'accord de 5 notes qui contient les 3 notes de l'accord initial dont 2 font partie des 3 premières notes
    r=0
    for i in range(len(L)):
        for j in range(len(M)):
            if L[i]==M[j]:
                r+=1
    if r==3 and reconnaissancebis(M[0:3],L):
        return(True)

    
def enlèvebémol (L): # supprime les bémols potentiellement présents dans un accord
    Lmod=L[:]
    for i in range (len(L)):
        for j in range (1,7):
            if L[i]==bémol[j]:
                Lmod[i]=nonbémol[j]
    return (Lmod)


def tonalité (): # choisit aléatoirement une tonalité (détermine les bémols à la clé) parmi les 24 possibles
    p=random.random()
    T=[]
    if p<=0.042:
        T=solbM
    elif 0.042<p<=0.084:
        T=lam
    elif 0.084<p<=0.126:
        T=solM
    elif 0.126<p<=0.168:
        T=mim
    elif 0.168<p<=0.21:
        T=réM
    elif 0.21<p<=0.252:
        T=sim
    elif 0.252<p<=0.294:
        T=laM
    elif 0.294<p<=0.336:
        T=solbm
    elif 0.336<p<=0.378:
        T=miM
    elif 0.378<p<=0.42:
        T=rébm
    elif 0.42<p<=0.462:
        T=siM
    elif 0.462<p<=0.504:
        T=labm
    elif 0.504<p<=0.546:
        T=rébM
    elif 0.546<p<=0.588:
        T=sibm
    elif 0.588<p<=0.63:
        T=labM
    elif 0.63<p<=0.672:
        T=fam
    elif 0.672<p<=0.714:
        T=mibM
    elif 0.714<p<=0.756:
        T=dom
    elif 0.756<p<=0.798:
        T=sibM
    elif 0.798<p<=0.84:
        T=solm
    elif 0.84<p<=0.882:
        T=faM
    elif 0.882<p<=0.924:
        T=rém
    elif 0.924<p<=0.966:
        T=mibm
    else:
        T=doM
    return(T)


def accordparfait(L): # transforme un accord quelconque de 3 notes en un accord parfait selon les règles d'harmonie
    L0=[(enlèvebémol(L))[0]]
    L1=[(enlèvebémol(L))[1]] # suppression des bémols potentiels pour n'avoir provisoirement que des bécarres
    L2=[(enlèvebémol(L))[2]]
    T=tonalité() # choix aléatoire la tonalité (détermination des bémols à la clé)
    for k in range(1,7):
        L0.append((3*nonbémol)[indice(nonbémol,(enlèvebémol(L))[0])+k*2])
        L1.append((3*nonbémol)[indice(nonbémol,(enlèvebémol(L))[1])+k*2])   # construction des 3 séries de tierces de 5 notes bécarres 
        L2.append((3*nonbémol)[indice(nonbémol,(enlèvebémol(L))[2])+k*2])
    for i in range(len(L)):
        for j in range (1,7):
            if L[i]==bémol[j]:
                if présence(L0,nonbémol[j]): # si présence de bémols dans l'accord initial: substitution des bécarres par les bémols correspondants dans les 3 séries de tierces
                    L0[indice(L0,nonbémol[j])]=bémol[j]
                if présence(L1,nonbémol[j]):
                    L1[indice(L1,nonbémol[j])]=bémol[j]
                if présence(L2,nonbémol[j]):
                    L2[indice(L2,nonbémol[j])]=bémol[j]                
    if T!=doM:# si la tonalité est différente de doM (aucune altération à la clé): substitution des bécarres restants par les bémols correspondants présents à la clé
        for t in range (len(T)):
            for j in range(1,7):
                for i in range (len(L)):
                    if T[t]==bémol[j] and L[i]!=nonbémol[j]:
                        if présence(L0,nonbémol[j]):
                            L0[indice(L0,nonbémol[j])]=bémol[j]
                        if présence(L1,nonbémol[j]):
                            L1[indice(L1,nonbémol[j])]=bémol[j]
                        if présence(L2,nonbémol[j]):  
                            L2[indice(L2,nonbémol[j])]=bémol[j]
    if reconnaissance(L0,L): 
        return(L0[0:3])
    elif reconnaissance(L1,L): # sélection  de l'accord de 5 notes qui contient les 3 notes de l'accord initial dont 2 font partie des 3 premières notes
        return(L1[0:3])        # troncation de l'accord de 5 notes pour ne garder que les 3 premières notes constituant l'accord fondamental
    else:
        return(L2[0:3])




#________________________________________Algorithmes de création de la nouvelle partition à partir de la matrice de passage________________________________________#

        
def EcrPuissN(k,N,n): # écrit un nombre k en puissance de N dans une liste contenant n caractères
    L=[]
    for i in range(n):
        L=[k%N]+L
        k=k//N
    return L


def CreationMatPass(L,n): # crée une matrice de passage(7*7) à partir de la liste numérique des notes de la partition initiale en considerant les n precedentes notes
    MatPass=np.zeros([19]*(n+1)) # initialisation de la matrice
    for i in range(n,len(L)):                      
        MatPass[tuple(L[i-n:i+1])]+=1 # incrémentation de 1 dans la matrice à l'endroit correspondant au n notes precedentes   
    for k in range((19**n)):          
        Inter=EcrPuissN(k,19,n)                     
        TotalLigne=0
        for j in range(19): 
            TotalLigne+=MatPass[tuple(Inter+[j])] # sommation de la totalité de la ligne
        if TotalLigne!=0:
            MatPass[tuple(Inter)]/=TotalLigne # division de chaque ligne par le total obtenu précédemment 
    return(MatPass)

    
def CreationPart(M,N,L,n): # crée une série de transitions de taille N à partir de la partition initiale L et de la matrice de passage M du programme précédent
    alea=random.randint(0,len(L)-1-n) # sélection aléatoire d'une partie de la partition originale
    Transitions=L[alea:alea+n]
    for i in range(N-n):
        a=random.random() # choix aléatoire d'un nombre entre 0 et 1
        note=18
        for k in range(18):                                 
            if a>M[tuple(Transitions[i:i+n]+[k])]: # prise en compte des parties correspondant à la n-ième note précédente, à la (n-1)-ième note,..., à la dernière note
                a=a-M[tuple(Transitions[i:i+n]+[k])]
            elif note==18:
                note=k
        if note==18 and M[tuple(Transitions[i:i+n]+[18])]==0: # prévention du cas où plus aucune note ne peut être jouée (probabilité de 0 partout dans la ligne correspondante)
            print("teminé en avance")
            return(Transitions)     
        Transitions.append(note) # ajout de la note choisie aléatoirement précédemment
    return(Transitions)




#________________________________________Algorithmes de production sonore________________________________________#


def MakeMelody(Transitions,acc1): # synthétise la nouvelle partition à partir de la première note et de la série de transitions
    Partition=[acc1]               
    for k in range(1,len(Transitions)):
        Partition.append(listeFonction[Transitions[k]](Partition[k-1]))
    return(Partition)

        
def JOUER(partition,rythme): # joue la partition créée
    for accord in partition:
        repnotes[accord[0]].play(0,rythme)
        repnotes[accord[1]].play(0,rythme)
        repnotes[accord[2]].play(0,rythme)
        while pygame.mixer.get_busy():
            pass




#________________________________________Programme principal________________________________________#


def MAIN(fichier,taille,rythme): # joue une mélodie aléatoire à partir d'une partition sous format '.txt'
    L,acc1=lectureFichier(fichier) # exploitation du fichier '.txt'
    MatPas=CreationMatPass(L,1) # création de la matrice de passage à partir de la partition initiale
    Trans=CreationPart(MatPas,taille,L,1) # création de la série de transitions sur laquelle sera basée la nouvelle partition
    part=MakeMelody(Trans,acc1) # création de la nouvelle partition
    JOUER(part,rythme) # sortie sonore de la nouvelle partition


            
