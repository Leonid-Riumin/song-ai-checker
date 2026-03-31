# AI Music Tracker

## Description

Ce projet est un prototype d’un outil capable d’analyser un fichier audio et d’estimer s’il a été créé par un humain ou généré par une intelligence artificielle.

Le projet s’inscrit dans une réflexion sur l’impact des technologies d’intelligence artificielle sur l’industrie musicale et sur le risque de standardisation du contenu musical.

## Problématique

Aujourd’hui, les intelligences artificielles peuvent produire de la musique très rapidement et en grande quantité. Cela peut entraîner une saturation du marché et rendre plus difficile l’émergence de nouveaux artistes.

La question principale est :
**Comment préserver la diversité musicale face à la production massive de musique générée par l’IA ?**

## Objectif du projet

L’objectif de ce projet est de proposer une solution technique partielle à ce problème, en développant un prototype capable de détecter les morceaux potentiellement générés par une IA.

## Fonctionnement

Le programme fonctionne en plusieurs étapes :

1. Chargement des fichiers audio (mp3 ou wav)
2. Extraction de caractéristiques audio (MFCC, spectre, etc.)
3. Entraînement d’un modèle de machine learning
4. Classification du fichier en :
   - Humain
   - IA
   - Incertain

## Technologies utilisées

- Python
- librosa (analyse audio)
- scikit-learn (machine learning)
- numpy


