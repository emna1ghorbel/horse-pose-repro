@"

\# Rapport de qualité du prétraitement — Sprint 2



\## Volumétrie finale

| Source | Frames brutes | Frames gardées | Frames rejetées | Taux de rejet |

|---|---|---|---|---|

| TigDog | 15 658 | 13 350 | 2 308 | 14.7% |

| YouTube (extraction) | 1 844 | — | — | — |

| YouTube (filtrage détection YOLOv8m) | 1 844 | 800 | 1 044 | \~57% |

| YouTube (masques YOLOv8-seg) | 800 | 735 | 21 | 2.6% |

| \*\*Total entraînement (processed/final)\*\* | — | \*\*14 085\*\* | — | — |



\## Méthodologie

\- TigDog : masques natifs fournis par le dataset (segmentation automatique, Papazoglou \& Ferrari 2013)

\- YouTube : détection cheval (YOLOv8m) puis masques (YOLOv8-seg), seuil de confiance 0.5

\- Redimensionnement : 128x128, ratio d'aspect préservé + padding noir

\- Filtre qualité : rejet si couverture masque <2% ou >85%



\## Vérifications effectuées

\- Vérification visuelle manuelle (image + masque), TigDog et YouTube

\- Suite de tests automatisés (tests/test\_data\_pipeline.py) : 9/9 PASSED



\## Limites connues

\- Masques TigDog imprécis sur les contours fins (méthode de 2013)

\- 9 shots TigDog entièrement rejetés : 238, 240, 488, 562, 704, 709, 711, 714, 1057

\- Répartition train/test stricte : TigDog+YouTube (entraînement) / Weizmann (test, Sprint 7)

"@ | Out-File -Encoding utf8 ..\\docs\\preprocessing\_quality\_report.md

