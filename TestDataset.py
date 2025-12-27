from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker
import numpy as np
import matplotlib.pyplot as plt

# 1. Télécharger un Atlas (Ceci définit les Nœuds de votre graphe)
# L'atlas MSDL est un atlas probabiliste couramment utilisé pour définir des régions fonctionnelles
atlas = datasets.fetch_atlas_msdl()
atlas_filename = atlas.maps

# 2. Télécharger des données fMRI (Exemple avec un dataset de développement, plus léger que HCP complet)
# Vous pouvez changer n_subjects pour avoir plus de données
data = datasets.fetch_development_fmri(n_subjects=1)
fmri_filename = data.func[0]

# 3. Extraction des séries temporelles (Création de la matrice X)
# Le masker va extraire le signal moyen dans chaque région de l'atlas
masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True, verbose=5)
time_series = masker.fit_transform(fmri_filename)

# 4. Adaptation au format du papier
# Nilearn sort une matrice (Temps x Régions)
# Le papier  définit X comme (Nœuds x Temps)
X = time_series.T 

print(f"Forme de la matrice X : {X.shape}")
# Résultat attendu : (39, 168) -> 39 régions (Nœuds), 168 points temporels (M observations)

x0 = X[0,:]
x1 = X[1,:]
x2 = X[2,:]

plt.subplot(3,1,1)
plt.plot(x0)
plt.subplot(3,1,2)
plt.plot(x1)
plt.subplot(3,1,3)
plt.plot(x2)
plt.show()