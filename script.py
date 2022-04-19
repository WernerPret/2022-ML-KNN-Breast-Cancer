import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1
breast_cancer_data = load_breast_cancer()
# for k,v in breast_cancer_data.items():
#   print(v)

# 2
data = breast_cancer_data.data[0]
feats = breast_cancer_data.feature_names

data_feature_names = pd.DataFrame({
  "Feature Names": feats,
  "Data Vals": data 
})
# print(data_feature_names)

# 3
target = breast_cancer_data.target
target_names = breast_cancer_data.target_names
# print(target_names)

# 5 + 6
training_set, validation_set, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=100)

# 7
# print(len(training_set))
# print(len(training_labels))

# 9
classifier = KNeighborsClassifier(3)

# 10
classifier.fit(training_set, training_labels)

score = classifier.score(validation_set, validation_labels)

x = []
scores = []
for i in range(100):
  classifier = KNeighborsClassifier(i+1)
  classifier.fit(training_set, training_labels)
  score = classifier.score(validation_set, validation_labels)
  scores.append(score)
  x.append(i+1)

plt.plot(x, scores)
plt.show()


  











