from sklearn.neural_network import MLPClassifier
import json

with open('data/car_training.json', 'r') as jsonfile:
    data = json.load(jsonfile)
    X = data['features']
    y = data['labels']

    
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X, y)