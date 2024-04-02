#Bibliotecas
from function import preprocessamento, modelo, metrics

#Dataset
file = 'online_shoppers_intention.csv'

#Funções
x_train, y_train, x_test, y_test, x_val, y_val = preprocessamento(file)
target_predict_test, target_predict_validation = modelo(x_train, y_train, x_test, y_test, x_val, y_val)
confusion_matrix_validation, confusion_matrix_test, accuracy, precision, recall, f1, report = metrics(y_val,target_predict_validation, y_test, target_predict_test)

for i in target_predict_test, target_predict_validation:
    print(i)

for i in confusion_matrix_validation, confusion_matrix_test, accuracy, precision, recall, f1, report:
    print(i)

