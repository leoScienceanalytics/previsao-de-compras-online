def preprocessamento(file):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    dados = pd.read_csv(file)
    dados = dados.drop(['Month', 'Weekend'], axis=1)
    
    colunas_features = ['Administrative', 'Administrative_Duration', 'Informational', 
               'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates',	
               'PageValues', 'SpecialDay',	'OperatingSystems', 'Browser', 'Region', 'TrafficType']
    x= dados[colunas_features]
    y = dados['Revenue']
    
    x, x_test, y, y_test = train_test_split(x, y, test_size=0.15, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x, y, random_state=0)
    
    return x_train, y_train, x_test, y_test, x_val, y_val

def modelo(x_train, y_train, x_test, y_test, x_val, y_val):
    from sklearn.ensemble import RandomForestClassifier
    
    random_forest_model = RandomForestClassifier(random_state=0, max_depth=5)
    random_forest_model.fit(x_train, y_train)
    target_predict_validation = random_forest_model.predict(x_val)
    target_predict_test = random_forest_model.predict(x_test)
    
    return target_predict_test,target_predict_validation

def metrics(y_val,target_predict_validation, y_test, target_predict_test):
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
    
    confusion_matrix_validation = confusion_matrix(y_val, target_predict_validation)
    confusion_matrix_test = confusion_matrix(y_test, target_predict_test)
    
    accuracy = accuracy_score(y_test, target_predict_test)
    precision = precision_score(y_test, target_predict_test, average='weighted')
    recall = recall_score(y_test, target_predict_test, average='weighted')  
    f1 = f1_score(y_test, target_predict_test, average='weighted')
    
    target_names = ['False', 'True']
    report = classification_report(y_test, target_predict_test, target_names=target_names)
    
    
    return confusion_matrix_validation, confusion_matrix_test, accuracy, precision, recall, f1, report
