Step 1 - Import the Data(csv)
Step 2 - Clean data: Remove duplicates, etc.
Step 3 - Split data into Training/test Sets.
Step 4 - Create a model: Select an Algorithm.
Step 5 - Train the model.
Step 6 - Make Predictions.
Step 7 - Evaluate and Improve.

# Libraries
jupyter for the code
Anaconda
Numpy
Pandas
MatPlotLib
Scikit-Learn

# Steps
- Install Anaconda
- Type CMD( or Anaconda Terminal): $ jupyter notebook
    --> Copy and paste localhost:8888 + token
    --> Desktop / New / Python 3 /

- Download a Dataset from Kaggle > Create an account.
    - Chose Dataset Video Games Sales and download CSV
    - Type in jupyter notebook webpage:
      
        import pandas as pd
        df = pd.read_csv('vgsales.csv')
        df
        df.shape    // (16598, 11)
        df.describe()
        df.values
      
      
# jupyter commands
click on command line and go on lock mode with Esc
type h

- You can run all cells on jupyter.
- To delete cell, use Esc on cells and use 'dd' twice

---------Project 2 ------------
- Always follow the 7 steps:
Step 1 - Import the Data(csv)
Step 2 - Clean data: Remove duplicates, etc.
Step 3 - Split data into Training/test Sets.
Step 4 - Create a model: Select an Algorithm.
Step 5 - Train the model.
Step 6 - Make Predictions.
Step 7 - Evaluate and Improve.
  
1 - import http://bit.ly/music-csv
    - In jupyter console:
            import pandas as pd
            music_data = pd.read_csv('music.csv')
            music_data

2 & 3- Cleaning data: Remove duplicates, remove null values

            music_data.drop( press shift + tab )
            X = music_data.drop(columns=['genre']) // This doesn't remove data from the file. Capital X is a convention.
            X
            y = music_data['genre']

4 - Select an Algorithm
    - 'Decision Tree' Algo should be imported from sklearn.

        import pandas as pd
        from sklearn.tree import DecisionTreeClassifier
        music_data = pd.read_csv('music.csv')
        X = music_data.drop(columns=['genre'])
        y = music_data['genre']
        model = DecisionTreeClassifier()
        model.fit(X, y)
        predictions = model.predict( [[21, 1], [22, 0]])
        predictions

// result = array(['HipHop', 'Dance'], dtype=object)


------
- Calculate the accuracy of a model. We need to split our data dataset into two sets:
    - One for Training
    - One for Testing
    
!! In our code currently, we are passing the whole dataset for Training the model and two samples for Predictions
That is not good to calculate the accuracy
    
General Rule: Allocation 70 or 80% of our data to Training and the remaining for Testing.
Example here: predictions = model.predict( [[21, 1], [22, 0]])
    - Instead of passing two values for predictions, we pass the dataset for testing, and we compare them to actual values
    in the test set.

        import pandas as pd
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        music_data = pd.read_csv('music.csv')
        X = music_data.drop(columns=['genre'])
        y = music_data['genre']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)   # 20% for testing
        
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)   # expected values(y_test) and actual values(predictions)
        score
    
0.75
0.5

!! We need a lot of data to train our sample. To check if a picture is really of a cat, we need millions of
samples to train a Model

-------
Persisting Models
- Once we trained our Model once, we save it because there is no need to train it every time
- we use joblib.dump()
        
       import pandas as pd
       from sklearn.tree import DecisionTreeClassifier
       from sklearn.externals import joblib
        
       music_data = pd.read_csv('music.csv')
       X = music_data.drop(columns=['genre'])
       y = music_data['genre']
       model = DecisionTreeClassifier()
       model.fit(X, y)
       joblib.dump(model, 'music-recommender.joblib')

- Now load it, ad check predictions:

        import pandas as pd
        from sklearn.tree import DecisionTreeClassifier
        import joblib
        
        # music_data = pd.read_csv('music.csv')
        # X = music_data.drop(columns=['genre'])
        # y = music_data['genre']
        # model = DecisionTreeClassifier()
        # model.fit(X, y)
        model = joblib.load('music-recommender.joblib')
        predictions = model.predict( [[21, 1], [22, 0]])
        predictions

array(['HipHop', 'Dance'], dtype=object)

-------------
Visualizing a Decision Tree:

    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    
    music_data = pd.read_csv('music.csv')
    X = music_data.drop(columns=['genre'])
    y = music_data['genre']
    
    model = DecisionTreeClassifier()
    model.fit(X, y)
    
    tree.export_graphviz(model, out_file='music-recommender.dot', 
                         feature_names=['age', 'gender'], 
                         class_names=sorted(y.unique()),
                         label='all',
                         rounded=True,
                         filled=True)

- This will export a d0t file with a decision Tree diagrams that
you can visualize with dot plugins

