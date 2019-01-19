import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
train_data = pd.read_table('news_train.txt', header=None)
data_test = pd.read_table('news_test.txt', header=None)
labels= train_data[0].unique()
y=train_data[0].replace(labels,range(len(labels))).values
text=train_data[2].values
text_test=data_test[1].values
vectorizer=TfidfVectorizer().fit(text)
x = vectorizer.transform(text)
x_test = vectorizer.transform(text_test)
clf = SVC(kernel='linear')
clf.fit(x,y)
res = clf.predict(x_test)
res = pd.DataFrame(res).replace(range(len(labels)),labels)
res.to_csv('output.txt', header=False, index=False)