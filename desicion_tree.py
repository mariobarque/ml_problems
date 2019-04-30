from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


def convert(data):
    number = LabelEncoder()
    data['Alt'] = number.fit_transform(data.Alt)
    data['Bar'] = number.fit_transform(data.Bar)
    data['Fri'] = number.fit_transform(data.Fri)
    data['Hun'] = number.fit_transform(data.Hun)
    data['Pat'] = number.fit_transform(data.Pat)
    data['Price'] = number.fit_transform(data.Price)
    data['Rain'] = number.fit_transform(data.Rain)
    data['Res'] = number.fit_transform(data.Res)
    data['Type'] = number.fit_transform(data.Type)
    data['Est'] = number.fit_transform(data.Est)
    return data


df = pd.read_csv('data.csv')
df = convert(df)

model = DecisionTreeClassifier()
le = LabelEncoder()

x = df.drop('Est', axis=1)
y = df['Est']

model.fit(x, y)

y_predict = model.predict(x)

print(y_predict)
