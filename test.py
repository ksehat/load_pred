from sklearn.preprocessing import LabelEncoder

ali = ['k', 'a', 'h', 'k', 'h', 'h']

le = LabelEncoder()
le.fit_transform(ali)