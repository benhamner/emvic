#!/usr/bin/env python

from sklearn.ensemble import RandomForestClassifier
import csv_io
import math

def main():
    training, target = csv_io.read_data("../Data/train.csv")
    training = [x[1:] for x in training]
    target = [float(x) for x in target]
    test, throwaway = csv_io.read_data("../Data/test.csv")
    test = [x[1:] for x in test]

    rf = RandomForestClassifier(n_estimators=100, min_split=2)
    rf.fit(training, target)
    predicted_probs = rf.predict_proba(test)
    predicted_probs = [[min(max(x,0.001),0.999) for x in y] 
                       for y in predicted_probs]
    predicted_probs = [["%f" % x for x in y] for y in predicted_probs]
    csv_io.write_delimited_file("../Submissions/rf_benchmark.csv",
                                predicted_probs)

if __name__=="__main__":
    main()
