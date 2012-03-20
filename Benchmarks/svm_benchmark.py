#!/usr/bin/env python

from sklearn import svm
import csv_io

def main():
    training, target = csv_io.read_data("../Data/train.csv")
    training = [x[1:] for x in training]
    target = [float(x) for x in target]
    test, throwaway = csv_io.read_data("../Data/test.csv")
    test = [x[1:] for x in test]

    svc = svm.SVC(probability=True)
    svc.fit(training, target)
    predicted_probs = svc.predict_proba(test)
    predicted_probs = [[min(max(x,0.001),0.999) for x in y] 
                       for y in predicted_probs]
    predicted_probs = [["%f" % x for x in y] for y in predicted_probs]
    csv_io.write_delimited_file("../Submissions/svm_benchmark.csv",
                                predicted_probs)

if __name__=="__main__":
    main()
