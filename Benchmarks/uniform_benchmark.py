#!/usr/bin/env python

import csv_io
import numpy as np

def main():
    training, target = csv_io.read_data("../Data/train.csv")
    test, throwaway = csv_io.read_data("../Data/test.csv")
    
    n_test = len(test)
    n_target = len(set(target))

    predicted_probs = [[0.001 for x in range(n_target)] 
                       for y in range(n_test)]
    predicted_probs = [["%f" % x for x in y] for y in predicted_probs]
    csv_io.write_delimited_file("../Submissions/uniform_benchmark.csv",
                                predicted_probs)

if __name__=="__main__":
    main()
