import numpy as np 
import os
import os.path
from collections import defaultdict
from experiments_analyzer import load_outputs
from scipy.stats import shapiro, normaltest
import json

def to_json(dictionary, file_name):
    with open(file_name, "w") as json_file:
        json.dump(dictionary, json_file, default=int)

def from_json(file_name):
    with open(file_name, "r") as json_file:
        return json.load(json_file)

def count_faults(golden_output, fault_output):
    fault_output_copy = np.copy(fault_output)
    nan_count = 0
    nan_map = np.isnan(fault_output)
    if nan_map.any():
        nan_count = np.sum(nan_map)
        fault_output_copy = np.nan_to_num(fault_output, nan=1E100)
    diff = golden_output - fault_output_copy
    diff_map = np.abs(diff) > 1E-3
    faults_count = np.sum(diff_map)
    non_nan_faults_map = np.logical_and(diff_map, np.logical_not(nan_map))
    values_diff = []
    if np.sum(non_nan_faults_map) > 0:
        indexes = np.vstack(np.where(non_nan_faults_map)).T
        for i in range(indexes.shape[0]):
            b, c, y, x = indexes[i, :]
            values_diff.append(diff[b, c, y, x])
    return faults_count, nan_count, np.array(values_diff)
    


def get_data_paths():
    root_path = "/home/aleto/experiments_data/convolution_S2"
    file_names = os.listdir(root_path)
    file_names = [file_name for file_name in file_names if "mode" not in file_name]
    return [os.path.join(root_path, file_name) for file_name in file_names]

def main():
    paths = get_data_paths()
    counters = {}
    diffs = {}
    nans = {}
    for path in paths:
        golden_output, fault_outputs = load_outputs([path])
        golden_output = golden_output[0]
        fault_outputs = fault_outputs[0]
        for fault_output in fault_outputs:
            faults, nan, values_diff = count_faults(golden_output, fault_output)
            if faults not in counters:
                counters[int(faults)] = 0
            counters[int(faults)] += faults
            if faults not in nans:
                nans[int(faults)] = 0
            nans[int(faults)] += nan
            if faults not in diffs:
                diffs[int(faults)] = np.zeros((0))
            diffs[int(faults)] = np.concatenate((diffs[faults], values_diff))
    dd = np.zeros((0))
    for faults in sorted(counters.keys()):
        nan_p = nans[faults] / counters[faults]
        print("Faults: {}".format(faults))
        print("NaN percentage: {:.5f}".format(nan_p))
        print("Min value: {:.5f}".format(diffs[faults].min()))
        print("Max value: {:.5f}".format(diffs[faults].max()))
        stat, p = shapiro(diffs[faults])
        print("Gaussian test (Saphiro-Wilk): {}".format(True if p > 0.05 else False))
        stat, p = normaltest(diffs[faults])
        print("Gaussian test (D'agostino): {}".format(True if p > 0.05 else False))
        print("\n")
        dd = np.concatenate((diffs[faults], dd))
    #np.save("counters", counters)
    #np.save("diffs", diffs)
    #np.save("nans", nans)
    to_json(counters, "counters.json")
    to_json(nans, "nans.json")
    #to_json(diffs, "diffs.json")
    np.save("global_diffs", dd)
    #print(np.sum(-1.0 <= dd <= 1.0) / dd.size())
    #print(sum(nans.values()) / sum(counters.values()))
    

if __name__ == "__main__":
    #main()
    global_diff = np.load("global_diffs.npy")
    nans = from_json("nans.json")
    counters = from_json("counters.json")
    less_than_1 = global_diff <= 1.0
    more_than_minus_1 = -1 <= global_diff
    between_interval = np.logical_and(less_than_1, more_than_minus_1)
    print(np.sum(between_interval) / sum(counters.values()))
    print(sum(nans.values()) / sum(counters.values()))
    print()
    