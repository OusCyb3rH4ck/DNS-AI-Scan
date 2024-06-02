#!/usr/bin/env python3

import requests
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import argparse
import math
from collections import Counter
from termcolor import colored

def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

def extract_features(subdomain):
    subdomain = subdomain.lower()  # Ensure lowercase
    segments = subdomain.split('.')
    avg_segment_length = sum(len(segment) for segment in segments) / len(segments)
    
    features = {
        'length': len(subdomain),
        'num_hyphens': subdomain.count('-'),
        'num_numbers': sum(c.isdigit() for c in subdomain),
        'char_freq': sum(c.isalpha() for c in subdomain) / len(subdomain),
        'num_special_chars': sum(not c.isalnum() for c in subdomain) - subdomain.count('.'),
        'entropy': entropy(subdomain),
        'num_segments': len(segments),
        'avg_segment_length': avg_segment_length
    }
    return features

def fetch_subdomains(domain):
    response = requests.get(f'https://api.hackertarget.com/hostsearch/?q={domain}')
    if response.status_code == 200:
        return [line.split(',')[0] for line in response.text.splitlines()]
    else:
        return []

def main(domain):
    subdomains = fetch_subdomains(domain)
    if not subdomains:
        print(f'No subdomains found for {domain}')
        return
    
    features_list = [extract_features(sub) for sub in subdomains]
    feature_names = list(features_list[0].keys())
    X = np.array([[features[feature] for feature in feature_names] for features in features_list])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_scaled)
    predictions = model.predict(X_scaled)
    
    for subdomain, prediction in zip(subdomains, predictions):
        if prediction == -1:
            print(f'{colored("Anomaly detected:", "red")} {colored(subdomain, "yellow")}')
        else:
            print(f'{colored("Normal subdomain:", "green")} {colored(subdomain, "white")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Determine subdomain anomalies using Machine Learning (AI), specifically Isolation Forest.')
    parser.add_argument('-d', '--domain', required=True, help='Domain to analyze')
    args = parser.parse_args()
    main(args.domain)
