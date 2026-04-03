

import random
import pandas as pd
import os

NUM_SAMPLES = 200_000   
OUT = "synthetic_complaints.csv"
os.makedirs("output", exist_ok=True)

complaint_topics = {
    "Water Supply Issues": [
        "No water supply since morning",
        "Water pressure very low",
        "Contaminated water from tap",
        "Irregular drinking water supply",
        "Water tanker delayed"
    ],
    "Electricity Problems": [
        "Frequent power cuts in locality",
        "Voltage fluctuations damaging devices",
        "Transformer not working",
        "Electricity outage for hours",
        "Street lights not working"
    ],
    "Road Infrastructure": [
        "Road is broken causing accidents",
        "Potholes everywhere on the street",
        "Sewer overflowing",
        "Footpath damaged heavily",
        "Construction work incomplete"
    ],
    "Public Transport": [
        "Bus always arrives late",
        "Metro service disrupted",
        "Auto drivers overcharging",
        "Crowded buses",
        "Poor condition of buses"
    ],
    "Healthcare Issues": [
        "Hospital staff not cooperating",
        "Long waiting time for OPD",
        "Ambulance not responding",
        "Medicines unavailable",
        "Poor cleanliness in ward"
    ],
    "Corruption & Admin Delays": [
        "Officials demanding bribe for approval",
        "Application pending for months",
        "No response from office despite reminders",
        "Corruption in issuing certificates",
        "Unnecessary delay in file processing"
    ]
}

noise_phrases = [
    "please resolve soon",
    "urgent attention needed",
    "same issue repeatedly",
    "requested many times",
    "situation getting worse",
    "please help"
]

states = ["Delhi", "Uttar Pradesh", "Maharashtra", "Bihar", "Punjab", "Rajasthan"]
districts = ["Central", "North", "South", "East", "West"]

def generate_complaint():
    topic = random.choice(list(complaint_topics.keys()))
    base = random.choice(complaint_topics[topic])
    if random.random() < 0.12:
        wrong_topic = random.choice(list(complaint_topics.keys()))
        wrong_sentence = random.choice(complaint_topics[wrong_topic])
        return f"{base}. {wrong_sentence}. {random.choice(noise_phrases)}"
    return f"{base}. {random.choice(noise_phrases)}"

def make_df(n):
    data = {"id": [], "complaint": [], "state": [], "district": [], "date": []}
    for i in range(n):
        data["id"].append(i+1)
        data["complaint"].append(generate_complaint())
        data["state"].append(random.choice(states))
        data["district"].append(random.choice(districts))
        data["date"].append(f"202{random.randint(0,2)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}")
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Generating synthetic dataset:", NUM_SAMPLES)
    df = make_df(NUM_SAMPLES)
    df.to_csv(OUT, index=False)
    print("Saved:", OUT)
