import random
import time
import requests
API_URL = "https://early-risk-alert-mobile-api.onrender.com/api/dashboard/overview"
TENANT_ID = "demo"

PATIENTS = [
    {
        "patient_id": "p100",
        "heart_rate": 78,
        "systolic_bp": 122,
        "diastolic_bp": 80,
        "spo2": 98,
    },
    {
        "patient_id": "p101",
        "heart_rate": 92,
        "systolic_bp": 138,
        "diastolic_bp": 88,
        "spo2": 96,
    },
    {
        "patient_id": "p102",
        "heart_rate": 105,
        "systolic_bp": 145,
        "diastolic_bp": 92,
        "spo2": 95,
    },
    {
        "patient_id": "p103",
        "heart_rate": 84,
        "systolic_bp": 128,
        "diastolic_bp": 82,
        "spo2": 97,
    },
]

def clamp(value, low, high):
    return max(low, min(high, value))

def vary_patient(p):
    return {
        "tenant_id": TENANT_ID,
        "patient_id": p["patient_id"],
        "heart_rate": clamp(p["heart_rate"] + random.randint(-3, 3), 55, 135),
        "systolic_bp": clamp(p["systolic_bp"] + random.randint(-4, 4), 95, 180),
        "diastolic_bp": clamp(p["diastolic_bp"] + random.randint(-3, 3), 60, 120),
        "spo2": clamp(p["spo2"] + random.randint(-1, 1), 88, 100),
    }

def send_vitals(payload):
    try:
        response = requests.get(API_URL, json=payload, timeout=15)
        print(f"Sent {payload['patient_id']} | {payload} | status={response.status_code}")
    except Exception as e:
        print(f"Error sending {payload['patient_id']}: {e}")

def main():
    print("Starting demo feeder...")
    while True:
        for patient in PATIENTS:
            payload = vary_patient(patient)
            send_vitals(payload)
            time.sleep(2)   # small pause between patients

        print("Cycle complete. Waiting 15 seconds...\n")
        time.sleep(15)

if __name__ == "__main__":
    main()
    main()
