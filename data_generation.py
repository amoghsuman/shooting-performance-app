import pandas as pd
import numpy as np

def generate_shooting_data(num_shooters=100, num_sessions=10000):
    np.random.seed(42)

    # Shooter base profile
    shooter_profiles = pd.DataFrame({
        "Shooter_ID": np.arange(1, num_shooters + 1),
        "Age": np.random.randint(18, 50, size=num_shooters),
        "Handedness": np.random.choice(["Left", "Right"], size=num_shooters, p=[0.1, 0.9]),
        "Experience_Level": np.random.choice(["Beginner", "Intermediate", "Expert"], size=num_shooters, p=[0.3, 0.4, 0.3])
    })

    # Sessions
    sessions = pd.DataFrame({
        "Shooter_ID": np.random.choice(shooter_profiles["Shooter_ID"], size=num_sessions),
        "Fatigue_Level": np.random.randint(0, 100, size=num_sessions),
        "Score": np.random.randint(60, 100, size=num_sessions),
        "Grouping_Size (cm)": np.random.uniform(5, 20, size=num_sessions),
        "Reaction_Time (sec)": np.random.normal(1.5, 0.3, size=num_sessions),
        "Pressure_Level (0-100)": np.random.randint(0, 100, size=num_sessions),
        "Wind_Speed (km/h)": np.random.randint(0, 20, size=num_sessions),
        "Temperature (°C)": np.random.randint(10, 35, size=num_sessions),
        "Humidity (%)": np.random.randint(20, 80, size=num_sessions),
        "Lighting_Conditions": np.random.choice(["Poor", "Average", "Good"], size=num_sessions, p=[0.2, 0.5, 0.3]),
        "Altitude (m)": np.random.randint(100, 2000, size=num_sessions),
        "Training_Type": np.random.choice(["Static Shooting", "Moving Target", "Competitive Match"], size=num_sessions),
        "Number_of_Shots": np.random.randint(5, 50, size=num_sessions),
    })

    # Merge static shooter data
    df = sessions.merge(shooter_profiles, on="Shooter_ID")

    # Simulate realistic target variable
    # Lower fatigue, tighter grouping, more experience → higher accuracy
    experience_score = df["Experience_Level"].map({"Beginner": 0, "Intermediate": 1, "Expert": 2})
    lighting_score = df["Lighting_Conditions"].map({"Poor": -5, "Average": 0, "Good": 5})
    training_penalty = df["Training_Type"].map({"Static Shooting": 5, "Moving Target": -3, "Competitive Match": -5})

    df["Shot_Accuracy (%)"] = (
        85
        - 0.3 * df["Fatigue_Level"]
        - 0.7 * df["Grouping_Size (cm)"]
        - 0.5 * df["Reaction_Time (sec)"] * 10
        + 3 * experience_score
        + lighting_score
        + training_penalty
        + np.random.normal(0, 3, size=len(df))  # some randomness
    ).clip(0, 100).round(2)

    df.to_csv("shooting_data.csv", index=False)
    print("✅ Generated improved synthetic dataset with meaningful relationships.")

if __name__ == "__main__":
    generate_shooting_data()
