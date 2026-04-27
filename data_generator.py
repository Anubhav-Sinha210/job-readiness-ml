import pandas as pd
import numpy as np
import os

def generate_student_data(num_samples=2500, random_state=42):
    np.random.seed(random_state)
    
    # 1. Base Feature Generation
    cgpa = np.random.normal(7.5, 1.2, num_samples)
    cgpa = np.clip(cgpa, 0.0, 10.0)
    
    tech_skills = np.random.normal(65, 15, num_samples)
    tech_skills = np.clip(tech_skills, 0, 100)
    
    aptitude = np.random.normal(70, 15, num_samples)
    aptitude = np.clip(aptitude, 0, 100)
    
    comm_skills = np.random.normal(65, 18, num_samples)
    comm_skills = np.clip(comm_skills, 0, 100)
    
    internships = np.random.poisson(0.8, num_samples)
    internships = np.clip(internships, 0, 3)
    
    projects = np.random.poisson(1.5, num_samples)
    projects = np.clip(projects, 0, 5)
    
    certifications = np.random.poisson(1.2, num_samples)
    certifications = np.clip(certifications, 0, 5)
    
    coding_score = np.random.normal(60, 20, num_samples)
    coding_score = np.clip(coding_score, 0, 100)
    
    # 2. Logic to define 'Job Ready' (Target Variable)
    # Higher weights on CGPA, Tech Skills, and Coding Score
    hidden_score = (
        (cgpa / 10) * 20 + 
        (tech_skills / 100) * 25 + 
        (coding_score / 100) * 25 + 
        (aptitude / 100) * 10 + 
        (comm_skills / 100) * 10 + 
        (internships / 3) * 5 + 
        (projects / 5) * 5
    )
    
    # Add random noise
    hidden_score += np.random.normal(0, 5, num_samples)
    
    # Threshold for Job Ready
    target = (hidden_score > 65).astype(int)
    
    # 3. Create DataFrame
    data = {
        'CGPA': cgpa,
        'Technical_Skills': tech_skills,
        'Aptitude_Score': aptitude,
        'Communication_Skills': comm_skills,
        'Internships': internships,
        'Projects': projects,
        'Certifications': certifications,
        'Coding_Score': coding_score,
        'Job_Ready': target
    }
    
    df = pd.DataFrame(data)
    
    # 4. Introduce Missing Values (Data Preprocessing Requirement)
    for col in ['Technical_Skills', 'Aptitude_Score', 'Communication_Skills']:
        mask = np.random.rand(num_samples) < 0.03 # 3% missing data
        df.loc[mask, col] = np.nan
        
    # 5. Introduce Outliers (Data Preprocessing Requirement)
    outlier_idx = np.random.choice(df.index, size=20, replace=False)
    df.loc[outlier_idx, 'CGPA'] = np.random.uniform(11, 15, size=20)
    
    outlier_idx2 = np.random.choice(df.index, size=15, replace=False)
    df.loc[outlier_idx2, 'Coding_Score'] = np.random.uniform(110, 150, size=15)
    
    # 6. Add categorical variable for handling (One-Hot Encoding)
    streams = ['Computer Science', 'Information Technology', 'Electronics', 'Mechanical']
    df['Degree_Stream'] = np.random.choice(streams, size=num_samples, p=[0.4, 0.3, 0.2, 0.1])
    
    # Reorder columns
    cols = ['Degree_Stream', 'CGPA', 'Technical_Skills', 'Aptitude_Score', 
            'Communication_Skills', 'Internships', 'Projects', 'Certifications', 
            'Coding_Score', 'Job_Ready']
    df = df[cols]
    
    return df

if __name__ == '__main__':
    print("Generating synthetic student dataset...")
    df = generate_student_data(num_samples=2500)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    filepath = 'data/student_data.csv'
    df.to_csv(filepath, index=False)
    print(f"Dataset generated successfully with {len(df)} records at {filepath}")
    
    print("\nMissing Values Introduced:")
    print(df.isnull().sum())
    
    print("\nTarget Class Distribution (Imbalanced):")
    print(df['Job_Ready'].value_counts())
