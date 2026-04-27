import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Job Readiness Predictor", layout="wide", page_icon="🎓")

# Function to load models
@st.cache_resource
def load_models():
    base_path = 'models'
    with open(f'{base_path}/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open(f'{base_path}/selector.pkl', 'rb') as f:
        selector = pickle.load(f)
    with open(f'{base_path}/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'{base_path}/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return preprocessor, selector, model, feature_names

def main():
    st.title("🎓 Student Job Readiness Prediction System")
    st.markdown("""
    Welcome to the readiness prediction platform. Enter your academic and technical details below to find out your probability of being job-ready!
    """)
    
    try:
        preprocessor, selector, model, feature_names = load_models()
    except Exception as e:
        st.error(f"Models not found! Please train the model first. Error: {e}")
        return

    # Sidebar for inputs
    st.sidebar.header("Student Metrics")
    degree = st.sidebar.selectbox("Degree Stream", ['Computer Science', 'Information Technology', 'Electronics', 'Mechanical'])
    cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 7.5, 0.1)
    tech_skills = st.sidebar.slider("Technical Skills Score", 0, 100, 65)
    apt_skills = st.sidebar.slider("Aptitude Score", 0, 100, 70)
    comm_skills = st.sidebar.slider("Communication Skills", 0, 100, 65)
    coding_score = st.sidebar.slider("Coding Score", 0, 100, 60)
    
    internships = st.sidebar.number_input("Number of Internships", 0, 5, 0)
    projects = st.sidebar.number_input("Number of Projects", 0, 10, 2)
    certs = st.sidebar.number_input("Number of Certifications", 0, 10, 1)
    
    if st.sidebar.button("Predict Readiness"):
        with st.spinner("Analyzing profile..."):
            input_df = pd.DataFrame({
                'Degree_Stream': [degree],
                'CGPA': [cgpa],
                'Technical_Skills': [tech_skills],
                'Aptitude_Score': [apt_skills],
                'Communication_Skills': [comm_skills],
                'Internships': [internships],
                'Projects': [projects],
                'Certifications': [certs],
                'Coding_Score': [coding_score]
            })
            
            # Predict
            prep_data = preprocessor.transform(input_df)
            sel_data = selector.transform(pd.DataFrame(prep_data))
            
            prob = model.predict_proba(sel_data)[0][1]
            pred = model.predict(sel_data)[0]
            
            st.markdown("---")
            st.header("Prediction Results")
            cols = st.columns(2)
            
            with cols[0]:
                if pred == 1:
                    st.success("## 🎉 Job Ready!")
                    st.balloons()
                else:
                    st.error("## ⚠️ Not Ready Yet")
                
                st.metric(label="Readiness Probability", value=f"{prob*100:.1f} %")
                
            # Recommendations
            with cols[1]:
                st.subheader("Personalized Recommendations")
                if prob > 0.8:
                    st.write("Excellent profile. Focus on specialized domain knowledge or advanced interview prep.")
                elif prob > 0.5:
                    st.write("You are on track but could improve your edge.")
                    if coding_score < 75:
                        st.markdown("- **Improve Coding**: Try solving more algorithm challenges on LeetCode/HackerRank.")
                    if tech_skills < 75:
                        st.markdown("- **Tech Skills**: Build a capstone project to boost technical depth.")
                    if comm_skills < 70:
                        st.markdown("- **Communication**: Participate in seminars or soft-skills workshops.")
                else:
                    st.write("Needs significant improvement in core areas.")
                    st.markdown("- Focus heavily on building core Technical and Coding capacities.")
                    if cgpa < 7:
                        st.markdown("- Improve academic scores.")
                    if internships == 0 or projects < 2:
                        st.markdown("- **Practical Experience**: Get an internship or build at least 2 strong projects.")

    st.markdown("---")
    st.header("Model Evaluation & Insights")
    tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Model Evaluation Metrics", "ROC Curves", "Clustering (K-Means)"])
    
    with tab1:
        st.write("Understanding what traits contribute the most toward being 'Job Ready'.")
        if os.path.exists("outputs/feature_importance.png"):
            st.image("outputs/feature_importance.png", width=700)
            
    with tab2:
        if os.path.exists("outputs/model_metrics.csv"):
            metrics = pd.read_csv("outputs/model_metrics.csv")
            st.dataframe(metrics.style.highlight_max(axis=0))
            
    with tab3:
        st.write("Receiver Operating Characteristic Curves evaluating the tradeoff between True Positives and False Positives.")
        # Load any ROC curve dynamically
        roc_images = [f for f in os.listdir("outputs") if f.startswith("roc_")] if os.path.exists("outputs") else []
        if roc_images:
            cols = st.columns(min(3, len(roc_images)))
            for i, rock in enumerate(roc_images[:3]): # Display top 3
                with cols[i]:
                    st.image(f"outputs/{rock}", caption=rock.replace("roc_", "").replace(".png", "").upper())

    with tab4:
        st.write("Unsupervised learning insights: discovering student cohorts using K-Means clustering.")
        if os.path.exists("outputs/kmeans_eval.png"):
            st.image("outputs/kmeans_eval.png", width=800)

if __name__ == "__main__":
    main()
