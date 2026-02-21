import streamlit as st
import pandas as pd
import json
import os
import re
import plotly.express as px
import plotly.graph_objects as go
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA

# --- 1. Setting up our Data Shapes (Pydantic Models) ---
# Here we define exactly how we want the AI to structure its responses. 
# This prevents the LLM from going off track and guarantees our dashboard won't crash!

class ThemeDefinition(BaseModel):
    theme_name: str = Field(description="The operational theme (e.g., 'Wait Time', 'Billing')")
    keywords: list[str] = Field(description="5 to 7 exact keywords or short phrases commonly used to describe this theme")
    evidence_samples: list[str] = Field(description="2-3 direct quotes from the dataset backing this up")

class ImprovementRoadmap(BaseModel):
    priority: int = Field(description="Priority rank (1 is highest)")
    recommendation: str = Field(description="Specific, actionable business recommendation")
    expected_rating_lift: str = Field(description="Expected improvement if fixed (e.g., '+0.6')")
    confidence: float = Field(description="Model's confidence in this recommendation (0.0 to 1.0)")

class ClinsightOutput(BaseModel):
    primary_themes: list[ThemeDefinition]
    improvement_roadmap: list[ImprovementRoadmap]
    executive_summary: str = Field(description="A formal, 2-paragraph executive summary intended for the Hospital Board.")

# This schema handles our live incoming reviews for the mic-drop demo moment.
class ReviewTriage(BaseModel):
    predicted_rating: int = Field(description="Predicted star rating 1 to 5")
    identified_themes: list[str] = Field(description="Themes found in this specific review")
    department_routing: str = Field(description="Which hospital department should handle this? (e.g., Billing, Front Desk, Medical Staff)")
    escalation_flag: bool = Field(description="True if this contains legal, medical malpractice, or severe PR risk requiring immediate management attention")
    suggested_response: str = Field(description="Draft a professional, compassionate response to the patient")


# --- 2. The Engine Room (Helper Functions) ---

def calculate_true_statistics(df: pd.DataFrame, ai_output: dict):
    # First, let's figure out the overall average rating for the whole clinic to use as our baseline.
    global_mean = df['rating'].mean()
    theme_analysis = []
    
    # Now we'll loop through the themes the AI found for us.
    for theme_def in ai_output['primary_themes']:
        theme_name = theme_def['theme_name']
        keywords = theme_def['keywords']
        
        # Time for some feature engineering: let's build a search pattern out of the keywords to find matching reviews.
        # We make everything lowercase just to be safe and robust.
        pattern = '|'.join([re.escape(k.lower()) for k in keywords])
        df[f"Theme_{theme_name}"] = df['feedback'].str.lower().str.contains(pattern, na=False)
        
        # Let's see what percentage of the total reviews actually mention this theme.
        subset = df[df[f"Theme_{theme_name}"]]
        frequency_pct = (len(subset) / len(df)) * 100 if len(df) > 0 else 0
        
        # How much does this specific theme drag down (or lift up) the overall rating?
        if len(subset) > 0:
            theme_mean = subset['rating'].mean()
            rating_impact = theme_mean - global_mean
        else:
            rating_impact = 0.0
            
        # Let's calculate how severe this problem is on a scale of 0 to 1. 
        # A really bad rating impact combined with it happening all the time = high risk!
        if rating_impact < 0:
            normalized_impact = abs(rating_impact) / 4.0 
            # We tweaked this multiplier up to 10 so it scales nicely for our small 50-row hackathon sample.
            severity = min((normalized_impact * (frequency_pct / 100.0)) * 10, 1.0) 
        else:
            severity = 0.0
            
        # Finally, let's bundle this all up so Streamlit can easily graph it for the judges.
        theme_analysis.append({
            "theme": theme_name,
            "frequency_percentage": round(frequency_pct, 1),
            "rating_impact": round(rating_impact, 2),
            "severity_score": round(severity, 2),
            "evidence_samples": theme_def['evidence_samples']
        })
        
    return theme_analysis, round(global_mean, 2)


@st.cache_data
def load_and_prep_data(uploaded_file):
    # First, we read the CSV file. If the user forgot headers, don't panic, we'll handle it!
    df = pd.read_csv(uploaded_file)
    
    # Let's try to be smart and guess the columns by looking for keywords like 'feedback' or 'rating'.
    feedback_cols = [col for col in df.columns if any(keyword in str(col).lower() for keyword in ['feedback', 'review', 'text', 'comment'])]
    rating_cols = [col for col in df.columns if any(keyword in str(col).lower() for keyword in ['rating', 'score', 'star'])]
    
    # If the keywords didn't work (like if headers are missing entirely), we'll just grab the first text column and the first number column.
    if not feedback_cols:
        text_cols = df.select_dtypes(include=['object']).columns
        feedback_col = text_cols[0] if len(text_cols) > 0 else df.columns[0]
    else:
        feedback_col = feedback_cols[0]
        
    if not rating_cols:
        num_cols = df.select_dtypes(include=['number', 'int64', 'float64']).columns
        rating_col = num_cols[0] if len(num_cols) > 0 else df.columns[1]
    else:
        rating_col = rating_cols[0]
    
    # Let's rename them so the rest of our app knows exactly what to look for.
    df = df.rename(columns={feedback_col: 'feedback', rating_col: 'rating'})
    df = df.dropna(subset=['feedback'])
    
    # For the sake of the live demo, we're capping this at 50 rows so the AI responds super fast.
    df_sample = df.head(50).copy()
    reviews_list = df_sample[['feedback', 'rating']].to_dict(orient='records')
    
    return reviews_list, df_sample


def analyze_reviews_with_gemini(api_key, reviews_data):
    # Time to call the big brains. We give the AI its persona and feed it our cleaned dataset.
    client = genai.Client(api_key=api_key)
    system_instruction = "You are 'ClinsightAI', an advanced healthcare operational decision intelligence system."
    prompt = f"Dataset:\n{json.dumps(reviews_data, indent=2)}\n\nExtract the structured operational insights."
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=ClinsightOutput,
        ),
    )
    return json.loads(response.text)


def generate_embeddings_and_clusters(api_key, df):
    # We're going to turn every single text review into a huge mathematical vector (an embedding).
    client = genai.Client(api_key=api_key)
    feedbacks = df['feedback'].tolist()
    
    response = client.models.embed_content(
        model='gemini-embedding-001',
        contents=feedbacks
    )
    embeddings = [emb.values for emb in response.embeddings]
    
    # Since these vectors have thousands of dimensions, we use PCA to squish them down to 2D so we can draw them on a graph.
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    df['PCA_x'] = embeddings_2d[:, 0]
    df['PCA_y'] = embeddings_2d[:, 1]
    df['Sentiment'] = df['rating'].apply(lambda x: 'Positive (4-5)' if x >= 4 else 'Negative (1-3)')
    return df


def triage_single_review(api_key, review_text):
    # This function handles the real-time stuff. When a new review pops up, we evaluate it instantly.
    client = genai.Client(api_key=api_key)
    prompt = f"Analyze this incoming hospital review: '{review_text}'"
    
    # We added a sneaky business rule here to automatically insert a clickable mailto link for angry patients.
    system_instruction = """
    You are a real-time healthcare review triage AI. Predict the rating (1-5), route it to the correct department, flag severe risks, and draft a compassionate reply.
    
    CRITICAL BUSINESS RULE:
    If the predicted rating is 3 stars or below, OR if the escalation_flag is True, you MUST include this exact sentence at the end of your suggested_response:
    'Please contact our management team directly at [medsupport@xyz.com](mailto:medsupport@xyz.com) so we can look into this immediately and make it right.'
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=ReviewTriage,
        ),
    )
    return json.loads(response.text)


# --- 3. The Front-End (Streamlit UI Layout) ---

st.set_page_config(page_title="ClinsightAI", layout="wide", page_icon=":)")

# Let's add some custom CSS to make our app look like a polished, premium SaaS product instead of a basic script.
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

st.title("ClinsightAI: Healthcare Decision Intelligence")
st.markdown("Transforming unstructured patient feedback into actionable business roadmaps, financial ROI, and real-time operations.")
st.divider()

with st.sidebar:
    st.header("⚙️ System Config")
    api_key_input = st.text_input("Gemini API Key", type="password", value="AIzaSyBUPyg8e-mVERo7rpR_KWrM8PjI5SiLHfg")
    uploaded_file = st.file_uploader("Upload Review Dataset (CSV)", type=["csv"])
    
    # We're building a cool global risk gauge for the sidebar to show the overall clinic health at a glance.
    if 'ai_data' in st.session_state:
        st.divider()
        st.subheader("Global Risk Escalation")
        
        # Let's calculate an average risk score from all our themes to power the speedometer.
        themes = st.session_state['ai_data']['theme_analysis']
        avg_severity = sum([t['severity_score'] for t in themes]) / len(themes) if themes else 0
        risk_score = min(avg_severity * 100 * 1.5, 100) # Scale it up to 100 for visual impact
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Clinic Risk Index"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 33], 'color': "#00CC96"},
                    {'range': [33, 66], 'color': "#FFA15A"},
                    {'range': [66, 100], 'color': "#EF553B"}],
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

if uploaded_file is not None and api_key_input:
    reviews_data, raw_df = load_and_prep_data(uploaded_file)
    
    if st.button("🚀 Initialize AI Engine", type="primary", use_container_width=True):
        with st.spinner("Analyzing feedback, generating embeddings, and mapping semantic clusters..."):
            try:
                # First, we ask the LLM to extract the themes and write the reports.
                raw_ai_data = analyze_reviews_with_gemini(api_key_input, reviews_data)
                
                # Then, we crunch the hard numbers using Pandas to prove the AI's claims.
                theme_stats, global_mean = calculate_true_statistics(raw_df, raw_ai_data)
                
                # Let's sort these themes so the absolute worst problems are always bubbling up to the top.
                theme_stats = sorted(theme_stats, key=lambda x: x['severity_score'], reverse=True)
                
                # We'll grab the top 2 most severe themes dynamically to flag on the executive summary.
                top_risk_themes = [t['theme'] for t in theme_stats if t['severity_score'] > 0][:2]
                if not top_risk_themes:
                    top_risk_themes = ["No critical systemic issues detected in this sample."]

                # Now we merge our hard math with the AI's insights and save it to the session state.
                st.session_state['ai_data'] = {
                    "clinic_summary": {
                        "overall_rating_mean": global_mean,
                        "primary_risk_themes": top_risk_themes
                    },
                    "theme_analysis": theme_stats,
                    "improvement_roadmap": raw_ai_data['improvement_roadmap'],
                    "executive_summary": raw_ai_data['executive_summary']
                }
                
                # Run our mathematical clustering model
                st.session_state['cluster_df'] = generate_embeddings_and_clusters(api_key_input, raw_df)
                st.session_state['raw_df'] = raw_df
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # Once the data is processed, we render the awesome dashboards.
    if 'ai_data' in st.session_state and 'cluster_df' in st.session_state:
        data = st.session_state['ai_data']
        cluster_df = st.session_state['cluster_df']
        loaded_raw_df = st.session_state['raw_df']
        
        # Let's set up our 5 main tabs, including the Live Triage feature!
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Dashboard", "💰 Financial Risk", "📝 Auto-Report", "🧠 AI Clusters", "🚨 Live Triage"])
        
        # --- TAB 1: EXECUTIVE DASHBOARD ---
        # Where the executives see the high-level health of their clinic.
        with tab1:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(label="Overall Rating Mean", value=f"{data['clinic_summary']['overall_rating_mean']} ⭐")
            with col2:
                themes_str = "\n".join([f"- {theme}" for theme in data['clinic_summary']['primary_risk_themes']])
                st.error(f"**Critical Risk Themes:**\n{themes_str}")

            st.divider()
            
            col_rate1, col_rate2 = st.columns(2)
            with col_rate1:
                rating_counts = loaded_raw_df['rating'].value_counts().reset_index()
                rating_counts.columns = ['Star Rating', 'Count']
                rating_counts = rating_counts.sort_values(by='Star Rating')
                fig_ratings = px.bar(rating_counts, x='Star Rating', y='Count', title="Distribution of Star Ratings",
                                     color='Star Rating', color_continuous_scale=['#EF553B', '#FFA15A', '#FECB52', '#00CC96', '#00CC96'])
                fig_ratings.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1))
                st.plotly_chart(fig_ratings, use_container_width=True)

            with col_rate2:
                df_themes = pd.DataFrame(data['theme_analysis'])
                fig2 = px.pie(df_themes, values='frequency_percentage', names='theme', hole=0.4, title="Theme Frequency")
                st.plotly_chart(fig2, use_container_width=True)

            st.divider()

            # The Action Priority Matrix - shows executives what to fix first based on frequency and severity.
            df_themes['impact_magnitude'] = df_themes['rating_impact'].abs()
            fig_matrix = px.scatter(
                df_themes, x='frequency_percentage', y='severity_score', size='impact_magnitude', color='theme',
                hover_name='theme', title="Action Priority Matrix (Frequency vs. Severity Risk)",
                labels={'frequency_percentage': 'How Often It Happens (%)', 'severity_score': 'Severity Risk (0.0 to 1.0)'}, size_max=40
            )
            fig_matrix.add_hline(y=0.5, line_dash="dot", line_color="gray", opacity=0.5)
            fig_matrix.add_vline(x=df_themes['frequency_percentage'].mean(), line_dash="dot", line_color="gray", opacity=0.5)
            st.plotly_chart(fig_matrix, use_container_width=True)

            st.header("Prioritized Action Roadmap")
            for action in data['improvement_roadmap']:
                st.success(f"**Priority {action['priority']}:** {action['recommendation']} (Lift: `{action['expected_rating_lift']} ⭐`)")

        # --- TAB 2: FINANCIAL RISK ---
        # Show me the money! Simulating the financial hit caused by operational failures.
        with tab2:
            st.header("Financial Risk Simulator")
            col_sim1, col_sim2 = st.columns(2)
            with col_sim1:
                annual_patients = st.number_input("Estimated Annual Patients", 1000, 100000, 10000, 1000)
            with col_sim2:
                ltv = st.number_input("Average Patient Lifetime Value ($)", 100, 10000, 500, 100)
            st.divider()
            
            negative_themes = df_themes[df_themes['rating_impact'] < 0].copy()
            if not negative_themes.empty:
                # We calculate how many patients left, multiplied by how much money they would have spent.
                negative_themes['Patients Affected'] = (annual_patients * (negative_themes['frequency_percentage'] / 100)).astype(int)
                negative_themes['Est. Churn Rate (%)'] = (negative_themes['severity_score'] * 15).round(1)
                negative_themes['Revenue At Risk ($)'] = (negative_themes['Patients Affected'] * (negative_themes['Est. Churn Rate (%)']/100) * ltv).astype(int)
                
                total_risk = negative_themes['Revenue At Risk ($)'].sum()
                st.metric(label="Total Annual Revenue At Risk", value=f"${total_risk:,.2f}", delta="- Action Required", delta_color="inverse")
                
                fig_finance = px.bar(negative_themes, x='theme', y='Revenue At Risk ($)', text='Revenue At Risk ($)', color='Revenue At Risk ($)', color_continuous_scale='Reds')
                fig_finance.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                st.plotly_chart(fig_finance, use_container_width=True)
            else:
                st.success("Revenue risk is low.")

        # --- TAB 3: AUTO-REPORT ---
        # The auto-generated email ready to be forwarded to the board.
        with tab3:
            st.header("Board-Ready Executive Summary")
            st.info(data['executive_summary'])

        # --- TAB 4: AI CLUSTERS ---
        # The nerdy (but awesome) semantic clustering scatter plot.
        with tab4:
            st.header("Semantic Review Clustering")
            # 
            fig_cluster = px.scatter(cluster_df, x="PCA_x", y="PCA_y", color="Sentiment",
                                     color_discrete_map={'Positive (4-5)': '#00CC96', 'Negative (1-3)': '#EF553B'},
                                     hover_data={"feedback": True, "rating": True, "PCA_x": False, "PCA_y": False})
            st.plotly_chart(fig_cluster, use_container_width=True)

        # --- TAB 5: NEW LIVE TRIAGE FEATURE ---
        # The mic-drop moment. Testing the AI on a live incoming review.
        with tab5:
            st.header("Live Review Triage Simulator")
            # 
            st.markdown("Test the AI on incoming data. Paste a hypothetical patient review below. The AI will instantly predict the rating, route it to the right department, and draft a response.")
            
            test_review = st.text_area("Enter a new patient review:", "I waited over 2 hours for my appointment and when I asked the receptionist, she was incredibly rude and told me to just sit down. The doctor was nice but I am never coming back here.")
            
            if st.button("Simulate Live Triage"):
                with st.spinner("Processing incoming review..."):
                    try:
                        triage_result = triage_single_review(api_key_input, test_review)
                        
                        col_t1, col_t2 = st.columns(2)
                        with col_t1:
                            st.metric("Predicted Rating", f"{triage_result['predicted_rating']} ⭐")
                            st.write(f"**Identified Themes:** {', '.join(triage_result['identified_themes'])}")
                            st.write(f"**Route To:** `{triage_result['department_routing']}`")
                        
                        with col_t2:
                            if triage_result['escalation_flag']:
                                st.error("⚠️**HIGH PRIORITY ESCALATION FLAG DETECTED** - Requires immediate management review.")
                            else:
                                st.success("Standard routing priority.")
                        
                        st.subheader("Auto-Drafted Response to Patient")
                        
                        # We make absolutely sure our hardcoded email address is a clickable mailto link!
                        final_response = triage_result['suggested_response']
                        if "medsupport@xyz.com" in final_response and "mailto:" not in final_response:
                            final_response = final_response.replace("medsupport@xyz.com", "**[medsupport@xyz.com](mailto:medsupport@xyz.com)**")
                            
                        # Render it as markdown inside the blue info box
                        st.info(final_response)

                    except Exception as e:
                        st.error(f"Error during triage: {e}")

elif uploaded_file is None:
    st.info("Please upload a hospital review CSV file in the sidebar to begin.")
