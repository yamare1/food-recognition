import streamlit as st

def food_card(food_name, confidence, nutrition_data):
    """Display a food card with nutrition information"""
    with st.container():
        st.subheader(f"{food_name.capitalize()} ({confidence*100:.1f}%)")
        st.write(f"Calories: {nutrition_data['calories']:.0f} kcal")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Protein", f"{nutrition_data['protein_g']:.1f}g")
            st.metric("Fat", f"{nutrition_data['fat_g']:.1f}g")
        with col2:
            st.metric("Carbs", f"{nutrition_data['carbs_g']:.1f}g")
            if 'fiber_g' in nutrition_data:
                st.metric("Fiber", f"{nutrition_data['fiber_g']:.1f}g")
