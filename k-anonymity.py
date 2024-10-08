import streamlit as st
import pandas as pd
import numpy as np

st.title('K-Anonymity with Generalization and Suppression')

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    # Read CSV into DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())
    
    columns = df.columns.tolist()
    
    # Select the value of k
    k = st.number_input("Select the value of k for k-anonymity", min_value=2, value=3)
    
    # Categorize columns
    st.write("Select Direct Identifiers (these will not be included in grouping):") # Modify to include handling of direct iden such as UMRN randomization?
    direct_identifiers = st.multiselect("Direct Identifiers", options=columns)
    
    remaining_columns = [col for col in columns if col not in direct_identifiers]
    
    st.write("Select Quasi-Identifiers:")
    quasi_identifiers = st.multiselect("Quasi-Identifiers", options=remaining_columns)
    
    remaining_columns = [col for col in remaining_columns if col not in quasi_identifiers]
    
    st.write(f"The remaining columns will be considered non-sensitive: {remaining_columns}")
    
    # Remove direct identifiers
    df_anonymized = df.drop(columns=direct_identifiers)
    
    # Generalization options for quasi-identifiers
    generalization_options = {}
    for qi in quasi_identifiers:
        st.write(f"Select generalization for {qi}:")
        if qi == 'Age at Colln':
            age_option = st.selectbox("Age Generalization", ['No generalization', '5-year intervals', '10-year intervals', '20-year intervals'], key=qi)
            generalization_options[qi] = age_option
        elif qi == 'Clinic Location':
            location_option = st.selectbox("Location Generalization", ['Exact Location', 'Region', 'Country'], key=qi)
            generalization_options[qi] = location_option
        else:
            st.write(f"No generalization options available for {qi}.")
    
    # Function to apply generalizations
    def apply_generalizations(df, generalization_options):
        df_generalized = df.copy()
        for qi, option in generalization_options.items():
            if qi == 'Age at Colln':
                if option == '5-year intervals':
                    df_generalized[qi] = df_generalized[qi].apply(lambda x: f"{(x//5)*5}-{((x//5)*5)+4}")
                elif option == '10-year intervals':
                    df_generalized[qi] = df_generalized[qi].apply(lambda x: f"{(x//10)*10}-{((x//10)*10)+9}")
                elif option == '20-year intervals':
                    df_generalized[qi] = df_generalized[qi].apply(lambda x: f"{(x//20)*20}-{((x//20)*20)+19}")
                else:
                    pass  # No generalization
            elif qi == 'Clinic Location': # Modify to include proper clinic location generalization
                if option == 'Region':
                    if 'Region' in df_generalized.columns:
                        df_generalized[qi] = df_generalized['Region']
                    else:
                        st.error("Column 'Region' not found in the dataset.")
                elif option == 'Country':
                    if 'Country' in df_generalized.columns:
                        df_generalized[qi] = df_generalized['Country']
                    else:
                        df_generalized[qi] = 'Country Name'  # Placeholder or actual mapping
                else:
                    pass  # Exact Location
            else:
                pass  # No generalization applied
        return df_generalized
    
    # Function to check and apply suppression
    def apply_suppression(df, quasi_identifiers, k):
        if not quasi_identifiers:
            st.error("No quasi-identifiers selected. Cannot apply k-anonymity.")
            return df, False, 0
        
        # Group by quasi-identifiers and get group sizes
        group_sizes = df.groupby(quasi_identifiers).size().reset_index(name='counts')
        
        # Merge group sizes back to the original dataframe
        df = df.merge(group_sizes, on=quasi_identifiers)
        
        # Suppress records in groups with counts less than k
        initial_record_count = len(df)
        df_suppressed = df[df['counts'] >= k].drop(columns=['counts'])
        suppressed_record_count = initial_record_count - len(df_suppressed)
        
        # Check if the resulting dataset is k-anonymous
        min_group_size = df_suppressed.groupby(quasi_identifiers).size().min()
        is_k_anonymous = min_group_size >= k if not df_suppressed.empty else False
        
        return df_suppressed, is_k_anonymous, suppressed_record_count
    
    # Apply generalizations
    df_anonymized = apply_generalizations(df_anonymized, generalization_options)
    
    # Apply suppression
    df_anonymized, is_k_anonymous, records_suppressed = apply_suppression(df_anonymized, quasi_identifiers, k)
    
    # Display results
    if is_k_anonymous:
        st.success(f"The dataset is {k}-anonymous after suppressing {records_suppressed} records.")
        st.write("Anonymized Data Preview:")
        st.dataframe(df_anonymized.head())
        # Offer download of anonymized dataset
        csv = df_anonymized.to_csv(index=False)
        st.download_button(label="Download Anonymized Data", data=csv, file_name='anonymized_data.csv', mime='text/csv')
    else:
        st.error(f"Unable to achieve {k}-anonymity with the current generalizations and suppression.")
