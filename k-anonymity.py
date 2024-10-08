import streamlit as st
import pandas as pd
import numpy as np
from itertools import product

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
    st.write("Select Direct Identifiers (these will not be included in grouping):")
    direct_identifiers = st.multiselect("Direct Identifiers", options=columns)
    
    remaining_columns = [col for col in columns if col not in direct_identifiers]
    
    st.write("Select Quasi-Identifiers:")
    quasi_identifiers = st.multiselect("Quasi-Identifiers", options=remaining_columns)
    
    remaining_columns = [col for col in remaining_columns if col not in quasi_identifiers]
    
    st.write(f"The remaining columns will be considered non-sensitive: {remaining_columns}")
    
    # Remove direct identifiers
    df_anonymized = df.drop(columns=direct_identifiers)
    
    # Define generalization hierarchies for each quasi-identifier
    generalization_hierarchies = {}
    max_generalization_levels = {}
    for qi in quasi_identifiers:
        if qi == 'Age at Colln':
            # Levels: 0 - Exact age, 1 - 5-year intervals, 2 - 10-year intervals, 3 - 20-year intervals
            generalization_hierarchies[qi] = [0, 1, 2, 3]
            max_generalization_levels[qi] = 3
        elif qi == 'Clinic Location':
            # Levels: 0 - Exact location, 1 - Region, 2 - Country
            generalization_hierarchies[qi] = [0, 1, 2]
            max_generalization_levels[qi] = 2
        else:
            # For other QIs, define generalization levels as needed
            generalization_hierarchies[qi] = [0]  # No generalization levels defined
            max_generalization_levels[qi] = 0
    
    # Generate all combinations of generalization levels
    generalization_level_combinations = list(product(*[generalization_hierarchies[qi] for qi in quasi_identifiers]))
    
    # Function to apply generalizations based on levels
    def apply_generalizations(df, quasi_identifiers, generalization_levels):
        df_generalized = df.copy()
        for qi, level in zip(quasi_identifiers, generalization_levels):
            if qi == 'Age at Colln':
                if level == 1:
                    df_generalized[qi] = df_generalized[qi].apply(lambda x: f"{(x//5)*5}-{((x//5)*5)+4}")
                elif level == 2:
                    df_generalized[qi] = df_generalized[qi].apply(lambda x: f"{(x//10)*10}-{((x//10)*10)+9}")
                elif level == 3:
                    df_generalized[qi] = df_generalized[qi].apply(lambda x: f"{(x//20)*20}-{((x//20)*20)+19}")
                else:
                    pass  # No generalization
            elif qi == 'Clinic Location':
                if level == 1:
                    if 'Region' in df_generalized.columns:
                        df_generalized[qi] = df_generalized['Region']
                    else:
                        st.error("Column 'Region' not found in the dataset.")
                elif level == 2:
                    if 'Country' in df_generalized.columns:
                        df_generalized[qi] = df_generalized['Country']
                    else:
                        df_generalized[qi] = 'Country Name'  # Placeholder or actual mapping
                else:
                    pass  # Exact Location
            else:
                pass  # No generalization applied
        return df_generalized
    
    # Function to calculate Discernibility Metric
    def calculate_discernibility_metric(df, quasi_identifiers, suppressed_count):
        N = len(df) + suppressed_count  # Total number of records including suppressed
        dm = 0
        # Group by quasi-identifiers to find equivalence classes
        equivalence_classes = df.groupby(quasi_identifiers).size().reset_index(name='counts')
        for count in equivalence_classes['counts']:
            dm += count ** 2  # Sum of squares of equivalence class sizes
        # Add penalty for suppressed records
        dm += suppressed_count * N
        return dm
    
    # Function to apply suppression
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
    
    # List to store results
    results = []
    
    # Iterate over all generalization level combinations
    for levels in generalization_level_combinations:
        # Apply generalizations
        df_generalized = apply_generalizations(df_anonymized, quasi_identifiers, levels)
        
        # Apply suppression
        df_suppressed, is_k_anonymous, records_suppressed = apply_suppression(df_generalized, quasi_identifiers, k)
        
        # Calculate suppression percentage
        suppression_percentage = (records_suppressed / len(df)) * 100
        
        # Calculate Discernibility Metric
        dm = calculate_discernibility_metric(df_suppressed, quasi_identifiers, records_suppressed)
        
        # Store result if k-anonymity is achieved
        if is_k_anonymous:
            results.append({
                'generalization_levels': levels,
                'df': df_suppressed,
                'discernibility_metric': dm,
                'suppression_percentage': suppression_percentage,
                'records_suppressed': records_suppressed
            })
    
    # Check if any k-anonymous solutions were found
    if results:
        st.success(f"Found {len(results)} generalization options that achieve {k}-anonymity.")
        
        # Sort results by Discernibility Metric (lower is better)
        results = sorted(results, key=lambda x: x['discernibility_metric'])
        
        # Display options for user to select
        st.write("Generalization Options (sorted by Discernibility Metric):")
        option_list = []
        for idx, res in enumerate(results):
            gen_levels = dict(zip(quasi_identifiers, res['generalization_levels']))
            option_desc = f"Option {idx+1}: "
            for qi in quasi_identifiers:
                level = gen_levels[qi]
                option_desc += f"{qi} Level {level}, "
            option_desc += f"DM: {res['discernibility_metric']}, Suppression: {res['suppression_percentage']:.2f}%"
            option_list.append(option_desc)
        
        selected_option = st.selectbox("Select a generalization option to apply:", options=option_list)
        
        # Get the selected option index
        selected_index = option_list.index(selected_option)
        selected_result = results[selected_index]
        
        # Display detailed metrics
        st.write(f"Selected Option Metrics:")
        st.write(f"- Generalization Levels: {dict(zip(quasi_identifiers, selected_result['generalization_levels']))}")
        st.write(f"- Discernibility Metric: {selected_result['discernibility_metric']}")
        st.write(f"- Suppression Percentage: {selected_result['suppression_percentage']:.2f}%")
        st.write(f"- Records Suppressed: {selected_result['records_suppressed']}")
        
        # Display the anonymized data
        st.write("Anonymized Data Preview:")
        st.dataframe(selected_result['df'].head())
        
        # Offer download of anonymized dataset
        csv = selected_result['df'].to_csv(index=False)
        st.download_button(label="Download Anonymized Data", data=csv, file_name='anonymized_data.csv', mime='text/csv')
    else:
        st.error(f"Unable to achieve {k}-anonymity with the available generalizations and suppression.")
