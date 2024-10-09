import streamlit as st
import pandas as pd
import numpy as np
from itertools import product
import hashlib
import random
import string

st.title('De-identification Tool: K-Anonymity')

# Function to generate a random encryption key
def generate_encryption_key():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=16))  # 16 characters long key

# Function to hash a value with an encryption key
def hash_value_with_key(value, key):
    value_with_key = f'{value}_{key}'
    return hashlib.sha256(value_with_key.encode()).hexdigest()

# Function to generate a pseudonym with the column name
def generate_pseudonym(index, column_name):
    return f'{column_name}_{chr(65 + index)}'

# Function to apply hashing to a column
def apply_hashing(df, column, encryption_key):
    if encryption_key:
        df[column] = df[column].apply(lambda val: hash_value_with_key(val, encryption_key))
    return df

# Function to apply pseudonymization to a column
def apply_pseudonymization(df, column, encryption_key):
    if encryption_key:
        pseudonym_mapping = {}
        unique_values = df[column].unique()
        for index, value in enumerate(unique_values):
            hashed_value = hash_value_with_key(value, encryption_key)
            pseudonym = generate_pseudonym(index, column)
            pseudonym_mapping[hashed_value] = pseudonym
        df[column] = df[column].apply(lambda val: pseudonym_mapping[hash_value_with_key(val, encryption_key)])
    return df

# Function to securely remove the encryption key
def secure_remove_key(encryption_key):
    encryption_key = '0' * len(encryption_key)  # Overwrite with zeros
    encryption_key = None  # Clear from memory
    del encryption_key  # Delete the variable

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
                #if 'Country' in df_generalized.columns: # Add if when Country column will be available
                df_generalized[qi] = 'Country Name'
                #else:
                #    st.error("Column 'Country' not found in the dataset.")
            else:
                pass  # Exact Clinic Location
        elif qi == 'DOM':
            df_generalized[qi] = pd.to_datetime(df_generalized[qi], errors='coerce')
            if level == 1:
                df_generalized[qi] = df_generalized[qi].dt.to_period('M').astype(str)  # 'YYYY-MM'
            elif level == 2:
                df_generalized[qi] = df_generalized[qi].dt.year.astype(str)  # 'YYYY'
            elif level == 3:
                years = df_generalized[qi].dt.year
                bins = range(int(years.min() // 10 * 10), int(years.max() // 10 * 10) + 10, 10)
                labels = [f"{b}-{b+9}" for b in bins[:-1]]
                df_generalized[qi] = pd.cut(years, bins=bins, labels=labels, right=False)
            else:
                df_generalized[qi] = df_generalized[qi].dt.strftime('%Y-%m-%d')  # Exact Date
        else:
            pass  # No generalization applied
    return df_generalized

# Function to calculate Discernibility Metric
def calculate_discernibility_metric(df, quasi_identifiers, suppressed_count):
    N = len(df) + suppressed_count  # Total number of records including suppressed
    dm = 0
    # Group by quasi-identifiers to find equivalence classes
    equivalence_classes = df.groupby(quasi_identifiers, observed=False).size().reset_index(name='counts')
    for count in equivalence_classes['counts']:
        dm += count ** 2  # Sum of squares of equivalence class sizes
    # Add penalty for suppressed records
    dm += suppressed_count * N
    return dm

# Function to apply suppression
def apply_suppression(df, quasi_identifiers, k):
    # Group by quasi-identifiers and get group sizes
    group_sizes = df.groupby(quasi_identifiers, observed=False).size().reset_index(name='counts')

    # Merge group sizes back to the original dataframe
    df = df.merge(group_sizes, on=quasi_identifiers)

    # Suppress records in groups with counts less than k
    initial_record_count = len(df)
    df_suppressed = df[df['counts'] >= k].drop(columns=['counts'])
    suppressed_record_count = initial_record_count - len(df_suppressed)

    # Check if the resulting dataset is k-anonymous
    if not df_suppressed.empty:
        min_group_size = df_suppressed.groupby(quasi_identifiers, observed=False).size().min()
        is_k_anonymous = min_group_size >= k
    else:
        is_k_anonymous = False

    return df_suppressed, is_k_anonymous, suppressed_record_count

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    # Read CSV into DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    columns = df.columns.tolist()

    # Select Direct Identifiers
    direct_identifiers = st.multiselect("Select Direct Identifiers:", options=columns)

    # For each direct identifier, select the de-identification technique
    direct_identifier_methods = {}
    if direct_identifiers:
        st.write("Select De-identification Technique for Each Direct Identifier:")
        for di in direct_identifiers:
            method = st.selectbox(f"De-identification method for {di}", options=['Drop', 'Hash', 'Pseudonymize'], key=di)
            direct_identifier_methods[di] = method

    # Create a copy of the DataFrame for anonymization
    df_anonymized = df.copy()

    # Generate encryption key for hashing and pseudonymization
    encryption_key = None
    if 'Hash' in direct_identifier_methods.values() or 'Pseudonymize' in direct_identifier_methods.values():
        encryption_key = generate_encryption_key()

    # Apply de-identification techniques to direct identifiers
    for di, method in direct_identifier_methods.items():
        if method == 'Drop':
            df_anonymized.drop(columns=[di], inplace=True)
        elif method == 'Hash':
            df_anonymized = apply_hashing(df_anonymized, di, encryption_key)
        elif method == 'Pseudonymize':
            df_anonymized = apply_pseudonymization(df_anonymized, di, encryption_key)

    # Exclude direct identifiers (except 'Performing Lab' and 'Clinic Name') from quasi-identifier options
    direct_identifiers_excl_special = [col for col in direct_identifiers if col not in ['Performing Lab', 'Clinic Name']]
    quasi_identifier_options = [col for col in df_anonymized.columns if col not in direct_identifiers_excl_special]

    # Select Quasi-Identifiers
    quasi_identifiers = st.multiselect("Select Quasi-Identifiers:", options=quasi_identifier_options)

    remaining_columns = [col for col in df_anonymized.columns if col not in quasi_identifiers and col not in direct_identifiers_excl_special]

    st.info(f"The remaining columns will be considered non-sensitive: {remaining_columns}")

    # Select the value of k
    k = st.number_input("Select the value of k for k-anonymity", min_value=2, value=3)

    if not quasi_identifiers:
        st.warning("No quasi-identifiers selected.")
    else:
        # Define generalization hierarchies for each quasi-identifier
        generalization_hierarchies = {}
        max_generalization_levels = {}
        generalization_descriptions = {}

        for qi in quasi_identifiers:
            if qi == 'Age at Colln':
                # Levels: 0 - Exact age, 1 - 5-year intervals, 2 - 10-year intervals, 3 - 20-year intervals
                generalization_hierarchies[qi] = [0, 1, 2, 3]
                max_generalization_levels[qi] = 3
                generalization_descriptions[qi] = {
                    0: 'Exact Age',
                    1: '5-year intervals',
                    2: '10-year intervals',
                    3: '20-year intervals'
                }
            elif qi == 'Clinic Location':
                # Levels: 0 - Clinic Location, 1 - Region, 2 - Country
                generalization_hierarchies[qi] = [0, 1, 2]
                max_generalization_levels[qi] = 2
                generalization_descriptions[qi] = {
                    0: 'Exact Location',
                    1: 'Region',
                    2: 'Country'
                }
            elif qi == 'DOM':
                # Levels: 0 - Exact Date, 1 - Month and Year, 2 - Year, 3 - Year Intervals (e.g., 1980-1989)
                generalization_hierarchies[qi] = [0, 1, 2, 3]
                max_generalization_levels[qi] = 3
                generalization_descriptions[qi] = {
                    0: 'Exact Date',
                    1: 'Month and Year',
                    2: 'Year',
                    3: 'Decade Intervals'
                }
            else:
                # For other QIs, define generalization levels as needed
                generalization_hierarchies[qi] = [0]  # No generalization levels defined
                max_generalization_levels[qi] = 0
                generalization_descriptions[qi] = {
                    0: 'No Generalization'
                }

        # Generate all combinations of generalization levels
        generalization_level_combinations = list(product(*[generalization_hierarchies[qi] for qi in quasi_identifiers]))

        # List to store results
        results = []

        # Iterate over all generalization level combinations
        for levels in generalization_level_combinations:
            # Apply generalizations
            df_generalized = apply_generalizations(df_anonymized, quasi_identifiers, levels)

            # Apply suppression
            df_suppressed, is_k_anonymous, records_suppressed = apply_suppression(df_generalized, quasi_identifiers, k)

            # Calculate suppression percentage
            suppression_percentage = (records_suppressed / len(df_anonymized)) * 100

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

            # Create a DataFrame to display options
            options_data = []
            for res in results:
                gen_levels = dict(zip(quasi_identifiers, res['generalization_levels']))
                gen_info = {}
                for qi in quasi_identifiers:
                    level = gen_levels[qi]
                    description = generalization_descriptions[qi][level]
                    gen_info[qi] = f"Level {level} ({description})"
                options_data.append({
                    **gen_info,
                    'DM': res['discernibility_metric'],
                    'Suppression (%)': f"{res['suppression_percentage']:.2f}"
                })

            options_df = pd.DataFrame(options_data)
            # Adjust index to start from 1
            options_df.index = options_df.index + 1

            # Display the options DataFrame
            st.write("Generalization Options (sorted by Discernibility Metric):")
            st.dataframe(options_df)

            # Allow user to select an option using the DataFrame index
            max_index = len(options_df)
            selected_index = st.number_input(
                "Enter the index of the selected option:",
                min_value=1,
                max_value=max_index,
                step=1
            )

            # Retrieve the selected result
            selected_result = results[int(selected_index) - 1]  # Subtract 1 to get the correct index in the results list

            # Display detailed metrics
            st.info(f"Generalization Levels:")
            gen_levels = dict(zip(quasi_identifiers, selected_result['generalization_levels']))
            for qi in quasi_identifiers:
                level = gen_levels[qi]
                description = generalization_descriptions[qi][level]
                st.write(f"- {qi}: Level {level} ({description})")
            st.info(f'''
            Selected Option Metrics:  
            - Discernibility Metric: {selected_result['discernibility_metric']}  
            - Suppression Percentage: {selected_result['suppression_percentage']:.2f}%  
            - Records Suppressed: {selected_result['records_suppressed']}
            ''')
           

            # Display the anonymized data
            st.write("Anonymized Data Preview:")
            st.dataframe(selected_result['df'].head())

            # Offer download of anonymized dataset
            csv = selected_result['df'].to_csv(index=False)
            st.download_button(label="Download Anonymized Data", data=csv, file_name='anonymized_data.csv', mime='text/csv')

        else:
            st.error(f"Unable to achieve {k}-anonymity with the available generalizations and suppression.")

    # Securely remove the encryption key after processing
    if encryption_key:
        secure_remove_key(encryption_key)

else:
    st.info("Please upload a CSV file to proceed.")
